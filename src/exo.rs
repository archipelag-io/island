//! Exo multi-Island cluster execution.
//!
//! Unlike pipeline/expert (which do layer/expert compute in-process), this runtime
//! delegates all sharding to a local **Exo** process. Every node in the cluster:
//!   1. starts `exo` in the cluster's namespace, with its own data dir and ports,
//!   2. waits until Exo's `/membership` reports the whole cluster as reachable,
//!   3. (primary only) asks Exo to place the model across all nodes,
//!   4. signals `ready` to the coordinator.
//!
//! The **primary** node (position 0) then receives the prompt, calls Exo's
//! OpenAI-compatible `/v1/chat/completions` (streaming), and relays each token to
//! the coordinator over NATS. Workers serve their shard silently until `stop`.
//!
//! Written against Exo's source as of Nov 2026 (zenoh networking, `/membership`,
//! `/place_instance`). Constraints of current Exo that callers must respect:
//! - **LAN only.** Exo discovers peers by link-local IPv6 multicast and manual
//!   peers are disabled, so every node of a cluster must share a network segment.
//! - **Same build.** Nodes only join peers running the same Exo build (and MLX).
//! - **No fault tolerance.** A node dropping mid-generation fails the request.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::select;
use tokio::sync::{watch, RwLock};
use tracing::{error, info, warn};

use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// How long to wait for all peers to become reachable.
const CLUSTER_FORM_TIMEOUT: Duration = Duration::from_secs(150);
/// How long the primary keeps retrying `/place_instance`. Right after the cluster
/// forms, the master may not have every node's memory yet and refuses placement.
const PLACEMENT_TIMEOUT: Duration = Duration::from_secs(60);
/// How long the primary waits for the placed model to become ready to serve
/// (download + load on every node). Exo answers 503 + Retry-After until then.
const MODEL_READY_TIMEOUT: Duration = Duration::from_secs(900);
/// Longest silence tolerated on the token stream. Exo sends an SSE keep-alive
/// comment every 10s, so this only trips when the cluster is really stuck.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
/// Grace period between SIGTERM and SIGKILL when stopping the Exo process group.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(15);

/// Per-cluster ports are derived from the group id so concurrent Exo jobs on one
/// Island don't collide. The UDP discovery port must be identical on every node
/// of a cluster, so it stays at Exo's default (52413).
const API_PORT_BASE: u16 = 53000;
const ZENOH_PORT_BASE: u16 = 53500;
const PORT_SLOTS: u32 = 500;

/// Exo configuration sent by the coordinator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExoConfig {
    pub group_id: String,
    /// "primary" | "worker"
    pub role: String,
    #[serde(default)]
    pub is_primary: bool,
    /// Shared namespace that isolates this cluster from all others.
    pub namespace: String,
    /// Exo model id, e.g. `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.
    pub model: String,
    /// Number of nodes the cluster should contain before we consider it formed.
    pub node_count: u32,
    #[serde(default)]
    pub peers: Vec<String>,
    pub subjects: ExoSubjects,
}

/// NATS subjects for exo cluster coordination.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExoSubjects {
    pub control: String,
    pub output: String,
    pub status: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExoPorts {
    api: u16,
    zenoh: u16,
}

impl ExoPorts {
    fn for_group(group_id: &str) -> Self {
        // FNV-1a: stable across builds and platforms (unlike std's hasher).
        let mut hash: u32 = 0x811c_9dc5;
        for b in group_id.bytes() {
            hash ^= b as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        let slot = (hash % PORT_SLOTS) as u16;
        Self {
            api: API_PORT_BASE + slot,
            zenoh: ZENOH_PORT_BASE + slot,
        }
    }
}

/// Execute an exo job — start the local Exo node, join the cluster, and (if
/// primary) drive inference.
pub async fn execute_exo_job(
    nats: &NatsAgent,
    _state: &Arc<RwLock<StateManager>>,
    job: &AssignJob,
    exo_config: ExoConfig,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &exo_config.group_id;

    let result = async {
        validate_identifier("group_id", group_id)?;
        validate_identifier("namespace", &exo_config.namespace)?;
        anyhow::ensure!(exo_config.node_count >= 1, "node_count must be at least 1");

        let ports = ExoPorts::for_group(group_id);
        let home = cluster_home(group_id)?;
        std::fs::create_dir_all(&home)
            .with_context(|| format!("Failed to create Exo home {}", home.display()))?;

        info!(
            job_id,
            group_id,
            role = %exo_config.role,
            api_port = ports.api,
            zenoh_port = ports.zenoh,
            "Starting exo node (namespace {}, {} nodes, model {})",
            exo_config.namespace,
            exo_config.node_count,
            exo_config.model,
        );

        let mut child = spawn_exo(&exo_config.namespace, ports, &home)?;
        let result = run_node(nats, job, &exo_config, ports.api, &mut cancel_rx).await;
        stop_process_group(&mut child).await;

        match &result {
            // Exo persists deployments and resumes them on start, so a cluster's
            // home must never be reused. Keep it on failure for its logs.
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&home);
            }
            Err(_) => warn!(job_id, "Keeping Exo home for debugging: {}", home.display()),
        }
        result
    }
    .await;

    if let Err(ref e) = result {
        error!(job_id, group_id, "Exo node failed: {:#}", e);
        let _ = publish_failed(nats, &exo_config, &format!("{:#}", e)).await;
    }

    result
}

/// The coordinator is the authority, but these strings become a path component
/// and a CLI argument, so refuse anything unexpected.
fn validate_identifier(field: &str, value: &str) -> Result<()> {
    let ok = !value.is_empty()
        && value.len() <= 128
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    anyhow::ensure!(ok, "invalid {}: {:?}", field, value);
    Ok(())
}

/// Per-cluster Exo home: `~/.island/exo/clusters/<group_id>`.
fn cluster_home(group_id: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().context("No home directory")?;
    Ok(home.join(".island/exo/clusters").join(group_id))
}

/// Exo's own default model directory, shared by all clusters so a model is
/// downloaded once per Island (and reused if the operator already runs Exo).
fn shared_models_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    Some(if cfg!(target_os = "linux") {
        home.join(".local/share/exo/models")
    } else {
        home.join(".exo/models")
    })
}

/// The `exo` executable; `EXO_BIN` overrides it (e.g. a source checkout's
/// `.venv/bin/exo`).
fn exo_bin() -> String {
    std::env::var("EXO_BIN").unwrap_or_else(|_| "exo".to_string())
}

fn spawn_exo(namespace: &str, ports: ExoPorts, home: &Path) -> Result<Child> {
    let mut cmd = Command::new(exo_bin());
    cmd.arg("--namespace")
        .arg(namespace)
        .arg("--api-port")
        .arg(ports.api.to_string())
        .arg("--zenoh-port")
        .arg(ports.zenoh.to_string())
        // Python's Path.home() / <absolute path> yields the absolute path.
        .env("EXO_HOME", home)
        .env_remove("EXO_ZENOH_NAMESPACE")
        .env_remove("EXO_LIBP2P_NAMESPACE")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    if let Some(models) = shared_models_dir() {
        cmd.env("EXO_DEFAULT_MODELS_DIR", models);
    }
    // Own process group, so shutdown reaches Exo's runner subprocesses too.
    #[cfg(unix)]
    cmd.process_group(0);

    cmd.spawn()
        .context("Failed to spawn `exo` — is it installed and on PATH (or EXO_BIN set)?")
}

/// SIGTERM the whole Exo process group (Exo shuts its runners down on SIGTERM),
/// then SIGKILL whatever is left after the grace period.
async fn stop_process_group(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id().and_then(|p| i32::try_from(p).ok()) {
        // SAFETY: plain kill(2) on the process group we created at spawn.
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
        let exited = tokio::time::timeout(SHUTDOWN_GRACE, child.wait())
            .await
            .is_ok();
        // SAFETY: as above; ESRCH if the group is already gone.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        if exited {
            return;
        }
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}

async fn run_node(
    nats: &NatsAgent,
    job: &AssignJob,
    cfg: &ExoConfig,
    port: u16,
    cancel_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let api = ExoApi::new(port)?;

    // 1. Wait for every node to be reachable.
    wait_for_cluster(&api, cfg.node_count, cancel_rx)
        .await
        .context("Cluster did not form in time")?;

    // 2. The primary places the model across all nodes. Chat requests are
    //    refused for a model without an instance.
    if cfg.is_primary {
        place_model(&api, &cfg.model, cfg.node_count, cancel_rx)
            .await
            .context("Failed to place model on the Exo cluster")?;
    }

    // 3. Signal ready.
    nats.publish_raw(
        &cfg.subjects.status,
        serde_json::to_vec(&json!({
            "host_id": nats.host_id(),
            "status": "ready",
            "role": cfg.role,
        }))?,
    )
    .await?;

    info!(job_id, role = %cfg.role, "Exo node ready, cluster formed");

    // 4. The primary waits for the prompt; workers wait for stop (their Exo
    //    process serves its shard as part of the cluster).
    let mut control_sub = nats
        .subscribe_ring(&cfg.subjects.control)
        .await
        .context("Failed to subscribe to exo control")?;

    loop {
        select! {
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        let action = serde_json::from_slice::<serde_json::Value>(&msg.payload)
                            .ok()
                            .and_then(|v| v.get("action").and_then(|a| a.as_str()).map(String::from));

                        match action.as_deref() {
                            Some("prompt") if cfg.is_primary => {
                                return run_primary_inference(nats, job, cfg, &api, cancel_rx).await;
                            }
                            Some("stop") => {
                                info!(job_id, "Exo node received stop");
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                    None => return Ok(()),
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!(job_id, "Exo node cancelled");
                    nats.publish_status(job_id, "cancelled", None).await?;
                    return Ok(());
                }
            }
        }
    }
}

/// Thin client for the local Exo node's HTTP API.
struct ExoApi {
    client: reqwest::Client,
    base: String,
}

impl ExoApi {
    fn new(port: u16) -> Result<Self> {
        Ok(Self {
            // No overall timeout: the chat stream is long-lived. Each call sets
            // its own, and the stream is guarded by STREAM_IDLE_TIMEOUT.
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(3))
                .build()?,
            base: format!("http://127.0.0.1:{}", port),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }
}

/// Sleep, but return early with an error if the job is cancelled.
async fn cancellable_sleep(d: Duration, cancel_rx: &mut watch::Receiver<bool>) -> Result<()> {
    if *cancel_rx.borrow() {
        anyhow::bail!("cancelled");
    }
    select! {
        _ = tokio::time::sleep(d) => Ok(()),
        _ = cancel_rx.changed() => {
            if *cancel_rx.borrow() {
                anyhow::bail!("cancelled");
            }
            Ok(())
        }
    }
}

/// Poll `/membership` until at least `node_count` nodes (this one included) are
/// reachable.
async fn wait_for_cluster(
    api: &ExoApi,
    node_count: u32,
    cancel_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + CLUSTER_FORM_TIMEOUT;
    let mut last_seen = 0;

    loop {
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "timed out waiting for {} nodes (last saw {}); nodes must share a LAN and run the same Exo build",
                node_count,
                last_seen
            );
        }

        let resp = api
            .client
            .get(api.url("/membership"))
            .timeout(Duration::from_secs(3))
            .send()
            .await;
        if let Ok(resp) = resp {
            if let Ok(view) = resp.json::<serde_json::Value>().await {
                if let Some(n) = reachable_nodes(&view) {
                    last_seen = n;
                    if n >= node_count {
                        return Ok(());
                    }
                }
            }
        }

        cancellable_sleep(Duration::from_millis(750), cancel_rx).await?;
    }
}

/// `reachable` from Exo's `GET /membership` (camelCase `MembershipView`): the
/// alive peers this node counts, plus itself.
fn reachable_nodes(view: &serde_json::Value) -> Option<u32> {
    view.get("reachable")
        .and_then(|v| v.as_u64())
        .map(|n| n.min(u32::MAX as u64) as u32)
}

/// Ask Exo to place `model` pipeline-sharded across exactly the whole cluster,
/// then wait until the instance shows up in the cluster state.
async fn place_model(
    api: &ExoApi,
    model: &str,
    node_count: u32,
    cancel_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let body = json!({
        "model_id": model,
        "sharding": "Pipeline",
        "instance_meta": "MlxRing",
        "min_nodes": node_count,
    });
    let deadline = tokio::time::Instant::now() + PLACEMENT_TIMEOUT;

    loop {
        let last_error = match api
            .client
            .post(api.url("/place_instance"))
            .timeout(Duration::from_secs(30))
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => break,
            Ok(resp) => {
                let status = resp.status();
                format!(
                    "HTTP {}: {}",
                    status,
                    error_detail(&resp.text().await.unwrap_or_default())
                )
            }
            Err(e) => e.to_string(),
        };
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("Exo refused placement of {}: {}", model, last_error);
        }
        info!("Exo placement not accepted yet ({}), retrying", last_error);
        cancellable_sleep(Duration::from_secs(2), cancel_rx).await?;
    }

    // `/instance/await` streams a single SSE event once an instance exists.
    let await_url = reqwest::Url::parse_with_params(
        &api.url("/instance/await"),
        &[("model_id", model), ("timeout_seconds", "60")],
    )?;
    let text = api
        .client
        .get(await_url)
        .timeout(Duration::from_secs(75))
        .send()
        .await
        .context("instance/await request failed")?
        .text()
        .await?;
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            let v: serde_json::Value = serde_json::from_str(data).unwrap_or_default();
            if v.get("instance").is_some() {
                return Ok(());
            }
            let msg = v.get("message").and_then(|m| m.as_str()).unwrap_or(data);
            anyhow::bail!("instance not created: {}", msg);
        }
    }
    anyhow::bail!("instance/await returned no event")
}

/// FastAPI errors come as `{"detail": ...}`; fall back to the raw body.
fn error_detail(body: &str) -> String {
    let detail = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("detail").cloned())
        .map(|d| {
            d.as_str()
                .map(String::from)
                .unwrap_or_else(|| d.to_string())
        });
    let mut s = detail.unwrap_or_else(|| body.to_string());
    s.truncate(500);
    s
}

/// Primary node: drive inference against the local Exo API and relay tokens.
async fn run_primary_inference(
    nats: &NatsAgent,
    job: &AssignJob,
    cfg: &ExoConfig,
    api: &ExoApi,
    cancel_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    let mut body = json!({
        "model": cfg.model,
        "messages": build_messages(&job.input),
        "stream": true,
    });
    if let Some(t) = job.model_temperature {
        body["temperature"] = json!(t);
    }
    if let Some(mt) = max_tokens(&job.input) {
        body["max_tokens"] = json!(mt);
    }

    info!(job_id, "Primary sending prompt to Exo cluster");

    let resp = match open_chat_stream(api, &body, cancel_rx).await {
        Ok(resp) => resp,
        Err(_) if *cancel_rx.borrow() => {
            nats.publish_status(job_id, "cancelled", None).await?;
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();

    loop {
        select! {
            chunk = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()) => {
                match chunk {
                    Err(_) => anyhow::bail!("Exo stream stalled for {:?}", STREAM_IDLE_TIMEOUT),
                    Ok(Some(Ok(bytes))) => {
                        buf.extend_from_slice(&bytes);
                        if drain_sse(nats, cfg, &mut buf).await? {
                            return finalize(nats, cfg).await;
                        }
                    }
                    Ok(Some(Err(e))) => anyhow::bail!("stream error: {}", e),
                    Ok(None) => anyhow::bail!("Exo stream ended without [DONE]"),
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    nats.publish_status(job_id, "cancelled", None).await?;
                    return Ok(());
                }
            }
        }
    }
}

/// POST the chat request, retrying while Exo answers 503 (model still
/// downloading/loading; Exo already holds each request ~30s before that).
async fn open_chat_stream(
    api: &ExoApi,
    body: &serde_json::Value,
    cancel_rx: &mut watch::Receiver<bool>,
) -> Result<reqwest::Response> {
    let deadline = tokio::time::Instant::now() + MODEL_READY_TIMEOUT;
    loop {
        let resp = api
            .client
            .post(api.url("/v1/chat/completions"))
            .json(body)
            .send()
            .await
            .context("Exo chat request failed")?;

        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let detail = error_detail(&resp.text().await.unwrap_or_default());

        if status != reqwest::StatusCode::SERVICE_UNAVAILABLE
            || tokio::time::Instant::now() >= deadline
        {
            anyhow::bail!("Exo returned HTTP {}: {}", status, detail);
        }
        info!("Exo model not ready yet ({}), retrying", detail);
        let wait = Duration::from_secs(retry_after.unwrap_or(5).clamp(1, 15));
        cancellable_sleep(wait, cancel_rx).await?;
    }
}

/// One parsed SSE line from Exo's chat stream.
#[derive(Debug, PartialEq)]
enum SseEvent {
    Skip,
    Content(String),
    Done,
    Error(String),
}

fn parse_sse_line(line: &str) -> SseEvent {
    // Comments: `: keep-alive`, `: prefill_progress {...}`, `: generation_stats {...}`.
    let Some(data) = line.strip_prefix("data:") else {
        return SseEvent::Skip;
    };
    let data = data.trim_start();
    if data == "[DONE]" {
        return SseEvent::Done;
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
        return SseEvent::Skip;
    };
    // Generation failures arrive as `data: {"error": {...}}` followed by [DONE].
    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .map(String::from)
            .unwrap_or_else(|| err.to_string());
        return SseEvent::Error(msg);
    }
    match v
        .pointer("/choices/0/delta/content")
        .and_then(|c| c.as_str())
    {
        Some(content) if !content.is_empty() => SseEvent::Content(content.to_string()),
        _ => SseEvent::Skip,
    }
}

/// Parse complete SSE lines from `buf`, publishing token chunks. Works on bytes
/// so a UTF-8 character split across network chunks isn't mangled. Returns
/// `true` when `data: [DONE]` is seen.
async fn drain_sse(nats: &NatsAgent, cfg: &ExoConfig, buf: &mut Vec<u8>) -> Result<bool> {
    while let Some(idx) = buf.iter().position(|&b| b == b'\n') {
        let raw: Vec<u8> = buf.drain(..=idx).collect();
        let line = String::from_utf8_lossy(&raw);
        match parse_sse_line(line.trim_end()) {
            SseEvent::Skip => {}
            SseEvent::Done => return Ok(true),
            SseEvent::Error(msg) => anyhow::bail!("Exo generation failed: {}", msg),
            SseEvent::Content(content) => {
                nats.publish_raw(
                    &cfg.subjects.output,
                    serde_json::to_vec(&json!({ "chunk": content, "is_final": false }))?,
                )
                .await?;
            }
        }
    }
    Ok(false)
}

/// Send the terminal output chunk + completion status.
async fn finalize(nats: &NatsAgent, cfg: &ExoConfig) -> Result<()> {
    nats.publish_raw(
        &cfg.subjects.output,
        serde_json::to_vec(&json!({ "chunk": "", "is_final": true }))?,
    )
    .await?;
    nats.publish_raw(
        &cfg.subjects.status,
        serde_json::to_vec(&json!({ "status": "complete" }))?,
    )
    .await?;
    Ok(())
}

async fn publish_failed(nats: &NatsAgent, cfg: &ExoConfig, error: &str) -> Result<()> {
    nats.publish_raw(
        &cfg.subjects.status,
        serde_json::to_vec(&json!({
            "host_id": nats.host_id(),
            "status": "failed",
            "error": error,
        }))?,
    )
    .await?;
    Ok(())
}

/// Build an OpenAI `messages` array from the job input, which may be a chat
/// payload (`{"messages": [...]}`), a `{"prompt": "..."}`, or a bare string.
fn build_messages(input: &serde_json::Value) -> serde_json::Value {
    if let Some(messages) = input.get("messages") {
        if messages.is_array() {
            return messages.clone();
        }
    }

    let content = input
        .get("prompt")
        .and_then(|p| p.as_str())
        .or_else(|| input.as_str())
        .unwrap_or("");

    json!([{ "role": "user", "content": content }])
}

fn max_tokens(input: &serde_json::Value) -> Option<u64> {
    input.get("max_tokens").and_then(|v| v.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_from_chat() {
        let input = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(
            build_messages(&input),
            json!([{"role": "user", "content": "hi"}])
        );
    }

    #[test]
    fn build_messages_from_prompt() {
        let input = json!({"prompt": "hello"});
        assert_eq!(
            build_messages(&input),
            json!([{"role": "user", "content": "hello"}])
        );
    }

    #[test]
    fn build_messages_from_string() {
        let input = json!("plain");
        assert_eq!(
            build_messages(&input),
            json!([{"role": "user", "content": "plain"}])
        );
    }

    #[test]
    fn reachable_from_membership_view() {
        // Shape of Exo's MembershipView (camelCase).
        let view = json!({
            "nodeId": "a", "selfState": "Alive", "quorum": "Met",
            "reachable": 2, "lastKnown": 3, "members": []
        });
        assert_eq!(reachable_nodes(&view), Some(2));
        assert_eq!(reachable_nodes(&json!({"topology": {}})), None);
    }

    #[test]
    fn ports_are_stable_distinct_and_in_range() {
        let a = ExoPorts::for_group("6f1c2e9a-0d4b-4e5f-9a7b-1c2d3e4f5a6b");
        assert_eq!(
            a,
            ExoPorts::for_group("6f1c2e9a-0d4b-4e5f-9a7b-1c2d3e4f5a6b")
        );
        assert!((API_PORT_BASE..API_PORT_BASE + PORT_SLOTS as u16).contains(&a.api));
        assert!((ZENOH_PORT_BASE..ZENOH_PORT_BASE + PORT_SLOTS as u16).contains(&a.zenoh));
        assert_ne!(a.api, a.zenoh);
        // Never Exo's defaults, so a locally running Exo app can't clash.
        assert!(![52413, 52414, 52415].contains(&a.api));
    }

    #[test]
    fn identifiers_are_validated() {
        assert!(validate_identifier("g", "exo-6f1c2e9a-0d4b").is_ok());
        assert!(validate_identifier("g", "").is_err());
        assert!(validate_identifier("g", "../etc").is_err());
        assert!(validate_identifier("g", "a b").is_err());
    }

    #[test]
    fn sse_content_chunk() {
        let line = r#"data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}"#;
        assert_eq!(parse_sse_line(line), SseEvent::Content("Hel".into()));
    }

    #[test]
    fn sse_comments_and_empty_deltas_are_skipped() {
        assert_eq!(parse_sse_line(": keep-alive"), SseEvent::Skip);
        assert_eq!(
            parse_sse_line(r#": prefill_progress {"processed":10,"total":40}"#),
            SseEvent::Skip
        );
        assert_eq!(parse_sse_line(": generation_stats {}"), SseEvent::Skip);
        assert_eq!(parse_sse_line(""), SseEvent::Skip);
        let stop = r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        assert_eq!(parse_sse_line(stop), SseEvent::Skip);
    }

    #[test]
    fn sse_done() {
        assert_eq!(parse_sse_line("data: [DONE]"), SseEvent::Done);
    }

    #[test]
    fn sse_error_is_surfaced() {
        let line = r#"data: {"error":{"message":"runner crashed","type":"InternalServerError","code":500}}"#;
        assert_eq!(
            parse_sse_line(line),
            SseEvent::Error("runner crashed".into())
        );
    }

    #[test]
    fn error_detail_extracts_fastapi_detail() {
        assert_eq!(
            error_detail(r#"{"detail":"not enough memory"}"#),
            "not enough memory"
        );
        assert_eq!(error_detail("plain text"), "plain text");
    }
}
