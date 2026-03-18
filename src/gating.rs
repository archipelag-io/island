//! Learned gating for MoE expert routing.
//!
//! Provides expert selection based on token embedding similarity, as an
//! approximation of learned MoE gating weights. When the router model is
//! loaded, tokens are embedded and routed to the experts whose centroids
//! are closest in embedding space.
//!
//! Two strategies:
//! - **Hash-based** (default): deterministic, no model needed, uniform distribution
//! - **Embedding-based** (when router model loaded): uses token embeddings to
//!   select experts based on learned similarity
//!
//! When MoE GGUF formats expose actual gating layer weights, both strategies
//! will be replaced with native gating network output.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Gating strategy for expert selection
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GatingStrategy {
    /// Consistent hashing on token text (default, no model needed)
    Hash,
    /// Embedding similarity against expert centroids
    Embedding,
    /// Round-robin across experts (for load testing)
    RoundRobin,
}

impl Default for GatingStrategy {
    fn default() -> Self {
        GatingStrategy::Hash
    }
}

/// Expert routing decision for a single token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Which experts were selected (top-K)
    pub expert_ids: Vec<u32>,
    /// Routing weights (softmax probabilities) for each selected expert
    pub weights: Vec<f32>,
    /// Which strategy was used
    pub strategy: GatingStrategy,
}

/// Native MoE gating weights extracted from a model's gating layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeGatingWeights {
    /// Gating weight matrix (n_vocab × n_experts or n_embd × n_experts)
    pub weights: Vec<Vec<f32>>,
    /// Bias vector (n_experts), if present
    pub bias: Option<Vec<f32>>,
    /// How to interpret the input (token_id lookup vs embedding dot product)
    pub input_mode: GatingInputMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GatingInputMode {
    /// Look up gating weights by token ID (fast, sparse)
    TokenLookup,
    /// Dot product of token embedding with weight matrix (dense)
    EmbeddingDot,
}

/// Gate that selects experts for tokens
pub struct ExpertGate {
    strategy: GatingStrategy,
    total_experts: u32,
    active_experts: u32,
    /// Expert centroids for embedding-based routing (expert_id → centroid vector)
    centroids: Vec<Vec<f32>>,
    /// Native gating weights extracted from MoE model (shared across layers)
    native_weights: Option<NativeGatingWeights>,
    /// Per-layer gating weights (layer_id → weights). When set, each layer
    /// uses its own learned gating function for expert selection.
    per_layer_weights: std::collections::HashMap<u32, NativeGatingWeights>,
    /// Round-robin counter
    rr_counter: std::sync::atomic::AtomicU64,
}

impl ExpertGate {
    /// Create a new gate with the given strategy
    pub fn new(strategy: GatingStrategy, total_experts: u32, active_experts: u32) -> Self {
        Self {
            strategy,
            total_experts,
            active_experts,
            centroids: Vec::new(),
            native_weights: None,
            per_layer_weights: std::collections::HashMap::new(),
            rr_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Initialize embedding centroids from expert weight data.
    /// Each centroid is the mean embedding vector for tokens that an expert specializes in.
    /// Call this after loading the router model's expert embeddings.
    pub fn set_centroids(&mut self, centroids: Vec<Vec<f32>>) {
        self.centroids = centroids;
        if !self.centroids.is_empty() {
            self.strategy = GatingStrategy::Embedding;
        }
    }

    /// Set native MoE gating weights extracted from the model (shared across all layers).
    /// When set, routing uses the learned gating function directly.
    pub fn set_native_weights(&mut self, weights: NativeGatingWeights) {
        self.native_weights = Some(weights);
    }

    /// Set per-layer gating weights. Each layer gets its own learned gating function.
    /// When routing, the layer_id parameter selects the appropriate weights.
    pub fn set_per_layer_weights(&mut self, layer_weights: std::collections::HashMap<u32, NativeGatingWeights>) {
        self.per_layer_weights = layer_weights;
    }

    /// Route a token using per-layer gating weights for a specific layer.
    /// Falls back to shared native weights, then to the configured strategy.
    pub fn route_for_layer(&self, token: &str, embedding: Option<&[f32]>, layer_id: u32) -> RoutingDecision {
        // Try per-layer weights first
        if let Some(layer_weights) = self.per_layer_weights.get(&layer_id) {
            let token_id = token.bytes().next().map(|b| b as i32);
            if let Some(decision) = self.route_with_weights(layer_weights, token_id, embedding) {
                return decision;
            }
        }

        // Fall back to shared route
        self.route(token, embedding)
    }

    /// Route a token using native gating weights (when available).
    fn route_native(&self, token_id: Option<i32>, embedding: Option<&[f32]>) -> Option<RoutingDecision> {
        let weights = self.native_weights.as_ref()?;
        self.route_with_weights(weights, token_id, embedding)
    }

    /// Route a token using specific gating weights.
    fn route_with_weights(&self, weights: &NativeGatingWeights, token_id: Option<i32>, embedding: Option<&[f32]>) -> Option<RoutingDecision> {

        let expert_scores: Vec<f32> = match weights.input_mode {
            GatingInputMode::TokenLookup => {
                let tid = token_id? as usize;
                if tid >= weights.weights.len() { return None; }
                weights.weights[tid].clone()
            }
            GatingInputMode::EmbeddingDot => {
                let emb = embedding?;
                // Dot product: scores[e] = sum(emb[i] * weights[i][e]) + bias[e]
                let n_experts = self.total_experts as usize;
                let mut scores = vec![0.0f32; n_experts];
                for (i, &e_val) in emb.iter().enumerate() {
                    if i < weights.weights.len() {
                        for (e, score) in scores.iter_mut().enumerate() {
                            if e < weights.weights[i].len() {
                                *score += e_val * weights.weights[i][e];
                            }
                        }
                    }
                }
                if let Some(ref bias) = weights.bias {
                    for (e, score) in scores.iter_mut().enumerate() {
                        if e < bias.len() {
                            *score += bias[e];
                        }
                    }
                }
                scores
            }
        };

        // Top-K selection with softmax
        let mut indexed: Vec<(u32, f32)> = expert_scores.iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.active_experts as usize;
        let selected: Vec<(u32, f32)> = indexed.into_iter().take(k).collect();

        let max_s = selected.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = selected.iter().map(|(_, s)| (s - max_s).exp()).sum();
        let weights_vec: Vec<f32> = selected.iter().map(|(_, s)| (s - max_s).exp() / exp_sum).collect();
        let expert_ids: Vec<u32> = selected.iter().map(|(id, _)| *id).collect();

        Some(RoutingDecision { expert_ids, weights: weights_vec, strategy: GatingStrategy::Embedding })
    }

    /// Route a token to top-K experts based on the current strategy.
    /// If native gating weights are set, they take priority over all strategies.
    pub fn route(&self, token: &str, embedding: Option<&[f32]>) -> RoutingDecision {
        // Try native gating first (learned weights from MoE model)
        if self.native_weights.is_some() {
            // Extract token_id from token string (simple hash for now)
            let token_id = token.bytes().next().map(|b| b as i32);
            if let Some(decision) = self.route_native(token_id, embedding) {
                return decision;
            }
        }

        match self.strategy {
            GatingStrategy::Hash => self.route_hash(token),
            GatingStrategy::Embedding => {
                if let Some(emb) = embedding {
                    self.route_embedding(emb)
                } else {
                    self.route_hash(token) // fallback
                }
            }
            GatingStrategy::RoundRobin => self.route_round_robin(),
        }
    }

    /// Hash-based routing: deterministic, uniform
    fn route_hash(&self, token: &str) -> RoutingDecision {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();

        let mut expert_ids = Vec::with_capacity(self.active_experts as usize);
        for k in 0..self.active_experts {
            let eid = ((hash.wrapping_add(k as u64 * 0x9e3779b97f4a7c15)) % self.total_experts as u64) as u32;
            if !expert_ids.contains(&eid) {
                expert_ids.push(eid);
            }
        }
        while expert_ids.len() < self.active_experts as usize {
            for e in 0..self.total_experts {
                if !expert_ids.contains(&e) { expert_ids.push(e); break; }
            }
        }

        let uniform_weight = 1.0 / expert_ids.len() as f32;
        let weights = vec![uniform_weight; expert_ids.len()];

        RoutingDecision { expert_ids, weights, strategy: GatingStrategy::Hash }
    }

    /// Embedding-based routing: select experts whose centroids are closest
    fn route_embedding(&self, embedding: &[f32]) -> RoutingDecision {
        if self.centroids.is_empty() {
            return self.route_hash("");
        }

        // Compute cosine similarity to each expert centroid
        let mut similarities: Vec<(u32, f32)> = self.centroids.iter()
            .enumerate()
            .map(|(idx, centroid)| {
                let sim = cosine_similarity(embedding, centroid);
                (idx as u32, sim)
            })
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top-K
        let k = self.active_experts as usize;
        let selected: Vec<(u32, f32)> = similarities.into_iter().take(k).collect();

        // Softmax over selected similarities for routing weights
        let max_sim = selected.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = selected.iter().map(|(_, s)| (s - max_sim).exp()).sum();
        let weights: Vec<f32> = selected.iter().map(|(_, s)| (s - max_sim).exp() / exp_sum).collect();
        let expert_ids: Vec<u32> = selected.iter().map(|(id, _)| *id).collect();

        RoutingDecision { expert_ids, weights, strategy: GatingStrategy::Embedding }
    }

    /// Round-robin routing: sequential expert assignment
    fn route_round_robin(&self) -> RoutingDecision {
        let counter = self.rr_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut expert_ids = Vec::with_capacity(self.active_experts as usize);

        for k in 0..self.active_experts {
            let eid = ((counter + k as u64) % self.total_experts as u64) as u32;
            expert_ids.push(eid);
        }

        let uniform_weight = 1.0 / expert_ids.len() as f32;
        let weights = vec![uniform_weight; expert_ids.len()];

        RoutingDecision { expert_ids, weights, strategy: GatingStrategy::RoundRobin }
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_routing() {
        let gate = ExpertGate::new(GatingStrategy::Hash, 8, 2);
        let decision = gate.route("hello", None);
        assert_eq!(decision.expert_ids.len(), 2);
        assert!(decision.expert_ids.iter().all(|&e| e < 8));
        assert_eq!(decision.strategy, GatingStrategy::Hash);

        // Deterministic
        let decision2 = gate.route("hello", None);
        assert_eq!(decision.expert_ids, decision2.expert_ids);
    }

    #[test]
    fn test_embedding_routing() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 4, 2);

        // Set centroids: 4 experts with distinct directions
        gate.set_centroids(vec![
            vec![1.0, 0.0, 0.0],  // expert 0: x-axis
            vec![0.0, 1.0, 0.0],  // expert 1: y-axis
            vec![0.0, 0.0, 1.0],  // expert 2: z-axis
            vec![1.0, 1.0, 0.0],  // expert 3: xy-diagonal
        ]);

        assert_eq!(gate.strategy, GatingStrategy::Embedding);

        // Token embedding closest to x-axis → should select expert 0
        let decision = gate.route("", Some(&[0.9, 0.1, 0.0]));
        assert_eq!(decision.expert_ids[0], 0); // Closest to x-axis
        assert_eq!(decision.strategy, GatingStrategy::Embedding);
        assert_eq!(decision.expert_ids.len(), 2);

        // Token embedding closest to y-axis → should select expert 1
        let decision = gate.route("", Some(&[0.0, 0.95, 0.05]));
        assert_eq!(decision.expert_ids[0], 1);
    }

    #[test]
    fn test_round_robin_routing() {
        let gate = ExpertGate::new(GatingStrategy::RoundRobin, 4, 2);

        let d1 = gate.route("", None);
        let d2 = gate.route("", None);
        let d3 = gate.route("", None);

        // Sequential expert assignment
        assert_eq!(d1.expert_ids, vec![0, 1]);
        assert_eq!(d2.expert_ids, vec![1, 2]);
        assert_eq!(d3.expert_ids, vec![2, 3]);
    }

    #[test]
    fn test_cosine_similarity() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_routing_decision_serialize() {
        let decision = RoutingDecision {
            expert_ids: vec![0, 3],
            weights: vec![0.7, 0.3],
            strategy: GatingStrategy::Embedding,
        };
        let json = serde_json::to_string(&decision).unwrap();
        assert!(json.contains("\"strategy\":\"embedding\""));
        assert!(json.contains("\"expert_ids\":[0,3]"));
    }

    // --- GatingStrategy tests ---

    #[test]
    fn test_gating_strategy_default_is_hash() {
        assert_eq!(GatingStrategy::default(), GatingStrategy::Hash);
    }

    #[test]
    fn test_gating_strategy_serde_roundtrip() {
        let strategies = vec![GatingStrategy::Hash, GatingStrategy::Embedding, GatingStrategy::RoundRobin];
        for s in strategies {
            let json = serde_json::to_string(&s).unwrap();
            let deserialized: GatingStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(s, deserialized);
        }
    }

    #[test]
    fn test_gating_strategy_snake_case_names() {
        assert_eq!(serde_json::to_string(&GatingStrategy::Hash).unwrap(), "\"hash\"");
        assert_eq!(serde_json::to_string(&GatingStrategy::Embedding).unwrap(), "\"embedding\"");
        assert_eq!(serde_json::to_string(&GatingStrategy::RoundRobin).unwrap(), "\"round_robin\"");
    }

    // --- cosine_similarity edge cases ---

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = vec![0.5, 0.5, 0.5];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    // --- Hash routing edge cases ---

    #[test]
    fn test_hash_routing_different_tokens_may_differ() {
        let gate = ExpertGate::new(GatingStrategy::Hash, 8, 2);
        let d1 = gate.route("apple", None);
        let d2 = gate.route("banana", None);
        // Both valid, both have 2 experts in range
        assert_eq!(d1.expert_ids.len(), 2);
        assert_eq!(d2.expert_ids.len(), 2);
        assert!(d1.expert_ids.iter().all(|&e| e < 8));
        assert!(d2.expert_ids.iter().all(|&e| e < 8));
    }

    #[test]
    fn test_hash_routing_uniform_weights() {
        let gate = ExpertGate::new(GatingStrategy::Hash, 4, 2);
        let decision = gate.route("test", None);
        assert_eq!(decision.weights.len(), 2);
        assert!((decision.weights[0] - 0.5).abs() < 1e-6);
        assert!((decision.weights[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hash_routing_single_active_expert() {
        let gate = ExpertGate::new(GatingStrategy::Hash, 8, 1);
        let decision = gate.route("hello", None);
        assert_eq!(decision.expert_ids.len(), 1);
        assert!((decision.weights[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hash_routing_all_experts_active() {
        let gate = ExpertGate::new(GatingStrategy::Hash, 4, 4);
        let decision = gate.route("test", None);
        assert_eq!(decision.expert_ids.len(), 4);
        // All 4 experts should appear (no duplicates)
        let mut sorted = decision.expert_ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);
    }

    // --- Round-robin edge cases ---

    #[test]
    fn test_round_robin_wraps_around() {
        let gate = ExpertGate::new(GatingStrategy::RoundRobin, 3, 1);
        let ids: Vec<u32> = (0..6).map(|_| gate.route("", None).expert_ids[0]).collect();
        assert_eq!(ids, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_round_robin_uniform_weights() {
        let gate = ExpertGate::new(GatingStrategy::RoundRobin, 4, 2);
        let decision = gate.route("", None);
        assert!((decision.weights[0] - 0.5).abs() < 1e-6);
    }

    // --- Embedding routing edge cases ---

    #[test]
    fn test_embedding_fallback_to_hash_without_centroids() {
        let gate = ExpertGate::new(GatingStrategy::Embedding, 4, 2);
        // No centroids set, should fall back to hash
        let decision = gate.route("test", Some(&[1.0, 0.0, 0.0]));
        assert_eq!(decision.strategy, GatingStrategy::Hash);
    }

    #[test]
    fn test_embedding_fallback_when_no_embedding_provided() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 4, 2);
        gate.set_centroids(vec![
            vec![1.0, 0.0], vec![0.0, 1.0],
            vec![-1.0, 0.0], vec![0.0, -1.0],
        ]);
        // Strategy is now Embedding, but no embedding provided → hash fallback
        let decision = gate.route("test", None);
        assert_eq!(decision.strategy, GatingStrategy::Hash);
    }

    #[test]
    fn test_embedding_routing_weights_sum_to_one() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 4, 2);
        gate.set_centroids(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]);
        let decision = gate.route("", Some(&[0.5, 0.5, 0.0]));
        let sum: f32 = decision.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Weights should sum to 1.0, got {}", sum);
    }

    // --- set_centroids side effect ---

    #[test]
    fn test_set_centroids_switches_strategy_to_embedding() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 4, 2);
        assert_eq!(gate.strategy, GatingStrategy::Hash);
        gate.set_centroids(vec![vec![1.0], vec![0.0]]);
        assert_eq!(gate.strategy, GatingStrategy::Embedding);
    }

    #[test]
    fn test_set_empty_centroids_keeps_strategy() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 4, 2);
        gate.set_centroids(vec![]);
        assert_eq!(gate.strategy, GatingStrategy::Hash);
    }

    // --- Native gating weights ---

    #[test]
    fn test_native_weights_take_priority_over_strategy() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 3, 2);
        // Set up token lookup weights: token 'a' (97) → scores [1.0, 0.5, 0.1]
        let mut weights_matrix = vec![vec![0.0; 3]; 128]; // 128 ASCII entries
        weights_matrix[97] = vec![1.0, 0.5, 0.1]; // 'a' = 97
        gate.set_native_weights(NativeGatingWeights {
            weights: weights_matrix,
            bias: None,
            input_mode: GatingInputMode::TokenLookup,
        });

        let decision = gate.route("abc", None); // first byte = 'a' = 97
        assert_eq!(decision.strategy, GatingStrategy::Embedding); // native uses Embedding strategy label
        assert_eq!(decision.expert_ids[0], 0); // expert 0 has highest score
    }

    #[test]
    fn test_native_weights_embedding_dot_with_bias() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 2, 1);
        // 2D embeddings, 2 experts
        // weights[dim][expert]: [[1.0, 0.0], [0.0, 1.0]]
        gate.set_native_weights(NativeGatingWeights {
            weights: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            bias: Some(vec![0.0, 0.5]), // bias expert 1
            input_mode: GatingInputMode::EmbeddingDot,
        });

        // embedding [1.0, 0.0] → scores: [1.0 + 0.0, 0.0 + 0.5] = [1.0, 0.5]
        let decision = gate.route("x", Some(&[1.0, 0.0]));
        assert_eq!(decision.expert_ids[0], 0);
    }

    // --- route_for_layer ---

    #[test]
    fn test_route_for_layer_uses_per_layer_weights() {
        let mut gate = ExpertGate::new(GatingStrategy::Hash, 3, 1);

        // Layer 0: expert 2 wins; Layer 1: expert 0 wins
        let mut layer0_weights = vec![vec![0.0; 3]; 128];
        layer0_weights[104] = vec![0.1, 0.2, 0.9]; // 'h' = 104
        let mut layer1_weights = vec![vec![0.0; 3]; 128];
        layer1_weights[104] = vec![0.9, 0.2, 0.1]; // 'h' = 104

        let mut per_layer = std::collections::HashMap::new();
        per_layer.insert(0, NativeGatingWeights {
            weights: layer0_weights,
            bias: None,
            input_mode: GatingInputMode::TokenLookup,
        });
        per_layer.insert(1, NativeGatingWeights {
            weights: layer1_weights,
            bias: None,
            input_mode: GatingInputMode::TokenLookup,
        });
        gate.set_per_layer_weights(per_layer);

        let d0 = gate.route_for_layer("hi", None, 0);
        assert_eq!(d0.expert_ids[0], 2);

        let d1 = gate.route_for_layer("hi", None, 1);
        assert_eq!(d1.expert_ids[0], 0);
    }

    #[test]
    fn test_route_for_layer_falls_back_to_shared_route() {
        let gate = ExpertGate::new(GatingStrategy::Hash, 4, 2);
        // No per-layer weights set → falls back to shared route (hash)
        let decision = gate.route_for_layer("test", None, 99);
        assert_eq!(decision.strategy, GatingStrategy::Hash);
        assert_eq!(decision.expert_ids.len(), 2);
    }

    // --- RoutingDecision deserialization ---

    #[test]
    fn test_routing_decision_deserialize() {
        let json = r#"{"expert_ids":[1,2],"weights":[0.6,0.4],"strategy":"round_robin"}"#;
        let decision: RoutingDecision = serde_json::from_str(json).unwrap();
        assert_eq!(decision.expert_ids, vec![1, 2]);
        assert_eq!(decision.strategy, GatingStrategy::RoundRobin);
    }
}
