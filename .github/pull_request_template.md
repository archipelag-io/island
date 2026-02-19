## What

<!-- Brief description of what this PR does. Link to related issue if applicable. -->

Closes #

## Why

<!-- Why is this change needed? What problem does it solve? -->

## How Tested

<!-- How did you verify this change works? -->

- [ ] `cargo test` passes locally
- [ ] Manually tested (describe what you did)
- [ ] New tests added for changed behavior

## Checklist

- [ ] CI passes (`cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`)
- [ ] No secrets, API keys, or credentials in code
- [ ] Tests added for new functionality
- [ ] Documentation updated if applicable

### Node Agent-specific

- [ ] New tests use `#[cfg(test)]` inline modules in the source file
- [ ] `ContainerConfig` construction sites updated (check both `agent.rs` and `executor.rs`)
- [ ] `#[instrument]` spans added to new async functions with relevant fields
- [ ] New config fields added to `config.example.toml` with documentation
- [ ] Security-critical code paths validated (seccomp, signing, registry allowlist)
- [ ] Error types defined with `thiserror` (not string errors)
