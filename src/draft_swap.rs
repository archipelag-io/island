//! Draft model hot-swapping for speculative decoding.
//!
//! Tracks acceptance rate trends and recommends swapping to a different
//! draft model when the current one diverges too much from the verifier.
//!
//! The coordinator can provide a list of candidate draft models ranked by
//! quality/speed tradeoff. This module monitors the rolling acceptance rate
//! and signals when a swap is warranted.
//!
//! Hot-swap flow:
//! 1. Monitor rolling acceptance rate over N rounds
//! 2. If rate drops below threshold, signal "swap_needed" to coordinator
//! 3. Coordinator picks a better-matched draft model from the candidate list
//! 4. Draft Island downloads new model, switches, resets tracking

use serde::{Deserialize, Serialize};
use tracing::info;

/// Tracks acceptance rate trends and recommends draft model swaps
pub struct DraftSwapMonitor {
    /// Rolling window of acceptance rates (last N rounds)
    window: Vec<f64>,
    /// Maximum window size
    window_size: usize,
    /// Threshold below which a swap is recommended
    swap_threshold: f64,
    /// Threshold above which the current model is excellent (no swap needed)
    excellent_threshold: f64,
    /// Current draft model URL
    current_model: String,
    /// Candidate models ordered by quality (best first)
    candidates: Vec<DraftCandidate>,
    /// Index of current model in candidates list
    current_index: usize,
    /// Number of rounds since last swap
    rounds_since_swap: u64,
    /// Minimum rounds before allowing another swap
    min_rounds_between_swaps: u64,
}

/// A candidate draft model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftCandidate {
    /// Model URL
    pub url: String,
    /// Human-readable name
    pub name: String,
    /// Expected tokens/second (higher = faster but possibly less accurate)
    pub speed_tok_s: f32,
    /// Model size in MB (for cache management)
    pub size_mb: u64,
}

/// Recommendation from the swap monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwapRecommendation {
    /// Current model is fine, keep going
    KeepCurrent,
    /// Swap to a different model (URL provided)
    SwapTo {
        url: String,
        name: String,
        reason: String,
    },
    /// Not enough data yet to make a recommendation
    InsufficientData,
}

impl DraftSwapMonitor {
    pub fn new(
        current_model: String,
        candidates: Vec<DraftCandidate>,
        window_size: usize,
    ) -> Self {
        let current_index = candidates.iter()
            .position(|c| c.url == current_model)
            .unwrap_or(0);

        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            swap_threshold: 0.4,
            excellent_threshold: 0.85,
            current_model,
            candidates,
            current_index,
            rounds_since_swap: 0,
            min_rounds_between_swaps: 10,
        }
    }

    /// Record a verification round result
    pub fn record_round(&mut self, drafted: usize, accepted: usize) {
        let rate = if drafted > 0 { accepted as f64 / drafted as f64 } else { 0.5 };

        if self.window.len() >= self.window_size {
            self.window.remove(0);
        }
        self.window.push(rate);
        self.rounds_since_swap += 1;
    }

    /// Get the rolling average acceptance rate
    pub fn rolling_rate(&self) -> Option<f64> {
        if self.window.is_empty() {
            None
        } else {
            Some(self.window.iter().sum::<f64>() / self.window.len() as f64)
        }
    }

    /// Check if a draft model swap is recommended
    pub fn check_swap(&self) -> SwapRecommendation {
        if self.window.len() < self.window_size / 2 {
            return SwapRecommendation::InsufficientData;
        }

        if self.rounds_since_swap < self.min_rounds_between_swaps {
            return SwapRecommendation::KeepCurrent;
        }

        let rate = self.rolling_rate().unwrap_or(0.5);

        if rate >= self.excellent_threshold {
            return SwapRecommendation::KeepCurrent;
        }

        if rate < self.swap_threshold && self.candidates.len() > 1 {
            // Try the next candidate (higher quality, possibly slower)
            let next_index = if self.current_index + 1 < self.candidates.len() {
                self.current_index + 1
            } else {
                0 // wrap around
            };

            let next = &self.candidates[next_index];
            if next.url != self.current_model {
                return SwapRecommendation::SwapTo {
                    url: next.url.clone(),
                    name: next.name.clone(),
                    reason: format!(
                        "acceptance rate {:.1}% below threshold {:.1}%",
                        rate * 100.0,
                        self.swap_threshold * 100.0,
                    ),
                };
            }
        }

        SwapRecommendation::KeepCurrent
    }

    /// Apply a swap — reset tracking and update current model
    pub fn apply_swap(&mut self, new_url: &str) {
        info!(
            "Draft model swap: {} → {}",
            self.current_model, new_url
        );
        self.current_model = new_url.to_string();
        self.current_index = self.candidates.iter()
            .position(|c| c.url == new_url)
            .unwrap_or(self.current_index);
        self.window.clear();
        self.rounds_since_swap = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_candidates() -> Vec<DraftCandidate> {
        vec![
            DraftCandidate { url: "fast.gguf".into(), name: "Fast".into(), speed_tok_s: 200.0, size_mb: 500 },
            DraftCandidate { url: "balanced.gguf".into(), name: "Balanced".into(), speed_tok_s: 100.0, size_mb: 1000 },
            DraftCandidate { url: "accurate.gguf".into(), name: "Accurate".into(), speed_tok_s: 50.0, size_mb: 2000 },
        ]
    }

    #[test]
    fn test_insufficient_data() {
        let monitor = DraftSwapMonitor::new("fast.gguf".into(), test_candidates(), 10);
        assert!(matches!(monitor.check_swap(), SwapRecommendation::InsufficientData));
    }

    #[test]
    fn test_keep_current_on_good_rate() {
        let mut monitor = DraftSwapMonitor::new("fast.gguf".into(), test_candidates(), 10);
        monitor.min_rounds_between_swaps = 0;
        for _ in 0..10 {
            monitor.record_round(5, 5); // 100% acceptance
        }
        assert!(matches!(monitor.check_swap(), SwapRecommendation::KeepCurrent));
    }

    #[test]
    fn test_swap_on_low_rate() {
        let mut monitor = DraftSwapMonitor::new("fast.gguf".into(), test_candidates(), 10);
        monitor.min_rounds_between_swaps = 0;
        for _ in 0..10 {
            monitor.record_round(5, 1); // 20% acceptance
        }
        match monitor.check_swap() {
            SwapRecommendation::SwapTo { name, .. } => assert_eq!(name, "Balanced"),
            other => panic!("Expected SwapTo, got {:?}", other),
        }
    }

    #[test]
    fn test_apply_swap_resets_window() {
        let mut monitor = DraftSwapMonitor::new("fast.gguf".into(), test_candidates(), 10);
        for _ in 0..5 {
            monitor.record_round(5, 1);
        }
        assert_eq!(monitor.window.len(), 5);

        monitor.apply_swap("balanced.gguf");
        assert_eq!(monitor.window.len(), 0);
        assert_eq!(monitor.current_model, "balanced.gguf");
        assert_eq!(monitor.rounds_since_swap, 0);
    }

    #[test]
    fn test_rolling_rate() {
        let mut monitor = DraftSwapMonitor::new("fast.gguf".into(), test_candidates(), 5);
        monitor.record_round(10, 8); // 0.8
        monitor.record_round(10, 6); // 0.6
        let rate = monitor.rolling_rate().unwrap();
        assert!((rate - 0.7).abs() < 1e-6);
    }
}
