//! Generic per-key circuit breaker for tool execution.
//!
//! Tracks consecutive failures per tool name (or any string key). When a tool
//! exceeds the configured failure threshold its circuit opens and subsequent
//! calls can check `is_open()` before attempting the operation.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

const DEFAULT_THRESHOLD: u32 = 5;

#[derive(Debug)]
struct BreakerState {
    consecutive_failures: u32,
    open: bool,
}

/// A thread-safe, per-key circuit breaker registry.
#[derive(Debug, Clone)]
pub struct CircuitBreakerRegistry {
    threshold: u32,
    state: Arc<Mutex<HashMap<String, BreakerState>>>,
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_THRESHOLD)
    }
}

impl CircuitBreakerRegistry {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a successful execution, resetting the failure counter.
    pub fn record_success(&self, key: &str) {
        let mut map = self.state.lock();
        if let Some(s) = map.get_mut(key) {
            s.consecutive_failures = 0;
            s.open = false;
        }
    }

    /// Record a failure. Returns `true` if the circuit just opened.
    pub fn record_failure(&self, key: &str) -> bool {
        let mut map = self.state.lock();
        let s = map.entry(key.to_string()).or_insert(BreakerState {
            consecutive_failures: 0,
            open: false,
        });
        s.consecutive_failures += 1;
        if s.consecutive_failures >= self.threshold && !s.open {
            s.open = true;
            return true;
        }
        false
    }

    /// Check whether the circuit is open (too many failures) for a given key.
    pub fn is_open(&self, key: &str) -> bool {
        let map = self.state.lock();
        map.get(key).is_some_and(|s| s.open)
    }

    /// Manually close (reset) the circuit for a key.
    pub fn reset(&self, key: &str) {
        let mut map = self.state.lock();
        map.remove(key);
    }

    /// Return a snapshot of all open circuits.
    pub fn open_circuits(&self) -> Vec<String> {
        let map = self.state.lock();
        map.iter()
            .filter(|(_, s)| s.open)
            .map(|(k, _)| k.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_after_threshold() {
        let cb = CircuitBreakerRegistry::new(3);
        assert!(!cb.record_failure("bash"));
        assert!(!cb.record_failure("bash"));
        assert!(cb.record_failure("bash"));
        assert!(cb.is_open("bash"));
    }

    #[test]
    fn success_resets() {
        let cb = CircuitBreakerRegistry::new(3);
        cb.record_failure("edit");
        cb.record_failure("edit");
        cb.record_success("edit");
        assert!(!cb.is_open("edit"));
        assert!(!cb.record_failure("edit"));
    }

    #[test]
    fn independent_keys() {
        let cb = CircuitBreakerRegistry::new(2);
        cb.record_failure("a");
        cb.record_failure("a");
        assert!(cb.is_open("a"));
        assert!(!cb.is_open("b"));
    }
}
