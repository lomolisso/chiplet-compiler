//! Cooperative cancellation. The scheduler sets the token on `stop_on_failure` or when
//! a pass exceeds `timeout_min`; passes check it between work units and runners kill their children.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::{CoreError, Result};

#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// `Err(Cancelled)` once cancelled; call between work units.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(CoreError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Sleep in short slices, returning early with `Err(Cancelled)`.
    pub fn sleep(&self, d: Duration) -> Result<()> {
        let deadline = std::time::Instant::now() + d;
        loop {
            self.check()?;
            let now = std::time::Instant::now();
            if now >= deadline {
                return Ok(());
            }
            std::thread::sleep((deadline - now).min(Duration::from_millis(10)));
        }
    }
}
