//! Core services shared by every pass: configuration, the artifact store, manifests and verify
//! reports, cache keys, cancellation and the capped tool runners (design §3.4–§3.6).

pub mod artifact;
pub mod cache;
pub mod cancel;
pub mod canon;
pub mod config;
pub mod error;
pub mod manifest;
pub mod runner;
pub mod schema;
pub mod verify;

pub use artifact::{ArtifactKind, ArtifactStore, KindDir, PassDir};
pub use cancel::CancelToken;
pub use config::{Config, LoadedConfig, PassResources, PassSection};
pub use error::{CoreError, Result};
pub use manifest::{Manifest, PassStatus};
pub use verify::{Check, CheckStatus, Evidence, VerifyReport};
