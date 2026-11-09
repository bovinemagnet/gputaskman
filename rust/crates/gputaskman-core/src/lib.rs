//! Platform-neutral core of gputaskman.
pub mod collector;
pub mod diff;
pub mod env;
pub mod gguf;
pub mod handy;
pub mod http;
pub mod llm;
pub mod model;
pub mod procinfo;
pub mod provenance;
pub mod report;
pub mod schema;
pub mod stores;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;
pub mod traits;
