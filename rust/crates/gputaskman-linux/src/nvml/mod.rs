//! NVIDIA GPUs through NVML: adapters, per-process memory and engines, device telemetry.
//!
//! The vendor library is loaded from the installed driver at run time; it is never bundled.
pub mod api;
pub mod backend;
pub mod real;
