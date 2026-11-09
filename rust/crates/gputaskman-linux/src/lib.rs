//! Linux backends for gputaskman.
pub mod drm;
pub mod nvml;
pub mod pidns;
pub mod procs;
pub mod sysfs;

use gputaskman_core::traits::{Clock, GpuBackend};
use std::path::PathBuf;
use std::sync::Arc;

/// The GPU backends this machine can offer, plus notes explaining what is missing.
///
/// NVML is added when the driver's library loads; a DRM backend is added when amdgpu cards exist and
/// not under WSL. DRM adapters whose key an NVML adapter already has are dropped. The kernel release
/// comes from `/proc/sys/kernel/osrelease`, which needs no `unsafe`.
pub fn linux_backends(clock: Arc<dyn Clock>) -> (Vec<Box<dyn GpuBackend>>, Vec<String>) {
    let mut backends: Vec<Box<dyn GpuBackend>> = Vec::new();
    let mut notes = Vec::new();
    let mut nvml_keys = Vec::new();

    match nvml::real::RealNvml::load() {
        Ok(api) => {
            let backend = nvml::backend::NvmlBackend::new(
                Box::new(api),
                Box::new(pidns::ProcNamespace::new("/proc")),
            );
            nvml_keys = backend.adapters().iter().map(|a| a.key).collect();
            backends.push(Box::new(backend));
        }
        Err(e) => notes.push(format!("NVML not loaded: {e}")),
    }

    let release = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if !release.to_lowercase().contains("microsoft") {
        let drm = drm::backend::DrmBackend::new(
            PathBuf::from("/sys"),
            PathBuf::from("/proc"),
            clock,
            release,
        );
        let drm = drm.without_keys(&nvml_keys);
        if !drm.adapters().is_empty() {
            backends.push(Box::new(drm));
        }
    }
    (backends, notes)
}
