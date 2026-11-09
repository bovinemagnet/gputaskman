//! Linux backends for gputaskman.
pub mod drm;
#[cfg(any(feature = "test-support", feature = "parity"))]
pub mod fixtures;
pub mod nvml;
#[cfg(feature = "parity")]
pub mod parity;
pub mod pidns;
pub mod procs;
pub mod sysfs;

use gputaskman_core::traits::{Clock, GpuBackend};
use std::path::Path;
use std::sync::Arc;

/// The note given instead of any GPU data under WSL.
const WSL_NOTE: &str = "WSL is not supported by the Rust build yet (milestone 1 is native Linux); no GPU data is collected";

/// The GPU backends this machine can offer, plus notes explaining what is missing.
///
/// NVML is added when the driver's library loads; a DRM backend is added when amdgpu cards exist.
/// DRM adapters whose key an NVML adapter already has are dropped. Under WSL no backend is built
/// (NVML there reports Windows host PIDs) and a note says so. The kernel release comes from
/// `/proc/sys/kernel/osrelease`, which needs no `unsafe`.
pub fn linux_backends(clock: Arc<dyn Clock>) -> (Vec<Box<dyn GpuBackend>>, Vec<String>) {
    linux_backends_at(Path::new("/sys"), Path::new("/proc"), clock)
}

/// As [`linux_backends`], reading from the given sysfs and procfs roots (tests use fixture trees).
pub fn linux_backends_at(
    sys_root: &Path,
    proc_root: &Path,
    clock: Arc<dyn Clock>,
) -> (Vec<Box<dyn GpuBackend>>, Vec<String>) {
    let release = std::fs::read_to_string(proc_root.join("sys/kernel/osrelease"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let lower = release.to_lowercase();
    if lower.contains("microsoft") || lower.contains("wsl") {
        return (Vec::new(), vec![WSL_NOTE.to_string()]);
    }

    let mut backends: Vec<Box<dyn GpuBackend>> = Vec::new();
    let mut notes = Vec::new();
    let mut nvml_keys = Vec::new();

    match nvml::real::RealNvml::load() {
        Ok(api) => {
            let backend = nvml::backend::NvmlBackend::new(
                Box::new(api),
                Box::new(pidns::ProcNamespace::new(proc_root)),
            );
            nvml_keys = backend.adapters().iter().map(|a| a.key).collect();
            backends.push(Box::new(backend));
        }
        Err(e) => notes.push(format!("NVML not loaded: {e}")),
    }

    let drm = drm::backend::DrmBackend::new(
        sys_root.to_path_buf(),
        proc_root.to_path_buf(),
        clock,
        release,
    );
    let drm = drm.without_keys(&nvml_keys);
    if !drm.adapters().is_empty() {
        backends.push(Box::new(drm));
    }
    (backends, notes)
}
