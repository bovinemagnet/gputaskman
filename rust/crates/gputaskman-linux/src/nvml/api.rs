//! The slice of NVML the backend uses, so the real library and a scripted fake are interchangeable.
use gputaskman_core::model::DeviceTelemetry;

/// One entry of NVML's compute or graphics process list. `used` is `None` when per-process memory is
/// hidden (`NVML_VALUE_NOT_AVAILABLE`, or any value of 2^63 or more).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvProcess {
    pub pid: u32,
    pub used: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvUtilSample {
    pub pid: u32,
    pub sm: u32,
    pub enc: u32,
    pub dec: u32,
    pub timestamp: u64,
}

pub trait NvmlApi: Send {
    fn device_count(&self) -> Result<u32, String>;
    fn pci_bus_id(&self, index: u32) -> Result<String, String>;
    fn name(&self, index: u32) -> Result<String, String>;
    /// `(total, used)` in bytes.
    fn memory(&self, index: u32) -> Result<(u64, u64), String>;
    fn compute_processes(&self, index: u32) -> Result<Vec<NvProcess>, String>;
    fn graphics_processes(&self, index: u32) -> Result<Vec<NvProcess>, String>;
    /// Utilisation samples newer than `since`. An error when nothing ran since (`NVML_ERROR_NOT_FOUND`).
    fn process_utilization(&self, index: u32, since: u64) -> Result<Vec<NvUtilSample>, String>;
    fn telemetry(&self, index: u32) -> DeviceTelemetry;
    fn driver_version(&self) -> Option<String>;
    /// `major.minor`, for example `12.4`.
    fn cuda_version(&self) -> Option<String>;
}

/// `cuda_version` text from NVML's integer, for example 12040 -> `12.4`.
pub fn format_cuda_version(v: i32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

#[cfg(any(test, feature = "test-support"))]
pub use fake::FakeNvml;

#[cfg(any(test, feature = "test-support"))]
mod fake {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, Default)]
    struct FakeDevice {
        bus: String,
        name: String,
        total: u64,
        used: u64,
        compute: Vec<NvProcess>,
        graphics: Vec<NvProcess>,
        samples: Vec<NvUtilSample>,
        failing: bool,
    }

    /// A scripted NVML: devices with process lists and utilisation samples.
    #[derive(Debug, Default)]
    pub struct FakeNvml {
        devices: Vec<FakeDevice>,
        count_fails: bool,
        util_calls: Arc<Mutex<Vec<u64>>>,
    }

    impl FakeNvml {
        pub fn device(mut self, bus: &str, name: &str, total: u64, used: u64) -> Self {
            self.devices.push(FakeDevice {
                bus: bus.to_string(),
                name: name.to_string(),
                total,
                used,
                ..FakeDevice::default()
            });
            self
        }

        fn with(mut self, i: usize, f: impl FnOnce(&mut FakeDevice)) -> Self {
            if let Some(d) = self.devices.get_mut(i) {
                f(d);
            }
            self
        }

        pub fn compute(self, i: usize, procs: Vec<NvProcess>) -> Self {
            self.with(i, |d| d.compute = procs)
        }

        pub fn graphics(self, i: usize, procs: Vec<NvProcess>) -> Self {
            self.with(i, |d| d.graphics = procs)
        }

        pub fn samples(self, i: usize, samples: Vec<NvUtilSample>) -> Self {
            self.with(i, |d| d.samples = samples)
        }

        /// Every read of this device fails, as for a GPU that fell off the bus.
        pub fn fail_handle(self, i: usize) -> Self {
            self.with(i, |d| d.failing = true)
        }

        /// `device_count` fails.
        pub fn fail_count(mut self) -> Self {
            self.count_fails = true;
            self
        }

        /// The `since` argument of every utilisation call so far.
        pub fn util_calls(&self) -> Vec<u64> {
            self.util_calls
                .lock()
                .map(|c| c.clone())
                .unwrap_or_default()
        }

        /// A handle that stays valid after the fake is boxed into a backend.
        pub fn util_calls_handle(&self) -> Arc<Mutex<Vec<u64>>> {
            Arc::clone(&self.util_calls)
        }

        fn dev(&self, i: u32) -> Result<&FakeDevice, String> {
            match self.devices.get(i as usize) {
                Some(d) if !d.failing => Ok(d),
                Some(_) => Err("NVML_ERROR_GPU_IS_LOST".to_string()),
                None => Err("NVML_ERROR_INVALID_ARGUMENT".to_string()),
            }
        }
    }

    impl NvmlApi for FakeNvml {
        fn device_count(&self) -> Result<u32, String> {
            if self.count_fails {
                return Err("NVML_ERROR_UNKNOWN".to_string());
            }
            Ok(self.devices.len() as u32)
        }
        fn pci_bus_id(&self, i: u32) -> Result<String, String> {
            self.dev(i).map(|d| d.bus.clone())
        }
        fn name(&self, i: u32) -> Result<String, String> {
            self.dev(i).map(|d| d.name.clone())
        }
        fn memory(&self, i: u32) -> Result<(u64, u64), String> {
            self.dev(i).map(|d| (d.total, d.used))
        }
        fn compute_processes(&self, i: u32) -> Result<Vec<NvProcess>, String> {
            self.dev(i).map(|d| d.compute.clone())
        }
        fn graphics_processes(&self, i: u32) -> Result<Vec<NvProcess>, String> {
            self.dev(i).map(|d| d.graphics.clone())
        }
        fn process_utilization(&self, i: u32, since: u64) -> Result<Vec<NvUtilSample>, String> {
            if let Ok(mut calls) = self.util_calls.lock() {
                calls.push(since);
            }
            self.dev(i).map(|d| d.samples.clone())
        }
        fn telemetry(&self, i: u32) -> DeviceTelemetry {
            let mut t = DeviceTelemetry {
                index: i,
                source: "NVML".to_string(),
                name: "?".to_string(),
                ..DeviceTelemetry::default()
            };
            if let Ok(d) = self.dev(i) {
                t.name = d.name.clone();
                t.pci_bus = d.bus.clone();
                t.mem_total = Some(d.total);
                t.mem_used = Some(d.used);
            }
            t
        }
        fn driver_version(&self) -> Option<String> {
            Some("550.00".to_string())
        }
        fn cuda_version(&self) -> Option<String> {
            Some(format_cuda_version(12040))
        }
    }
}
