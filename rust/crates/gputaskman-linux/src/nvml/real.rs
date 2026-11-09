//! NVML over `nvml-wrapper`, which loads the installed `libnvidia-ml.so` at run time. The library is
//! never bundled (AGENTS.md rule 8). Not exercised on hardware by the tests: this machine has no NVIDIA
//! driver, so the mapping below is compile-checked only.
use crate::nvml::api::{NvProcess, NvUtilSample, NvmlApi, format_cuda_version};
use gputaskman_core::model::DeviceTelemetry;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{
    Clock, PcieUtilCounter, PerformanceState, TemperatureSensor,
};
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::struct_wrappers::device::ProcessInfo;

pub struct RealNvml {
    nvml: Nvml,
}

impl RealNvml {
    /// `Err` (never a panic) when the driver's library is absent or NVML cannot initialise.
    pub fn load() -> Result<Self, String> {
        Nvml::init()
            .map(|nvml| Self { nvml })
            .map_err(|e| e.to_string())
    }
}

/// NVML reports `NVML_VALUE_NOT_AVAILABLE` (unsigned -1) when per-process memory is hidden.
fn used(m: &UsedGpuMemory) -> Option<u64> {
    match m {
        UsedGpuMemory::Used(n) if *n < (1 << 63) => Some(*n),
        _ => None,
    }
}

fn procs(list: Vec<ProcessInfo>) -> Vec<NvProcess> {
    list.iter()
        .map(|p| NvProcess {
            pid: p.pid,
            used: used(&p.used_gpu_memory),
        })
        .collect()
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Python's `_THROTTLE` table.
const THROTTLE: [(u64, &str); 6] = [
    (0x2, "app-clocks"),
    (0x4, "sw-power-cap"),
    (0x8, "hw-slowdown"),
    (0x20, "sw-thermal"),
    (0x40, "hw-thermal"),
    (0x80, "power-brake"),
];

impl NvmlApi for RealNvml {
    fn device_count(&self) -> Result<u32, String> {
        self.nvml.device_count().map_err(err)
    }

    fn pci_bus_id(&self, i: u32) -> Result<String, String> {
        let d = self.nvml.device_by_index(i).map_err(err)?;
        d.pci_info().map(|p| p.bus_id).map_err(err)
    }

    fn name(&self, i: u32) -> Result<String, String> {
        self.nvml
            .device_by_index(i)
            .map_err(err)?
            .name()
            .map_err(err)
    }

    fn memory(&self, i: u32) -> Result<(u64, u64), String> {
        let m = self
            .nvml
            .device_by_index(i)
            .map_err(err)?
            .memory_info()
            .map_err(err)?;
        Ok((m.total, m.used))
    }

    fn compute_processes(&self, i: u32) -> Result<Vec<NvProcess>, String> {
        let d = self.nvml.device_by_index(i).map_err(err)?;
        d.running_compute_processes().map(procs).map_err(err)
    }

    fn graphics_processes(&self, i: u32) -> Result<Vec<NvProcess>, String> {
        let d = self.nvml.device_by_index(i).map_err(err)?;
        d.running_graphics_processes().map(procs).map_err(err)
    }

    fn process_utilization(&self, i: u32, since: u64) -> Result<Vec<NvUtilSample>, String> {
        let d = self.nvml.device_by_index(i).map_err(err)?;
        let samples = d.process_utilization_stats(since).map_err(err)?;
        Ok(samples
            .iter()
            .map(|s| NvUtilSample {
                pid: s.pid,
                sm: s.sm_util,
                enc: s.enc_util,
                dec: s.dec_util,
                timestamp: s.timestamp,
            })
            .collect())
    }

    /// Port of `nvml.py` `Nvml.sample`: every field that cannot be read stays `None`.
    fn telemetry(&self, i: u32) -> DeviceTelemetry {
        let mut t = DeviceTelemetry {
            index: i,
            name: "?".to_string(),
            source: "NVML".to_string(),
            ..DeviceTelemetry::default()
        };
        let Ok(d) = self.nvml.device_by_index(i) else {
            return t;
        };
        if let Ok(name) = d.name() {
            t.name = name;
        }
        if let Ok(p) = d.pci_info() {
            t.pci_bus = p.bus_id;
        }
        if let Ok(m) = d.memory_info() {
            t.mem_total = Some(m.total);
            t.mem_used = Some(m.used);
            t.mem_reserved = Some(m.reserved);
        }
        if let Ok(u) = d.utilization_rates() {
            t.util_gpu = Some(f64::from(u.gpu));
            t.util_mem = Some(f64::from(u.memory));
        }
        t.enc = d
            .encoder_utilization()
            .ok()
            .map(|u| f64::from(u.utilization));
        t.dec = d
            .decoder_utilization()
            .ok()
            .map(|u| f64::from(u.utilization));
        t.temp = d.temperature(TemperatureSensor::Gpu).ok().map(f64::from);
        t.power_w = d.power_usage().ok().map(|mw| f64::from(mw) / 1000.0);
        t.power_limit_w = d
            .enforced_power_limit()
            .ok()
            .map(|mw| f64::from(mw) / 1000.0);
        t.fan = d.fan_speed(0).ok();
        t.pstate = d.performance_state().ok().and_then(|p| match p {
            PerformanceState::Unknown => None,
            other => Some(other.as_c()),
        });
        t.sm_clock = d.clock_info(Clock::SM).ok();
        t.sm_clock_max = d.max_clock_info(Clock::SM).ok();
        t.mem_clock = d.clock_info(Clock::Memory).ok();
        t.video_clock = d.clock_info(Clock::Video).ok();
        t.pcie_rx_kbs = d
            .pcie_throughput(PcieUtilCounter::Receive)
            .ok()
            .map(u64::from);
        t.pcie_tx_kbs = d.pcie_throughput(PcieUtilCounter::Send).ok().map(u64::from);
        t.pcie_gen = d.current_pcie_link_gen().ok();
        t.pcie_width = d.current_pcie_link_width().ok();
        if let Ok(reasons) = d.current_throttle_reasons() {
            let bits = reasons.bits();
            t.throttle = THROTTLE
                .iter()
                .filter(|(bit, _)| bits & bit != 0)
                .map(|(_, n)| (*n).to_string())
                .collect();
        }
        if let Ok(cc) = d.cuda_compute_capability() {
            t.compute_cap = format!("{}.{}", cc.major, cc.minor);
        }
        t
    }

    fn driver_version(&self) -> Option<String> {
        self.nvml
            .sys_driver_version()
            .ok()
            .filter(|v| !v.is_empty())
    }

    fn cuda_version(&self) -> Option<String> {
        self.nvml
            .sys_cuda_driver_version()
            .ok()
            .filter(|v| *v != 0)
            .map(format_cuda_version)
    }
}
