//! Core data model: adapters, per-process GPU memory with provenance, and model evidence.
use crate::provenance::{Bytes, Confidence, MeasuredTotal, Reason, Resident, Shown};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;

pub type Pid = u32;

pub const VENDOR_NVIDIA: u16 = 0x10de;
pub const VENDOR_AMD: u16 = 0x1002;
pub const MIB: u64 = 1 << 20;
pub const GIB: u64 = 1 << 30;

/// Runtime tags that mean a process is doing AI work (`procinfo.AI_RUNTIMES`).
pub const AI_RUNTIMES: &[&str] = &[
    "llama.cpp",
    "whisper.cpp",
    "ggml",
    "PyTorch",
    "MLX",
    "TensorRT",
    "ONNX RT",
    "DirectML",
    "cuDNN",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Nvml,
    Drm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Adapter {
    pub key: u64,
    pub pci_bus: String,
    pub name: String,
    pub vendor_id: u16,
    pub dedicated_vram: u64,
    pub shared_system: u64,
    pub integrated: bool,
    pub backend: BackendKind,
}

impl Adapter {
    pub fn vendor(&self) -> String {
        match self.vendor_id {
            VENDOR_NVIDIA => "NVIDIA".to_string(),
            VENDOR_AMD => "AMD".to_string(),
            0x8086 => "Intel".to_string(),
            other => format!("{other:#x}"),
        }
    }
}

/// Stable integer adapter key from a PCI bus ID (the Linux stand-in for a WDDM LUID).
pub fn pci_key(bus_id: &str) -> u64 {
    let hex: String = bus_id
        .to_lowercase()
        .chars()
        .filter(|c| matches!(c, '0'..='9' | 'a'..='f'))
        .collect();
    u64::from_str_radix(&hex, 16).unwrap_or(0)
}

pub fn ordered_adapters(mut adapters: Vec<Adapter>) -> Vec<Adapter> {
    adapters.sort_by_key(|a| {
        (
            a.vendor_id != VENDOR_NVIDIA,
            Reverse(a.dedicated_vram),
            a.key,
        )
    });
    adapters
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectError {
    OutOfRange { index: usize, available: usize },
}

impl fmt::Display for SelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectError::OutOfRange { index, available } => {
                write!(
                    f,
                    "--gpu {index} is out of range; {available} GPU adapter(s) available"
                )
            }
        }
    }
}

impl std::error::Error for SelectError {}

pub fn select_adapter(adapters: &[Adapter], index: usize) -> Result<Option<Adapter>, SelectError> {
    let choices = ordered_adapters(adapters.to_vec());
    match choices.get(index) {
        Some(a) => Ok(Some(a.clone())),
        None if choices.is_empty() && index == 0 => Ok(None),
        None => Err(SelectError::OutOfRange {
            index,
            available: choices.len(),
        }),
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DeviceTelemetry {
    pub index: u32,
    pub name: String,
    pub pci_bus: String,
    pub driver_model: String,
    pub mem_total: Option<u64>,
    pub mem_used: Option<u64>,
    pub mem_reserved: Option<u64>,
    pub util_gpu: Option<f64>,
    pub util_mem: Option<f64>,
    pub enc: Option<f64>,
    pub dec: Option<f64>,
    pub temp: Option<f64>,
    pub power_w: Option<f64>,
    pub power_kind: String,
    pub power_limit_w: Option<f64>,
    pub fan: Option<u32>,
    pub pstate: Option<u32>,
    pub sm_clock: Option<u32>,
    pub sm_clock_max: Option<u32>,
    pub gfx_clock: Option<u32>,
    pub mem_clock: Option<u32>,
    pub video_clock: Option<u32>,
    pub pcie_rx_kbs: Option<u64>,
    pub pcie_tx_kbs: Option<u64>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    pub throttle: Vec<String>,
    pub compute_cap: String,
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdapterGpu {
    pub vram_used: Bytes,
    pub shared_used: Bytes,
}

impl Default for AdapterGpu {
    fn default() -> Self {
        Self {
            vram_used: Bytes::unavailable(Reason::NotOnBackend),
            shared_used: Bytes::unavailable(Reason::NotOnBackend),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Engines {
    Measured { by_engine: BTreeMap<String, f64> },
    Unavailable { reason: Reason },
}

impl Engines {
    /// Busiest engine percentage; `Some(0.0)` when measured but idle, `None` when unavailable.
    pub fn busiest(&self) -> Option<f64> {
        match self {
            Engines::Measured { by_engine } => {
                Some(by_engine.values().copied().fold(0.0, f64::max))
            }
            Engines::Unavailable { .. } => None,
        }
    }

    pub fn get(&self, name: &str) -> f64 {
        match self {
            Engines::Measured { by_engine } => by_engine.get(name).copied().unwrap_or(0.0),
            Engines::Unavailable { .. } => 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcGpu {
    pub vram: Bytes,
    pub shared: Bytes,
    pub committed: Bytes,
    pub engines: Engines,
    pub owner: Option<Confidence>,
    pub resident: Option<Resident>,
    pub kfd: bool,
}

impl ProcGpu {
    pub fn measured(vram: u64) -> Self {
        Self {
            vram: Bytes::measured(vram),
            shared: Bytes::unavailable(Reason::NotOnBackend),
            committed: Bytes::unavailable(Reason::NotOnBackend),
            engines: Engines::Measured {
                by_engine: BTreeMap::new(),
            },
            owner: None,
            resident: None,
            kfd: false,
        }
    }

    /// Measured or inferred VRAM for display, sorting and scan priority. Never summed.
    pub fn shown_vram(&self) -> Option<Shown> {
        match self.vram {
            Bytes::Measured { bytes } | Bytes::Inferred { bytes, .. } => Some(Shown::new(bytes)),
            _ => None,
        }
    }

    /// Committed memory not resident on the device or shared: clamped at zero, unknown without all inputs.
    pub fn paged_out(&self) -> Bytes {
        match (self.vram, self.shared, self.committed) {
            (
                Bytes::Measured { bytes: v },
                Bytes::Measured { bytes: s },
                Bytes::Measured { bytes: c },
            ) => Bytes::measured(c.saturating_sub(v.saturating_add(s))),
            (a, b, c) => {
                let reason = [a, b, c]
                    .iter()
                    .find_map(Bytes::reason)
                    .unwrap_or(Reason::NotOnBackend);
                Bytes::unavailable(reason)
            }
        }
    }

    /// Port of `collector._is_live`: worth showing when not attributed or visibly using the GPU.
    pub fn is_live(&self) -> bool {
        let at_least = |b: Bytes, floor: u64| b.as_measured().is_some_and(|m| m.get() >= floor);
        !self.vram.is_measured()
            || at_least(self.vram, MIB)
            || self.engines.busiest().is_some_and(|p| p >= 0.5)
            || at_least(self.shared, 16 * MIB)
            || at_least(self.paged_out(), 16 * MIB)
    }
}

/// Sum of measured VRAM across rows, or unavailable when a row's counter is missing.
/// Inferred, estimated and reported bytes are never summed.
pub fn attributed_total<'a>(rows: impl IntoIterator<Item = &'a ProcGpu>) -> Bytes {
    let mut total = MeasuredTotal::new();
    for row in rows {
        if let Bytes::Unavailable { reason } = row.vram
            && reason.voids_attribution()
        {
            return Bytes::unavailable(reason);
        }
        if let Some(m) = row.vram.as_measured() {
            total.add(m);
        }
    }
    Bytes::measured(total.total())
}

/// Evidence a model is in use, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    Api,
    Mapped,
    Cmdline,
    Installed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MappedModel {
    pub path: String,
    pub size: u64,
    pub resident: u64,
    pub gguf: bool,
    pub sources: Vec<Evidence>,
    pub store_name: Option<String>,
    pub kind: String,
    pub kind_hint: String,
}

impl MappedModel {
    pub fn add_source(&mut self, source: Evidence) {
        if !self.sources.contains(&source) {
            self.sources.push(source);
        }
    }

    pub fn kind_label(&self) -> String {
        if self.kind_hint.is_empty() {
            self.kind.clone()
        } else {
            format!("likely {} (name)", self.kind_hint)
        }
    }

    pub fn evidence_label(&self) -> String {
        let has = |e: Evidence| self.sources.contains(&e);
        let how = [
            (Evidence::Api, "api"),
            (Evidence::Mapped, "mapped"),
            (Evidence::Cmdline, "cmdline"),
        ]
        .iter()
        .filter(|(e, _)| has(*e))
        .map(|(_, n)| *n)
        .collect::<Vec<_>>()
        .join("+");
        if self.sources.len() >= 2 && (has(Evidence::Api) || has(Evidence::Mapped)) {
            return format!("confirmed ({how})");
        }
        if how.is_empty() && has(Evidence::Installed) {
            return "installed, not proven loaded".to_string();
        }
        match how.as_str() {
            "mapped" => "in use (mapped)".to_string(),
            "cmdline" => "launch arg".to_string(),
            "api" => "reported (api)".to_string(),
            _ => how,
        }
    }
}

pub fn is_diffusion_cmdline(cmd: &[String]) -> bool {
    cmd.iter()
        .any(|a| a == "--diffusion-model" || a.starts_with("--diffusion-model="))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcDetail {
    pub pid: Pid,
    pub name: String,
    pub exe: String,
    pub cmdline: Vec<String>,
    pub start_time: f64,
    pub start_ticks: Option<u64>,
    pub runtimes: Vec<String>,
    pub models: Vec<MappedModel>,
    pub candidates: Vec<MappedModel>,
    pub ports: Vec<u16>,
    pub parents: Option<Vec<Pid>>,
    pub access_denied: bool,
    pub maps_time: f64,
    pub scan_mem: f64,
}

impl ProcDetail {
    pub fn new(pid: Pid, name: impl Into<String>) -> Self {
        Self {
            pid,
            name: name.into(),
            exe: String::new(),
            cmdline: Vec::new(),
            start_time: 0.0,
            start_ticks: None,
            runtimes: Vec::new(),
            models: Vec::new(),
            candidates: Vec::new(),
            ports: Vec::new(),
            parents: None,
            access_denied: false,
            maps_time: 0.0,
            scan_mem: 0.0,
        }
    }

    pub fn is_ai(&self) -> bool {
        !self.models.is_empty()
            || self
                .runtimes
                .iter()
                .any(|r| AI_RUNTIMES.contains(&r.as_str()))
    }

    pub fn is_diffusion(&self) -> bool {
        is_diffusion_cmdline(&self.cmdline)
    }
}
