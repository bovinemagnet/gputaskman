//! How every number is known. Only measured bytes are ever summed (AGENTS.md rule 1), and unknown is
//! never zero (rule 2). These types make the first rule a compile-time check.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// Only one explanation fits, or every explanation gives the same bytes.
    Consistent,
    /// A heuristic owner: several fit, paired by order.
    Order,
}

impl Confidence {
    pub fn marker(self) -> &'static str {
        match self {
            Confidence::Consistent => "*",
            Confidence::Order => "?",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateSource {
    Gguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reporter {
    LoadLog,
    LlamaServerApi,
    OllamaApi,
    LmStudioApi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    Legacy,
    SharedClient,
    SharedBuffers,
    Primary,
    Incomplete,
    PartialScan,
    NvmlNotLoaded,
    NvmlHidden,
    ProcessUnreadable,
    NotOnBackend,
    OutsideNamespace,
    KfdUntimed,
    NoBaseline,
    TornCounter,
    V1Unknown,
}

impl Reason {
    pub fn code(self) -> &'static str {
        match self {
            Reason::Legacy => "legacy",
            Reason::SharedClient => "shared_client",
            Reason::SharedBuffers => "shared_buffers",
            Reason::Primary => "primary",
            Reason::Incomplete => "incomplete",
            Reason::PartialScan => "partial_scan",
            Reason::NvmlNotLoaded => "nvml_not_loaded",
            Reason::NvmlHidden => "nvml_hidden",
            Reason::ProcessUnreadable => "process_unreadable",
            Reason::NotOnBackend => "not_on_backend",
            Reason::OutsideNamespace => "outside_namespace",
            Reason::KfdUntimed => "kfd_untimed",
            Reason::NoBaseline => "no_baseline",
            Reason::TornCounter => "torn_counter",
            Reason::V1Unknown => "v1_unknown",
        }
    }

    /// Python counts an NVML row with hidden memory as 0 in the attribution total (it is part of the
    /// unattributed remainder) but voids the total when a counter is missing. Keep that behaviour.
    pub fn voids_attribution(self) -> bool {
        !matches!(self, Reason::NvmlHidden)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Bytes {
    Measured {
        bytes: u64,
    },
    /// Exact size, inferred owner, already counted under another row: never summed.
    Inferred {
        bytes: u64,
        by: Confidence,
    },
    Estimated {
        bytes: u64,
        from: EstimateSource,
    },
    Reported {
        bytes: u64,
        by: Reporter,
    },
    Unavailable {
        reason: Reason,
    },
}

impl Bytes {
    pub fn measured(bytes: u64) -> Self {
        Bytes::Measured { bytes }
    }
    pub fn unavailable(reason: Reason) -> Self {
        Bytes::Unavailable { reason }
    }
    pub fn as_measured(&self) -> Option<Measured> {
        match *self {
            Bytes::Measured { bytes } => Some(Measured(bytes)),
            _ => None,
        }
    }
    pub fn is_measured(&self) -> bool {
        matches!(self, Bytes::Measured { .. })
    }
    /// Any known byte count, for display only. Never sum this.
    pub fn value(&self) -> Option<u64> {
        match *self {
            Bytes::Measured { bytes }
            | Bytes::Inferred { bytes, .. }
            | Bytes::Estimated { bytes, .. }
            | Bytes::Reported { bytes, .. } => Some(bytes),
            Bytes::Unavailable { .. } => None,
        }
    }
    pub fn reason(&self) -> Option<Reason> {
        match *self {
            Bytes::Unavailable { reason } => Some(reason),
            _ => None,
        }
    }
}

/// A measured byte count. Only `Bytes::as_measured` makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measured(u64);

impl Measured {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Default)]
pub struct MeasuredTotal(u64);

impl MeasuredTotal {
    pub fn new() -> Self {
        Self(0)
    }
    pub fn add(&mut self, m: Measured) {
        self.0 = self.0.saturating_add(m.0);
    }
    pub fn total(&self) -> u64 {
        self.0
    }
}

/// Measured or inferred bytes for display, sorting and scan priority. Has no path into `MeasuredTotal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shown(u64);

impl Shown {
    pub(crate) fn new(bytes: u64) -> Self {
        Self(bytes)
    }
    pub fn get(self) -> u64 {
        self.0
    }
}

/// amdgpu: resident bytes in every buffer a process can reach, shared ones included. A reference only:
/// the same buffer can appear under several processes, so these are never added up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resident {
    pub vram: Bytes,
    pub gtt: Bytes,
}
