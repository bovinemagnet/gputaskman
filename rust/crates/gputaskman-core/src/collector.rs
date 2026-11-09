//! Assembles one Snapshot from the selected GPU backend, process inspection and LLM introspection
//! (`collector.py`, without the WSL, Windows and macOS branches).
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::env::Paths;
use crate::handy::{HandyEvidence, HandyProbe};
use crate::llm::gb;
use crate::llm::logs::ollama_log_path;
use crate::llm::servers::ServerProber;
use crate::llm::view::{LlmView, build_view};
use crate::model::{
    Adapter, BackendKind, DeviceTelemetry, Pid, ProcDetail, ProcGpu, SelectError, attributed_total,
    ordered_adapters, select_adapter,
};
use crate::procinfo::classify::{AI_KINDS, classify};
use crate::procinfo::inspector::ProcInspector;
use crate::procinfo::installed::Installed;
use crate::procinfo::models::ModelContext;
use crate::provenance::{Bytes, Reason};
use crate::traits::{Clock, Deadline, GpuBackend, HttpGet, ProcessSource, ScanPolicy};

/// Memory-map candidates: the top N VRAM users plus anything busy.
pub const DEEP_TOP_N: usize = 25;
/// Memory-map scanning per sample; leftovers wait for the next sample.
pub const SCAN_BUDGET: Duration = Duration::from_millis(150);

const OUTSIDE_NAME: &str = "(outside this container)";
const SIDE_KINDS: &[&str] = &["vision projector", "LoRA adapter"];

/// Where a row's PID comes from. An outside PID is from another PID namespace and never a local one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Local,
    Outside,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcRow {
    pub pid: Pid,
    pub name: String,
    pub kind: String,
    pub gpu: ProcGpu,
    pub detail: ProcDetail,
    pub llm: Option<LlmView>,
    pub handy: Option<HandyEvidence>,
    pub origin: Origin,
}

fn basename(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
}

fn last_part(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

impl ProcRow {
    /// Unique row id: PIDs from other namespaces and local PIDs can collide.
    pub fn key(&self) -> String {
        match self.origin {
            Origin::Local => self.pid.to_string(),
            Origin::Outside => format!("o{}", self.pid),
        }
    }

    pub fn pid_label(&self) -> String {
        match self.origin {
            Origin::Local => self.pid.to_string(),
            Origin::Outside => format!("H:{}", self.pid),
        }
    }

    pub fn runtime(&self) -> String {
        self.detail
            .runtimes
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("+")
    }

    /// One-line "what's in there" summary for the process table.
    pub fn inside(&self) -> String {
        if let Some(h) = &self.handy {
            if let Some(m) = h.last_model.as_deref().filter(|m| !m.is_empty()) {
                return format!(
                    "Handy last log ({}): {}",
                    h.last_event.as_deref().unwrap_or(""),
                    last_part(m)
                );
            }
            if let Some(m) = h.selected_model.as_deref().filter(|m| !m.is_empty()) {
                return format!("Handy selected (not load proof): {}", last_part(m));
            }
        }
        if let Some(llm) = self.llm.as_ref().filter(|l| !l.models.is_empty()) {
            return llm
                .models
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    if SIDE_KINDS.contains(&g.kind()) {
                        format!("+{}", g.kind())
                    } else {
                        format!("{} {} {}", llm.label(i), g.params_label(), g.quant())
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
        }
        let name = |m: &crate::model::MappedModel| {
            m.store_name
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| basename(&m.path).to_string())
        };
        let hint = |m: &crate::model::MappedModel| {
            if m.kind_hint.is_empty() {
                String::new()
            } else {
                format!(" ({})", m.kind_label())
            }
        };
        if !self.detail.models.is_empty() {
            return self
                .detail
                .models
                .iter()
                .take(2)
                .map(|m| format!("{}{}", name(m), hint(m)))
                .collect::<Vec<_>>()
                .join(", ");
        }
        if !self.detail.candidates.is_empty() {
            let listed = self
                .detail
                .candidates
                .iter()
                .take(2)
                .map(|m| format!("{} {}{}", name(m), gb(m.size as f64), hint(m)))
                .collect::<Vec<_>>()
                .join(", ");
            return format!("installed (unproven): {listed}");
        }
        match self.kind.as_str() {
            "compositor" => "desktop back buffers + composed frames",
            "external" => "a process in another PID namespace (host or another container)",
            "kernel" => "driver/kernel allocations",
            "browser" => "page textures, video, canvas/WebGL",
            "video" => "video frames (decode/encode)",
            "3D/game" => "textures, meshes, render targets",
            _ => "",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub taken: f64,
    pub adapter: Option<Adapter>,
    pub adapters: Vec<Adapter>,
    pub device: Option<DeviceTelemetry>,
    /// Why device telemetry could not be read; `None` when it was.
    pub device_status: Option<Reason>,
    pub driver: Option<String>,
    pub cuda: Option<String>,
    pub vram_total: Bytes,
    pub vram_used: Bytes,
    pub shared_used: Bytes,
    pub procs: Vec<ProcRow>,
    pub backend: Option<BackendKind>,
    /// What the backend's process scan could not cover; `None` when it saw everything.
    pub process_status: Option<String>,
    /// False on an APU (amdgpu iGPU): its GTT is ordinary GPU memory, not spill from a full VRAM.
    pub shared_is_spill: bool,
    pub notes: Vec<String>,
}

impl Snapshot {
    /// Measured VRAM over every row; unavailable when a row's counter is missing (rule 1).
    pub fn attributed(&self) -> Bytes {
        attributed_total(self.procs.iter().map(|r| &r.gpu))
    }
}

pub struct CollectorDeps {
    pub backends: Vec<Box<dyn GpuBackend>>,
    pub procs: Box<dyn ProcessSource>,
    pub http: Box<dyn HttpGet>,
    pub clock: Arc<dyn Clock>,
    pub paths: Paths,
    pub notes: Vec<String>,
}

/// Every backend's adapters in selection order (NVIDIA first, then by VRAM, then key). Linux
/// backends never report software adapters, so there is nothing further to exclude.
pub fn list_adapters(backends: &[Box<dyn GpuBackend>]) -> Vec<Adapter> {
    ordered_adapters(backends.iter().flat_map(|b| b.adapters()).collect())
}

pub struct Collector {
    /// Only the backend owning the selected adapter; `None` when there is no supported GPU.
    backend: Option<Box<dyn GpuBackend>>,
    adapter: Option<Adapter>,
    adapters: Vec<Adapter>,
    procs: Box<dyn ProcessSource>,
    http: Box<dyn HttpGet>,
    clock: Arc<dyn Clock>,
    paths: Paths,
    notes: Vec<String>,
    gpu_index: usize,
    policy: ScanPolicy,
    inspector: ProcInspector,
    prober: ServerProber,
    models: ModelContext,
    installed: Installed,
    handy: HandyProbe,
}

/// Python's `{"dedicated", "shared"} <= available`: a counter the backend lacks entirely
/// (`NotOnBackend`) or that NVML hides (`NvmlHidden`) still counts as available there; one the
/// backend could not attribute to this process does not.
fn counter_known(b: Bytes) -> bool {
    !matches!(
        b,
        Bytes::Unavailable { reason } if !matches!(reason, Reason::NotOnBackend | Reason::NvmlHidden)
    )
}

fn shown(g: &ProcGpu) -> u64 {
    g.shown_vram().map_or(0, |s| s.get())
}

fn shared(g: &ProcGpu) -> u64 {
    g.shared.value().unwrap_or(0)
}

impl Collector {
    pub fn new(
        deps: CollectorDeps,
        gpu_index: usize,
        policy: ScanPolicy,
    ) -> Result<Collector, SelectError> {
        let owned: Vec<(Adapter, usize)> = deps
            .backends
            .iter()
            .enumerate()
            .flat_map(|(i, b)| b.adapters().into_iter().map(move |a| (a, i)))
            .collect();
        let adapters = ordered_adapters(owned.iter().map(|(a, _)| a.clone()).collect());
        let adapter = select_adapter(&adapters, gpu_index)?;
        let owner = adapter
            .as_ref()
            .and_then(|a| owned.iter().find(|(o, _)| o == a).map(|(_, i)| *i));
        // Keep only the owning backend: DRM never walks fds while an NVIDIA GPU is selected.
        let mut backend = owner.and_then(|i| deps.backends.into_iter().nth(i));
        if let (Some(b), Some(a)) = (backend.as_mut(), adapter.as_ref()) {
            b.start(a, policy);
        }
        Ok(Collector {
            backend,
            adapter,
            adapters,
            procs: deps.procs,
            http: deps.http,
            clock: deps.clock,
            inspector: ProcInspector::new(),
            prober: ServerProber::default(),
            models: ModelContext::new(deps.paths.clone()),
            installed: Installed::new(deps.paths.clone()),
            handy: HandyProbe::new(deps.paths.clone()),
            paths: deps.paths,
            notes: deps.notes,
            gpu_index,
            policy,
        })
    }

    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }

    /// Exclusive ports of the process's parents (an Ollama or LM Studio front-end), nearest first.
    fn parent_ports(&mut self, pid: Pid, now: f64) -> Vec<u16> {
        let procs = self.procs.as_ref();
        let parents = self.inspector.parents(procs, pid, 2);
        parents
            .into_iter()
            .flat_map(|ppid| self.inspector.exclusive_ports(procs, ppid, now))
            .collect()
    }

    #[allow(clippy::too_many_lines)]
    pub fn sample(&mut self) -> Snapshot {
        let taken = self.clock.wall();
        let now = taken;
        let mut snap = Snapshot {
            taken,
            adapter: self.adapter.clone(),
            adapters: self.adapters.clone(),
            device: None,
            device_status: None,
            driver: None,
            cuda: None,
            vram_total: Bytes::unavailable(Reason::NotOnBackend),
            vram_used: Bytes::unavailable(Reason::NotOnBackend),
            shared_used: Bytes::unavailable(Reason::NotOnBackend),
            procs: Vec::new(),
            backend: None,
            process_status: None,
            shared_is_spill: true,
            notes: self.notes.clone(),
        };
        let (Some(backend), Some(adapter)) = (self.backend.as_mut(), self.adapter.as_ref()) else {
            return snap;
        };
        let sample = backend.sample();
        let (device, device_status) = match backend.device(self.gpu_index) {
            Ok(d) => (Some(d), None),
            Err(reason) => (None, Some(reason)),
        };
        let total = device
            .as_ref()
            .and_then(|d| d.mem_total)
            .filter(|t| *t > 0)
            .or(Some(adapter.dedicated_vram).filter(|t| *t > 0));
        snap.vram_total = total.map_or(Bytes::unavailable(Reason::NotOnBackend), Bytes::measured);
        // Versions belong to the telemetry that was read; without it the header shows none.
        if device.is_some() {
            snap.driver = backend.driver();
            snap.cuda = backend.cuda();
        }
        snap.backend = Some(backend.kind());
        snap.device = device;
        snap.device_status = device_status;
        snap.vram_used = sample.adapter.vram_used;
        snap.shared_used = sample.adapter.shared_used;
        snap.process_status = sample.status.clone();
        snap.shared_is_spill = !(adapter.backend == BackendKind::Drm && adapter.integrated);

        let procs = self.procs.as_ref();
        let mut live: Vec<(Pid, &ProcGpu)> = sample
            .procs
            .iter()
            .filter(|(_, g)| g.is_live())
            .map(|(pid, g)| (*pid, g))
            .collect();
        // Largest first, as Python's -(shown_vram or 0): unknown VRAM ties with 0 and the stable
        // sort keeps pid order.
        live.sort_by_key(|(_, g)| Reverse(shown(g)));
        for (pid, g) in live {
            let d = self.inspector.get(procs, pid, now).clone();
            snap.procs.push(ProcRow {
                pid,
                name: d.name.clone(),
                kind: classify(&d, &g.engines),
                gpu: g.clone(),
                detail: d,
                llm: None,
                handy: None,
                origin: Origin::Local,
            });
        }
        let local = snap.procs.len();
        for (pid, g) in &sample.outside {
            // A container sees the host's PIDs from NVML; unmatched ones are other containers or the host.
            snap.procs.push(ProcRow {
                pid: *pid,
                name: OUTSIDE_NAME.to_string(),
                kind: "external".to_string(),
                gpu: g.clone(),
                detail: ProcDetail::new(*pid, OUTSIDE_NAME),
                llm: None,
                handy: None,
                origin: Origin::Outside,
            });
        }

        let rows = &mut snap.procs[..local];
        let deep: BTreeSet<Pid> = rows
            .iter()
            .take(DEEP_TOP_N)
            .map(|r| r.pid)
            .chain(
                rows.iter()
                    .filter(|r| r.gpu.engines.busiest().is_some_and(|p| p >= 2.0))
                    .map(|r| r.pid),
            )
            .collect();
        let clock = self.clock.as_ref();
        let deadline = match self.policy {
            ScanPolicy::Budgeted => Deadline::after(clock, SCAN_BUDGET),
            ScanPolicy::Unbounded => Deadline::none(),
        };
        let mut ollama_runners: Vec<Pid> = Vec::new();
        for row in rows.iter_mut() {
            // Largest VRAM first, so the budget goes to what matters.
            let mem = (shown(&row.gpu) + shared(&row.gpu)) as f64;
            if deep.contains(&row.pid)
                && !deadline.passed(clock)
                && self.inspector.needs_scan(&row.detail, mem, now)
            {
                self.inspector.scan(
                    row.pid,
                    procs,
                    &mut self.models,
                    &mut self.installed,
                    mem,
                    now,
                );
                row.detail = self.inspector.get(procs, row.pid, now).clone();
            }
            let d = &row.detail;
            row.kind = classify(d, &row.gpu.engines);
            // Old Ollama: "ollama runner ..."; newer Ollama: its own lib/ollama/llama-server.
            let exe = d.exe.to_lowercase();
            if AI_KINDS.contains(&row.kind.as_str())
                && (exe.contains("\\ollama\\")
                    || exe.contains("/ollama/")
                    || (d.name.to_lowercase().starts_with("ollama")
                        && d.cmdline.iter().any(|a| a == "runner")))
            {
                ollama_runners.push(row.pid);
            }
        }

        let spill = snap.shared_is_spill;
        for i in 0..local {
            let row = &snap.procs[i];
            let handy = matches!(
                basename(&row.detail.exe).to_lowercase().as_str(),
                "handy.exe" | "handy"
            )
            .then(|| self.handy.inspect(row.detail.start_time))
            .flatten();
            let wants_llm = !row.detail.is_diffusion()
                && AI_KINDS.contains(&row.kind.as_str())
                && (!row.detail.models.is_empty() || row.kind == "LLM")
                && counter_known(row.gpu.vram)
                && counter_known(row.gpu.shared);
            let pid = row.pid;
            let llm = if wants_llm {
                let own_ports = self
                    .inspector
                    .exclusive_ports(self.procs.as_ref(), pid, now);
                let own = self.prober.get(self.http.as_ref(), &own_ports, now);
                // Ollama / LM Studio front-end process.
                let host_ports = self.parent_ports(pid, now);
                let host = self.prober.get(self.http.as_ref(), &host_ports, now);
                let log = (ollama_runners.len() == 1 && ollama_runners[0] == pid)
                    .then(|| ollama_log_path(&self.paths));
                let row = &snap.procs[i];
                let (vram, gtt) = (shown(&row.gpu), shared(&row.gpu));
                // On an APU, GTT is the GPU's ordinary memory (often most of it), not spill: count it as held memory.
                let held = vram + if spill { 0 } else { gtt };
                let host_info = if own.is_some() { host.clone() } else { None };
                Some(build_view(
                    &row.detail,
                    held as f64,
                    if spill { gtt as f64 } else { 0.0 },
                    own.or(host),
                    log,
                    host_info,
                    row.gpu.shown_vram().is_some(),
                    &mut self.models,
                    now,
                ))
            } else {
                None
            };
            let row = &mut snap.procs[i];
            row.handy = handy;
            row.llm = llm;
        }

        self.inspector
            .prune(&sample.procs.keys().copied().collect());
        snap
    }
}
