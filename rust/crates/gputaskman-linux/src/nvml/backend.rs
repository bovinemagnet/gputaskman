//! The NVML backend: adapters, per-process memory and engines, and container PID matching.
use crate::nvml::api::{NvProcess, NvUtilSample, NvmlApi};
use crate::pidns::{NamespaceProbe, PidMatcher};
use gputaskman_core::model::{
    Adapter, AdapterGpu, BackendKind, DeviceTelemetry, Engines, Pid, ProcGpu, VENDOR_NVIDIA,
    pci_key,
};
use gputaskman_core::provenance::{Bytes, Reason};
use gputaskman_core::traits::{BackendSample, GpuBackend, ScanPolicy};
use std::collections::{BTreeMap, BTreeSet};

pub struct NvmlBackend {
    api: Box<dyn NvmlApi>,
    ns: Box<dyn NamespaceProbe>,
    matcher: PidMatcher,
    /// The NVML index of the selected GPU, once `start()` picked one.
    selected: Option<u32>,
    #[allow(dead_code)] // recorded for the collector's policy; NVML has no scan to bound
    policy: ScanPolicy,
    /// Newest utilisation timestamp seen: the next poll asks only for newer samples.
    last_ts: u64,
}

impl NvmlBackend {
    pub fn new(api: Box<dyn NvmlApi>, ns: Box<dyn NamespaceProbe>) -> Self {
        Self {
            api,
            ns,
            matcher: PidMatcher::default(),
            selected: None,
            policy: ScanPolicy::Budgeted(gputaskman_core::collector::SCAN_BUDGET),
            last_ts: 0,
        }
    }

    /// `(NVML index, adapter)` per GPU NVML can open; a lost or failing GPU is skipped, not fatal.
    fn devices(&self) -> Vec<(u32, Adapter)> {
        let Ok(count) = self.api.device_count() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for i in 0..count {
            let (Ok(name), Ok((total, _))) = (self.api.name(i), self.api.memory(i)) else {
                continue;
            };
            let (key, pci_bus) = match self.api.pci_bus_id(i) {
                Ok(bus) => (pci_key(&bus), bus),
                Err(_) => (u64::from(i) + 1, String::new()),
            };
            out.push((
                i,
                Adapter {
                    key,
                    pci_bus,
                    name,
                    vendor_id: VENDOR_NVIDIA,
                    dedicated_vram: total,
                    shared_system: 0,
                    integrated: false,
                    backend: BackendKind::Nvml,
                },
            ));
        }
        out
    }

    /// Latest utilisation sample per PID since the previous call.
    fn utilisation(&mut self, index: u32) -> BTreeMap<u32, NvUtilSample> {
        let Ok(samples) = self.api.process_utilization(index, self.last_ts) else {
            return BTreeMap::new(); // nothing ran since the last sample
        };
        let mut latest: BTreeMap<u32, NvUtilSample> = BTreeMap::new();
        for s in samples {
            match latest.get(&s.pid) {
                Some(have) if s.timestamp <= have.timestamp => {}
                _ => {
                    latest.insert(s.pid, s);
                }
            }
        }
        if let Some(newest) = latest.values().map(|s| s.timestamp).max() {
            self.last_ts = newest;
        }
        latest
    }

    /// Re-keys NVML's PIDs to local ones. In the host PID namespace they already are local, including
    /// other users' processes whose `/proc/<pid>/fd` this monitor cannot read. In a container's own
    /// namespace every NVML PID is a host PID, even one that equals a local PID, so all of them are
    /// matched by creation order. Unmatched host PIDs go to the second map and are never looked up locally.
    fn resolve_namespaces(
        &mut self,
        procs: BTreeMap<Pid, ProcGpu>,
        compute_pids: &BTreeSet<u32>,
    ) -> (BTreeMap<Pid, ProcGpu>, BTreeMap<Pid, ProcGpu>) {
        let nv_pids: BTreeSet<u32> = procs.keys().copied().collect();
        let unresolved: BTreeSet<u32> = match self.ns.in_child_pid_namespace() {
            // Unknown: PIDs that do not exist here cannot be local.
            None => nv_pids
                .iter()
                .copied()
                .filter(|p| !self.ns.exists(*p))
                .collect(),
            Some(true) => nv_pids.clone(),
            Some(false) => BTreeSet::new(),
        };
        if unresolved.is_empty() {
            self.matcher.update(&[], &[]);
            return (procs, BTreeMap::new());
        }
        let holders = self.ns.nvidia_holders();
        let me = self.ns.self_pid();
        let resolved_locally: BTreeSet<u32> = nv_pids.difference(&unresolved).copied().collect();
        let local: Vec<u32> = holders
            .difference(&resolved_locally)
            .copied()
            .filter(|p| *p != me && self.ns.has_cuda(*p))
            .collect();
        let compute: Vec<u32> = unresolved
            .iter()
            .copied()
            .filter(|p| compute_pids.contains(p))
            .collect(); // CUDA, so local libcuda users
        let (matched, _) = self.matcher.update(&compute, &local);
        let mut kept = BTreeMap::new();
        let mut moved = BTreeMap::new(); // kept apart: a local PID may equal a host PID not yet moved
        let mut outside = BTreeMap::new();
        for (pid, mut g) in procs {
            if !unresolved.contains(&pid) {
                kept.insert(pid, g);
            } else if let Some((lpid, confidence)) = matched.get(&pid) {
                g.owner = Some(*confidence);
                moved.insert(*lpid, g);
            } else {
                outside.insert(pid, g);
            }
        }
        kept.extend(moved);
        (kept, outside)
    }
}

/// Largest reported use over both lists, or hidden when any entry hides it.
fn row(max_used: Option<u64>, hidden: bool, util: Option<(&NvUtilSample, bool)>) -> ProcGpu {
    let mut g = ProcGpu::measured(max_used.unwrap_or(0));
    if hidden {
        g.vram = Bytes::unavailable(Reason::NvmlHidden);
    }
    if let Some((s, in_compute)) = util {
        let by_engine = BTreeMap::from([
            (
                if in_compute { "compute" } else { "3d" }.to_string(),
                f64::from(s.sm),
            ),
            ("encode".to_string(), f64::from(s.enc)),
            ("decode".to_string(), f64::from(s.dec)),
        ]);
        g.engines = Engines::Measured { by_engine };
    }
    g
}

impl GpuBackend for NvmlBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Nvml
    }

    fn adapters(&self) -> Vec<Adapter> {
        self.devices().into_iter().map(|(_, a)| a).collect()
    }

    /// Records the selected GPU and the policy. No sample is taken: the first utilisation poll asks for
    /// everything the driver has buffered.
    fn start(&mut self, adapter: &Adapter, policy: ScanPolicy) {
        self.selected = self
            .devices()
            .into_iter()
            .find(|(_, a)| a.key == adapter.key)
            .map(|(i, _)| i);
        self.policy = policy;
        self.last_ts = 0;
        self.matcher = PidMatcher::default();
    }

    fn sample(&mut self) -> BackendSample {
        let Some(index) = self.selected else {
            return BackendSample::default();
        };
        // NVML offers adapter memory, so a failed read is unreadable, not absent from the backend.
        let adapter = AdapterGpu {
            vram_used: self
                .api
                .memory(index)
                .map_or(Bytes::unavailable(Reason::Unreadable), |(_, used)| {
                    Bytes::measured(used)
                }),
            ..AdapterGpu::default()
        };
        let compute = self.api.compute_processes(index).unwrap_or_default();
        let graphics = self.api.graphics_processes(index).unwrap_or_default();
        let compute_pids: BTreeSet<u32> = compute.iter().map(|p| p.pid).collect();
        // A process in both lists reports the same allocation twice; take the larger, never the sum.
        let mut seen: BTreeMap<u32, (Option<u64>, bool)> = BTreeMap::new();
        for NvProcess { pid, used } in compute.iter().chain(graphics.iter()) {
            let e = seen.entry(*pid).or_insert((None, false));
            match used {
                Some(n) => e.0 = Some(e.0.map_or(*n, |m| m.max(*n))),
                None => e.1 = true,
            }
        }
        let util = self.utilisation(index);
        let procs: BTreeMap<Pid, ProcGpu> = seen
            .into_iter()
            .map(|(pid, (max, hidden))| {
                // Utilisation for a process NVML no longer lists (or another PID namespace) is ignored.
                let u = util.get(&pid).map(|s| (s, compute_pids.contains(&pid)));
                (pid, row(max, hidden, u))
            })
            .collect();
        let (procs, outside) = self.resolve_namespaces(procs, &compute_pids);
        BackendSample {
            procs,
            outside,
            adapter,
            status: None,
        }
    }

    fn device(&mut self, _index: usize) -> Result<DeviceTelemetry, Reason> {
        match self.selected {
            Some(i) => Ok(self.api.telemetry(i)),
            None => Err(Reason::NotOnBackend),
        }
    }

    fn driver(&self) -> Option<String> {
        self.api.driver_version()
    }

    fn cuda(&self) -> Option<String> {
        self.api.cuda_version()
    }
}
