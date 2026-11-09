//! The amdgpu DRM backend: cards from sysfs, processes from DRM fdinfo.
use crate::drm::cards::{self, DrmCard};
use crate::drm::counters::DrmCounters;
use crate::drm::device::read_device;
use gputaskman_core::model::{Adapter, BackendKind, DeviceTelemetry};
use gputaskman_core::provenance::Reason;
use gputaskman_core::traits::{BackendSample, Clock, GpuBackend, ScanPolicy};
use std::path::PathBuf;
use std::sync::Arc;

pub struct DrmBackend {
    cards: Vec<DrmCard>,
    proc_root: PathBuf,
    clock: Arc<dyn Clock>,
    kernel_release: String,
    /// The selected card and its counters, once `start()` picked an amdgpu adapter.
    selected: Option<(DrmCard, DrmCounters)>,
}

impl DrmBackend {
    /// Lists the amdgpu cards under `<sys_root>/class/drm`; no process is scanned until `start()`.
    pub fn new(
        sys_root: PathBuf,
        proc_root: PathBuf,
        clock: Arc<dyn Clock>,
        kernel_release: String,
    ) -> Self {
        Self {
            cards: cards::cards(&sys_root.join("class").join("drm"), None),
            proc_root,
            clock,
            kernel_release,
            selected: None,
        }
    }

    /// Drops cards whose adapter key another backend (NVML) already reports.
    pub fn without_keys(mut self, keys: &[u64]) -> Self {
        self.cards.retain(|c| !keys.contains(&c.key));
        self
    }
}

impl GpuBackend for DrmBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Drm
    }

    fn adapters(&self) -> Vec<Adapter> {
        cards::adapters(&self.cards)
    }

    /// Keeps only the selected card (another vendor's GPU means no fd walk at all), applies the
    /// policy and takes the engine baseline over every process.
    fn start(&mut self, adapter: &Adapter, policy: ScanPolicy) {
        self.selected = None;
        let Some(card) = self.cards.iter().find(|c| c.key == adapter.key).cloned() else {
            return;
        };
        let mut counters = DrmCounters::new(
            vec![card.clone()],
            self.proc_root.clone(),
            Arc::clone(&self.clock),
        );
        counters.start();
        if policy == ScanPolicy::Unbounded {
            counters.set_budget(None); // one-shot output sees every process
        }
        self.selected = Some((card, counters));
    }

    fn sample(&mut self) -> BackendSample {
        let Some((card, counters)) = self.selected.as_mut() else {
            return BackendSample::default();
        };
        let (mut procs, mut adapters) = counters.sample();
        let status = counters.status();
        BackendSample {
            procs: procs.remove(&card.key).unwrap_or_default(),
            outside: Default::default(),
            adapter: adapters.remove(&card.key).unwrap_or_default(),
            status: (!status.is_empty()).then(|| status.to_string()),
        }
    }

    fn device(&mut self, index: usize) -> Result<DeviceTelemetry, Reason> {
        match &self.selected {
            Some((card, _)) => Ok(read_device(card, index, &self.kernel_release)),
            None => Err(Reason::NotOnBackend),
        }
    }

    fn driver(&self) -> Option<String> {
        Some(format!("kernel {}", self.kernel_release))
    }

    fn cuda(&self) -> Option<String> {
        None
    }
}
