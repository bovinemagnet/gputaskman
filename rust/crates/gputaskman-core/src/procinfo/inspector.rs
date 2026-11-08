//! Per-process details cache and memory-map scans (`procinfo.ProcInspector`).
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::gguf::is_gguf;
use crate::model::{AI_RUNTIMES, Evidence, MIB, MappedModel, Pid, ProcDetail};
use crate::procinfo::installed::Installed;
use crate::procinfo::models::{
    MODEL_EXTS, ModelContext, NOT_MODELS, is_model_file, models_from_cmdline, skip_path,
};
use crate::procinfo::runtimes::{lib_name, match_runtimes, splitext};
use crate::traits::ProcessSource;

/// Caches per-process details. Memory-map scans are the only costly step (1-50 ms each), so they
/// run for new processes, when a process's GPU memory changes noticeably, and on a slow backstop.
#[derive(Default)]
pub struct ProcInspector {
    cache: BTreeMap<Pid, ProcDetail>,
    ports: BTreeMap<Pid, Vec<u16>>,
    ports_time: Option<f64>,
}

impl ProcInspector {
    /// Never rescan one process more often than this.
    pub const SCAN_MIN_GAP: f64 = 5.0;
    /// Rescan even without a memory change.
    pub const SCAN_BACKSTOP: f64 = 120.0;
    /// Or 10% of the process's GPU memory, whichever is larger.
    pub const SCAN_DELTA: f64 = (64 * MIB) as f64;
    pub const PORTS_INTERVAL: f64 = 5.0;

    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn refresh_ports(&mut self, procs: &dyn ProcessSource, now: f64) {
        if self
            .ports_time
            .is_some_and(|t| now - t < Self::PORTS_INTERVAL)
        {
            return;
        }
        self.ports = procs
            .listening_ports()
            .into_iter()
            .map(|(pid, mut p)| {
                p.sort_unstable();
                p.dedup();
                (pid, p)
            })
            .collect();
        self.ports_time = Some(now);
    }

    /// Cheap details: name, exe, command line, listening ports. Built once per process and
    /// re-checked for PID reuse on each call.
    pub fn get(&mut self, procs: &dyn ProcessSource, pid: Pid, now: f64) -> &mut ProcDetail {
        self.refresh_ports(procs, now);
        let reused = self.cache.get(&pid).is_some_and(|d| {
            d.start_ticks
                .is_some_and(|t| procs.start_time(pid).map(|s| s.ticks) != Some(t))
        });
        if reused || !self.cache.contains_key(&pid) {
            self.cache.insert(pid, Self::new_detail(procs, pid));
        }
        let ports = self.ports.get(&pid).cloned().unwrap_or_default();
        let d = self
            .cache
            .entry(pid)
            .or_insert_with(|| ProcDetail::new(pid, "?"));
        d.ports = ports;
        d
    }

    pub fn exclusive_ports(&mut self, procs: &dyn ProcessSource, pid: Pid, now: f64) -> Vec<u16> {
        self.refresh_ports(procs, now);
        let others: BTreeSet<u16> = self
            .ports
            .iter()
            .filter(|(p, _)| **p != pid)
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        self.ports
            .get(&pid)
            .map(|v| v.iter().copied().filter(|p| !others.contains(p)).collect())
            .unwrap_or_default()
    }

    fn new_detail(procs: &dyn ProcessSource, pid: Pid) -> ProcDetail {
        if pid == 0 {
            return ProcDetail::new(0, "System Idle");
        }
        let Some(b) = procs.basics(pid) else {
            let mut d = ProcDetail::new(pid, "?");
            d.access_denied = true;
            return d;
        };
        let mut d = ProcDetail::new(pid, b.name.unwrap_or_else(|| "?".into()));
        d.access_denied = b.exe.is_none() || b.cmdline.is_none() || b.start.is_none();
        d.exe = b.exe.unwrap_or_default();
        d.cmdline = b.cmdline.unwrap_or_default();
        if let Some(s) = b.start {
            d.start_time = s.wall;
            d.start_ticks = Some(s.ticks);
        }
        d
    }

    #[must_use]
    pub fn needs_scan(&self, d: &ProcDetail, mem: f64, now: f64) -> bool {
        if d.access_denied || d.pid == 0 {
            return false;
        }
        if d.maps_time == 0.0 {
            return true;
        }
        let age = now - d.maps_time;
        if age < Self::SCAN_MIN_GAP {
            return false;
        }
        age > Self::SCAN_BACKSTOP
            || (mem - d.scan_mem).abs() > Self::SCAN_DELTA.max(0.1 * d.scan_mem)
    }

    /// Memory maps give loaded libraries (runtimes) and mapped model files. Never lists open files.
    pub fn scan(
        &mut self,
        pid: Pid,
        procs: &dyn ProcessSource,
        ctx: &mut ModelContext,
        installed: &mut Installed,
        mem: f64,
        now: f64,
    ) {
        let Some(d) = self.cache.get_mut(&pid) else {
            return;
        };
        d.maps_time = now;
        d.scan_mem = mem;
        let Ok(maps) = procs.memory_maps(pid) else {
            d.access_denied = true;
            return;
        };
        let mut dlls = BTreeSet::new();
        let mut models: Vec<MappedModel> = Vec::new();
        for m in maps {
            let path = m.path.as_str();
            let base = Path::new(path)
                .file_name()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let lib = lib_name(&base);
            if let Some(lib) = &lib
                && base.starts_with("agxmetal")
            {
                // The Metal driver lives under /System: record it before skipping system paths.
                dlls.insert(lib.clone());
                continue;
            }
            if skip_path(path) {
                continue;
            }
            let ext = splitext(&base);
            if let Some(lib) = lib {
                dlls.insert(lib);
                continue;
            }
            if NOT_MODELS.contains(&ext) {
                continue;
            }
            let Ok(size) = fs::metadata(path).map(|md| md.len()) else {
                continue;
            };
            let gguf =
                (ext == ".gguf" || ext.is_empty() || size > 64 * MIB) && is_gguf(Path::new(path));
            if !(gguf || MODEL_EXTS.contains(&ext)) {
                continue;
            }
            let store = ctx.stores.store_name(path, now);
            if is_model_file(ext, size, gguf, store.as_deref()) {
                // Keyed by exact path: Linux paths are case-sensitive (Python lower-cases for Windows).
                let model = ctx.make_model(path, size, m.rss, gguf, Evidence::Mapped, store, now);
                match models.iter_mut().find(|x| x.path == path) {
                    Some(x) => *x = model,
                    None => models.push(model),
                }
            }
        }
        for path in models_from_cmdline(&d.cmdline) {
            if let Some(x) = models.iter_mut().find(|x| x.path == path) {
                x.add_source(Evidence::Cmdline);
            } else if let Ok(md) = fs::metadata(&path) {
                let gguf = is_gguf(Path::new(&path));
                models.push(ctx.make_model(&path, md.len(), 0, gguf, Evidence::Cmdline, None, now));
            }
        }
        d.runtimes = match_runtimes(&dlls);
        models.sort_by_key(|m| std::cmp::Reverse(m.size));
        d.models = models;
        let ai = d.runtimes.iter().any(|r| AI_RUNTIMES.contains(&r.as_str()));
        d.candidates = if ai && d.models.is_empty() {
            installed.models(&d.exe, ctx, now)
        } else {
            Vec::new()
        };
    }

    /// Ancestors of `pid`, nearest first, up to `depth`; cached in the process detail.
    pub fn parents(&mut self, procs: &dyn ProcessSource, pid: Pid, depth: usize) -> Vec<Pid> {
        if let Some(p) = self.cache.get(&pid).and_then(|d| d.parents.clone()) {
            return p;
        }
        let mut chain = Vec::new();
        let mut cur = pid;
        for _ in 0..depth {
            match procs.parent(cur) {
                Some(p) if p != 0 && p != pid && !chain.contains(&p) => {
                    chain.push(p);
                    cur = p;
                }
                _ => break,
            }
        }
        if let Some(d) = self.cache.get_mut(&pid) {
            d.parents = Some(chain.clone());
        }
        chain
    }

    pub fn prune(&mut self, live: &BTreeSet<Pid>) {
        self.cache.retain(|pid, _| live.contains(pid));
    }
}
