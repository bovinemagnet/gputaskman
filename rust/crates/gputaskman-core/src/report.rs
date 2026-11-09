//! Plain-text snapshot output (`--once`), a port of `report.py` `to_text` for the `nvml` and `drm`
//! backends. The wording and markers (`*`, `?`, `n/a`) are Python's.
use std::fmt::Write as _;
use std::path::Path;

use crate::collector::{Origin, ProcRow, Snapshot};
use crate::llm::gb;
use crate::model::{BackendKind, DeviceTelemetry, Engines, ProcGpu};
use crate::provenance::{Bytes, Confidence, Reason};

pub const INFERRED_NOTE: &str = "* matched: the size is exact (Windows' counter for a WSL VM GPU context, or NVML for a host PID seen from a container); which local process owns it is inferred, and only one explanation fits (or every one that fits gives the same size).";
pub const ORDER_NOTE: &str = "? order guess: the size is exact but several owners fit, so newer processes are paired with newer GPU contexts / PIDs and outside ones are assumed oldest. Cached foreign-size hints only rank extra candidates; they never prove ownership. Becomes * with evidence, such as a process starting while gputaskman runs.";
pub const DRM_RESIDENT_NOTE: &str = "Resident: every buffer the process can reach, shared ones included. The same buffer can appear under several processes, so these are not added up.";

/// Why an amdgpu row's VRAM is unknown, in words. `None` for reasons with no DRM explanation.
pub fn drm_note(g: &ProcGpu) -> Option<&'static str> {
    Some(match g.vram.reason()? {
        Reason::Legacy => {
            "VRAM n/a: this kernel's amdgpu fdinfo has incomplete or unparseable drm-shared-* stats, so resident memory may include buffers imported from other processes and would be counted once per importer."
        }
        Reason::SharedClient => {
            "VRAM and GPU% n/a: this process holds a DRM client other processes hold too (fd passing, fork). fdinfo counts the client once and doesn't say which of them uses it."
        }
        Reason::SharedBuffers => {
            "VRAM n/a: this process's DRM client shares buffers with other clients (dma-buf, e.g. windows and video frames). fdinfo counts shared buffers by where they prefer to live, not where they are, so its private part can't be measured."
        }
        Reason::Primary => {
            "VRAM n/a: this process opened the GPU through /dev/dri/card*, which root-owned services such as logind may also hold where a normal user can't see them; run as root to check."
        }
        Reason::Incomplete => {
            "VRAM and GPU% n/a: some of this process's file descriptors or fdinfo couldn't be read."
        }
        Reason::PartialScan => {
            "VRAM and GPU% n/a this refresh: the scan budget ran out before every process's fds were listed, so another holder of this DRM client can't be ruled out."
        }
        _ => return None,
    })
}

/// Measured size, with `*` or `?` when the size is exact but its local owner is inferred.
pub fn vram_label(g: &ProcGpu) -> String {
    let mark = g.owner.map_or("", Confidence::marker);
    match g.shown_vram() {
        None => "n/a".to_string(),
        Some(s) => format!("{}{mark}", gb(s.get() as f64)),
    }
}

fn gb_or_na(b: Bytes) -> String {
    b.value()
        .map_or_else(|| "N/A".to_string(), |v| gb(v as f64))
}

fn opt<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or_else(|| "n/a".to_string(), |v| v.to_string())
}

/// Python's `format_metric`: a missing value is `N/A`. `prec` is the `.Nf` precision; without one an
/// integral value prints without a decimal point, as the integer counters do in Python.
fn metric(v: Option<f64>, prec: Option<usize>, unit: &str) -> String {
    let Some(x) = v else {
        return "N/A".to_string();
    };
    match prec {
        Some(p) => format!("{x:.p$}{unit}"),
        None if x.fract() == 0.0 && x.abs() < 1e15 => format!("{x:.0}{unit}"),
        None => format!("{x}{unit}"),
    }
}

fn device_line(nv: &DeviceTelemetry) -> String {
    let rx = nv.pcie_rx_kbs.map(|k| k as f64 / 1024.0);
    let tx = nv.pcie_tx_kbs.map(|k| k as f64 / 1024.0);
    let pstate = nv
        .pstate
        .map_or_else(|| "P N/A".to_string(), |p| format!("P{p}"));
    let amd = nv.source.starts_with("AMD ");
    let label = if amd { "AMD gfx" } else { "GPU" };
    let clock = if amd {
        format!("gfx {}", metric(nv.gfx_clock.map(f64::from), None, ""))
    } else {
        format!("SM {}", metric(nv.sm_clock.map(f64::from), None, ""))
    };
    let power_kind = if nv.power_kind.is_empty() {
        String::new()
    } else {
        format!(" ({})", nv.power_kind)
    };
    format!(
        "{label} {}  mem-bandwidth {}  {}  {}/{} W{power_kind}  {pstate}  {clock} MHz  video {} MHz  PCIe rx {} MB/s tx {} MB/s",
        metric(nv.util_gpu, None, "%"),
        metric(nv.util_mem, None, "%"),
        metric(nv.temp, None, "C"),
        metric(nv.power_w, Some(0), ""),
        metric(nv.power_limit_w, Some(0), ""),
        metric(nv.video_clock.map(f64::from), None, ""),
        metric(rx, Some(0), ""),
        metric(tx, Some(0), ""),
    )
}

fn row_line(s: &Snapshot, p: &ProcRow, w: usize) -> String {
    let drm_local = s.backend == Some(BackendKind::Drm) && p.origin == Origin::Local;
    let spill = if !drm_local {
        String::new()
    } else {
        match p.gpu.shared {
            Bytes::Unavailable { .. } => "N/A".to_string(),
            Bytes::Measured { bytes } if bytes >= 1 << 20 => gb(bytes as f64),
            _ => String::new(),
        }
    };
    let pct = match &p.gpu.engines {
        e @ Engines::Measured { .. } => format!("{:.0}", e.busiest().unwrap_or(0.0)),
        Engines::Unavailable { .. } => "n/a".to_string(),
    };
    let name: String = p.name.chars().take(24).collect();
    let runtime: String = p.runtime().chars().take(24).collect();
    format!(
        "{:>w$}  {name:<24} {:<10} {runtime:<24} {:>9} {spill:>8} {:>8} {pct:>5}  {}",
        p.pid_label(),
        p.kind,
        vram_label(&p.gpu),
        "",
        p.inside()
    )
}

fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
}

fn detail_sections(out: &mut Vec<String>, procs: &[ProcRow]) {
    for p in procs {
        if let Some(h) = &p.handy {
            out.push(String::new());
            if let Some(sel) = h.selected_model.as_deref().filter(|m| !m.is_empty()) {
                out.push(format!(
                    "== {} ({}) Handy selected (settings, not load proof): {sel}",
                    p.name, p.pid
                ));
            }
            if let Some(ev) = h.last_event.as_deref().filter(|m| !m.is_empty()) {
                out.push(format!(
                    "   last Handy event: {ev} {} at {} (runtime-reported history, not current GPU residency)",
                    h.last_model.as_deref().unwrap_or(""),
                    h.last_event_local()
                ));
            }
        }
        let Some(v) = &p.llm else { continue };
        out.push(String::new());
        out.push(format!(
            "== {} ({})  VRAM {}",
            p.name,
            p.pid_label(),
            vram_label(&p.gpu)
        ));
        for (i, g) in v.models.iter().enumerate() {
            let ev = v
                .files
                .get(i)
                .map(|f| f.evidence_label())
                .unwrap_or_default();
            out.push(format!(
                "   {}: {}  [{}, {} params, {}, {} layers, ctx_train {}]  {}  evidence: {ev}",
                v.kind_label(i),
                v.label(i),
                g.arch(),
                g.params_label(),
                g.quant(),
                g.n_layers(),
                opt(g.ctx_train()),
                basename(&g.path),
            ));
        }
        for f in &v.other_files {
            let kind = f.kind_label();
            let kind = if kind.is_empty() {
                String::new()
            } else {
                format!(", {kind}")
            };
            out.push(format!(
                "   file: {}  ({}{kind})  evidence: {}",
                f.store_name
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&f.path),
                gb(f.size as f64),
                f.evidence_label()
            ));
        }
        for seg in &v.segments {
            out.push(format!("   {:>9}  {}", gb_or_na(seg.bytes), seg.label));
        }
        if let Some(sv) = &v.server {
            let mut line = format!(
                "   server: {} {} ctx={} slots={}/{}",
                sv.kind,
                sv.url,
                opt(sv.n_ctx),
                opt(sv.slots_busy),
                opt(sv.slots_total)
            );
            if let Some(g) = sv.gen_tok_s {
                let _ = write!(line, " gen={g:.1} tok/s");
            }
            out.push(line);
        }
        if let Some(src) = &v.log_source {
            out.push(format!("   exact sizes from log: {src}"));
        }
        for n in &v.notes {
            out.push(format!("   - {n}"));
        }
    }
}

pub fn to_text(s: &Snapshot, limit: usize) -> String {
    let mut out: Vec<String> = Vec::new();
    let nv = s.device.as_ref();
    let name = s.adapter.as_ref().map_or("GPU", |a| a.name.as_str());
    let mut info = s
        .driver
        .as_deref()
        .filter(|d| !d.is_empty())
        .map_or_else(String::new, |d| format!("  driver {d}"));
    if let Some(c) = s.cuda.as_deref().filter(|c| !c.is_empty()) {
        let _ = write!(info, "  CUDA {c}");
    }
    out.push(format!(
        "{name}{info}  {}",
        nv.map_or("", |n| n.driver_model.as_str())
    ));
    let mut memory = format!(
        "VRAM {} / {}",
        gb_or_na(s.vram_used),
        gb_or_na(s.vram_total)
    );
    if s.backend == Some(BackendKind::Drm) {
        let _ = write!(
            memory,
            "   shared (in system RAM) {}",
            gb_or_na(s.shared_used)
        );
    }
    out.push(memory);
    out.extend(s.notes.iter().cloned());
    if let Some(nv) = nv {
        out.push(device_line(nv));
    }
    out.push(String::new());
    let shown = &s.procs[..s.procs.len().min(limit)];
    let w = shown
        .iter()
        .map(|p| p.pid_label().len())
        .fold(7, usize::max);
    out.push(format!(
        "{:>w$}  {:<24} {:<10} {:<24} {:>9} {:>8} {:>8} {:>5}  INSIDE",
        "PID", "PROCESS", "KIND", "RUNTIME", "VRAM", "SPILL", "PAGED", "GPU%"
    ));
    out.extend(shown.iter().map(|p| row_line(s, p, w)));
    let owner = |c: Confidence| shown.iter().any(|p| p.gpu.owner == Some(c));
    if owner(Confidence::Consistent) {
        out.push(INFERRED_NOTE.to_string());
    }
    if owner(Confidence::Order) {
        out.push(ORDER_NOTE.to_string());
    }
    if s.backend == Some(BackendKind::Drm) {
        let mut seen: Vec<&str> = Vec::new();
        for note in shown
            .iter()
            .filter(|p| p.origin == Origin::Local)
            .filter_map(|p| drm_note(&p.gpu))
        {
            if !seen.contains(&note) {
                seen.push(note);
                out.push(note.to_string());
            }
        }
    }
    if let Some(status) = s.process_status.as_deref().filter(|p| !p.is_empty()) {
        out.push(format!("DRM process scan: {status}."));
    }
    detail_sections(&mut out, &s.procs);
    out.join("\n")
}
