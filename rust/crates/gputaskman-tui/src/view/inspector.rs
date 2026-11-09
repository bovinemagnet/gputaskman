//! The inspector (`app.py` `render_inspector`, Linux branches): what is known about the selected
//! process's GPU memory, and how it is known.
use std::path::Path;

use gputaskman_core::collector::{Origin, ProcRow, Snapshot};
use gputaskman_core::llm::gb;
use gputaskman_core::llm::view::LlmView;
use gputaskman_core::model::{BackendKind, Engines, MappedModel};
use gputaskman_core::provenance::{Bytes, Confidence};
use gputaskman_core::report::{DRM_RESIDENT_NOTE, INFERRED_NOTE, ORDER_NOTE, drm_note, vram_label};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::app::App;
use crate::format::{
    CYAN, GREEN, GREY42, GREY50, GREY62, GREY70, MAGENTA, RED, bold, fg, segment_color, thousands,
};
use crate::rows::ShownRow;
use crate::widgets::stacked_bar;

pub const PROMPT: &str = "Select a process (↑/↓) to see what's inside its GPU memory.  Press ? to learn what's knowable.";

const KFD_NOTE: &str = "Engines n/a: this process uses ROCm/HIP (/dev/kfd), whose compute queues the kernel doesn't time per process. The device's AMD gfx load includes its work.";

/// Python's `EXPLAIN`, by kind.
fn explain(kind: &str) -> &'static str {
    match kind {
        "wsl-vm" => {
            "The WSL virtual machine as Windows sees it. Windows charges all GPU memory used by Linux processes in every WSL distro (and WSLg) to vmwp.exe, so this is the exact total. The split across Linux processes is not exposed."
        }
        "compositor" => {
            "Desktop Window Manager: its own back buffers and the composed image for each monitor. Windows' 'Dedicated Usage' counter shows dwm as huge (it counts every app surface it can see); this tool prefers Local Usage for attribution. The adapter counter is the total; per-process samples can overlap or differ in timing."
        }
        "kernel" => {
            "Driver/kernel-owned allocations: paging buffers, display scan-out, firmware, scheduler state."
        }
        "system" => "Session/system process holding display and cursor surfaces.",
        "browser" => {
            "Chromium GPU process: page tiles, images, video frames, canvas, WebGL/WebGPU buffers. Deeper: edge://gpu, or DevTools > Performance with the GPU track."
        }
        "3D/game" => {
            "Textures, meshes, render targets and descriptor heaps. Windows only sees opaque allocations; per-resource names need a graphics debugger (PIX on Windows, NVIDIA Nsight Graphics)."
        }
        "video" => "Decoded/encoded video frames and codec surfaces (NVIDIA NVDEC/ENC)",
        "compute" => {
            "Compute allocations are opaque to the OS. Runtime-specific profilers can provide more detail (CUDA: cudaMalloc with Nsight Systems)."
        }
        "diffusion" => {
            "Model components are identified from launch arguments; GPU allocations cannot be split by component."
        }
        "AI" => {
            "ML runtime allocations (tensors, workspaces). PyTorch: torch.cuda.memory_summary(); ONNX Runtime/DirectML pools are opaque from outside."
        }
        "app" => "UI surfaces: swap chains, window back-buffers, cached bitmaps/glyphs.",
        _ => "",
    }
}

fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
}

pub fn title(app: &App) -> String {
    let Some(r) = app.selected_row() else {
        return String::new();
    };
    let row = r.row;
    let host = if row.origin == Origin::Outside {
        "host PID "
    } else {
        ""
    };
    let mut t = format!("{} ({host}{}) - {}", row.name, row.pid, row.kind);
    if let Some(i) = r.gpu {
        t.push_str(&format!(" - GPU {i}"));
    }
    t
}

pub fn lines(app: &App, term_width: u16) -> Vec<Line<'static>> {
    let Some(r) = app.selected_row() else {
        return vec![Line::styled(PROMPT, fg(GREY62))];
    };
    let mut out = vec![header(&r)];
    out.extend(provenance_notes(r.row, r.snap));
    out.extend(handy_lines(r.row));
    match &r.row.llm {
        Some(v) => out.extend(llm_lines(&r, v, term_width)),
        None => out.extend(other_lines(r.row)),
    }
    let detail = &r.row.detail;
    if !detail.ports.is_empty() {
        let ports: Vec<String> = detail.ports.iter().take(8).map(u16::to_string).collect();
        out.push(Line::styled(
            format!("listening: {}", ports.join(", ")),
            fg(GREY50),
        ));
    }
    if !detail.cmdline.is_empty() {
        let cmd: String = detail.cmdline.join(" ").chars().take(400).collect();
        out.push(Line::styled(cmd, fg(GREY42)));
    } else if detail.access_denied {
        out.push(Line::styled(
            "(protected process - run elevated for runtime/file details)",
            fg(GREY42),
        ));
    }
    out
}

fn header(r: &ShownRow<'_>) -> Line<'static> {
    let g = &r.row.gpu;
    let drm_local = r.snap.backend == Some(BackendKind::Drm) && r.row.origin == Origin::Local;
    // Ruling R5: rows without a spill counter print n/a here, as Python does.
    let spill = if drm_local {
        g.shared
            .value()
            .map_or_else(|| "n/a".to_string(), |v| gb(v as f64))
    } else {
        "n/a".to_string()
    };
    let spill_style = if r.snap.shared_is_spill && g.shared.value().is_some_and(|v| v > 256 << 20) {
        fg(RED).add_modifier(Modifier::BOLD)
    } else {
        Style::new()
    };
    let engines = match &g.engines {
        Engines::Measured { by_engine } => {
            let busy: Vec<String> = by_engine
                .iter()
                .filter(|(_, v)| **v >= 0.5)
                .map(|(k, v)| format!("{k} {v:.0}%"))
                .collect();
            if busy.is_empty() {
                "idle".to_string()
            } else {
                busy.join(" ")
            }
        }
        Engines::Unavailable { .. } => "n/a".to_string(),
    };
    let mut spans = vec![
        Span::styled("VRAM ", bold()),
        Span::styled(vram_label(g), bold()),
        Span::raw("   spill(RAM) "),
        Span::styled(spill, spill_style),
        Span::raw("   paged(est) n/a"),
        Span::raw(format!("   engines {engines}")),
    ];
    if !r.row.detail.runtimes.is_empty() {
        spans.push(Span::styled("   runtime ", fg(GREY62)));
        spans.push(Span::styled(r.row.detail.runtimes.join(" "), fg(CYAN)));
    }
    Line::from(spans)
}

fn owner_note(owner: Option<Confidence>) -> &'static str {
    if owner == Some(Confidence::Order) {
        ORDER_NOTE
    } else {
        INFERRED_NOTE
    }
}

fn provenance_notes(row: &ProcRow, s: &Snapshot) -> Vec<Line<'static>> {
    let g = &row.gpu;
    let local = row.origin == Origin::Local;
    let drm = s.backend == Some(BackendKind::Drm);
    let mut out = Vec::new();
    if local && g.vram.is_measured() && g.owner.is_some() {
        out.push(Line::styled(owner_note(g.owner), fg(GREY62)));
    }
    if local && !g.vram.is_measured() {
        if let Bytes::Inferred { by, .. } = g.vram {
            out.push(Line::styled(owner_note(Some(by)), fg(GREY62)));
        } else if drm && let Some(note) = drm_note(g) {
            out.push(Line::styled(note, fg(GREY62)));
        }
    }
    if drm
        && local
        && let Some(res) = &g.resident
    {
        let size = |b: Bytes| {
            b.value()
                .map_or_else(|| "n/a".to_string(), |v| gb(v as f64))
        };
        out.push(Line::styled(
            format!(
                "Resident VRAM {}, GTT {}. {DRM_RESIDENT_NOTE}",
                size(res.vram),
                size(res.gtt)
            ),
            fg(GREY62),
        ));
    }
    if drm && g.kfd {
        out.push(Line::styled(KFD_NOTE, fg(GREY62)));
    }
    out
}

fn handy_lines(row: &ProcRow) -> Vec<Line<'static>> {
    let Some(h) = &row.handy else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(m) = h.selected_model.as_deref().filter(|m| !m.is_empty()) {
        out.push(Line::styled(
            format!("Handy selected (settings; not load proof): {m}"),
            fg(GREY70),
        ));
    }
    if let Some(ev) = h.last_event.as_deref().filter(|e| !e.is_empty()) {
        out.push(Line::styled(
            format!(
                "Handy last event: {ev} {} at {} (runtime-reported history, not current GPU residency)",
                h.last_model.as_deref().unwrap_or(""),
                h.last_event_local()
            ),
            fg(GREEN),
        ));
    }
    out
}

fn file_line(
    m: &MappedModel,
    color: ratatui::style::Color,
    prefix: &str,
    evidence: bool,
) -> Line<'static> {
    let name = m
        .store_name
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| m.path.clone());
    let kind = m.kind_label();
    let kind = if kind.is_empty() {
        String::new()
    } else {
        format!(", {kind}")
    };
    let how = if evidence {
        format!(", {}", m.evidence_label())
    } else {
        String::new()
    };
    Line::styled(
        format!("{prefix}{name}  ({}{how}{kind})", gb(m.size as f64)),
        fg(color),
    )
}

fn llm_lines(r: &ShownRow<'_>, v: &LlmView, term_width: u16) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    if r.gpu.is_some() {
        out.push(Line::styled(
            "Model files, launch flags and runtime buffer logs can span GPUs. The breakdown below is not a measured per-device model placement.",
            fg(GREY70),
        ));
    }
    let width = usize::from(term_width.saturating_sub(8).clamp(20, 100));
    let seg_total: u64 = v.segments.iter().filter_map(|x| x.bytes.value()).sum();
    let shown = r.row.gpu.shown_vram().map_or(0, |s| s.get());
    let total = shown.max(seg_total).max(1) as f64;
    let parts: Vec<_> = v
        .segments
        .iter()
        .map(|x| (x.bytes.value().unwrap_or(0) as f64, segment_color(x.kind)))
        .collect();
    out.push(Line::from(stacked_bar(&parts, total, width)));
    let mut legend = Vec::new();
    for x in &v.segments {
        legend.push(Span::styled("■ ", fg(segment_color(x.kind))));
        let exact = matches!(x.bytes, Bytes::Reported { .. });
        legend.push(Span::styled(
            format!(
                "{} {}",
                x.label,
                x.bytes
                    .value()
                    .map_or_else(|| "n/a".to_string(), |b| gb(b as f64))
            ),
            if exact { bold() } else { Style::new() },
        ));
        legend.push(Span::raw("  "));
    }
    if let Some(src) = &v.log_source {
        legend.push(Span::styled(
            format!("(allocated sizes from {})", basename(src)),
            fg(GREEN),
        ));
    }
    out.push(Line::from(legend));
    if !v.models.is_empty() {
        out.extend(model_table(v));
    }
    for f in &v.other_files {
        out.push(file_line(f, GREY62, "file: ", true));
    }
    if let Some(sv) = &v.server {
        let mut spans = vec![
            Span::styled("server ", bold()),
            Span::styled(format!("{} {}", sv.kind, sv.url), fg(CYAN)),
        ];
        if let Some(n) = sv.n_ctx.filter(|n| *n > 0) {
            spans.push(Span::raw(format!("  ctx {}", thousands(n.unsigned_abs()))));
        }
        if let Some(total) = sv.slots_total {
            spans.push(Span::raw(format!(
                "  slots busy {}/{total}",
                sv.slots_busy
                    .map_or_else(|| "?".to_string(), |b| b.to_string())
            )));
        }
        if let Some(g) = sv.gen_tok_s.filter(|g| *g != 0.0) {
            spans.push(Span::styled(
                format!("  gen {g:.1} tok/s"),
                fg(GREEN).add_modifier(Modifier::BOLD),
            ));
        }
        if let Some(p) = sv.prompt_tok_s.filter(|p| *p != 0.0) {
            spans.push(Span::raw(format!("  prompt {p:.0} tok/s")));
        }
        out.push(Line::from(spans));
        if sv.kind == "lmstudio" {
            for m in sv.models.iter().take(4) {
                out.push(Line::raw(format!(
                    "  loaded: {} {} {} ctx {}",
                    m.name.as_deref().unwrap_or(""),
                    m.r#type.as_deref().unwrap_or(""),
                    m.quant.as_deref().unwrap_or(""),
                    m.ctx.map_or_else(|| "?".to_string(), |c| c.to_string())
                )));
            }
        }
    }
    for n in &v.notes {
        let style = if n.starts_with('!') {
            fg(RED).add_modifier(Modifier::BOLD)
        } else {
            fg(GREY70)
        };
        out.push(Line::styled(format!("• {n}"), style));
    }
    out
}

/// The model table, columns padded to their widest cell and separated by two spaces.
fn model_table(v: &LlmView) -> Vec<Line<'static>> {
    let headers = [
        "",
        "model",
        "arch",
        "params",
        "quant",
        "layers",
        "train ctx",
        "on GPU (est)",
        "evidence",
        "file",
    ];
    let mut rows: Vec<[String; 10]> = Vec::new();
    for (i, g) in v.models.iter().enumerate() {
        let f = v.files.get(i);
        rows.push([
            v.kind_label(i),
            v.label(i),
            g.arch().to_string(),
            g.params_label(),
            g.quant(),
            if g.n_layers() == 0 {
                String::new()
            } else {
                g.n_layers().to_string()
            },
            g.ctx_train()
                .filter(|c| *c != 0)
                .map_or_else(String::new, thousands),
            gb(g.gpu_weight_bytes(None) as f64),
            f.map_or_else(String::new, MappedModel::evidence_label),
            basename(&g.path),
        ]);
    }
    let mut widths = headers.map(|h| h.chars().count());
    for r in &rows {
        for (w, c) in widths.iter_mut().zip(r) {
            *w = (*w).max(c.chars().count());
        }
    }
    let pad = |s: &str, w: usize| format!("{s:<w$}");
    let mut out = vec![Line::styled(
        headers
            .iter()
            .zip(widths)
            .map(|(h, w)| pad(h, w))
            .collect::<Vec<_>>()
            .join("  "),
        fg(GREY70).add_modifier(Modifier::BOLD),
    )];
    let styles = [
        fg(MAGENTA),
        Style::new(),
        Style::new(),
        Style::new(),
        Style::new(),
        Style::new(),
        Style::new(),
        Style::new(),
        fg(GREEN),
        fg(GREY50),
    ];
    for r in rows {
        let mut spans = Vec::new();
        for (j, ((c, w), st)) in r.iter().zip(widths).zip(styles).enumerate() {
            if j > 0 {
                spans.push(Span::raw("  "));
            }
            spans.push(Span::styled(pad(c, w), st));
        }
        out.push(Line::from(spans));
    }
    out
}

fn other_lines(row: &ProcRow) -> Vec<Line<'static>> {
    let mut out = vec![Line::styled(explain(&row.kind), fg(GREY70))];
    for m in &row.detail.models {
        out.push(file_line(m, MAGENTA, "file: ", true));
    }
    if !row.detail.candidates.is_empty() {
        out.push(Line::styled(
            "No mapped model file was observed. Some runtimes read then close model files. Installed candidates (not proven loaded):",
            fg(GREY70),
        ));
        for m in &row.detail.candidates {
            out.push(file_line(m, GREY62, "  ", false));
        }
    }
    out
}
