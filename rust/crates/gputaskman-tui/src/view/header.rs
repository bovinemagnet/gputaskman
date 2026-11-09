//! The device box (`app.py` `render_device`): name and diagnostics, the kind-coloured VRAM bar,
//! its legend and notes, and the device metric lines.
use gputaskman_core::collector::Snapshot;
use gputaskman_core::llm::gb;
use gputaskman_core::model::DeviceTelemetry;
use gputaskman_core::provenance::MeasuredTotal;
use gputaskman_core::report::metric;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};

use crate::app::App;
use crate::format::{GREY35, GREY62, GREY70, MIB, RED, YELLOW, bar_color, bold, fg};
use crate::history::Chart;
use crate::widgets::stacked_bar;

pub fn device_title(app: &App) -> &'static str {
    if app.slim {
        "Slim monitoring"
    } else {
        "Full diagnostics"
    }
}

/// The device box's lines for every GPU; `term_width` sizes the VRAM bar as Python does.
pub fn device_lines(app: &App, term_width: u16) -> Vec<Line<'static>> {
    if app.snapshots.is_empty() {
        return vec![Line::from("Sampling GPU counters...")];
    }
    let multi = app.multi();
    let mut out = Vec::new();
    for (i, s) in app.snapshots.iter().enumerate() {
        if multi {
            out.push(Line::styled(format!("GPU {i}"), bold()));
        }
        out.extend(device(app, s, term_width));
    }
    out
}

fn device(app: &App, s: &Snapshot, term_width: u16) -> Vec<Line<'static>> {
    let nv = s.device.as_ref();
    let mut lines = vec![head(app, s, nv)];
    lines.extend(vram(s, term_width));
    if let Some(d) = nv {
        lines.push(metrics_line(d));
        lines.push(status_line(d, app.slim));
    }
    lines
}

fn head(app: &App, s: &Snapshot, nv: Option<&DeviceTelemetry>) -> Line<'static> {
    let name = s
        .adapter
        .as_ref()
        .map(|a| a.name.clone())
        .or_else(|| nv.map(|d| d.name.clone()))
        .unwrap_or_else(|| "GPU".to_string());
    let mut spans = vec![Span::styled(name, bold())];
    if !app.slim {
        let mut info = String::new();
        if let Some(d) = s.driver.as_deref().filter(|d| !d.is_empty()) {
            info.push_str(&format!("   driver {d}"));
        }
        if let Some(c) = s.cuda.as_deref().filter(|c| !c.is_empty()) {
            info.push_str(&format!("  CUDA {c}"));
        }
        spans.push(Span::raw(format!("{info}  ")));
        spans.push(Span::styled(
            nv.map(|d| d.driver_model.clone()).unwrap_or_default(),
            fg(YELLOW).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw(format!(
            "  {}",
            nv.map(|d| d.pci_bus.clone()).unwrap_or_default()
        )));
    }
    if app.paused {
        spans.push(Span::styled(
            "   [PAUSED]",
            fg(RED).add_modifier(Modifier::BOLD),
        ));
    }
    Line::from(spans)
}

/// The VRAM bar and its legend. Only measured row bytes are summed (rule 1); rows whose VRAM is
/// unknown leave their share in "unattributed".
fn vram(s: &Snapshot, term_width: u16) -> Vec<Line<'static>> {
    let mut by_kind: Vec<(String, MeasuredTotal)> = Vec::new();
    let mut all = MeasuredTotal::new();
    for r in &s.procs {
        if let Some(m) = r.gpu.vram.as_measured() {
            match by_kind.iter_mut().find(|(k, _)| *k == r.kind) {
                Some((_, total)) => total.add(m),
                None => {
                    let mut total = MeasuredTotal::new();
                    total.add(m);
                    by_kind.push((r.kind.clone(), total));
                }
            }
            all.add(m);
        }
    }
    let mut order: Vec<(String, f64)> = by_kind
        .into_iter()
        .map(|(k, t)| (k, t.total() as f64))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1)); // stable: ties keep first-seen order
    let attributed = all.total() as f64;
    let used = s.vram_used.value().map(|u| u as f64);
    let total = s.vram_total.value().map(|t| t as f64);
    let scale = match used {
        Some(u) if attributed > 0.0 => (u / attributed).min(1.0),
        _ => 1.0,
    };
    let mut parts: Vec<(f64, ratatui::style::Color)> = match used {
        Some(_) => order
            .iter()
            .map(|(k, v)| (v * scale, bar_color(k)))
            .collect(),
        None => Vec::new(),
    };
    let other = used.map_or(0.0, |u| (u - attributed * scale).max(0.0));
    parts.push((other, GREY35));
    let width = usize::from(term_width.saturating_sub(40).clamp(20, 90));
    let mut bar = vec![Span::raw("VRAM ")];
    bar.extend(stacked_bar(&parts, total.unwrap_or(0.0), width));
    bar.push(Span::styled(
        format!(
            " {} / {}",
            used.map_or_else(|| "N/A".to_string(), gb),
            total.map_or_else(|| "N/A".to_string(), gb)
        ),
        bold(),
    ));

    let mut legend = vec![Span::raw("     ")];
    for (k, v) in order.iter().take(7) {
        legend.push(Span::styled("■ ", fg(bar_color(k))));
        legend.push(Span::raw(format!("{k} {}  ", gb(*v))));
    }
    if other > (32 * MIB) as f64 {
        legend.push(Span::styled("■ ", fg(GREY35)));
        legend.push(Span::raw(format!("unattributed {}  ", gb(other))));
    }
    if let (Some(u), Some(t)) = (used, total) {
        legend.push(Span::styled(
            format!("free {}", gb((t - u).max(0.0))),
            fg(GREY62),
        ));
    }
    if let Some(shared) = s.shared_used.value().filter(|v| *v > 64 * MIB) {
        legend.push(Span::styled(
            format!("   shared(RAM) {}", gb(shared as f64)),
            fg(YELLOW),
        ));
    }
    if scale < 1.0 {
        legend.push(Span::styled(
            "   [bar normalized; labels are raw]",
            fg(GREY62),
        ));
    }
    let mut lines = vec![Line::from(bar), Line::from(legend)];
    for note in &s.notes {
        lines.push(Line::styled(format!("     {note}"), fg(GREY62)));
    }
    if let Some(p) = s.process_status.as_deref().filter(|p| !p.is_empty()) {
        lines.push(Line::styled(
            format!("     DRM process scan: {p}."),
            fg(GREY62),
        ));
    }
    lines
}

fn is_amd_source(d: &DeviceTelemetry) -> bool {
    d.source.starts_with("AMD ")
}

fn metrics_line(d: &DeviceTelemetry) -> Line<'static> {
    let rx = d.pcie_rx_kbs.map(|k| k as f64 / 1024.0);
    let tx = d.pcie_tx_kbs.map(|k| k as f64 / 1024.0);
    let label = if is_amd_source(d) { "AMD gfx" } else { "GPU" };
    let mut spans = vec![
        Span::styled(
            format!("{label} {}", metric(d.util_gpu, None, "%")),
            fg(Chart::Gpu.color()).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  Mem activity {}", metric(d.util_mem, None, "%")),
            fg(Chart::MemActivity.color()),
        ),
        Span::raw(format!(
            "  PCIe↓{:>5} ↑{:>4} MB/s",
            metric(rx, Some(0), ""),
            metric(tx, Some(0), "")
        )),
    ];
    let nonzero = |v: Option<f64>| v.is_some_and(|v| v != 0.0);
    if nonzero(d.enc) || nonzero(d.dec) {
        spans.push(Span::raw(format!(
            "  enc {} dec {}",
            metric(d.enc, None, "%"),
            metric(d.dec, None, "%")
        )));
    }
    Line::from(spans)
}

fn status_line(d: &DeviceTelemetry, slim: bool) -> Line<'static> {
    let amd = is_amd_source(d);
    let u = |v: Option<u32>| v.map(f64::from);
    let power_kind = if d.power_kind.is_empty() {
        String::new()
    } else {
        format!(" {}", d.power_kind)
    };
    let text = if slim {
        let (label, clock) = if amd {
            ("gfx", d.gfx_clock)
        } else {
            ("SM", d.sm_clock)
        };
        format!(
            "{}  {} W{power_kind}  {label} {} MHz",
            metric(d.temp, None, "°C"),
            metric(d.power_w, Some(0), ""),
            metric(u(clock), None, "")
        )
    } else {
        let pstate = d
            .pstate
            .map_or_else(|| "P N/A".to_string(), |p| format!("P{p}"));
        let clock = if amd {
            format!("gfx {} MHz", metric(u(d.gfx_clock), None, ""))
        } else {
            format!(
                "SM {}/{} MHz",
                metric(u(d.sm_clock), None, ""),
                metric(u(d.sm_clock_max), None, "")
            )
        };
        let tail = if d.source == "NVML" {
            format!(
                "CC {}",
                if d.compute_cap.is_empty() {
                    "N/A"
                } else {
                    &d.compute_cap
                }
            )
        } else {
            d.source.clone()
        };
        format!(
            "{}  {}/{} W{power_kind}  {pstate}  {clock}  mem {} MHz  video {} MHz  PCIe gen{} x{}  {tail}",
            metric(d.temp, None, "°C"),
            metric(d.power_w, Some(0), ""),
            metric(d.power_limit_w, Some(0), ""),
            metric(u(d.mem_clock), None, ""),
            metric(u(d.video_clock), None, ""),
            metric(u(d.pcie_gen), None, ""),
            metric(u(d.pcie_width), None, ""),
        )
    };
    let mut spans = vec![Span::styled(text, fg(GREY70))];
    if !d.throttle.is_empty() {
        spans.push(Span::styled(
            format!("  throttle: {}", d.throttle.join(",")),
            fg(YELLOW),
        ));
    }
    Line::from(spans)
}
