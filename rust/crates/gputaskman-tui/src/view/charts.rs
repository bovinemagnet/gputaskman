//! The charts box (`app.py` `render_charts`): the selector header, a grid of sparkline cells per
//! GPU, and the scale notes.
use gputaskman_core::collector::Snapshot;
use gputaskman_core::report::metric;
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};

use crate::app::{App, CHART_MODES};
use crate::format::{CYAN, GREY62, bold, fg};
use crate::history::{CHARTS, Chart, History, is_amd};
use crate::widgets::spark;

pub const LABEL: &str = "Charts  ";
const HINT: &str = "  Click to choose · c cycles";

fn first_is_amd(app: &App) -> bool {
    app.snapshots.first().is_some_and(is_amd)
}

/// The selector's text: `All charts` or the chosen chart's name.
pub fn selector_label(app: &App) -> String {
    match app.chart {
        None => "All charts".to_string(),
        Some(c) => c.menu_name(first_is_amd(app), app.multi()).to_string(),
    }
}

/// Width of `[<label> ▾]`.
pub fn selector_width(app: &App) -> u16 {
    u16::try_from(selector_label(app).chars().count() + 4).unwrap_or(u16::MAX)
}

/// The chart menu's entries, in `CHART_MODES` order.
pub fn menu_entries(app: &App) -> Vec<String> {
    let (amd, multi) = (first_is_amd(app), app.multi());
    CHART_MODES
        .iter()
        .map(|m| match m {
            None => "All charts".to_string(),
            Some(c) => c.menu_name(amd, multi).to_string(),
        })
        .collect()
}

/// Python's grid: at most 3 columns, one per 46 cells of width (after 6 for border and padding).
pub fn grid_columns(count: usize, term_width: u16) -> usize {
    let width = usize::from(term_width.saturating_sub(6).max(1));
    count.min(3).min(((width + 2) / 46).max(1)).max(1)
}

pub fn chart_lines(app: &App, term_width: u16) -> Vec<Line<'static>> {
    let mut out = vec![Line::from(vec![
        Span::styled(LABEL, bold()),
        Span::styled(format!("[{} ▾]", selector_label(app)), fg(CYAN)),
        Span::styled(HINT, fg(GREY62)),
    ])];
    if app.snapshots.is_empty() {
        return out;
    }
    let multi = app.multi();
    let charts: Vec<Chart> = match app.chart {
        None => CHARTS.to_vec(),
        Some(c) => vec![c],
    };
    let columns = grid_columns(charts.len(), term_width);
    let width = usize::from(term_width.saturating_sub(6).max(1));
    let cell_width = (width.saturating_sub(2 * (columns - 1)) / columns).max(1);
    let empty = History::default();
    let mut notes = Vec::new();
    for (i, s) in app.snapshots.iter().enumerate() {
        let history = app.histories.get(i).unwrap_or(&empty);
        let amd = is_amd(s);
        if multi {
            out.push(Line::styled(format!("GPU {i}"), bold()));
        }
        let cells: Vec<Vec<Span<'static>>> = charts
            .iter()
            .map(|c| cell(*c, history, s, amd, cell_width))
            .collect();
        for chunk in cells.chunks(columns) {
            let mut spans = Vec::new();
            for (j, c) in chunk.iter().enumerate() {
                if j > 0 {
                    spans.push(Span::raw("  "));
                }
                spans.extend(c.iter().cloned());
            }
            out.push(Line::from(spans));
        }
        let note = note(history, s, amd);
        notes.push(if multi {
            format!("GPU {i}: {note}")
        } else {
            note
        });
    }
    for n in notes {
        out.push(Line::styled(n, fg(GREY62)));
    }
    out
}

fn cell(c: Chart, h: &History, s: &Snapshot, amd: bool, width: usize) -> Vec<Span<'static>> {
    let style = fg(c.color());
    let name = format!("{:<12} ", c.name(amd));
    let value = format!("{:>8} ", metric(h.current(c), Some(0), c.unit()));
    let line = spark(
        h.values(c),
        width.saturating_sub(22).max(1),
        h.scale(c, s.device.as_ref()),
    );
    let used = name.chars().count() + value.chars().count() + line.chars().count();
    let mut spans = vec![
        Span::styled(name, style),
        Span::styled(value, style.add_modifier(Modifier::BOLD)),
        Span::styled(line, style),
    ];
    if used < width {
        spans.push(Span::raw(" ".repeat(width - used)));
    }
    spans
}

fn note(h: &History, s: &Snapshot, amd: bool) -> String {
    let d = s.device.as_ref();
    format!(
        "Recent samples · blank = no reading · power 0–{:.0} W · {} clock 0–{:.0} MHz",
        h.scale(Chart::Power, d),
        if amd { "GFX" } else { "SM" },
        h.scale(Chart::Clock, d)
    )
}
