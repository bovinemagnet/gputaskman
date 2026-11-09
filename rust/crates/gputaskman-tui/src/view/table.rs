//! The process table (`app.py` `render_table`): cells per row, in Python's columns and colours.
use gputaskman_core::model::Engines;
use gputaskman_core::provenance::{Bytes, Reason};
use gputaskman_core::report::vram_label;
use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Cell, Row, Table};

use crate::app::App;
use crate::format::{
    GREY50, GREY62, GREY70, MAGENTA, MIB, RED, YELLOW, bold, bytes_cell, fg, kind_style, pct_cell,
};
use crate::rows::{SORTS, ShownRow};

/// Name, width (0 fills the rest), right-aligned.
pub const COLUMNS: [(&str, u16, bool); 12] = [
    ("PID", 8, true),
    ("Process", 22, false),
    ("Kind", 11, false),
    ("Runtime", 20, false),
    ("VRAM", 10, true),
    ("Spill", 9, true),
    ("Paged", 9, true),
    ("GPU%", 5, true),
    ("3D", 4, true),
    ("Copy", 5, true),
    ("Video", 6, true),
    ("What's inside", 0, false),
];

/// The extra column with several GPUs.
pub const GPU_COLUMN: (&str, u16, bool) = ("GPU", 3, true);

pub fn subtitle(app: &App) -> String {
    format!(
        "sort: {}  ·  {} processes",
        SORTS[app.sort % SORTS.len()],
        if app.show_all { "all" } else { "top" }
    )
}

pub fn cells(r: &ShownRow<'_>, multi: bool) -> Vec<(String, Style)> {
    let row = r.row;
    let g = &row.gpu;
    let local = row.origin == gputaskman_core::collector::Origin::Local;
    let engine = |f: &dyn Fn(&Engines) -> f64| match &g.engines {
        e @ Engines::Measured { .. } => pct_cell(Some(f(e))),
        Engines::Unavailable { .. } => pct_cell(None),
    };

    let spill_text = bytes_cell(&g.shared, MIB);
    let spill_style = if spill_text == "n/a" {
        fg(GREY62)
    } else if g.shared.value().is_some_and(|v| v >= 256 * MIB)
        && matches!(row.kind.as_str(), "LLM" | "AI")
        && r.snap.shared_is_spill
    {
        fg(RED).add_modifier(Modifier::BOLD)
    } else {
        fg(YELLOW)
    };
    // Ruling R4: no committed counter means no paged estimate at all.
    let paged = match g.committed {
        Bytes::Unavailable {
            reason: Reason::NotOnBackend,
        } => String::new(),
        _ => bytes_cell(&g.paged_out(), MIB),
    };

    let mut out = vec![
        (
            row.pid_label(),
            if local { Style::new() } else { fg(GREY62) },
        ),
        (row.name.clone(), Style::new()),
        (row.kind.clone(), kind_style(&row.kind)),
        (
            row.detail
                .runtimes
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(" "),
            fg(GREY70),
        ),
        (
            vram_label(g),
            if g.vram.is_measured() {
                bold()
            } else {
                fg(GREY62)
            },
        ),
        (spill_text, spill_style),
        (paged, fg(GREY62)),
        engine(&|e| e.busiest().unwrap_or(0.0)),
        engine(&|e| e.get("3d") + e.get("compute")),
        engine(&|e| e.get("copy")),
        engine(&|e| e.get("decode") + e.get("encode") + e.get("video")),
        (
            row.inside(),
            fg(if row.llm.is_some() { MAGENTA } else { GREY62 }),
        ),
    ];
    if multi {
        out.push((
            r.gpu.map_or_else(String::new, |i| i.to_string()),
            Style::new(),
        ));
    }
    out
}

/// The text of `column` in the visible row `key` (for tests).
pub fn cell(app: &App, key: &str, column: &str) -> Option<String> {
    let multi = app.multi();
    let index = COLUMNS
        .iter()
        .map(|c| c.0)
        .chain(multi.then_some(GPU_COLUMN.0))
        .position(|c| c == column)?;
    let rows = app.rows();
    let row = rows.iter().find(|r| r.key() == key)?;
    cells(row, multi).into_iter().nth(index).map(|c| c.0)
}

/// Draws the table in `area`; returns the data rows' area and the first row drawn.
pub fn draw(app: &App, f: &mut Frame, area: Rect) -> (Rect, usize) {
    let rows = app.rows();
    let multi = app.multi();
    let block = Block::new()
        .borders(Borders::BOTTOM)
        .border_style(fg(GREY50))
        .title_bottom(Line::styled(subtitle(app), fg(GREY62)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let body = Rect {
        y: inner.y.saturating_add(1),
        height: inner.height.saturating_sub(1),
        ..inner
    };
    let visible = usize::from(body.height).max(1);
    let selected = app
        .selected
        .as_deref()
        .and_then(|k| rows.iter().position(|r| r.key() == k));
    let focus = selected.unwrap_or(app.cursor);
    let mut offset = app
        .areas
        .table_offset
        .min(rows.len().saturating_sub(visible));
    if focus < offset {
        offset = focus;
    } else if focus >= offset + visible {
        offset = focus + 1 - visible;
    }
    let mut columns: Vec<(&str, u16, bool)> = COLUMNS.to_vec();
    if multi {
        columns.push(GPU_COLUMN);
    }
    let widths: Vec<Constraint> = columns
        .iter()
        .map(|c| {
            if c.1 == 0 {
                Constraint::Fill(1)
            } else {
                Constraint::Length(c.1)
            }
        })
        .collect();
    let header = Row::new(columns.iter().map(|c| {
        let line = Line::from(c.0);
        Cell::from(if c.2 { line.right_aligned() } else { line })
    }))
    .style(bold());
    let body_rows: Vec<Row<'static>> = rows
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
        .map(|(i, r)| {
            let row = Row::new(
                cells(r, multi)
                    .into_iter()
                    .zip(&columns)
                    .map(|((t, s), c)| {
                        let line = Line::styled(t, s);
                        Cell::from(if c.2 { line.right_aligned() } else { line })
                    }),
            );
            if Some(i) == selected {
                row.style(
                    Style::new()
                        .bg(Color::Indexed(25))
                        .add_modifier(Modifier::BOLD),
                )
            } else if i % 2 == 1 {
                row.style(Style::new().bg(Color::Indexed(235)))
            } else {
                row
            }
        })
        .collect();
    f.render_widget(
        Table::new(body_rows, widths)
            .header(header)
            .column_spacing(1),
        inner,
    );
    (body, offset)
}
