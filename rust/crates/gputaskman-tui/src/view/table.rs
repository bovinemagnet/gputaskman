//! The process table (`app.py` `render_table`): cells per row, in Python's columns and colours.
use gputaskman_core::model::Engines;
use gputaskman_core::provenance::{Bytes, Reason};
use gputaskman_core::report::vram_label;
use ratatui::style::{Modifier, Style};

use crate::app::App;
use crate::format::{
    GREY62, GREY70, MAGENTA, MIB, RED, YELLOW, bold, bytes_cell, fg, kind_style, pct_cell,
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
