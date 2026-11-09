//! Which processes the table shows, in what order, and their keys (`app.py` `visible_rows`).
use std::cmp::Reverse;

use gputaskman_core::collector::{ProcRow, Snapshot};

use crate::format::MIB;

/// Sort names, cycled by `s`.
pub const SORTS: [&str; 5] = ["VRAM", "GPU%", "spill", "paged", "name"];

/// A row as the table shows it: with its snapshot and, when there are several GPUs, its GPU.
#[derive(Debug, Clone, Copy)]
pub struct ShownRow<'a> {
    pub row: &'a ProcRow,
    pub snap: &'a Snapshot,
    pub gpu: Option<usize>,
}

impl ShownRow<'_> {
    /// `ProcRow::key()`, prefixed `<gpu>:` when there are several GPUs.
    pub fn key(&self) -> String {
        match self.gpu {
            Some(i) => format!("{i}:{}", self.row.key()),
            None => self.row.key(),
        }
    }
}

fn shown(r: &ProcRow) -> u64 {
    r.gpu.shown_vram().map_or(0, |s| s.get())
}

fn busiest(r: &ProcRow) -> f64 {
    r.gpu.engines.busiest().unwrap_or(0.0)
}

fn spill(r: &ProcRow) -> u64 {
    r.gpu.shared.value().unwrap_or(0)
}

fn paged(r: &ProcRow) -> u64 {
    r.gpu.paged_out().value().unwrap_or(0)
}

/// Python's "top" filter: unknown VRAM, real VRAM, load, spill, paging or an LLM view keeps a row.
pub fn in_top(r: &ProcRow) -> bool {
    !r.gpu.vram.is_measured()
        || shown(r) >= 32 * MIB
        || busiest(r) >= 1.0
        || spill(r) >= 64 * MIB
        || paged(r) >= 128 * MIB
        || r.llm.is_some()
}

/// Stable sort by `SORTS[sort]`: largest first, names A to Z; unknown values count as 0.
pub fn sort_rows(rows: &mut [ShownRow<'_>], sort: usize) {
    match sort % SORTS.len() {
        0 => rows.sort_by_key(|r| Reverse(shown(r.row))),
        1 => rows.sort_by(|a, b| busiest(b.row).total_cmp(&busiest(a.row))),
        2 => rows.sort_by_key(|r| Reverse(spill(r.row))),
        3 => rows.sort_by_key(|r| Reverse(paged(r.row))),
        _ => rows.sort_by_key(|r| r.row.name.to_lowercase()),
    }
}

/// The table's rows over every snapshot, tagged with their GPU when there are several.
pub fn table_rows(snaps: &[Snapshot], show_all: bool, sort: usize) -> Vec<ShownRow<'_>> {
    let multi = snaps.len() > 1;
    let mut rows: Vec<ShownRow<'_>> = snaps
        .iter()
        .enumerate()
        .flat_map(|(i, snap)| {
            snap.procs
                .iter()
                .filter(move |r| show_all || in_top(r))
                .map(move |row| ShownRow {
                    row,
                    snap,
                    gpu: multi.then_some(i),
                })
        })
        .collect();
    sort_rows(&mut rows, sort);
    rows
}

/// The largest LLM row on any GPU (Python's one-time auto-selection).
pub fn first_llm_key(snaps: &[Snapshot]) -> Option<String> {
    table_rows(snaps, true, 0)
        .into_iter()
        .find(|r| r.row.llm.is_some())
        .map(|r| r.key())
}
