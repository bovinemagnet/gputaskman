#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::BTreeMap;

use gputaskman_core::model::{Engines, ProcGpu};
use gputaskman_core::provenance::{Bytes, Reason};
use gputaskman_tui::fixtures::*;
use gputaskman_tui::rows::*;

fn keys(rows: &[ShownRow<'_>]) -> Vec<String> {
    rows.iter().map(ShownRow::key).collect()
}

#[test]
fn rows_sort_by_vram_by_default_and_cycle_through_five_sorts() {
    let s = snapshot();
    let snaps = vec![s];
    assert_eq!(keys(&table_rows(&snaps, false, 0)), ["10", "20"]);
    assert_eq!(keys(&table_rows(&snaps, false, 4)), ["10", "20"]); // llama-server < xorg
    assert_eq!(SORTS, ["VRAM", "GPU%", "spill", "paged", "name"]);
}

#[test]
fn unknown_vram_sorts_as_zero_and_ties_keep_their_order() {
    let mut s = snapshot();
    let mut hidden = row(30, "hidden", "AI", ProcGpu::measured(0));
    hidden.gpu.vram = Bytes::unavailable(Reason::NvmlHidden);
    let small = row(40, "small", "app", ProcGpu::measured(0));
    s.procs.push(hidden);
    s.procs.push(small);
    let snaps = vec![s];
    assert_eq!(keys(&table_rows(&snaps, true, 0)), ["10", "20", "30", "40"]);
}

#[test]
fn gpu_sort_puts_the_busiest_first() {
    let mut s = snapshot();
    s.procs[1].gpu.engines = Engines::Measured {
        by_engine: BTreeMap::from([("3d".to_string(), 40.0)]),
    };
    let snaps = vec![s];
    assert_eq!(keys(&table_rows(&snaps, false, 1)), ["20", "10"]);
}

#[test]
fn the_top_filter_keeps_unknown_busy_big_and_llm_rows() {
    let mut s = snapshot();
    s.procs.push(row(30, "tiny", "app", ProcGpu::measured(MIB)));
    let mut unknown = row(31, "unknown", "app", ProcGpu::measured(0));
    unknown.gpu.vram = Bytes::unavailable(Reason::SharedClient);
    s.procs.push(unknown);
    let mut busy = row(32, "busy", "app", ProcGpu::measured(0));
    busy.gpu.engines = Engines::Measured {
        by_engine: BTreeMap::from([("compute".to_string(), 1.0)]),
    };
    s.procs.push(busy);
    let mut spill = row(33, "spill", "app", ProcGpu::measured(0));
    spill.gpu.shared = Bytes::measured(64 * MIB);
    s.procs.push(spill);
    let snaps = vec![s];
    let top = keys(&table_rows(&snaps, false, 4));
    assert!(!top.contains(&"30".to_string()));
    for k in ["31", "32", "33", "10", "20"] {
        assert!(top.contains(&k.to_string()), "{k} missing from {top:?}");
    }
    assert!(keys(&table_rows(&snaps, true, 0)).contains(&"30".to_string()));
}

#[test]
fn several_gpus_prefix_keys_with_the_gpu_and_sort_together() {
    let first = snapshot();
    let mut second = snapshot();
    second.procs[0].gpu.vram = Bytes::measured(3 * GIB);
    let snaps = vec![first, second];
    let rows = table_rows(&snaps, false, 0);
    assert_eq!(keys(&rows), ["0:10", "1:10", "0:20", "1:20"]);
    assert_eq!(rows[1].gpu, Some(1));
}

#[test]
fn first_llm_key_picks_the_largest_llm_on_any_gpu() {
    let first = snapshot();
    let mut second = snapshot();
    second.procs[0].gpu.vram = Bytes::measured(9 * GIB);
    assert_eq!(first_llm_key(&[snapshot()]), Some("10".to_string()));
    assert_eq!(first_llm_key(&[first, second]), Some("1:10".to_string()));
    let mut none = snapshot();
    none.procs[0].llm = None;
    assert_eq!(first_llm_key(&[none]), None);
}
