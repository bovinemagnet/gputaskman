#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::provenance::{Bytes, Confidence, EstimateSource, Reason, Reporter};
use gputaskman_tui::format::*;
use gputaskman_tui::widgets::{spark, stacked_bar};
use ratatui::style::{Color, Modifier};

#[test]
fn spark_distinguishes_zero_from_missing_and_clamps_values() {
    assert_eq!(
        spark(&[None, Some(0.0), Some(50.0), Some(100.0)], 4, 100.0),
        " ▁▅█"
    );
    assert_eq!(spark(&[Some(-10.0), Some(200.0)], 2, 100.0), "▁█");
    assert_eq!(spark(&[Some(100.0), None, Some(0.0)], 2, 100.0), " ▁");
}

fn bar_text(spans: &[ratatui::text::Span<'_>]) -> String {
    spans.iter().map(|s| s.content.as_ref()).collect()
}

#[test]
fn stacked_bar_fills_proportionally_and_pads_with_shade() {
    let spans = stacked_bar(&[(4.0, MAGENTA), (2.0, GREY35)], 16.0, 8);
    assert_eq!(bar_text(&spans), "███░░░░░");
    assert_eq!(spans[0].style.fg, Some(MAGENTA));
    assert_eq!(spans[1].style.fg, Some(GREY35));
    assert_eq!(spans[2].style.fg, Some(GREY27));
}

#[test]
fn stacked_bar_never_overflows_and_handles_a_zero_total() {
    assert_eq!(bar_text(&stacked_bar(&[(20.0, MAGENTA)], 10.0, 5)), "█████");
    assert_eq!(bar_text(&stacked_bar(&[(3.0, MAGENTA)], 0.0, 4)), "░░░░");
}

#[test]
fn stacked_bar_rounds_half_to_even_like_python() {
    // 1/8 of 4 cells is 0.5: Python's round() gives 0.
    assert_eq!(bar_text(&stacked_bar(&[(1.0, MAGENTA)], 8.0, 4)), "░░░░");
}

#[test]
fn bytes_text_follows_provenance() {
    let gib = 1u64 << 30;
    assert_eq!(bytes_text(&Bytes::measured(gib)), "1.00 GB");
    assert_eq!(
        bytes_text(&Bytes::Inferred {
            bytes: gib,
            by: Confidence::Order
        }),
        "1.00 GB?"
    );
    assert_eq!(
        bytes_text(&Bytes::Estimated {
            bytes: gib,
            from: EstimateSource::Gguf
        }),
        "1.00 GB (est)"
    );
    assert_eq!(
        bytes_text(&Bytes::Reported {
            bytes: gib,
            by: Reporter::LoadLog
        }),
        "1.00 GB (log)"
    );
    assert_eq!(bytes_text(&Bytes::unavailable(Reason::NotOnBackend)), "");
    assert_eq!(bytes_text(&Bytes::unavailable(Reason::SharedClient)), "n/a");
}

#[test]
fn bytes_cell_hides_tiny_measured_values_only() {
    assert_eq!(bytes_cell(&Bytes::measured(512 * 1024), MIB), "");
    assert_eq!(bytes_cell(&Bytes::measured(0), MIB), "");
    assert_eq!(
        bytes_cell(&Bytes::unavailable(Reason::Incomplete), MIB),
        "n/a"
    );
    assert_eq!(bytes_cell(&Bytes::measured(2 * MIB), MIB), "2 MB");
}

#[test]
fn pct_cell_matches_the_python_table() {
    assert_eq!(pct_cell(None).0, "n/a");
    assert_eq!(pct_cell(None).1.fg, Some(GREY62));
    assert_eq!(pct_cell(Some(0.0)).0, "");
    assert_eq!(pct_cell(Some(0.4)).0, "");
    assert_eq!(pct_cell(Some(65.0)).0, "65");
    assert!(pct_cell(Some(65.0)).1.add_modifier.contains(Modifier::BOLD));
    assert!(!pct_cell(Some(5.0)).1.add_modifier.contains(Modifier::BOLD));
}

#[test]
fn kind_colours_follow_python_and_unknown_kinds_are_plain() {
    assert_eq!(kind_style("LLM").fg, Some(MAGENTA));
    assert!(kind_style("LLM").add_modifier.contains(Modifier::BOLD));
    assert_eq!(kind_style("browser").fg, Some(Color::Indexed(27)));
    assert_eq!(kind_style("nonsense").fg, None);
    assert_eq!(bar_color("nonsense"), WHITE);
}

#[test]
fn thousands_groups_digits() {
    assert_eq!(thousands(4096), "4,096");
    assert_eq!(thousands(40960), "40,960");
    assert_eq!(thousands(999), "999");
    assert_eq!(thousands(1_000_000), "1,000,000");
}

#[test]
fn reported_values_name_their_reporter() {
    let gib = 1u64 << 30;
    for (by, label) in [
        (Reporter::LoadLog, "(log)"),
        (Reporter::LlamaServerApi, "(llama-server)"),
        (Reporter::OllamaApi, "(ollama)"),
        (Reporter::LmStudioApi, "(lmstudio)"),
    ] {
        assert_eq!(
            bytes_text(&Bytes::Reported { bytes: gib, by }),
            format!("1.00 GB {label}")
        );
    }
}
