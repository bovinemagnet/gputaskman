#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::model::{Engines, ProcGpu, VENDOR_AMD};
use gputaskman_core::provenance::{Bytes, Reason};
use gputaskman_tui::app::App;
use gputaskman_tui::fixtures::*;
use gputaskman_tui::format::GREY35;
use gputaskman_tui::history::Chart;
use gputaskman_tui::view::header::{device_lines, device_title};
use std::collections::BTreeMap;

fn full_device() -> gputaskman_core::model::DeviceTelemetry {
    let mut d = device("Test GPU");
    d.driver_model = "Linux".into();
    d.pci_bus = "00000000:01:00.0".into();
    d.util_gpu = Some(47.0);
    d.util_mem = Some(44.0);
    d.temp = Some(59.0);
    d.power_w = Some(126.0);
    d.power_limit_w = Some(300.0);
    d.sm_clock = Some(2595);
    d.sm_clock_max = Some(3090);
    d.mem_clock = Some(15801);
    d.video_clock = Some(2242);
    d.pstate = Some(1);
    d.pcie_gen = Some(5);
    d.pcie_width = Some(8);
    d.compute_cap = "12.0".into();
    d.pcie_rx_kbs = Some(35 * 1024);
    d.pcie_tx_kbs = Some(31 * 1024);
    d.throttle = vec!["thermal".into()];
    d
}

#[test]
fn monitoring_toggle_hides_diagnostics_without_changing_live_state() {
    let mut s = snapshot();
    s.driver = Some("621.01".into());
    s.cuda = Some("13.5".into());
    s.device = Some(full_device());
    s.notes = vec!["Counters unavailable for one protected process".into()];
    let mut app = App::default();
    app.apply_all(vec![s]);
    let full = plain(&device_lines(&app, 180));
    assert_eq!(device_title(&app), "Full diagnostics");
    for diagnostic in [
        "driver 621.01",
        "CUDA 13.5",
        "Linux",
        "00000000:01:00.0",
        "PCIe gen5 x8",
        "CC 12.0",
        "126/300 W",
        "P1",
        "2595/3090 MHz",
        "mem 15801 MHz",
        "video 2242 MHz",
    ] {
        assert!(
            full.contains(diagnostic),
            "{diagnostic} missing from\n{full}"
        );
    }
    app.paused = true;
    app.slim = true;
    let slim = plain(&device_lines(&app, 180));
    assert_eq!(device_title(&app), "Slim monitoring");
    for hidden in [
        "621.01",
        "13.5",
        "00000000:01:00.0",
        "PCIe gen",
        "CC ",
        "/300 W",
        "P1",
        "/3090",
        "mem 15801",
        "video 2242",
    ] {
        assert!(!slim.contains(hidden), "{hidden} still in\n{slim}");
    }
    for live in [
        "Test GPU",
        "VRAM",
        "10.00 GB / 16.00 GB",
        "shared(RAM) 1.00 GB",
        "GPU 47%",
        "Mem activity 44%",
        "PCIe↓",
        "59°C",
        "126 W",
        "SM 2595 MHz",
        "throttle: thermal",
        "Counters unavailable for one protected process",
        "[PAUSED]",
    ] {
        assert!(slim.contains(live), "{live} missing from\n{slim}");
    }
    app.slim = false;
    app.paused = false;
    assert_eq!(plain(&device_lines(&app, 180)), full);
}

#[test]
fn monitoring_applies_to_every_gpu() {
    let mut first = snapshot();
    let mut second = snapshot();
    first.driver = Some("test-driver".into());
    second.driver = Some("test-driver".into());
    first.device = Some(device("First GPU"));
    second.device = Some(device("Second GPU"));
    let mut app = App::default();
    app.slim = true;
    app.apply_all(vec![first, second]);
    let text = plain(&device_lines(&app, 140));
    for name in ["GPU 0", "GPU 1", "First GPU", "Second GPU"] {
        assert!(text.contains(name), "{name} missing");
    }
    assert!(!text.contains("test-driver"));
    assert!(!text.contains("PCIe gen"));
    app.slim = false;
    assert!(plain(&device_lines(&app, 140)).contains("test-driver"));
}

#[test]
fn monitoring_keeps_missing_values_and_provenance() {
    let mut s = snapshot();
    s.driver = Some("hidden-driver".into());
    s.vram_used = Bytes::unavailable(Reason::NotOnBackend);
    let mut d = device("GPU");
    d.util_gpu = Some(0.0);
    s.device = Some(d);
    s.notes = vec!["Counter pairing unavailable".into()];
    let mut app = App::default();
    app.slim = true;
    app.apply_all(vec![s]);
    let lines = plain(&device_lines(&app, 140));
    assert!(!lines.contains("hidden-driver"));
    assert!(lines.contains("N/A / 16.00 GB"));
    assert!(lines.contains("GPU 0%"));
    assert!(lines.contains("Counter pairing unavailable"));
    assert!(lines.contains("N/A  N/A W  SM N/A MHz"));
    assert!(!lines.contains("None"));

    let mut amd = snapshot();
    amd.adapter = Some(adapter(VENDOR_AMD, "Radeon"));
    let mut d = device("Radeon");
    d.source = "AMD sysfs".into();
    d.util_gpu = Some(25.0);
    d.gfx_clock = Some(2200);
    d.power_w = Some(115.0);
    d.power_kind = "ASIC".into();
    amd.device = Some(d);
    app.apply_all(vec![amd.clone()]);
    let lines = plain(&device_lines(&app, 140));
    assert!(lines.contains("AMD gfx 25%"));
    assert!(lines.contains("gfx 2200 MHz"));
    assert!(lines.contains("115 W ASIC"));
    amd.device = None;
    app.apply_all(vec![amd]);
    assert!(plain(&device_lines(&app, 140)).contains("VRAM"));
}

#[test]
fn unavailable_device_metrics_render_as_na() {
    let mut s = snapshot();
    let mut d = device("GPU");
    d.util_mem = Some(0.0);
    d.power_w = Some(2.4);
    d.pcie_tx_kbs = Some(0);
    s.device = Some(d);
    let mut app = App::default();
    app.apply_all(vec![s]);
    let lines = plain(&device_lines(&app, 140));
    assert!(lines.contains("2/N/A W"), "{lines}");
    assert!(lines.contains("P N/A"));
    assert!(lines.contains("PCIe↓  N/A"));
    assert!(!lines.contains("None"));
}

#[test]
fn the_metrics_line_uses_the_chart_colours() {
    let mut s = snapshot();
    let mut d = device("GPU");
    d.util_gpu = Some(0.0);
    d.util_mem = Some(0.0);
    s.device = Some(d);
    let mut app = App::default();
    app.apply_all(vec![s]);
    let lines = device_lines(&app, 140);
    assert_eq!(
        style_of(&lines, "Mem activity").unwrap().fg,
        Some(Chart::MemActivity.color())
    );
    assert_eq!(
        style_of(&lines, "GPU 0%").unwrap().fg,
        Some(Chart::Gpu.color())
    );
}

#[test]
fn unknown_process_memory_stays_unattributed() {
    let mut s = snapshot();
    s.procs[0].gpu.vram = Bytes::measured(2 * GIB);
    let mut unknown = row(30, "unknown", "app", ProcGpu::measured(0));
    unknown.gpu.vram = Bytes::unavailable(Reason::SharedClient);
    unknown.gpu.engines = Engines::Measured {
        by_engine: BTreeMap::from([("3d".to_string(), 20.0)]),
    };
    s.procs.push(unknown);
    let mut app = App::default();
    app.apply_all(vec![s]);
    let lines = device_lines(&app, 180);
    let text = plain(&lines);
    // 10 GiB used, 4 GiB measured in rows: the other 6 GiB is unattributed, not spread or zeroed.
    assert!(text.contains("unattributed 6.00 GB"), "{text}");
    assert!(text.contains("LLM 2.00 GB"));
    assert!(
        lines[1]
            .spans
            .iter()
            .any(|s| s.style.fg == Some(GREY35) && s.content.contains('█'))
    );
}

#[test]
fn the_bar_is_normalised_when_rows_exceed_the_device_total() {
    let mut s = snapshot();
    s.vram_used = Bytes::measured(5 * GIB); // rows hold 10 GiB
    let mut app = App::default();
    app.apply_all(vec![s]);
    let text = plain(&device_lines(&app, 180));
    assert!(text.contains("[bar normalized; labels are raw]"));
    assert!(text.contains("LLM 8.00 GB"));
}

#[test]
fn notes_and_scan_status_reach_the_device_box() {
    let mut s = snapshot();
    s.notes = vec!["NVML not loaded: libnvidia-ml.so.1 missing".into()];
    s.process_status = Some("2 processes could not be read".into());
    let mut app = App::default();
    app.apply_all(vec![s]);
    let text = plain(&device_lines(&app, 180));
    assert!(text.contains("NVML not loaded: libnvidia-ml.so.1 missing"));
    assert!(text.contains("DRM process scan: 2 processes could not be read."));
}

#[test]
fn before_the_first_sample_the_box_says_so() {
    let app = App::default();
    assert_eq!(plain(&device_lines(&app, 120)), "Sampling GPU counters...");
}
