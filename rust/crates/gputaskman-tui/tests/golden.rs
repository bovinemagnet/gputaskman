#![allow(clippy::unwrap_used, clippy::expect_used)]
use gputaskman_core::llm::view::{Segment, SegmentKind};
use gputaskman_core::model::{
    BackendKind, Engines, Evidence, MappedModel, ProcGpu, VENDOR_AMD, VENDOR_NVIDIA,
};
use gputaskman_core::provenance::{Bytes, EstimateSource, Reason, Reporter};
use gputaskman_tui::app::App;
use gputaskman_tui::fixtures::*;

fn nvidia() -> gputaskman_core::collector::Snapshot {
    let mut s = snapshot();
    s.adapter = Some(adapter(VENDOR_NVIDIA, "NVIDIA GeForce RTX 3060 Ti"));
    s.driver = Some("580.95".into());
    s.cuda = Some("13.0".into());
    let mut d = device("NVIDIA GeForce RTX 3060 Ti");
    d.util_gpu = Some(15.0);
    d.util_mem = Some(30.0);
    d.temp = Some(37.0);
    d.power_w = Some(26.0);
    d.power_limit_w = Some(200.0);
    d.sm_clock = Some(390);
    d.sm_clock_max = Some(2100);
    s.device = Some(d);
    s
}

#[test]
fn golden_nvml_single_gpu_with_an_llm_selected() {
    let mut s = nvidia();
    let mut view = server_view();
    view.models = vec![gguf("qwen3", "Qwen3-1.7B")];
    view.files = vec![MappedModel {
        path: "/models/Qwen3-1.7B.gguf".into(),
        size: GIB,
        resident: 0,
        gguf: true,
        sources: vec![Evidence::Mapped],
        store_name: None,
        kind: "LLM".into(),
        kind_hint: String::new(),
    }];
    view.segments = vec![
        Segment {
            label: "weights (CUDA0)".into(),
            bytes: Bytes::Reported {
                bytes: 604 * MIB,
                by: Reporter::LoadLog,
            },
            kind: SegmentKind::Weights,
        },
        Segment {
            label: "KV cache".into(),
            bytes: Bytes::Estimated {
                bytes: 448 * MIB,
                from: EstimateSource::Gguf,
            },
            kind: SegmentKind::Kv,
        },
    ];
    view.log_source = Some("/var/log/llama/server.log".into());
    s.procs[0].llm = Some(view);
    let mut app = App::default();
    app.apply_all(vec![s]);
    insta::assert_snapshot!(screen(&render(&mut app, 140, 42)));
}

#[test]
fn golden_drm_with_a_shared_client() {
    let mut s = snapshot();
    s.backend = Some(BackendKind::Drm);
    s.adapter = Some(adapter(VENDOR_AMD, "AMD Radeon RX 9070"));
    let mut shared = row(30, "firefox", "browser", ProcGpu::measured(0));
    shared.gpu.vram = Bytes::unavailable(Reason::SharedClient);
    shared.gpu.shared = Bytes::unavailable(Reason::SharedClient);
    shared.gpu.engines = Engines::Unavailable {
        reason: Reason::SharedClient,
    };
    s.procs.push(shared);
    let mut app = App::default();
    app.selected = Some("30".into());
    app.apply_all(vec![s]);
    insta::assert_snapshot!(screen(&render(&mut app, 140, 42)));
}

#[test]
fn golden_two_gpus() {
    let mut app = App::default();
    app.apply_all(vec![nvidia(), snapshot()]);
    insta::assert_snapshot!(screen(&render(&mut app, 140, 50)));
}

#[test]
fn golden_no_gpu() {
    let mut s = snapshot();
    s.procs.clear();
    s.backend = None;
    s.vram_total = Bytes::unavailable(Reason::NotOnBackend);
    s.vram_used = Bytes::unavailable(Reason::NotOnBackend);
    s.shared_used = Bytes::unavailable(Reason::NotOnBackend);
    s.notes = vec![
        "NVML not loaded: libnvidia-ml.so.1: cannot open shared object file".into(),
        "no supported GPU found (NVIDIA through NVML, or an amdgpu card)".into(),
    ];
    let mut app = App::default();
    app.apply_all(vec![s]);
    insta::assert_snapshot!(screen(&render(&mut app, 120, 30)));
}
