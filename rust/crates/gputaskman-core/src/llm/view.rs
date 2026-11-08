//! The LLM memory breakdown for one process (`llm.py` `build_view`): which models are loaded, what
//! the runtime reported, and how the process's VRAM splits. Segments from a load log are
//! runtime-reported; everything else here is an estimate, never a measurement.
use std::path::{Path, PathBuf};

use super::flags::{Flags, parse_flags};
use super::gb;
use super::logs::{is_gpu_device, parse_load_log};
use super::servers::{ServerInfo, ServerModel};
use crate::gguf::{GgufInfo, MetaValue, is_gguf};
use crate::model::{Evidence, MIB, MappedModel, ProcDetail};
use crate::procinfo::models::{ModelContext, abspath};
use crate::provenance::{Bytes, EstimateSource, Reporter};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentKind {
    Weights,
    Kv,
    Compute,
    Runtime,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Segment {
    pub label: String,
    pub bytes: Bytes,
    pub kind: SegmentKind,
}

impl Segment {
    fn reported(label: String, bytes: u64, kind: SegmentKind) -> Self {
        Self {
            label,
            bytes: Bytes::Reported {
                bytes,
                by: Reporter::LoadLog,
            },
            kind,
        }
    }

    fn estimated(label: impl Into<String>, bytes: u64, kind: SegmentKind) -> Self {
        Self {
            label: label.into(),
            bytes: Bytes::Estimated {
                bytes,
                from: EstimateSource::Gguf,
            },
            kind,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LlmView {
    pub models: Vec<GgufInfo>,
    /// Evidence for `models[i]` (same order).
    pub files: Vec<MappedModel>,
    pub other_files: Vec<MappedModel>,
    pub flags: Flags,
    pub server: Option<ServerInfo>,
    /// Managing front-end (Ollama or LM Studio) when the runner has its own server.
    pub host: Option<ServerInfo>,
    pub log_source: Option<String>,
    pub segments: Vec<Segment>,
    pub notes: Vec<String>,
    pub kv_per_token: Option<u64>,
    pub kv_type: String,
    pub n_ctx: Option<i64>,
}

impl LlmView {
    /// Best human name for `models[i]`: model-store name, else GGUF basename or name.
    #[must_use]
    pub fn label(&self, i: usize) -> String {
        let f = self.files.get(i);
        let g = &self.models[i];
        let store = f
            .and_then(|f| f.store_name.clone())
            .filter(|s| !s.is_empty());
        if let Some(f) = f
            && !f.kind_hint.is_empty()
        {
            return store.unwrap_or_else(|| basename(&f.path));
        }
        store
            .or_else(|| {
                g.metadata
                    .get("general.basename")
                    .and_then(MetaValue::as_str)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
            })
            .unwrap_or_else(|| g.name())
    }

    #[must_use]
    pub fn kind_label(&self, i: usize) -> String {
        let g = &self.models[i];
        if let Some(f) = self.files.get(i)
            && !f.kind_hint.is_empty()
        {
            return f.kind_label();
        }
        match g.decision_type().filter(|d| !d.is_empty()) {
            Some(t) => format!("{} ({t})", g.kind()),
            None => g.kind().to_string(),
        }
    }
}

fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
}

/// Bytes per KV-cache element for each cache type name; unknown names count as f16.
fn kv_type_bytes(name: &str) -> f64 {
    match name {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q4_0" | "iq4_nl" => 18.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "q5_0" => 22.0 / 32.0,
        "q5_1" => 24.0 / 32.0,
        _ => 2.0,
    }
}

const SIDE_KINDS: [&str; 2] = ["vision projector", "LoRA adapter"];

/// Python's `f"{n:,}"`.
fn group_thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

fn same_path(a: &str, b: &str) -> bool {
    match (abspath(Path::new(a)), abspath(Path::new(b))) {
        (Some(a), Some(b)) => a == b,
        _ => a == b,
    }
}

/// Marks files a runtime API also reports; returns the API entries that matched this process.
fn merge_api(
    files: &mut Vec<MappedModel>,
    server: Option<&ServerInfo>,
    host: Option<&ServerInfo>,
    ctx: &mut ModelContext,
    now: f64,
) -> Vec<ServerModel> {
    let mut matched = Vec::new();
    if let Some(s) = server
        && s.kind == "llama-server"
        && let Some(mp) = s.model_path.as_deref().filter(|p| !p.is_empty())
    {
        if let Some(hit) = files.iter_mut().find(|f| same_path(&f.path, mp)) {
            hit.add_source(Evidence::Api);
        } else if let Ok(meta) = std::fs::metadata(mp)
            && meta.is_file()
        {
            // Loaded without mmap or via the router: API-only evidence.
            let m = ctx.make_model(
                mp,
                meta.len(),
                0,
                is_gguf(Path::new(mp)),
                Evidence::Api,
                None,
                now,
            );
            files.insert(0, m);
        }
    }
    for front in [server, host]
        .into_iter()
        .flatten()
        .filter(|s| s.kind == "ollama")
    {
        for f in files.iter_mut() {
            let store = f.store_name.clone().unwrap_or_default();
            let names: Vec<&str> = store.split(", ").collect();
            for om in &front.models {
                if om.name.as_deref().is_some_and(|n| names.contains(&n)) {
                    f.add_source(Evidence::Api);
                    matched.push(om.clone());
                }
            }
        }
    }
    let lms = match (host, server) {
        (Some(h), _) if h.kind == "lmstudio" => Some(h),
        (_, Some(s)) if s.kind == "lmstudio" => Some(s),
        _ => None,
    };
    if let Some(l) = lms
        && l.models.len() == 1
    {
        // LM Studio keys don't map to paths; only pair 1:1.
        let mut primary = files
            .iter_mut()
            .filter(|f| !SIDE_KINDS.contains(&f.kind.as_str()));
        if let (Some(only), None) = (primary.next(), primary.next()) {
            only.add_source(Evidence::Api);
            matched.push(l.models[0].clone());
        }
    }
    matched
}

fn nonzero(n: Option<i64>) -> Option<i64> {
    n.filter(|n| *n != 0)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn build_view(
    d: &ProcDetail,
    vram: f64,
    shared: f64,
    server: Option<ServerInfo>,
    log_path: Option<PathBuf>,
    host: Option<ServerInfo>,
    attributed: bool,
    ctx: &mut ModelContext,
    now: f64,
) -> LlmView {
    let flags = parse_flags(&d.cmdline);
    // Cached details stay untouched: merge into a copy.
    let mut files = d.models.clone();
    let matched = merge_api(&mut files, server.as_ref(), host.as_ref(), ctx, now);
    let mut v = LlmView {
        models: Vec::new(),
        files: Vec::new(),
        other_files: Vec::new(),
        flags,
        server,
        host,
        log_source: None,
        segments: Vec::new(),
        notes: Vec::new(),
        kv_per_token: None,
        kv_type: "f16".to_string(),
        n_ctx: None,
    };
    for m in files {
        let g = if m.gguf {
            ctx.gguf.read(Path::new(&m.path))
        } else {
            None
        };
        match g {
            Some(g) if g.error.is_none() => {
                v.models.push(g);
                v.files.push(m);
            }
            _ => v.other_files.push(m),
        }
    }

    let log_path = log_path.filter(|p| !p.as_os_str().is_empty()).or_else(|| {
        v.flags
            .log_file
            .clone()
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
    });
    let mut log = None;
    if let Some(p) = log_path {
        log = parse_load_log(&p, d.start_time - 5.0);
        if log.is_some() {
            v.log_source = Some(p.to_string_lossy().into_owned());
        }
    }

    // Only causal LLMs grow a KV cache.
    let main = v
        .models
        .iter()
        .find(|g| g.kind() == "LLM" && g.n_layers() != 0)
        .cloned();
    let ctk = v
        .flags
        .ctk
        .clone()
        .unwrap_or_else(|| "f16".into())
        .to_lowercase();
    let ctv = v
        .flags
        .ctv
        .clone()
        .unwrap_or_else(|| "f16".into())
        .to_lowercase();
    v.kv_type = if ctk == ctv {
        ctk.clone()
    } else {
        format!("K {ctk}/V {ctv}")
    };
    if let Some(m) = &main
        && let Some(elems) = m.kv_bytes_per_token(1.0).filter(|e| *e != 0)
    {
        // K+V elements per token; K and V are usually equal width.
        v.kv_per_token =
            Some((elems as f64 * (kv_type_bytes(&ctk) + kv_type_bytes(&ctv)) / 2.0) as u64);
    }
    v.n_ctx = nonzero(v.flags.ctx).or_else(|| {
        v.server
            .as_ref()
            .filter(|s| s.kind == "llama-server")
            .and_then(|s| s.n_ctx)
    });
    if main.is_some() {
        let candidates: &[ServerModel] = if matched.is_empty() {
            v.server
                .as_ref()
                .filter(|s| s.kind == "ollama")
                .map_or(&[], |s| s.models.as_slice())
        } else {
            &matched
        };
        for om in candidates {
            if let Some(c) = nonzero(om.ctx) {
                v.n_ctx = nonzero(v.n_ctx).or(Some(c));
            }
        }
    }
    let primaries: Vec<&GgufInfo> = v
        .models
        .iter()
        .filter(|g| !SIDE_KINDS.contains(&g.kind()))
        .collect();
    let no_kv = !primaries.is_empty() && primaries.iter().all(|g| !g.causal());
    let no_kv_kinds = {
        let mut kinds: Vec<String> = primaries
            .iter()
            .map(|g| match g.decision_type().filter(|t| !t.is_empty()) {
                Some(t) => format!("{t} {}", g.kind()),
                None => g.kind().to_string(),
            })
            .collect();
        kinds.sort();
        kinds.dedup();
        kinds.join(", ")
    };

    // None: the OS can't say how much VRAM this process holds.
    let total: Option<i64> = attributed.then_some(vram as i64);
    if let Some(log) = &log {
        let gpu: Vec<(&String, &std::collections::BTreeMap<String, u64>)> =
            log.iter().filter(|(dev, _)| is_gpu_device(dev)).collect();
        let mut acc: u64 = 0;
        for (kind, label, seg) in [
            ("weights", "weights", SegmentKind::Weights),
            ("kv", "KV cache", SegmentKind::Kv),
            ("compute", "compute buffers", SegmentKind::Compute),
        ] {
            let n: u64 = gpu
                .iter()
                .map(|(_, k)| k.get(kind).copied().unwrap_or(0))
                .sum();
            if n != 0 {
                let devs = gpu
                    .iter()
                    .filter(|(_, k)| k.get(kind).copied().unwrap_or(0) != 0)
                    .map(|(dev, _)| dev.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                v.segments
                    .push(Segment::reported(format!("{label} ({devs})"), n, seg));
                acc += n;
            }
        }
        let cpu_w: u64 = log
            .iter()
            .filter(|(dev, _)| !is_gpu_device(dev))
            .map(|(_, k)| k.get("weights").copied().unwrap_or(0))
            .sum();
        if cpu_w != 0
            && let Some(m) = &main
            && cpu_w as f64 > m.token_embd_bytes as f64 * 1.5
        {
            v.notes.push(format!(
                "{} of weights are on the CPU side (partial offload or CPU-mapped).",
                gb(cpu_w as f64)
            ));
        }
        if let Some(t) = total
            && t > acc as i64
        {
            v.segments.push(Segment::estimated(
                "runtime and other overhead",
                (t - acc as i64) as u64,
                SegmentKind::Runtime,
            ));
        }
    } else {
        let ngl = if v.models.len() == 1 {
            v.flags.ngl
        } else {
            None
        };
        let mut weights: u64 = v
            .models
            .iter()
            .filter(|g| g.kind() != "LoRA adapter")
            .map(|g| g.gpu_weight_bytes(ngl))
            .sum();
        weights += v
            .models
            .iter()
            .filter(|g| g.kind() == "LoRA adapter")
            .map(|g| g.tensor_bytes)
            .sum::<u64>();
        if let Some(t) = total
            && t > 0
            && weights as f64 > t as f64 * 1.03
        {
            v.notes.push(format!(
                "Weights need ~{} but the process holds {} of VRAM: \
                 partial offload, paging, or placement on another GPU may explain the gap.",
                gb(weights as f64),
                gb(t as f64)
            ));
            weights = t as u64;
        }
        if weights != 0 {
            let label = if v.models.len() == 1 {
                "weights".to_string()
            } else {
                format!("weights ({} files)", v.models.len())
            };
            v.segments
                .push(Segment::estimated(label, weights, SegmentKind::Weights));
        }
        // A non-positive context cannot grow a cache.
        let kv: i64 = match (v.kv_per_token, v.n_ctx) {
            (Some(k), Some(c)) if k != 0 && c > 0 => (k as i64).saturating_mul(c),
            _ => 0,
        };
        let n_ctx = group_thousands(v.n_ctx.unwrap_or(0));
        let rest = total.map(|t| t - weights as i64);
        match rest {
            None => {
                if kv != 0 {
                    v.segments.push(Segment::estimated(
                        format!("KV cache ({n_ctx} ctx, {}) est.", v.kv_type),
                        kv as u64,
                        SegmentKind::Kv,
                    ));
                }
                v.notes.push(
                    "This process's VRAM total is unknown (WSL charges it to the VM on the Windows host, \
                     or the driver hides per-process memory); segments are estimates from the model file and launch flags."
                        .to_string(),
                );
            }
            Some(rest) if kv != 0 && kv <= rest => {
                let swa = main
                    .as_ref()
                    .and_then(GgufInfo::sliding_window)
                    .filter(|s| *s != 0);
                v.segments.push(Segment::estimated(
                    format!(
                        "KV cache ({n_ctx} ctx, {}) est.{}",
                        v.kv_type,
                        if swa.is_some() { " max" } else { "" }
                    ),
                    kv as u64,
                    SegmentKind::Kv,
                ));
                if let Some(swa) = swa {
                    v.notes.push(format!(
                        "Sliding-window attention ({swa} tokens): most layers keep a short KV window, \
                         so the real KV cache is smaller than this estimate."
                    ));
                }
                let rest = rest - kv;
                if rest > 0 {
                    v.segments.push(Segment::estimated(
                        "compute buffers + runtime",
                        rest as u64,
                        SegmentKind::Runtime,
                    ));
                }
            }
            Some(rest) if rest > 0 && no_kv => v.segments.push(Segment::estimated(
                "compute buffers + runtime",
                rest as u64,
                SegmentKind::Runtime,
            )),
            Some(rest) if rest > 0 => v.segments.push(Segment::estimated(
                "KV cache + compute + runtime",
                rest as u64,
                SegmentKind::Unknown,
            )),
            Some(_) => {}
        }
    }

    if v.models.is_empty() && log.is_none() && total.is_some_and(|t| t != 0) {
        v.notes.push(
            "No GGUF mapped. Model may be loaded with --no-mmap or via another format.".to_string(),
        );
    }
    if no_kv {
        v.notes.push(format!(
            "Non-causal encoder ({no_kv_kinds}): no KV cache grows with use; VRAM is weights + compute."
        ));
    }
    if main.is_some()
        && let Some(kpt) = v.kv_per_token.filter(|k| *k != 0)
    {
        v.notes.push(format!(
            "KV cost {:.0} KB/token ({}): 4k ctx = {}, 32k ctx = {}.",
            kpt as f64 / 1024.0,
            v.kv_type,
            gb((kpt * 4096) as f64),
            gb((kpt * 32768) as f64)
        ));
    }
    if shared > (256 * MIB) as f64 {
        v.notes.push(format!(
            "! {} is in SHARED GPU memory (system RAM over PCIe). \
             May include pinned host buffers; under VRAM pressure, fallback can slow inference.",
            gb(shared)
        ));
    }
    if total.is_some_and(|t| t < (64 * MIB) as i64) && !v.models.is_empty() {
        v.notes.push(
            "Model is mapped but almost nothing is in VRAM: CPU inference, or another GPU."
                .to_string(),
        );
    }
    let fronts: Vec<&ServerInfo> = [v.server.as_ref(), v.host.as_ref()]
        .into_iter()
        .flatten()
        .filter(|s| s.kind == "ollama")
        .collect();
    for front in fronts {
        let mine: Vec<&ServerModel> = front
            .models
            .iter()
            .filter(|om| matched.contains(om))
            .collect();
        let shown: Vec<&ServerModel> = if mine.is_empty() {
            front.models.iter().collect()
        } else {
            mine.clone()
        };
        for om in shown {
            if let Some(size) = om.size.filter(|s| *s != 0) {
                let pct = 100.0 * om.size_vram.unwrap_or(0) as f64 / size as f64;
                v.notes.push(format!(
                    "Ollama{}: {} {} total, {pct:.0}% on GPU, ctx {}.",
                    if mine.is_empty() {
                        " (not matched to this runner)"
                    } else {
                        ""
                    },
                    om.name.as_deref().unwrap_or(""),
                    gb(size as f64),
                    nonzero(om.ctx).map_or_else(|| "?".to_string(), |c| c.to_string())
                ));
            }
        }
    }
    if let Some(h) = v.host.as_ref().filter(|h| h.kind == "lmstudio") {
        for m in h.models.iter().take(4) {
            v.notes.push(format!(
                "LM Studio: {} {} {} ctx {}.",
                m.name.as_deref().unwrap_or(""),
                m.r#type.as_deref().unwrap_or(""),
                m.quant.as_deref().unwrap_or(""),
                nonzero(m.ctx).map_or_else(|| "?".to_string(), |c| c.to_string())
            ));
        }
    }
    v
}
