//! Compare two saved snapshots: which measured memory buckets changed, next to what runtimes reported.
//!
//! Measured counters are compared on their own. Runtime-reported and estimated breakdowns are listed
//! beside them, never added to measured totals. A model file appearing in a process is evidence that
//! the process has the file, not that it is resident in GPU memory. An absent row is `not listed`,
//! not zero, and a counter that is unavailable on either side is unknown, not zero.
use std::collections::BTreeMap;

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use crate::collector::Origin;
use crate::llm::gb;
use crate::llm::view::SegmentKind;
use crate::model::Evidence;
use crate::provenance::{Bytes, Reporter};
use crate::schema::v2::{ProcessV2, SnapshotV2};

const MIB: i64 = 1 << 20;
/// Smaller per-process moves are allocator and compositor churn, not a model loading.
pub const NOISE: i64 = 16 * MIB;
const SEGMENT_ORDER: [&str; 5] = ["weights", "kv", "compute", "runtime", "unknown"];
const PLACEHOLDER_NAMES: [&str; 2] = ["?", "(outside this container)"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Change {
    pub before: Option<i64>,
    pub after: Option<i64>,
    pub delta: Option<i64>,
}

fn change(before: Option<i64>, after: Option<i64>) -> Change {
    Change {
        before,
        after,
        delta: before.zip(after).map(|(b, a)| a.saturating_sub(b)),
    }
}

/// How a row was paired. Serialises as Python's strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matched {
    Pid,
    NamePorts,
    Name,
    New,
    Gone,
}

impl Matched {
    pub fn as_str(self) -> &'static str {
        match self {
            Matched::Pid => "pid",
            Matched::NamePorts => "name+ports",
            Matched::Name => "name",
            Matched::New => "new",
            Matched::Gone => "gone",
        }
    }
}

impl Serialize for Matched {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

/// A map that keeps insertion order, so `segments` serialises in the same order Python lists them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ordered<T>(pub Vec<(String, T)>);

impl<T> Ordered<T> {
    pub fn get(&self, key: &str) -> Option<&T> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn iter(&self) -> impl Iterator<Item = &(String, T)> {
        self.0.iter()
    }
}

impl<T: Serialize> Serialize for Ordered<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeviceChanges {
    pub vram_used: Change,
    pub shared_used: Change,
    pub attributed: Change,
    pub unattributed: Change,
}

/// Measured buckets only. `vram_resident_bytes` and `gtt_resident_bytes` are amdgpu references that
/// include shared buffers and are never summed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MeasuredChanges {
    pub vram_bytes: Change,
    pub shared_bytes: Change,
    pub committed_bytes: Change,
    pub paged_out_bytes: Change,
    pub vram_resident_bytes: Change,
    pub gtt_resident_bytes: Change,
}

impl MeasuredChanges {
    fn all(&self) -> [&Change; 6] {
        [
            &self.vram_bytes,
            &self.shared_bytes,
            &self.committed_bytes,
            &self.paged_out_bytes,
            &self.vram_resident_bytes,
            &self.gtt_resident_bytes,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InferredChanges {
    pub vram_inferred_bytes: Change,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SegmentChange {
    #[serde(flatten)]
    pub change: Change,
    /// `runtime-reported` or `estimated`.
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuntimeChanges {
    pub n_ctx: Change,
    pub segments: Ordered<SegmentChange>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FileRef {
    pub path: String,
    pub sources: Vec<Evidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FileChanges {
    pub added: Vec<FileRef>,
    pub removed: Vec<FileRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProcessChange {
    pub name: String,
    pub host: Option<String>,
    #[serde(rename = "match")]
    pub matched: Matched,
    pub pid_before: Option<u32>,
    pub pid_after: Option<u32>,
    pub measured: MeasuredChanges,
    pub inferred: InferredChanges,
    pub runtime: Option<RuntimeChanges>,
    pub model_files: FileChanges,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiffResult {
    pub gpu: Option<String>,
    pub backend: Option<String>,
    pub seconds: Option<f64>,
    pub device: DeviceChanges,
    pub processes: Vec<ProcessChange>,
}

fn host_name(o: Origin) -> Option<&'static str> {
    match o {
        Origin::Local => None,
        Origin::Outside => Some("outside"),
        Origin::Windows => Some("windows"),
    }
}

fn named(p: &ProcessV2) -> bool {
    p.origin != Origin::Outside && !PLACEHOLDER_NAMES.contains(&p.name.as_str())
}

fn origin_tag(o: Origin) -> u8 {
    match o {
        Origin::Local => 0,
        Origin::Outside => 1,
        Origin::Windows => 2,
    }
}

type Pair<'a> = (Option<&'a ProcessV2>, Option<&'a ProcessV2>, Matched);

/// Pair rows by PID, then by name and ports, then by name, when each pairing is unambiguous.
///
/// A model reload often restarts the server under a new PID. Host rows have their own PID space.
/// Every row comes back exactly once: paired, new, or gone.
pub fn match_rows<'a>(before: &'a [ProcessV2], after: &'a [ProcessV2]) -> Vec<Pair<'a>> {
    let keys: Vec<(u8, Option<u32>)> = before
        .iter()
        .map(|p| (origin_tag(p.origin), p.pid))
        .collect();
    let mut left: BTreeMap<usize, &ProcessV2> = before.iter().enumerate().collect();
    let by_pid: BTreeMap<(u8, u32), usize> = keys
        .iter()
        .enumerate()
        .filter_map(|(i, k)| {
            let pid = k.1?;
            (keys.iter().filter(|o| *o == k).count() == 1).then_some(((k.0, pid), i))
        })
        .collect();
    let mut pairs: Vec<Pair<'a>> = Vec::new();
    let mut rest: Vec<&'a ProcessV2> = Vec::new();
    for a in after {
        let hit = a
            .pid
            .and_then(|pid| by_pid.get(&(origin_tag(a.origin), pid)))
            .copied();
        match hit {
            Some(i) if left.get(&i).is_some_and(|b| b.name == a.name) => {
                if let Some(b) = left.remove(&i) {
                    pairs.push((Some(b), Some(a), Matched::Pid));
                }
            }
            _ => rest.push(a),
        }
    }

    type KeyFn = fn(&ProcessV2) -> Option<String>;
    let by_ports: KeyFn = |p| {
        if p.ports.is_empty() || !named(p) {
            return None;
        }
        let mut ports = p.ports.clone();
        ports.sort_unstable();
        Some(format!("{}\0{}\0{ports:?}", origin_tag(p.origin), p.name))
    };
    let by_name: KeyFn = |p| named(p).then(|| format!("{}\0{}", origin_tag(p.origin), p.name));
    for (how, key) in [(Matched::NamePorts, by_ports), (Matched::Name, by_name)] {
        let mut olds: BTreeMap<Option<String>, Vec<usize>> = BTreeMap::new();
        for (i, b) in &left {
            olds.entry(key(b)).or_default().push(*i);
        }
        let mut news: Vec<(Option<String>, Vec<&'a ProcessV2>)> = Vec::new();
        for a in &rest {
            let k = key(a);
            match news.iter_mut().find(|(nk, _)| *nk == k) {
                Some((_, v)) => v.push(a),
                None => news.push((k, vec![a])),
            }
        }
        for (k, found) in news {
            let Some(k) = k else { continue };
            let [only] = found[..] else { continue };
            let Some([old]) = olds.get(&Some(k)).map(Vec::as_slice) else {
                continue;
            };
            if let Some(b) = left.remove(old) {
                pairs.push((Some(b), Some(only), how));
                rest.retain(|r| !std::ptr::eq(*r, only));
            }
        }
    }
    pairs.extend(rest.into_iter().map(|a| (None, Some(a), Matched::New)));
    pairs.extend(left.into_values().map(|b| (Some(b), None, Matched::Gone)));
    pairs
}

fn int(b: &Bytes) -> Option<i64> {
    b.as_measured()
        .map(|m| i64::try_from(m.get()).unwrap_or(i64::MAX))
}

fn kind_name(k: SegmentKind) -> &'static str {
    match k {
        SegmentKind::Weights => "weights",
        SegmentKind::Kv => "kv",
        SegmentKind::Compute => "compute",
        SegmentKind::Runtime => "runtime",
        SegmentKind::Unknown => "unknown",
    }
}

type Sums = (BTreeMap<&'static str, i64>, BTreeMap<&'static str, bool>);

fn segments(p: Option<&ProcessV2>) -> Sums {
    let mut sums = BTreeMap::new();
    let mut exact = BTreeMap::new();
    for s in p
        .and_then(|p| p.llm.as_ref())
        .map_or(&[][..], |l| l.segments.as_slice())
    {
        let k = kind_name(s.kind);
        let bytes = s
            .bytes
            .value()
            .map_or(0, |v| i64::try_from(v).unwrap_or(i64::MAX));
        let sum = sums.entry(k).or_insert(0i64);
        *sum = sum.saturating_add(bytes);
        let was = exact.get(k).copied().unwrap_or(true);
        exact.insert(
            k,
            was && matches!(
                s.bytes,
                Bytes::Reported {
                    by: Reporter::LoadLog,
                    ..
                }
            ),
        );
    }
    (sums, exact)
}

fn runtime(b: Option<&ProcessV2>, a: Option<&ProcessV2>) -> Option<RuntimeChanges> {
    let lb = b.and_then(|p| p.llm.as_ref());
    let la = a.and_then(|p| p.llm.as_ref());
    if lb.is_none() && la.is_none() {
        return None;
    }
    let (sb, eb) = segments(b);
    let (sa, ea) = segments(a);
    let mut kinds: Vec<&'static str> = sb.keys().chain(sa.keys()).copied().collect();
    kinds.sort_by_key(|k| SEGMENT_ORDER.iter().position(|o| o == k).unwrap_or(99));
    kinds.dedup();
    Some(RuntimeChanges {
        n_ctx: change(lb.and_then(|l| l.n_ctx), la.and_then(|l| l.n_ctx)),
        segments: Ordered(
            kinds
                .into_iter()
                .map(|k| {
                    let reported =
                        eb.get(k).copied().unwrap_or(true) && ea.get(k).copied().unwrap_or(true);
                    (
                        k.to_string(),
                        SegmentChange {
                            change: change(sb.get(k).copied(), sa.get(k).copied()),
                            source: if reported {
                                "runtime-reported"
                            } else {
                                "estimated"
                            }
                            .to_string(),
                        },
                    )
                })
                .collect(),
        ),
    })
}

fn file_map(p: Option<&ProcessV2>) -> Vec<FileRef> {
    let mut out: Vec<FileRef> = Vec::new();
    for m in p.map_or(&[][..], |p| p.model_files.as_slice()) {
        let f = FileRef {
            path: m.path.clone(),
            sources: m.evidence.clone(),
        };
        match out.iter_mut().find(|o| o.path == f.path) {
            Some(o) => *o = f,
            None => out.push(f),
        }
    }
    out
}

fn files(b: Option<&ProcessV2>, a: Option<&ProcessV2>) -> FileChanges {
    let (fb, fa) = (file_map(b), file_map(a));
    let only = |from: &[FileRef], other: &[FileRef]| -> Vec<FileRef> {
        from.iter()
            .filter(|f| !other.iter().any(|o| o.path == f.path))
            .cloned()
            .collect()
    };
    FileChanges {
        added: only(&fa, &fb),
        removed: only(&fb, &fa),
    }
}

fn resident(p: Option<&ProcessV2>, gtt: bool) -> Option<i64> {
    let r = p?.resident.as_ref()?;
    int(if gtt { &r.gtt } else { &r.vram })
}

fn inferred(p: Option<&ProcessV2>) -> Option<i64> {
    match p?.vram {
        Bytes::Inferred { bytes, .. } => Some(i64::try_from(bytes).unwrap_or(i64::MAX)),
        _ => None,
    }
}

fn process_change(b: Option<&ProcessV2>, a: Option<&ProcessV2>, how: Matched) -> ProcessChange {
    let p = a.or(b);
    let pair = |f: fn(&ProcessV2) -> Option<i64>| change(b.and_then(f), a.and_then(f));
    ProcessChange {
        name: p.map(|p| p.name.clone()).unwrap_or_default(),
        host: p.and_then(|p| host_name(p.origin)).map(String::from),
        matched: how,
        pid_before: b.and_then(|p| p.pid),
        pid_after: a.and_then(|p| p.pid),
        measured: MeasuredChanges {
            vram_bytes: pair(|p| int(&p.vram)),
            shared_bytes: pair(|p| int(&p.shared)),
            committed_bytes: pair(|p| int(&p.committed)),
            paged_out_bytes: pair(|p| int(&p.paged_out)),
            vram_resident_bytes: change(resident(b, false), resident(a, false)),
            gtt_resident_bytes: change(resident(b, true), resident(a, true)),
        },
        inferred: InferredChanges {
            vram_inferred_bytes: change(inferred(b), inferred(a)),
        },
        runtime: runtime(b, a),
        model_files: files(b, a),
    }
}

fn size(p: &ProcessChange) -> u64 {
    let v = &p.measured.vram_bytes;
    match v.delta {
        Some(d) => d.unsigned_abs(),
        None => v.after.or(v.before).map_or(0, |n| n.unsigned_abs()),
    }
}

fn unattributed(s: &SnapshotV2) -> Option<i64> {
    let used = int(&s.memory.used)?;
    let attributed = int(&s.memory.attributed)?;
    Some(used.saturating_sub(attributed))
}

fn gpu_name(s: &SnapshotV2) -> Option<String> {
    s.adapter.as_ref().map(|a| a.name.clone())
}

fn backend_name(s: &SnapshotV2) -> Option<String> {
    s.backend
        .map(|b| serde_json::to_value(b).ok())
        .and_then(|v| v?.as_str().map(String::from))
}

fn shown(o: &Option<String>) -> String {
    o.clone().unwrap_or_else(|| "None".to_string())
}

pub fn diff(before: &SnapshotV2, after: &SnapshotV2) -> Result<DiffResult, String> {
    let (gb_, ga) = (gpu_name(before), gpu_name(after));
    let (bb, ba) = (backend_name(before), backend_name(after));
    if gb_ != ga || bb != ba {
        return Err(format!(
            "snapshots are from different GPUs or backends: {} ({}) vs {} ({})",
            shown(&gb_),
            shown(&bb),
            shown(&ga),
            shown(&ba)
        ));
    }
    let both = |f: fn(&SnapshotV2) -> &Bytes| change(int(f(before)), int(f(after)));
    let device = DeviceChanges {
        vram_used: both(|s| &s.memory.used),
        shared_used: both(|s| &s.memory.shared_used),
        attributed: both(|s| &s.memory.attributed),
        unattributed: change(unattributed(before), unattributed(after)),
    };
    let mut processes: Vec<ProcessChange> = match_rows(&before.processes, &after.processes)
        .into_iter()
        .map(|(b, a, how)| process_change(b, a, how))
        .collect();
    processes.sort_by_key(|p| std::cmp::Reverse(size(p)));
    let seconds = before
        .taken
        .zip(after.taken)
        .filter(|(b, a)| b.is_finite() && a.is_finite())
        .map(|(b, a)| ((a - b) * 10.0).round() / 10.0);
    Ok(DiffResult {
        gpu: ga,
        backend: ba,
        seconds,
        device,
        processes,
    })
}

// ---------------------------------------------------------------- text

fn gb_or_na(v: Option<i64>) -> String {
    v.map_or_else(|| "N/A".to_string(), |v| gb(v as f64))
}

fn signed(v: Option<i64>) -> String {
    let Some(v) = v else {
        return "N/A".to_string();
    };
    if v.abs() < MIB / 2 {
        return "0 MB".to_string();
    }
    format!("{}{}", if v >= 0 { "+" } else { "-" }, gb(v.abs() as f64))
}

/// A value appeared, disappeared, or moved by at least the noise threshold.
fn moved(c: &Change) -> bool {
    if c.before == c.after {
        return false;
    }
    c.delta.is_none_or(|d| d.abs() >= NOISE)
}

fn changed(p: &ProcessChange) -> bool {
    if matches!(p.matched, Matched::New | Matched::Gone) {
        return true;
    }
    if p.measured
        .all()
        .iter()
        .any(|c| c.delta.unwrap_or(0).abs() >= NOISE)
    {
        return true;
    }
    if p.inferred.vram_inferred_bytes.delta.unwrap_or(0).abs() >= NOISE {
        return true;
    }
    // Remainder segments follow the measured total, so they move with every sample; apply the same threshold.
    if let Some(r) = &p.runtime
        && (r.n_ctx.before != r.n_ctx.after || r.segments.iter().any(|(_, s)| moved(&s.change)))
    {
        return true;
    }
    !p.model_files.added.is_empty() || !p.model_files.removed.is_empty()
}

fn pid_label(p: &ProcessChange) -> String {
    let pid = match (p.pid_before, p.pid_after) {
        (None, None) => "None".to_string(),
        (None, Some(a)) => a.to_string(),
        (Some(b), None) => b.to_string(),
        (Some(b), Some(a)) if a == b => b.to_string(),
        (Some(b), Some(a)) => format!("{b}->{a}"),
    };
    match p.host.as_deref() {
        Some(h) => format!("{}:{pid}", if h == "windows" { 'W' } else { 'H' }),
        None => pid,
    }
}

fn opt(n: Option<i64>) -> String {
    n.map_or_else(|| "N/A".to_string(), |n| n.to_string())
}

pub fn to_text(d: &DiffResult) -> String {
    let when = d
        .seconds
        .map_or(String::new(), |s| format!(", {s:?} s apart"));
    let mut out: Vec<String> = vec![
        format!(
            "gputaskman diff: {} ({}){when}",
            d.gpu.as_deref().unwrap_or("unknown GPU"),
            shown(&d.backend)
        ),
        "Measured counters are compared directly. Runtime-reported and estimated values are listed"
            .into(),
        "separately and are not added to measured totals.".into(),
        String::new(),
        format!(
            "{:<26}{:>11}{:>11}{:>11}",
            "DEVICE (measured)", "BEFORE", "AFTER", "CHANGE"
        ),
    ];
    for (label, c) in [
        ("dedicated used", &d.device.vram_used),
        ("shared used", &d.device.shared_used),
        ("attributed to processes", &d.device.attributed),
        ("unattributed", &d.device.unattributed),
    ] {
        out.push(format!(
            "{label:<26}{:>11}{:>11}{:>11}",
            gb_or_na(c.before),
            gb_or_na(c.after),
            signed(c.delta)
        ));
    }

    let procs: Vec<&ProcessChange> = d.processes.iter().filter(|p| changed(p)).collect();
    out.push(String::new());
    out.push(format!(
        "{:<22}{:>14}  {:<11}{:>11}{:>11}{:>11}{:>11}",
        "PROCESSES (dedicated)", "PID", "MATCH", "BEFORE", "AFTER", "CHANGE", "SHARED +/-"
    ));
    for p in &procs {
        let (v, s) = (&p.measured.vram_bytes, &p.measured.shared_bytes);
        let before = if p.matched == Matched::New {
            "not listed".to_string()
        } else {
            gb_or_na(v.before)
        };
        let after = if p.matched == Matched::Gone {
            "not listed".to_string()
        } else {
            gb_or_na(v.after)
        };
        let name = if p.name.is_empty() { "?" } else { &p.name };
        let name: String = name.chars().take(21).collect();
        out.push(format!(
            "{name:<22}{:>14}  {:<11}{before:>11}{after:>11}{:>11}{:>11}",
            pid_label(p),
            p.matched.as_str(),
            signed(v.delta),
            signed(s.delta)
        ));
    }
    if procs.is_empty() {
        out.push("(no process changed by 16 MB or more)".into());
    }
    if procs.len() < d.processes.len() {
        out.push(format!(
            "{} process(es) changed by less than 16 MB are not shown; --json lists all.",
            d.processes.len() - procs.len()
        ));
    }

    for p in &procs {
        let mut detail: Vec<String> = Vec::new();
        let inf = &p.inferred.vram_inferred_bytes;
        if moved(inf) {
            detail.push(format!(
                "  inferred VRAM (WSL context match)  {} -> {}  {}",
                gb_or_na(inf.before),
                gb_or_na(inf.after),
                signed(inf.delta)
            ));
        }
        if let Some(r) = &p.runtime {
            if r.n_ctx.before != r.n_ctx.after {
                detail.push(format!(
                    "  context length  {} -> {}",
                    opt(r.n_ctx.before),
                    opt(r.n_ctx.after)
                ));
            }
            for (kind, s) in r.segments.iter() {
                if moved(&s.change) {
                    detail.push(format!(
                        "  {kind:<8} {:>9} -> {:>9}  {:>9}  {}",
                        gb_or_na(s.change.before),
                        gb_or_na(s.change.after),
                        signed(s.change.delta),
                        s.source
                    ));
                }
            }
        }
        for (verb, list) in [
            ("added", &p.model_files.added),
            ("removed", &p.model_files.removed),
        ] {
            for f in list {
                let sources: Vec<String> = f
                    .sources
                    .iter()
                    .filter_map(|e| serde_json::to_value(e).ok()?.as_str().map(String::from))
                    .collect();
                detail.push(format!(
                    "  model file {verb}: {} [{}]",
                    f.path,
                    sources.join(", ")
                ));
            }
        }
        if !detail.is_empty() {
            out.push(String::new());
            out.push(format!("{} ({})", p.name, pid_label(p)));
            out.extend(detail);
        }
    }
    if procs
        .iter()
        .any(|p| !p.model_files.added.is_empty() || !p.model_files.removed.is_empty())
    {
        out.push(String::new());
        out.push(
            "Model files show what a process opened or named, not what is resident in GPU memory."
                .into(),
        );
    }
    if procs
        .iter()
        .any(|p| matches!(p.matched, Matched::New | Matched::Gone))
    {
        out.push(
            "'not listed' means the process was absent from that snapshot: not running, or using too \
             little GPU to be listed."
                .into(),
        );
    }
    out.join("\n")
}
