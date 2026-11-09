//! Exact buffer sizes from llama.cpp and Ollama load logs.
use super::MIB;
use crate::env::Paths;
use regex::Regex;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::UNIX_EPOCH;

const TAIL: u64 = 512 << 10;

fn llama_buf() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
            .unwrap_or_else(|_| unreachable!("static pattern"))
    })
}

fn ollama_buf() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r#"msg="(model weights|kv cache|compute graph)"\s+device=(\S+)\s+size="([\d.]+)\s*(GiB|MiB|KiB|B)""#)
            .unwrap_or_else(|_| unreachable!("static pattern"))
    })
}

fn unit(u: &str) -> u64 {
    match u {
        "GiB" => 1 << 30,
        "MiB" => 1 << 20,
        "KiB" => 1 << 10,
        _ => 1,
    }
}

fn kind(k: &str) -> &'static str {
    match k {
        "model" | "model weights" => "weights",
        "KV" | "kv cache" | "RS" => "kv",
        _ => "compute",
    }
}

pub fn is_gpu_device(dev: &str) -> bool {
    let d = dev.to_uppercase();
    !(d.starts_with("CPU") || d.contains("HOST") || d.starts_with("MMAP"))
}

fn is_weights_line(l: &str) -> bool {
    l.contains("model buffer size") || l.contains("msg=\"model weights\"")
}

/// Most recent model-load block in a llama.cpp or Ollama log, as `{device: {kind: bytes}}`.
/// `not_before` is seconds since the epoch; an older file yields `None`.
pub fn parse_load_log(
    path: &Path,
    not_before: f64,
) -> Option<BTreeMap<String, BTreeMap<String, u64>>> {
    let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
    let secs = mtime
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    if secs < not_before {
        return None;
    }
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.seek(SeekFrom::End(0)).ok()?;
    f.seek(SeekFrom::Start(len.saturating_sub(TAIL))).ok()?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw).ok()?;
    let text = String::from_utf8_lossy(&raw);
    let lines: Vec<&str> = text.lines().collect();
    let mut start = lines.iter().rposition(|l| is_weights_line(l))?;
    while start > 0 && is_weights_line(lines[start - 1]) {
        start -= 1;
    }
    let mut out: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for line in &lines[start..] {
        let (dev, k, n) = if let Some(m) = llama_buf().captures(line) {
            let Ok(f) = m[3].parse::<f64>() else { continue };
            (
                m[1].trim_end_matches(':').to_string(),
                kind(&m[2]),
                (f * MIB as f64) as u64,
            )
        } else if let Some(m) = ollama_buf().captures(line) {
            let Ok(f) = m[3].parse::<f64>() else { continue };
            (
                m[2].to_string(),
                kind(&m[1]),
                (f * unit(&m[4]) as f64) as u64,
            )
        } else {
            continue;
        };
        *out.entry(dev)
            .or_default()
            .entry(k.to_string())
            .or_insert(0) += n;
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Linux service installs log to the systemd journal; a user-run `ollama serve` may write here.
pub fn ollama_log_path(paths: &Paths) -> PathBuf {
    paths.home.join(".ollama").join("logs").join("server.log")
}
