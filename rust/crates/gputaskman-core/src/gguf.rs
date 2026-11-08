//! GGUF header reader: what model is this file, and what will it cost in VRAM?
//!
//! Only the header and tensor table are read (a few MB at most), never the weights.
//! Tensor byte sizes come from the tensor offset table, so new quant types need no lookup table.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const ARRAY_INLINE_LIMIT: u64 = 4096;
const MAX_DEPTH: u32 = 16;
const CACHE_ENTRIES: usize = 64;

/// A GGUF metadata value. Unsigned integers map to `U`, signed to `I`, floats to `F`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Integer value, accepting either signedness. Negative values are not valid sizes.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U(n) => Some(*n),
            MetaValue::I(n) => u64::try_from(*n).ok(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[must_use]
pub fn file_type_name(ft: u64) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufInfo {
    pub path: String,
    pub file_size: u64,
    pub version: u32,
    pub metadata: BTreeMap<String, MetaValue>,
    /// Block index to bytes.
    pub layer_bytes: BTreeMap<u32, u64>,
    /// `output.*` tensors, plus a copy of the token embeddings when they are tied.
    pub output_bytes: u64,
    pub token_embd_bytes: u64,
    pub tensor_bytes: u64,
    /// `None` when the tensor table could not be read.
    pub n_params: Option<u64>,
    pub error: Option<String>,
}

impl GgufInfo {
    /// Architecture-prefixed metadata lookup, for example `block_count` reads `llama.block_count`.
    #[must_use]
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        let arch = self.metadata_str("general.architecture").unwrap_or("");
        self.metadata.get(&format!("{arch}.{key}"))
    }

    fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(MetaValue::as_str)
    }

    #[must_use]
    pub fn arch(&self) -> &str {
        self.metadata_str("general.architecture").unwrap_or("?")
    }

    /// What the file is, from header keys only (most specific first).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        let t = self.metadata_str("general.type").unwrap_or("model");
        if self.arch() == "clip" || t == "mmproj" {
            return "vision projector";
        }
        if t == "adapter" {
            return "LoRA adapter";
        }
        if self.decision_type().is_some_and(|d| !d.is_empty()) {
            return "decision model";
        }
        if let Some(pooling) = self.meta("pooling_type") {
            // 4 = LLAMA_POOLING_TYPE_RANK
            return if pooling.as_u64() == Some(4) {
                "reranker"
            } else {
                "embedding model"
            };
        }
        if self.metadata.keys().any(|k| k.starts_with("stt.")) {
            return "speech model";
        }
        if self.n_layers() != 0 && self.ctx_train().is_some_and(|c| c != 0) && self.causal() {
            return "LLM";
        }
        "model"
    }

    #[must_use]
    pub fn decision_type(&self) -> Option<&str> {
        self.meta("decision.type").and_then(MetaValue::as_str)
    }

    #[must_use]
    pub fn causal(&self) -> bool {
        !matches!(self.meta("attention.causal"), Some(MetaValue::Bool(false)))
    }

    #[must_use]
    pub fn name(&self) -> String {
        match self.metadata_str("general.name") {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => Path::new(&self.path)
                .file_name()
                .map_or_else(|| self.path.clone(), |n| n.to_string_lossy().into_owned()),
        }
    }

    #[must_use]
    pub fn quant(&self) -> String {
        match self.metadata.get("general.file_type") {
            None => "?".into(),
            Some(v) => match v.as_u64() {
                Some(ft) => file_type_name(ft).map_or_else(|| format!("type{ft}"), String::from),
                None => "?".into(),
            },
        }
    }

    #[must_use]
    pub fn n_layers(&self) -> u32 {
        self.meta("block_count")
            .and_then(MetaValue::as_u64)
            .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
    }

    #[must_use]
    pub fn ctx_train(&self) -> Option<u64> {
        self.meta("context_length").and_then(MetaValue::as_u64)
    }

    #[must_use]
    pub fn sliding_window(&self) -> Option<u64> {
        self.meta("attention.sliding_window")
            .and_then(MetaValue::as_u64)
    }

    #[must_use]
    pub fn params_label(&self) -> String {
        match self.n_params {
            None => "?".into(),
            Some(n) if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
            Some(n) if n >= 1_000_000 => format!("{:.0}M", n as f64 / 1e6),
            Some(n) => n.to_string(),
        }
    }

    /// Bytes llama.cpp would place on the GPU. Input embeddings always stay on the CPU.
    #[must_use]
    pub fn gpu_weight_bytes(&self, n_gpu_layers: Option<i64>) -> u64 {
        if self.layer_bytes.is_empty() {
            // Non-transformer (for example CLIP or whisper): assume everything goes up.
            return self.tensor_bytes;
        }
        let declared = i64::from(self.n_layers());
        let l = if declared != 0 {
            declared
        } else {
            self.layer_bytes.len() as i64
        };
        let ngl = match n_gpu_layers {
            Some(n) if n >= 0 => n,
            _ => l + 1,
        };
        let first_gpu = (l - ngl).max(0);
        let mut total: u64 = self
            .layer_bytes
            .iter()
            .filter(|(i, _)| i64::from(**i) >= first_gpu)
            .map(|(_, b)| *b)
            .sum();
        if ngl > l {
            total = total.saturating_add(self.output_bytes);
        }
        total
    }

    /// Per-token KV cache cost (K+V, all layers) at the given element size (f16 = 2).
    #[must_use]
    pub fn kv_bytes_per_token(&self, type_bytes: f64) -> Option<u64> {
        let layers = self.n_layers() as usize;
        let n_head = self.meta("attention.head_count");
        let n_kv = self.meta("attention.head_count_kv").or(n_head);
        let n_embd = self.meta("embedding_length").and_then(MetaValue::as_u64);
        let n_kv = n_kv?;
        if layers == 0 {
            return None;
        }
        let n_head = match n_head {
            Some(MetaValue::Array(a)) => a.iter().filter_map(MetaValue::as_u64).max(),
            Some(v) => v.as_u64(),
            None => None,
        };
        let head_dim = match (n_embd, n_head) {
            (Some(e), Some(h)) if e != 0 && h != 0 => Some(e / h),
            _ => None,
        };
        let k_len = self
            .meta("attention.key_length")
            .and_then(MetaValue::as_u64)
            .or(head_dim)?;
        let v_len = self
            .meta("attention.value_length")
            .and_then(MetaValue::as_u64)
            .or(head_dim)?;
        if k_len == 0 || v_len == 0 {
            return None;
        }
        let per_token = k_len + v_len;
        let elems: u64 = match n_kv {
            MetaValue::Array(a) => a
                .iter()
                .take(layers)
                .map(|h| h.as_u64().unwrap_or(0).saturating_mul(per_token))
                .fold(0u64, u64::saturating_add),
            v => v
                .as_u64()?
                .saturating_mul(per_token)
                .saturating_mul(layers as u64),
        };
        Some((elems as f64 * type_bytes) as u64)
    }
}

#[must_use]
pub fn is_gguf(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
        && &magic == b"GGUF"
}

struct Reader<R> {
    r: R,
    version: u32,
    pos: u64,
    size: u64,
}

type Res<T> = Result<T, String>;

impl<R: Read + Seek> Reader<R> {
    fn remaining(&self) -> u64 {
        self.size.saturating_sub(self.pos)
    }

    fn bytes<const N: usize>(&mut self) -> Res<[u8; N]> {
        let mut buf = [0u8; N];
        self.r
            .read_exact(&mut buf)
            .map_err(|_| "truncated GGUF header".to_string())?;
        self.pos += N as u64;
        Ok(buf)
    }

    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn length(&mut self) -> Res<u64> {
        if self.version >= 2 {
            self.u64()
        } else {
            self.u32().map(u64::from)
        }
    }

    /// A length that must fit in what is left of the file, so corrupt headers cannot
    /// trigger large allocations.
    fn bounded(&mut self, bytes_per_item: u64) -> Res<u64> {
        let n = self.length()?;
        match n.checked_mul(bytes_per_item) {
            Some(total) if total <= self.remaining() => Ok(n),
            _ => Err("truncated GGUF header".into()),
        }
    }

    fn skip(&mut self, n: u64) -> Res<()> {
        let off = i64::try_from(n).map_err(|_| "truncated GGUF header".to_string())?;
        self.r
            .seek(SeekFrom::Current(off))
            .map_err(|e| e.to_string())?;
        self.pos += n;
        Ok(())
    }

    fn string(&mut self) -> Res<String> {
        let n = self.bounded(1)?;
        let mut buf = vec![0u8; n as usize];
        self.r
            .read_exact(&mut buf)
            .map_err(|_| "truncated GGUF header".to_string())?;
        self.pos += n;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn skip_string(&mut self) -> Res<()> {
        let n = self.bounded(1)?;
        self.skip(n)
    }

    fn scalar(&mut self, vtype: u32) -> Res<Option<MetaValue>> {
        Ok(Some(match vtype {
            0 => MetaValue::U(u64::from(self.bytes::<1>()?[0])),
            1 => MetaValue::I(i64::from(i8::from_le_bytes(self.bytes()?))),
            2 => MetaValue::U(u64::from(u16::from_le_bytes(self.bytes()?))),
            3 => MetaValue::I(i64::from(i16::from_le_bytes(self.bytes()?))),
            4 => MetaValue::U(u64::from(self.u32()?)),
            5 => MetaValue::I(i64::from(i32::from_le_bytes(self.bytes()?))),
            6 => MetaValue::F(f64::from(f32::from_le_bytes(self.bytes()?))),
            7 => MetaValue::Bool(self.bytes::<1>()?[0] != 0),
            10 => MetaValue::U(self.u64()?),
            11 => MetaValue::I(i64::from_le_bytes(self.bytes()?)),
            12 => MetaValue::F(f64::from_le_bytes(self.bytes()?)),
            _ => return Ok(None),
        }))
    }

    fn value(&mut self, vtype: u32, depth: u32) -> Res<MetaValue> {
        if let Some(v) = self.scalar(vtype)? {
            return Ok(v);
        }
        match vtype {
            8 => self.string().map(MetaValue::Str),
            9 => {
                if depth >= MAX_DEPTH {
                    return Err("GGUF arrays nested too deeply".into());
                }
                let etype = self.u32()?;
                if let Some(size) = scalar_size(etype) {
                    let n = self.bounded(size)?;
                    if n <= ARRAY_INLINE_LIMIT {
                        let mut out = Vec::with_capacity(n as usize);
                        for _ in 0..n {
                            out.push(self.scalar(etype)?.ok_or("bad array type")?);
                        }
                        return Ok(MetaValue::Array(out));
                    }
                    self.skip(n * size)?;
                    return Ok(MetaValue::Str(format!("<array[{n}]>")));
                }
                if etype == 8 {
                    let n = self.bounded(if self.version >= 2 { 8 } else { 4 })?;
                    for _ in 0..n {
                        self.skip_string()?;
                    }
                    return Ok(MetaValue::Str(format!("<string[{n}]>")));
                }
                let n = self.bounded(1)?;
                for _ in 0..n {
                    self.value(etype, depth + 1)?;
                }
                Ok(MetaValue::Str(format!("<array[{n}]>")))
            }
            _ => Err(format!("unknown GGUF value type {vtype}")),
        }
    }
}

fn scalar_size(t: u32) -> Option<u64> {
    match t {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn block_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (digits, _) = rest.split_once('.')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn parse(path: &Path, size: u64) -> GgufInfo {
    let mut info = GgufInfo {
        path: path.to_string_lossy().into_owned(),
        file_size: size,
        version: 0,
        metadata: BTreeMap::new(),
        layer_bytes: BTreeMap::new(),
        output_bytes: 0,
        token_embd_bytes: 0,
        tensor_bytes: 0,
        n_params: None,
        error: None,
    };
    let Ok(file) = File::open(path) else {
        info.error = Some("cannot open file".into());
        return info;
    };
    let mut r = Reader {
        r: BufReader::with_capacity(1 << 20, file),
        version: 0,
        pos: 0,
        size,
    };
    match read_header(&mut r, &mut info) {
        Ok(tensors) => finish(&mut info, tensors, size),
        Err(e) => info.error = Some(e),
    }
    info
}

/// Returns the sorted tensor table as (offset, name, element count), plus the data start offset.
type Tensors = (Vec<(u64, String, u64)>, u64);

fn read_header<R: Read + Seek>(r: &mut Reader<R>, info: &mut GgufInfo) -> Res<Tensors> {
    if r.bytes::<4>()? != *b"GGUF" {
        return Err("not a GGUF file".into());
    }
    info.version = r.u32()?;
    r.version = info.version;
    let n_tensors = r.length()?;
    let n_kv = r.length()?;
    if n_tensors > r.remaining() || n_kv > r.remaining() {
        return Err("truncated GGUF header".into());
    }
    for _ in 0..n_kv {
        let key = r.string()?;
        let vtype = r.u32()?;
        let value = r.value(vtype, 0)?;
        info.metadata.insert(key, value);
    }
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if u64::from(n_dims) > r.remaining() {
            return Err("truncated GGUF header".into());
        }
        let mut n = 1u64;
        for _ in 0..n_dims {
            n = n.saturating_mul(r.length()?);
        }
        r.u32()?; // ggml type
        let offset = r.u64()?;
        tensors.push((offset, name, n));
    }
    let align = match info.metadata.get("general.alignment") {
        None => 32,
        Some(v) => v
            .as_u64()
            .filter(|a| *a != 0)
            .ok_or("invalid general.alignment")?,
    };
    let data_start = r.pos.div_ceil(align).saturating_mul(align);
    tensors.sort();
    Ok((tensors, data_start))
}

fn finish(info: &mut GgufInfo, (tensors, data_start): Tensors, size: u64) {
    let mut n_params = 0u64;
    let mut has_output_weight = false;
    for (i, (offset, name, n)) in tensors.iter().enumerate() {
        let end = tensors
            .get(i + 1)
            .map_or_else(|| size.saturating_sub(data_start), |t| t.0);
        let nbytes = end.saturating_sub(*offset);
        n_params = n_params.saturating_add(*n);
        info.tensor_bytes = info.tensor_bytes.saturating_add(nbytes);
        has_output_weight |= name == "output.weight";
        if let Some(idx) = block_index(name) {
            let e = info.layer_bytes.entry(idx).or_insert(0);
            *e = e.saturating_add(nbytes);
        } else if name.starts_with("token_embd.") {
            info.token_embd_bytes = info.token_embd_bytes.saturating_add(nbytes);
        } else if name.starts_with("output") {
            info.output_bytes = info.output_bytes.saturating_add(nbytes);
        }
    }
    info.n_params = Some(n_params);
    if !has_output_weight && !info.layer_bytes.is_empty() {
        // Tied embeddings: llama.cpp puts a copy on the GPU.
        info.output_bytes = info.output_bytes.saturating_add(info.token_embd_bytes);
    }
}

type CacheKey = (PathBuf, u64, SystemTime);

/// Reads GGUF headers, remembering the last few by (path, size, mtime).
#[derive(Default)]
pub struct GgufCache {
    entries: Vec<(CacheKey, GgufInfo)>,
}

impl GgufCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `None` only when the file cannot be inspected at all; a bad header gives `error`.
    pub fn read(&mut self, path: &Path) -> Option<GgufInfo> {
        let meta = std::fs::metadata(path).ok()?;
        let key = (
            path.to_path_buf(),
            meta.len(),
            meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        );
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            let entry = self.entries.remove(pos);
            let info = entry.1.clone();
            self.entries.push(entry);
            return Some(info);
        }
        let info = parse(path, meta.len());
        if self.entries.len() >= CACHE_ENTRIES {
            self.entries.remove(0);
        }
        self.entries.push((key, info.clone()));
        Some(info)
    }
}
