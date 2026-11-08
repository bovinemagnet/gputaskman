//! Small, read-only Handy settings and model-event probe (Linux paths).
//!
//! The selected model comes from app settings and means *selected*, not loaded. Log events are
//! timestamped history, not current residency. Only lines matching the timestamp, model-event and
//! unload patterns are inspected; nothing else from the log is stored.
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::UNIX_EPOCH;

use chrono::{DateTime, Local, NaiveDateTime, SecondsFormat};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::env::Paths;

const MAX_SETTINGS_BYTES: u64 = 256 << 10;
const MAX_LOG_BYTES: u64 = 512 << 10;

static TIMESTAMP: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"^\[(\d{4}-\d{2}-\d{2})\]\[(\d{2}:\d{2}:\d{2})\]").ok());
static MODEL_EVENT: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(
        r"(?:Loaded whisper model |Live preview: model |Live streaming transcription started \(model )'([^']+)'",
    )
    .ok()
});
static UNLOAD_EVENT: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(?i)Model unloaded due to inactivity|Model idle for \d+s.*unloading").ok()
});

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandyEvidence {
    pub selected_model: Option<String>,
    pub last_model: Option<String>,
    pub last_event: Option<String>,
    /// UTC ISO 8601, as Handy logs it.
    pub last_event_at: Option<String>,
}

impl HandyEvidence {
    /// The last event time in the local time zone, the raw string if it does not parse, or empty.
    pub fn last_event_local(&self) -> String {
        let Some(at) = self.last_event_at.as_deref().filter(|s| !s.is_empty()) else {
            return String::new();
        };
        match DateTime::parse_from_rfc3339(at) {
            Ok(t) => t
                .with_timezone(&Local)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string(),
            Err(_) => at.to_string(),
        }
    }
}

type Signature = (i128, u64);
type Event = (String, String, String);

fn signature(path: &Path) -> Option<Signature> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    let nanos = match mtime.duration_since(UNIX_EPOCH) {
        Ok(d) => i128::try_from(d.as_nanos()).ok()?,
        Err(e) => -i128::try_from(e.duration().as_nanos()).ok()?,
    };
    Some((nanos, meta.len()))
}

pub struct HandyProbe {
    settings_path: PathBuf,
    log_path: PathBuf,
    settings_cache: Option<(Signature, Option<String>)>,
    log_cache: HashMap<u64, (Signature, Option<Event>)>,
}

impl HandyProbe {
    pub fn new(paths: Paths) -> Self {
        let base = paths.xdg_data.join("com.pais.handy");
        Self {
            settings_path: base.join("settings_store.json"),
            log_path: base.join("logs").join("handy.log"),
            settings_cache: None,
            log_cache: HashMap::new(),
        }
    }

    /// Selected model plus the latest model event after the process started (wall seconds).
    pub fn inspect(&mut self, process_started: f64) -> Option<HandyEvidence> {
        let selected = self.selected_model();
        let event = self.last_model_event(process_started);
        if selected.is_none() && event.is_none() {
            return None;
        }
        Some(HandyEvidence {
            selected_model: selected,
            last_model: event
                .as_ref()
                .and_then(|e| Some(e.1.clone()).filter(|m| !m.is_empty())),
            last_event: event.as_ref().map(|e| e.0.clone()),
            last_event_at: event.map(|e| e.2),
        })
    }

    fn selected_model(&mut self) -> Option<String> {
        let sig = signature(&self.settings_path)?;
        if let Some((cached, value)) = &self.settings_cache
            && *cached == sig
        {
            return value.clone();
        }
        let value = if sig.1 <= MAX_SETTINGS_BYTES {
            read_selected(&self.settings_path)
        } else {
            None
        };
        self.settings_cache = Some((sig, value.clone()));
        value
    }

    fn last_model_event(&mut self, process_started: f64) -> Option<Event> {
        let sig = signature(&self.log_path)?;
        let key = process_started.to_bits();
        if let Some((cached, value)) = self.log_cache.get(&key)
            && *cached == sig
        {
            return value.clone();
        }
        let latest = scan_log(&read_tail(&self.log_path), process_started);
        self.log_cache.insert(key, (sig, latest.clone()));
        latest
    }
}

fn read_selected(path: &Path) -> Option<String> {
    let mut buf = Vec::new();
    File::open(path)
        .ok()?
        .take(MAX_SETTINGS_BYTES + 1)
        .read_to_end(&mut buf)
        .ok()?;
    let data: serde_json::Value = serde_json::from_slice(&buf).ok()?;
    let selected = data.get("settings")?.get("selected_model")?.as_str()?;
    Some(selected.to_string()).filter(|s| !s.is_empty())
}

fn read_tail(path: &Path) -> String {
    let read = || -> std::io::Result<Vec<u8>> {
        let mut f = File::open(path)?;
        let end = f.seek(SeekFrom::End(0))?;
        f.seek(SeekFrom::Start(end.saturating_sub(MAX_LOG_BYTES)))?;
        let mut buf = Vec::new();
        f.take(MAX_LOG_BYTES).read_to_end(&mut buf)?;
        Ok(buf)
    };
    read()
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default()
}

fn scan_log(text: &str, process_started: f64) -> Option<Event> {
    let (ts_re, model_re, unload_re) = (
        TIMESTAMP.as_ref()?,
        MODEL_EVENT.as_ref()?,
        UNLOAD_EVENT.as_ref()?,
    );
    let mut latest: Option<Event> = None;
    for line in text.lines() {
        let Some(m) = ts_re.captures(line) else {
            continue;
        };
        let stamp = format!("{} {}", &m[1], &m[2]);
        let Ok(naive) = NaiveDateTime::parse_from_str(&stamp, "%Y-%m-%d %H:%M:%S") else {
            continue;
        };
        let occurred = naive.and_utc();
        if process_started != 0.0 && (occurred.timestamp() as f64) < process_started - 2.0 {
            continue;
        }
        let iso = occurred.to_rfc3339_opts(SecondsFormat::Secs, false);
        if let Some(mm) = model_re.captures(line) {
            let event = if line.contains("Loaded whisper model ") {
                "loaded"
            } else if line.contains("Live streaming transcription started ") {
                "streaming started"
            } else {
                "preview"
            };
            latest = Some((event.to_string(), mm[1].to_string(), iso));
        } else if unload_re.is_match(line) {
            let event = if line.to_lowercase().contains("unloaded") {
                "unloaded"
            } else {
                "unloading"
            };
            let model = latest.as_ref().map(|l| l.1.clone()).unwrap_or_default();
            latest = Some((event.to_string(), model, iso));
        }
    }
    latest
}
