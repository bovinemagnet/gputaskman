//! Process classification (`procinfo.py` `classify`).
use crate::model::{Engines, ProcDetail};

pub const BROWSERS: &[&str] = &[
    "msedge.exe",
    "chrome.exe",
    "firefox.exe",
    "msedgewebview2.exe",
    "brave.exe",
    "opera.exe",
    "arc.exe",
    "vivaldi.exe",
    "chrome",
    "chromium",
    "chromium-browser",
    "firefox",
    "firefox-bin",
    "brave",
    "msedge",
    "opera",
    "vivaldi-bin",
    // macOS: browsers draw through a GPU helper process
    "google chrome helper (gpu)",
    "microsoft edge helper (gpu)",
    "brave browser helper (gpu)",
    "arc helper (gpu)",
    "chromium helper (gpu)",
    "com.apple.webkit.gpu",
    "safari",
    "firefox gpu helper",
];

/// Display servers and compositors: they own the desktop back buffers, like `dwm.exe` on Windows.
pub const COMPOSITORS: &[&str] = &[
    "dwm.exe",
    "xorg",
    "xwayland",
    "gnome-shell",
    "kwin_wayland",
    "kwin_x11",
    "sway",
    "hyprland",
    "mutter",
    "weston",
    "wayfire",
    "labwc",
    "niri",
    "cosmic-comp",
    "picom",
    "compiz",
    "windowserver",
];

pub const LLM_NAMES: &[&str] = &[
    "llama-",
    "llama.",
    "ollama",
    "lm studio",
    "lmstudio",
    "llmster",
    "koboldcpp",
    "jan",
    "gpt4all",
    "llamafile",
];

/// GGUF file kind to process kind, in display priority order.
pub const MODEL_KINDS: &[(&str, &str)] = &[
    ("LLM", "LLM"),
    ("decision model", "decision"),
    ("embedding model", "embedding"),
    ("reranker", "embedding"),
    ("speech model", "speech"),
];

pub const AI_KINDS: &[&str] = &[
    "LLM",
    "decision",
    "decision?",
    "embedding",
    "speech",
    "diffusion",
    "AI",
];

fn process_kind(model_kind: &str) -> Option<&'static str> {
    MODEL_KINDS
        .iter()
        .find(|(k, _)| *k == model_kind)
        .map(|(_, p)| *p)
}

/// Classify a process. PID 4 is an ordinary process on Linux, so there is no `kernel` branch.
/// Name-based hints only reach `decision?`; they never count as model evidence.
#[must_use]
pub fn classify(d: &ProcDetail, engines: &Engines) -> String {
    let n = d.name.to_lowercase();
    let n = n.as_str();
    if COMPOSITORS.contains(&n) {
        return "compositor".into();
    }
    if matches!(
        n,
        "csrss.exe" | "winlogon.exe" | "lsass.exe" | "svchost.exe"
    ) {
        return "system".into();
    }
    if d.is_diffusion() {
        return "diffusion".into();
    }
    let kinds: Vec<&str> = d
        .models
        .iter()
        .filter(|m| m.kind_hint.is_empty())
        .filter_map(|m| process_kind(&m.kind))
        .collect();
    for k in ["LLM", "decision", "embedding", "speech"] {
        if kinds.contains(&k) {
            return k.into();
        }
    }
    if d.models.iter().any(|m| m.kind_hint == "decision model") {
        return "decision?".into();
    }
    let has = |r: &str| d.runtimes.iter().any(|x| x == r);
    if has("llama.cpp") || LLM_NAMES.iter().any(|p| n.starts_with(p)) {
        return "LLM".into();
    }
    let video = engines.get("decode") + engines.get("encode") + engines.get("video") > 1.0;
    if d.is_ai() && !video {
        return "AI".into();
    }
    if BROWSERS.contains(&n) {
        return "browser".into();
    }
    if video || has("NVENC") {
        return "video".into();
    }
    if has("CUDA") || has("HIP") {
        return "compute".into();
    }
    if (has("D3D12") || has("Vulkan")) && engines.get("3d") > 10.0 {
        return "3D/game".into();
    }
    // macOS: Metal reports one GPU time, no 3D/compute split.
    if has("Metal") && engines.get("gpu") > 10.0 {
        return "3D/game".into();
    }
    "app".into()
}
