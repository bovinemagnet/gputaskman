//! The help popup: Python's `HELP_MD`, rendered as styled lines.
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Padding, Paragraph, Wrap};

use crate::app::App;
use crate::format::{ACCENT, CYAN, bold, fg};
use crate::view::scaled;

pub const HELP: &str = r##"# What can be known about GPU memory?

Press **m** (or click **Slim/Full** in the footer) to switch between full diagnostics
and slim monitoring. Slim hides hardware identifiers and secondary telemetry details;
live readings, charts, processes, inspection, and provenance/availability warnings stay visible.
This display-only choice applies to all GPUs and resets to full when the app restarts.

| Layer | Source | Works for |
|---|---|---|
| **Device** telemetry | NVML, AMD ADL, or optional AMD SMI; amdgpu sysfs on Linux | availability varies by GPU and driver |
| **Per-process** VRAM, spill to RAM, per-engine load | Windows WDDM counters (dxgkrnl); DRM fdinfo for amdgpu on Linux | workloads visible to WDDM or DRM |
| **Which runtime** (CUDA, HIP, Vulkan, D3D12, ggml, ONNX, PyTorch...) | DLLs mapped in the process | everything |
| **Which model file** | memory-mapped files + command line | llama.cpp, stable-diffusion.cpp, LM Studio, Ollama |
| **Model name & kind** (LLM, decision, embedding, speech) | GGUF header, Ollama/LM Studio/Hugging Face stores, local server APIs | GGUF family |
| **Weights / KV cache / compute** split | GGUF header math, llama.cpp/Ollama logs, local server APIs | GGUF/llama.cpp family |
| Per-texture / per-buffer names | needs a debugger (PIX, Nsight) attached to the app | games, 3D apps |

**Evidence** shows how a model was identified: `api` (the runtime says it is loaded), `mapped`
(Windows shows the file mapped into the process right now), `cmdline` (a launch argument only).
Two sources, including `api` or `mapped`, is shown as *confirmed*.

**Local workloads only:** inference, decision models, vision, speech, training, graphics and video.
Cloud-hosted models are outside this monitor's scope. DLLs identify runtimes, not models or tasks.
`decision?` / *likely decision model (name)* is a filename or model-folder hint when a specific
GGUF kind is missing. It does not confirm the kind, GPU residency, or execution. Installed
candidates remain unproven; engine activity is measured per process, not per model.

**Linux / WSL:** on native Linux, per-process VRAM and SM/encoder/decoder load come from NVML
(no spill or commit counters exist there, so those columns stay blank). **AMD GPUs on Linux**
(`amdgpu`) use sysfs for the device and DRM fdinfo (`/proc/<pid>/fdinfo`) per process: VRAM,
GTT (GPU memory in system RAM, shown as SPILL), and gfx/compute/video engine time (`n/a` for a new client's
first sample and for ROCm/HIP processes: their compute runs on `/dev/kfd` queues the kernel doesn't time per
process). VRAM and SPILL are charged to a process only when its DRM clients are provably private: one holder,
opened through a render node, and no shared buffers. Otherwise they show `n/a` and the bytes stay unattributed;
the inspector still shows the resident bytes it can reach (shared buffers included, so not added up). Only processes you can inspect are
listed, and the header says how many it couldn't; run with `sudo` to see other users'. Under **WSL**, Windows
charges all Linux GPU memory to the VM (`vmwp.exe`), so VRAM and load are device-wide only; rows
are Linux processes with `/dev/dxg` open and show `n/a` rather than a made-up share. When WSL
interop is enabled, the Windows host's counters are read through `powershell.exe`: **WSL VM**
(`vmwp.exe`) is then the exact total of Linux GPU memory, and Windows apps appear as `W:<pid>`
rows. `--no-host` turns this off. Linux rows get their VRAM by matching the VM's GPU contexts
(newest first) to Linux processes: `*` when candidates agree on the size, `?` for a heuristic owner.
Cached foreign sizes only help rank guesses; they never prove ownership across runs.

**Why does nvidia-smi say N/A?** GeForce cards on Windows run in *WDDM* mode: Windows, not the
NVIDIA driver, owns video memory, so NVML can't attribute it to processes. The Windows counters can.

**dwm.exe looks huge elsewhere?** Windows' per-process "Dedicated Usage" counts every surface a
process can *see*, so dwm can show nearly the whole card. VRAM here is what's charged to each
process ("Local Usage" when available). The adapter counter is the total, not the sum of rows.

**SPILL** = GPU-accessible system RAM. This can include pinned host buffers or intentional
allocations, not just overflow. Large usage under VRAM pressure can indicate fallback;
the counter alone does not prove it.

**PAGED** = max(Total Committed - Local Usage - Non Local Usage, 0). This is an estimate of
non-resident committed memory, not a measured eviction count or proof of pagefile activity.

**Inside** labels are inferred from runtimes and files. Log sizes describe allocated buffers,
not necessarily current residency. A game's textures cannot be enumerated here.

**GPU%** = busiest engine (like Task Manager). On GeForce/WDDM, CUDA kernels are usually reported
on the **3D** engine. AMD device GPU charts show graphics-engine activity, which is not
necessarily equivalent to NVML GPU utilization. Vulkan does not expose another process's HIP internals.

Keys: `↑/↓` select · `s` sort · `c` chart · `a` show all · `p` pause · `?` this help · `q` quit

All eight trend charts are visible by default. They stack in narrow terminals and
share rows when there is room. Click the **Charts** dropdown to show all charts or
one metric; `c` cycles the same choices. A blank sample means no reading; zero is
drawn at the baseline. Percentage charts use 0–100%, temperature uses 0–120°C,
and the power and SM clock scales are shown below the charts. In short terminals,
scroll the dashboard or inspector without losing the process list.
"##;

/// One markdown line: `**bold**`, `*italic*` and `` `code` `` become styled spans.
fn inline(text: &str) -> Line<'static> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    let (mut strong, mut em, mut code) = (false, false, false);
    let style = |strong: bool, em: bool, code: bool| {
        let mut s = if code { fg(CYAN) } else { Style::new() };
        if strong {
            s = s.add_modifier(Modifier::BOLD);
        }
        if em {
            s = s.add_modifier(Modifier::ITALIC);
        }
        s
    };
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        let marker = if code {
            (c == '`').then_some(1)
        } else if rest.starts_with("**") {
            Some(2)
        } else if c == '`' || c == '*' {
            Some(1)
        } else {
            None
        };
        match marker {
            Some(len) => {
                if !buf.is_empty() {
                    spans.push(Span::styled(
                        std::mem::take(&mut buf),
                        style(strong, em, code),
                    ));
                }
                match (len, c) {
                    (2, _) => strong = !strong,
                    (_, '`') => code = !code,
                    _ => em = !em,
                }
                rest = &rest[len..];
            }
            None => {
                buf.push(c);
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, style(strong, em, code)));
    }
    Line::from(spans)
}

fn table_lines(rows: &[Vec<String>]) -> Vec<Line<'static>> {
    let clean = |s: &str| s.replace("**", "").replace('`', "");
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|r| r.iter().map(|c| clean(c)).collect())
        .collect();
    let columns = cells.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..columns)
        .map(|i| {
            cells
                .iter()
                .filter_map(|r| r.get(i))
                .map(|c| c.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    cells
        .iter()
        .enumerate()
        .map(|(n, r)| {
            let text = r
                .iter()
                .zip(&widths)
                .map(|(c, w)| format!("{c:<w$}"))
                .collect::<Vec<_>>()
                .join("  ");
            if n == 0 {
                Line::styled(text, bold())
            } else {
                Line::raw(text)
            }
        })
        .collect()
}

pub fn help_lines() -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut table: Vec<Vec<String>> = Vec::new();
    for raw in HELP.lines() {
        if raw.starts_with('|') {
            if !raw.contains("---") {
                table.push(
                    raw.trim_matches('|')
                        .split('|')
                        .map(|c| c.trim().to_string())
                        .collect(),
                );
            }
            continue;
        }
        if !table.is_empty() {
            out.extend(table_lines(&table));
            table.clear();
        }
        match raw.strip_prefix("# ") {
            Some(h) => out.push(Line::styled(
                h.to_string(),
                bold().add_modifier(Modifier::UNDERLINED),
            )),
            None => out.push(inline(raw)),
        }
    }
    if !table.is_empty() {
        out.extend(table_lines(&table));
    }
    out
}

/// Draws the popup centred in `area`; returns how far it can scroll.
pub fn draw(_app: &App, f: &mut Frame, area: Rect, scroll: u16) -> u16 {
    let width = scaled(area.width, 95, 100).min(140);
    let height = scaled(area.height, 85, 100);
    let popup = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    let block = Block::bordered()
        .border_type(BorderType::Thick)
        .border_style(fg(ACCENT))
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(popup);
    let text = Paragraph::new(help_lines()).wrap(Wrap { trim: false });
    let total = u16::try_from(text.line_count(inner.width)).unwrap_or(u16::MAX);
    let max = total.saturating_sub(inner.height);
    f.render_widget(Clear, popup);
    f.render_widget(text.scroll((scroll.min(max), 0)).block(block), popup);
    max
}
