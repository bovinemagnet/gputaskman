# Rust port milestone 2: the interactive TUI

Status: approved design, 2026-11-09. Builds on milestone 1
(`docs/superpowers/specs/2026-11-09-rust-linux-port-design.md`), whose rules and schema still bind.

## Goal

`gputaskman` with no `--once`/`--json` opens a live terminal UI with the same behaviour as the
Python Textual UI (`src/gputaskman/app.py`, v0.3.4) on native Linux: device header with a
kind-coloured VRAM bar, eight trend charts, the process table, the inspector with the LLM
breakdown, Slim/Full monitoring, help, multi-GPU, keyboard and mouse.

Out of scope: WDDM, WSL and Metal rendering semantics (paged column values, WSL VM row ordering,
Metal labels) arrive with those backends' milestones; Textual's `^p` command palette.

## Global constraints

- Toolchain 1.97.0 (pinned in `rust/rust-toolchain.toml`); edition 2024; workspace lints
  (`unwrap_used` and `expect_used` denied) apply to the new crate.
- New dependencies, exact versions: `ratatui = "0.30.2"`, `crossterm = "0.29.0"`; dev-only
  `insta = "1.49.0"`. No other new crates.
- The TUI crate does not depend on `gputaskman-linux`; it builds and tests on Ubuntu, Windows and
  macOS CI.
- The AGENTS.md provenance rules bind every screen element (see "Provenance").
- The UI never reads `/proc`, `/sys`, NVML or the network itself; everything it shows comes from
  `Snapshot` (and the `LlmView` inside its rows).
- Python stays unchanged as the reference.
- Documentation uses British spelling. Author Paul Snow; version 0.0.0.

## Architecture

New library crate `rust/crates/gputaskman-tui`, used by the `gputaskman` binary.

```
gputaskman (bin) ── no --once/--json ──► gputaskman_tui::run(collectors, Config { interval })
                                              │
   sampler thread (owns Vec<Collector>)       │ main thread
   wait 0.5 s, then every interval:           │  loop:
     unless paused: sample each ──── mpsc ───►│    snapshots → app.apply_all(Vec<Snapshot>)
     stop on request, then exit               │    crossterm event (key/mouse/resize) → Action → app.update
                                              │    redraw when dirty: view::draw(&App, &mut Frame)
```

### Modules

| Module | Responsibility |
|---|---|
| `lib.rs` | `pub fn run(collectors: Vec<Collector>, config: Config) -> Result<(), TuiError>`; the event loop |
| `app.rs` | `App` state: latest snapshots, per-GPU chart histories, selected row key, sort index, show-all, paused, slim, chart mode, open popup, scroll offsets, auto-selected flag, the screen areas from the last draw (for mouse hit-testing); `apply_all` |
| `action.rs` | `Action` enum; `fn from_key(KeyEvent, &App) -> Option<Action>`, `fn from_mouse(MouseEvent, &App) -> Option<Action>`; `App::update(Action)` |
| `rows.rs` | the five sorts, the "top" filter, multi-GPU row merge, LLM auto-select |
| `history.rs` | `History`: eight ring buffers of `Option<f64>`, capacity 60; chart series, scales and labels |
| `format.rs` | the one helper that turns a `Bytes`/`Option<f64>` into a styled cell (see "Provenance"); colour constants |
| `widgets.rs` | `spark()` and `stacked_bar()` (pure, unit-tested) |
| `view/` | `header`, `charts`, `chart_menu`, `table`, `inspector`, `help`, `footer`; each draws from `&App` and mutates nothing but the recorded hit areas |
| `sampler.rs` | `Sampler::spawn(collectors, interval, paused: Arc<AtomicBool>) -> (Sampler, Receiver<Vec<Snapshot>>)`; `Drop` signals stop and joins |
| `terminal.rs` | raw mode, alternate screen and mouse capture behind a guard; a panic hook restores the terminal before the default hook prints |

### Core changes (`gputaskman-core`)

- `ScanPolicy::Budgeted` becomes `ScanPolicy::Budgeted(Duration)`; the collector uses the given
  budget. One-shot output keeps `Unbounded`.
- `report::metric` becomes `pub` (the Python `format_metric` equivalent), unchanged in behaviour.
- `Collector` must be `Send` (it moves to the sampler thread); a compile-time assertion test
  guards this.

### Binary changes (`gputaskman`)

- Without `--once`/`--json`: if stdout is not a terminal, print
  `the TUI needs an interactive terminal; use --once or --json` to stderr and exit 2.
- With `--gpu N`: one collector for adapter N, budget `SCAN_BUDGET`.
- Without `--gpu`: one collector per adapter from `list_adapters`, each built from its own
  `linux_backends_at` call, each with budget `SCAN_BUDGET / n`; with no adapters, one collector
  for index 0 (it reports the no-GPU notes).
- A `TuiError` prints to stderr and exits 1.

## Screen

Top to bottom, as in `app.py`.

### Dashboard (height ≤ 50%, scrolls when taller)

Device box, border title `Full diagnostics` or `Slim monitoring`; one block per GPU, headed
`GPU <i>` when there are several.

- Line 1: adapter name (bold); in full mode `   driver <d>  CUDA <c>  <driver_model>  <pci_bus>`
  (each part only when known); `   [PAUSED]` in bold red when paused.
- VRAM bar: `VRAM ` + stacked bar of width `clamp(width − 40, 20, 90)` + ` <used> / <total>`.
  Segments: per kind, the sum of **measured** dedicated bytes of that kind's rows, largest first,
  scaled by `min(1, used / attributed)` when the device used total is known; then
  `max(used − attributed × scale, 0)` as unattributed (grey35); the rest `░`. No segments when the
  used total is unknown.
- Legend: up to seven `■ <kind> <size>` entries, `■ unattributed <size>` when over 32 MiB,
  `free <size>` when used is known, `shared(RAM) <size>` in yellow when over 64 MiB,
  `[bar normalized; labels are raw]` when the scale is below 1; then each snapshot note,
  the counter status and `DRM process scan: <status>.` on their own lines.
- Metrics line (when device telemetry exists): `GPU <util>%` (`AMD gfx` for an AMD source),
  `Mem activity <util_mem>%`, `PCIe↓<rx> ↑<tx> MB/s`, `enc/dec` when either is non-zero.
- Status line: full mode `<temp>°C  <power>/<limit> W <kind>  P<state>  <clock>  mem <m> MHz
  video <v> MHz  PCIe gen<g> x<w>  CC <cc>` (`<source>` instead of CC when not NVML); slim mode
  `<temp>°C  <power> W <kind>  SM|gfx <clock> MHz`; `  throttle: a,b` in yellow when throttled.
  Unknown values print as `N/A` via `report::metric`.

Charts box:

- Header: `Charts` (bold), the selector `[<mode> ▾]`, `Click to choose · c cycles` (muted).
- Grid: the selected chart, or all eight in order GPU, SM clock (labelled `GFX clock` for AMD,
  `SM/GFX clock` with several GPUs), VRAM, Mem activity, Temperature, Power, Encoder, Decoder.
  Columns `min(count, 3, max(1, (width + 2) / 46))`; cell `<name:12> <current:>8> <spark>`, spark
  width `cell_width − 22`.
- Scales: percentage charts 0–100, temperature 0–120, power `max(power_limit, max seen, 1)`, clock
  `max(sm_clock_max, max seen, 1)`. Blank sample = no reading; zero at the baseline.
- Note: `Recent samples · blank = no reading · power 0–<p> W · SM|GFX clock 0–<c> MHz`; one note
  per GPU with several.
- History per GPU index: util, clock (gfx clock for AMD), mem activity, VRAM %
  (`used / total × 100` when both known), temperature, power, encoder, decoder. A snapshot without
  telemetry appends `None` to all but VRAM.

### Process table (fills the rest, at least 4 rows)

Columns `PID Process Kind Runtime VRAM Spill Paged GPU% 3D Copy Video What's inside`, plus `GPU`
with several GPUs. Zebra stripes. Bottom border: `sort: <name>  ·  all|top processes`.

- Top filter (when not show-all): a row stays if its VRAM is not measured, or VRAM ≥ 32 MiB, or
  GPU% ≥ 1, or spill ≥ 64 MiB, or paged ≥ 128 MiB, or it has an LLM view.
- Sorts, cycled by `s`: VRAM (shown VRAM, descending, unknown as 0), GPU%, spill, paged, name
  (case-insensitive). Stable. With several GPUs, rows from all snapshots merge and re-sort.
- Cells: PID right-aligned (grey for host rows); kind coloured by `KIND_STYLE`; runtime is the first
  three runtimes (grey70); VRAM via `report::vram_label`, bold when measured, grey otherwise;
  Spill/Paged/engine cells via the provenance helper; spill bold red when ≥ 256 MiB on an LLM/AI
  row and `shared_is_spill`, else yellow; percentages blank below 0.5, bold at 20 or more;
  3D = 3d + compute, Video = decode + encode + video; What's inside magenta for LLM rows, grey62
  otherwise, truncated with an ellipsis.

### Inspector (height ≤ 30%, scrolls)

Border title `<name> (<pid>) - <kind>` (`host PID` for outside rows; ` - GPU <i>` with several).
With no selection: `Select a process (↑/↓) to see what's inside its GPU memory.  Press ? to learn
what's knowable.` in grey.

Content follows `render_inspector` line for line, restricted to the Linux branches: the
VRAM/spill/paged/engines/runtime header; inferred-owner note; DRM private-client note; resident
VRAM/GTT with `DRM_RESIDENT_NOTE`; the kfd engines note; Handy selected model and last event; for
an LLM row the multi-GPU caveat, the segment bar (width `clamp(width − 8, 20, 100)`) and legend
(exact segments bold, log source in green), the model table, other files, the server line and
the notes (`!`-prefixed in bold red); otherwise the kind explanation, model files and installed
candidates; then listening ports (first 8) and the command line (first 400 characters) or the
protected-process line.

### Footer, help and chart menu

- Footer: `q Quit  s Sort  c Chart  a All/Top  m Slim/Full  p Pause  ? Explain`; each hint is
  clickable.
- Help: centred popup (width ≤ 140, ≤ 95% of the screen; height 85%), scrollable, Python's
  `HELP_MD` rendered with headings, bold, inline code and the table laid out as columns. `Esc`,
  `q` or `?` closes it.
- Chart menu: a list popup under the selector: `All charts`, then each chart. `↑/↓` and `Enter`,
  or a click, choose; `Esc` or a click outside closes it.

### Keys and mouse

| Input | Action |
|---|---|
| `q`, `Ctrl+C` | quit |
| `s` | next sort |
| `c` | next chart mode (All, then each chart) |
| `a` | toggle all/top |
| `m` | toggle slim/full |
| `p` | toggle pause (the sampler stops sampling; the UI keeps responding) |
| `?` | help |
| `↑ ↓ PgUp PgDn Home End` | move the table selection |
| click a row | select it |
| wheel over table / inspector / dashboard | scroll that pane |
| click a footer hint / the chart selector | run it / open the chart menu |

### Behaviour carried over

- The first LLM row by VRAM sort is selected once, on the first snapshot that has one.
- The selection is held by `ProcRow::key()`; with several GPUs it is prefixed `<i>:`, and the
  prefix is added or dropped when the GPU count changes. When the selected process disappears,
  the selection is cleared and the inspector shows its prompt (Python's behaviour) until the user
  moves the cursor, which then starts from the row at the same position (clamped).
- Resize redraws immediately; layouts reflow while paused.
- Below 40×10 the screen shows `terminal too small` instead of the layout.

### Colours

`KIND_STYLE`, `SEG_STYLE` and the greys map Rich names to xterm-256 indices (for example
`magenta` 5, `orchid` 170, `plum2` 183, `light_sky_blue1` 153, `medium_purple1` 141, `orange3` 172,
`green3` 40, `dodger_blue2` 27, `spring_green2` 47, `grey35` 240, `grey42` 242, `grey50` 244,
`grey58` 246, `grey62` 247, `grey70` 249). Chart colours are Python's hex values as RGB.

## Provenance

- The bar adds only `Measured` bytes, through `MeasuredTotal`; `Shown`/`Resident` never reach it.
- One helper, `format::bytes_cell(&Bytes)`, renders every byte cell: measured → size (bold where
  Python bolds); inferred → size in grey with the `*`/`?` marker from `vram_label`; estimated →
  size with `(est)`; reported → size with its source; `Unavailable { not_on_backend }` → blank;
  any other `Unavailable` → `n/a`.
- Percentages: `None` → `n/a` in grey62 where Python prints `n/a`; a measured 0 → blank in the
  table (Python's `< 0.5` rule) and `0` in the header.
- Chart history keeps `None` for a missing reading; it is drawn blank, never as 0.
- The UI invents no figures: every number comes from `Snapshot`, `ProcRow` or `LlmView`.

## Error handling

| Situation | Behaviour |
|---|---|
| No supported GPU, or WSL | TUI starts; the device box shows the snapshot notes; empty table |
| stdout not a terminal | exit 2, `the TUI needs an interactive terminal; use --once or --json` |
| Terminal below 40×10 | `terminal too small` |
| Sampler thread panics | the channel disconnects; the main loop restores the terminal, prints `gputaskman: sampling stopped: <panic message>` to stderr, exit 1 |
| Terminal write error | restore, report on stderr, exit 1 |
| `q` / `Ctrl+C` | stop and join the sampler, restore, exit 0 |

The terminal guard restores raw mode, the alternate screen and mouse capture on every exit path,
including panics on the main thread.

## Testing

All tests run without a GPU or a terminal.

- Unit: `spark` (zero vs missing, clamping), `stacked_bar` (rounding, overflow, normalisation),
  history scales, chart grid columns, top filter and sorts, multi-GPU merge, LLM auto-select,
  selection keys across GPU-count changes, key and mouse mapping, chart menu navigation,
  `bytes_cell` for every `Bytes` state.
- Render: `ratatui::backend::TestBackend` at 180×50, 140×42, 117×59 and 240×50, fed snapshots
  built with `gputaskman_core::testing` and the Linux fixture builders. Assertions on buffer text
  and cell styles. Four `insta` golden screens: NVML single GPU with an LLM selected; DRM with a
  shared client; two GPUs; no GPU.
- Ported Python UI tests (`tests/test_core.py`, `ChartTests` and `UiTests`), adapted to NVML/DRM
  snapshots:
  `spark_distinguishes_zero_from_missing_and_clamps_values`,
  `chart_palette_uses_distinct_explicit_colors_consistently`,
  `monitoring_toggle_hides_diagnostics_without_changing_live_state`,
  `monitoring_applies_to_every_gpu_and_survives_new_samples`,
  `monitoring_keeps_missing_values_and_backend_provenance`,
  `native_hidden_vram_keeps_engine_reading_in_ui`,
  `unknown_process_memory_preserves_unattributed_adapter_usage`,
  `eighth_chart_records_sm_clock_with_scale_and_missing_readings`,
  `unavailable_device_metrics_render_as_na`,
  `chart_cycle_shows_selected_metric_and_preserves_gaps`,
  `chart_dropdown_is_clickable_and_keyboard_accessible`,
  `multi_gpu_rows_histories_and_selection_stay_separate`,
  `poll_collects_all_adapters_and_closes_each_collector` (as a sampler test),
  `explicit_gpu_keeps_single_device_layout`,
  `chart_layout_reflows_while_paused_and_keeps_processes_visible`,
  `all_charts_work_without_nvml_or_vram_capacity`,
  `handy_evidence_is_explicit_in_inspector`,
  `likely_kind_is_explicit_in_process_and_inspector`,
  `navigation_sort_pause_and_help`.
  Not ported (WSL only): `wsl_sort_selection_and_bar_use_inferred_vram_without_double_counting`,
  `multi_gpu_wsl_does_not_claim_process_device_ownership`.
- Sampler: fake backends, 20 ms interval; it delivers snapshots, delivers none while paused, and
  `Drop` joins within one interval.
- CLI: no-argument run under `assert_cmd` (no terminal) exits 2 with the message above.
- Manual: run the TUI against the DRM fixture tree via `GPUTASKMAN_SYS_ROOT` /
  `GPUTASKMAN_PROC_ROOT` in a real terminal. Reported as checked against fixtures, not validated
  on hardware.

## Documentation

`rust/README.md`: TUI usage and keys, the validation table (TUI: fixtures only), the dropped
command palette in the deliberate-differences table, and the ported/skipped UI test list.
