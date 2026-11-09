# gputaskman (Rust port)

A Rust port of gputaskman's collector, text report, JSON output and snapshot diff for native Linux.
The Python implementation in `src/gputaskman/` remains the reference: where the two disagree, the
Rust side is wrong unless the Python behaviour breaks one of the provenance rules in
[`AGENTS.md`](../AGENTS.md).

## Build and run

Requires Rust 1.97 or later. `rust-toolchain.toml` pins 1.97.0, the compiler the compile-fail
(`trybuild`) snapshots were generated with; rustup installs it on first use in `rust/`.

```sh
cd rust
cargo build --release
target/release/gputaskman --once        # one text snapshot
target/release/gputaskman --json        # one JSON snapshot (schema v2)
target/release/gputaskman --list-gpus
target/release/gputaskman diff before.json after.json
```

`diff` reads both Rust (schema v2) and Python (v1) `--json` files.

Without `--once` or `--json`, `gputaskman` opens the live TUI (all GPUs, or one with `--gpu N`;
refresh with `-i SECONDS`). Keys: `↑ ↓ PgUp PgDn Home End` select, `s` sort, `c` chart, `a` all
or top processes, `m` slim or full diagnostics, `p` pause, `?` help, `q` or `Ctrl+C` quit. The
mouse selects rows, scrolls panes, opens the chart selector and runs the footer's key hints.
The TUI needs an interactive terminal; with stdout redirected it exits with status 2.

`GPUTASKMAN_PROC_ROOT` and `GPUTASKMAN_SYS_ROOT` replace `/proc` and `/sys`. They exist for tests,
which point them at fixture trees; they are not a supported way to inspect another machine.

## Status

- **Milestone 1.** The collector, both Linux backends, the text report, JSON output and `diff`.
- **Milestone 2.** The interactive TUI (crate `gputaskman-tui`), with the same screen and keys as
  the Python Textual UI on Linux. Checked against fixtures only, not on GPU hardware.
- **Native Linux only.** NVML for NVIDIA GPUs and DRM fdinfo for amdgpu cards. WSL, Windows and
  macOS are not supported; under WSL the command collects nothing and says why in a note. CI is
  configured to test the platform-neutral `gputaskman-core` and `gputaskman-tui` crates on Linux,
  Windows and macOS, but has not run yet (workflows are not yet enabled on the fork), so only Linux has been checked.
- **JSON schema v2 is unstable.** Field names and shapes may still change. Every number carries its
  provenance (`measured`, `inferred`, `estimated`, `reported` or `unavailable` with a reason).

Crates:

| Crate | Contents |
|---|---|
| `gputaskman-core` | Provenance types, model layer, process inspection, LLM view, text and JSON reports, v1 reader, diff |
| `gputaskman-linux` | NVML backend, amdgpu DRM backend, PID namespace matching, `/proc` readers |
| `gputaskman-tui` | The interactive terminal UI: state, actions, screen layout, sampler thread |
| `gputaskman` | The command-line interface |

### Deliberate differences from Python

| Difference | Why |
|---|---|
| The text report prints `n/a` where Python prints `None` (missing `ctx_train`, context or slots) | Rule 2: unknown is `n/a` or blank in the UI; Python's `None` is an artefact |
| A GGUF length field longer than the remaining file is rejected at once with `EOFError: truncated GGUF header`, and arrays nest at most 16 levels | Bounded allocation on hostile files; Python's error text there depends on the interpreter |
| Reason code `unreadable` for an amdgpu sysfs counter that exists but cannot be read | Python has no reason codes outside DRM attribution; rule 9 wants the reason |
| Schema v2 has a top-level `notes` array and `status.device` | Rule 9: degradation reasons (a vendor library that did not load, unreadable device telemetry) must reach the JSON as well as the text |
| Rows with unknown VRAM sort as 0 with a stable sort | The same order as Python's `-(shown_vram or 0)`; recorded because the design first placed unknown rows last |
| Local API probes never follow redirects | Rule 6: a redirect could send the `GET` to another host or port; Python's `urllib` follows them |
| A non-positive `n_ctx` produces no KV segment | A negative context is not a size; Python multiplies it through and can show a negative KV estimate |
| An NVML PID hidden (no usage) in one process list and measured in the other is unknown as a whole | Rule 2: part of its memory is unknown. Python counts the measured part in `attributed` although no row shows it |
| Unknown `vram_total` prints `N/A` in the text report | Rule 2; Python prints `0 MB` |
| A v1 inferred row without `vram_inferred_by` reads as an order guess (`?`) | Rule 5: without the confidence, `*` cannot be claimed |
| No `^p` command palette | Textual's built-in palette; `app.py` adds nothing of its own to it |
| The mouse wheel over the process table moves the selection | ratatui has no scroll view; moving the selection scrolls the table |
| The first LLM is auto-selected on the first refresh that has one, unless something is already selected | Python only tries on the first refresh |
| WSL is refused with a note and no GPU data | NVML under WSL reports Windows host PIDs, which would be charged to the wrong processes; WSL support is later work |

## Validation

Rule 10: hardware claims need hardware. The development machine has only an Intel `i915` GPU,
which gputaskman does not support, so no backend has been checked live.

| Backend | Hardware | Status |
|---|---|---|
| NVML, bare metal | none | **not validated on hardware** |
| NVML, in a container (own PID namespace) | none | **not validated on hardware** |
| amdgpu DRM fdinfo | none | **not validated on hardware** |

The TUI is checked against fixture snapshots and the fixture `/sys` and `/proc` trees; it has no
hardware validation.

Development machine: Intel i915 only (unsupported). Every behaviour above is covered by tests on
fakes, fixture trees and replayed measurements only, plus parity with the Python implementation.

### Live-check checklist

Before removing a "not validated" label, on an NVIDIA or amdgpu machine:

1. Run a bounded, known workload (a llama-server with a known model, or a fixed-size CUDA or HIP
   allocation).
2. Capture `gputaskman --json` from both Python and Rust.
3. Compare with `nvidia-smi` (NVIDIA) or Python's DRM output (amdgpu), and run
   `gputaskman diff python.json rust.json`; measured buckets must agree.
4. Record hardware, driver, kernel and results before removing the "not validated" label.

## Rule-to-test map

Each rule in [`AGENTS.md`](../AGENTS.md) and the Rust tests that cover it.

| Rule | Tests |
|---|---|
| 1. Only measured bytes are summed | `only_measured_bytes_reach_the_total`, `attribution_total_rules`, `provenance_misuse_does_not_compile` (compile-fail: only a `Measured` value enters a `MeasuredTotal`, never a `Bytes` or `Shown`; resident bytes cannot be added; a `Measured` cannot be forged), `drm_unattributed_rows_void_the_total_and_explain_why` |
| 2. Unknown is not zero | `measured_zero_is_not_unknown`, `hidden_per_process_memory_is_unknown_not_zero`, `measured_zero_and_unknown_stay_apart`, `drm_json_and_text_keep_unknown_apart_from_zero`, `best_vram_keeps_measured_and_inferred_zero_distinct_from_unknown`, `native_hidden_vram_does_not_hide_measured_engine_load`, `test_missing_sysfs_memory_is_unknown`, `text_shows_unavailable_device_metrics_without_fabricating_zeroes` |
| 3. Per-process memory from the right counters (amdgpu: provably private clients only) | `private_render_client_is_measured_and_engine_load_needs_two_samples`, `client_held_by_several_processes_is_charged_to_none`, `any_shared_buffer_leaves_the_process_unattributed`, `primary_node_is_not_charged`, `test_old_kernel_without_shared_keys_is_not_summed`, `test_parse_fdinfo_shared_needs_all_memory_regions`, `test_private_and_co_owned_clients_leave_the_whole_process_unattributed`, `test_incomplete_process_has_no_partial_totals` |
| 4. Model evidence ladder | `evidence_ladder_and_labels`, `measured_runtime_and_mapped_evidence_stay_separate`, `installed_models_are_candidates_only`, `selected_model_is_distinct_from_runtime_load_and_unload_events`, `selected_model_without_fresh_log_is_not_reported_as_loaded`, `openjev_hint_does_not_change_header_kv_or_loading_evidence`, `installed_kind_hints_do_not_classify_the_process` |
| 5. Markers mean what they say | `markers`, `inferred_markers_and_notes_appear_in_text`, `host_pids_are_matched_to_container_pids_by_creation_order`, `extra_outside_compute_is_a_guess_until_a_start_event`, `outside_rows_never_borrow_a_local_identity` |
| 6. Look, don't touch | `local_url_is_loopback_only`, `provenance_misuse_does_not_compile` (a `LocalUrl` cannot be built for another host), `loopback_path_cannot_change_the_host`, `ureq_get_is_a_get_and_only_accepts_200`, `ureq_get_does_not_follow_redirects`, `probes_are_get_only_on_the_given_port`, `shared_listen_port_is_not_used_for_server_attribution` |
| 7. Stay cheap | `deep_scans_stop_at_the_budget`, `test_scan_budget_keeps_rows_and_resumes`, `test_one_huge_process_does_not_stall_the_rotation`, `test_budget_is_checked_inside_one_process`, `ports_refresh_at_most_every_five_seconds`, `prober_caches_hits_and_misses_per_port`, `caches_last_sixty_seconds`, `corrupt_lengths_do_not_allocate` |
| 8. Never bundle driver libraries | Not a test: NVML is loaded at runtime from the installed driver by `nvml-wrapper`; no library is vendored. `loading_the_real_library_never_panics` checks that a missing library does not crash |
| 9. Degrade with a reason | `linux_backends_does_not_panic_without_drivers`, `no_supported_gpu_gives_an_empty_snapshot_with_a_reason`, `json_without_a_supported_gpu_is_valid_v2`, `snapshot_notes_and_process_status_reach_json_and_text`, `device_failure_is_recorded_and_hides_driver_versions`, `a_failing_gpu_is_skipped_not_fatal`, `test_uninspectable_processes_are_reported`, `backend_status_reports_unreadable_proc_root` |
| 10. Hardware claims need hardware | Not a test: the validation table above. `backends_match_python` and `report_matches_python` compare with the Python implementation, not with hardware |

## TUI tests ported from Python

These tests from `tests/test_core.py` (`ChartTests` and `UiTests`) are ported, adapted to NVML and
DRM snapshots. Rust tests are in `crates/gputaskman-tui/tests/`.

| Python test | Rust test (file::name) |
|---|---|
| `spark_distinguishes_zero_from_missing_and_clamps_values` | `format.rs::spark_distinguishes_zero_from_missing_and_clamps_values` |
| `chart_palette_uses_distinct_explicit_colors_consistently` | `history.rs::chart_palette_uses_distinct_explicit_colours`, `header.rs::the_metrics_line_uses_the_chart_colours` |
| `monitoring_toggle_hides_diagnostics_without_changing_live_state` | `header.rs::monitoring_toggle_hides_diagnostics_without_changing_live_state` |
| `monitoring_applies_to_every_gpu_and_survives_new_samples` | `header.rs::monitoring_applies_to_every_gpu` |
| `monitoring_keeps_missing_values_and_backend_provenance` | `header.rs::monitoring_keeps_missing_values_and_provenance` |
| `native_hidden_vram_keeps_engine_reading_in_ui` | `table.rs::native_hidden_vram_keeps_the_engine_reading`, `inspector.rs::native_hidden_vram_keeps_the_engine_reading` |
| `unknown_process_memory_preserves_unattributed_adapter_usage` | `header.rs::unknown_process_memory_stays_unattributed` |
| `eighth_chart_records_sm_clock_with_scale_and_missing_readings` | `history.rs::eighth_chart_records_sm_clock_with_scale_and_missing_readings`, `charts.rs::eighth_chart_shows_sm_clock_with_its_scale` |
| `unavailable_device_metrics_render_as_na` | `header.rs::unavailable_device_metrics_render_as_na`, `history.rs::unavailable_device_metrics_record_as_missing_and_zero_stays_zero` |
| `chart_cycle_shows_selected_metric_and_preserves_gaps` | `charts.rs::chart_cycle_shows_the_selected_metric_alone_and_preserves_gaps`, `app.rs::chart_cycle_visits_all_then_each_chart` |
| `chart_dropdown_is_clickable_and_keyboard_accessible` | `screen.rs::chart_dropdown_is_clickable_and_keyboard_accessible`, `app.rs::chart_menu_is_keyboard_accessible`, `charts.rs::the_selector_names_the_mode_and_the_menu_lists_every_mode` |
| `multi_gpu_rows_histories_and_selection_stay_separate` | `rows.rs::several_gpus_prefix_keys_with_the_gpu_and_sort_together`, `app.rs::selection_keys_follow_the_gpu_count`, `app.rs::each_gpu_keeps_its_own_history_and_chart_values`, `charts.rs::several_gpus_get_their_own_grids_and_notes`, `screen.rs::several_gpus_show_both_devices_and_a_gpu_column` |
| `poll_collects_all_adapters_and_closes_each_collector` | `sampler.rs::the_sampler_collects_every_gpu_and_drops_each_collector` |
| `explicit_gpu_keeps_single_device_layout` | `table.rs::explicit_gpu_keeps_single_device_layout` |
| `chart_layout_reflows_while_paused_and_keeps_processes_visible` | `screen.rs::chart_layout_reflows_while_paused_and_keeps_processes_visible`, `charts.rs::chart_layout_reflows_with_the_width` |
| `all_charts_work_without_nvml_or_vram_capacity` | `charts.rs::all_charts_work_without_device_telemetry_or_vram_capacity` |
| `handy_evidence_is_explicit_in_inspector` | `inspector.rs::handy_evidence_is_explicit` |
| `likely_kind_is_explicit_in_process_and_inspector` | `table.rs::likely_kind_is_explicit_in_the_kind_column`, `inspector.rs::likely_kind_and_evidence_are_explicit_in_the_model_table` |
| `navigation_sort_pause_and_help` | `app.rs::navigation_sort_pause_and_help` |

Two are skipped because WSL is not supported by the Rust build yet:

- `wsl_sort_selection_and_bar_use_inferred_vram_without_double_counting`
- `multi_gpu_wsl_does_not_claim_process_device_ownership`

## Parity with Python

The parity tests run the same scenarios through the Python implementation and the Rust one and
require equal results. They need [`uv`](https://docs.astral.sh/uv/) and the project's Python
environment:

```sh
uv sync --locked                      # from the repository root
cd rust
cargo test --workspace --features gputaskman-linux/parity,gputaskman-core/parity
```

- **Backend parity** (`backends_match_python`): each scenario in `parity/scenarios/` describes
  amdgpu cards with process fds and fdinfo, or fake NVML data and a PID namespace.
  `parity/run_python.py` builds the tree with the helpers from Python's `tests/test_drm.py` and
  `tests/test_linux.py`, runs `DrmCounters` or `LinuxCounters`, and prints each process's VRAM,
  GTT, reason, owner and engines; the Rust backends must give the same on the same tree.
- **Report parity** (`report_matches_python`): `report_mixed.json` goes through Python's
  `report.to_dict`, then the Rust v1 reader, and must equal the Rust v2 snapshot of the same rows,
  ignoring `taken`. v1 writes `null` for every unknown value and names a reason only for amdgpu VRAM,
  so the Rust side is compared after dropping the reasons v1 cannot carry: unknown stays unknown.

The harness imports the Python modules directly. If their internals change, the harness changes
with them in the same commit.

Other checks:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
