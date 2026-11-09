//! Argument handling and the one-shot flow, mirroring the Python `cli.py`.
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, Subcommand};
use gputaskman_core::collector::{Collector, CollectorDeps, SCAN_BUDGET, list_adapters};
use gputaskman_core::env::Paths;
use gputaskman_core::http::UreqGet;
use gputaskman_core::traits::{Clock, ScanPolicy, SystemClock};
use gputaskman_core::{diff, report, schema};
use gputaskman_linux::{linux_backends_at, procs::LinuxProcs};
use gputaskman_tui::{Config, Source};

#[derive(Parser)]
#[command(
    name = "gputaskman",
    about = "What's inside your GPU: per-process VRAM, engines, and LLM weights/KV-cache breakdown.",
    after_help = "gputaskman diff BEFORE.json AFTER.json compares two --json snapshots."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// print one snapshot and exit (no TUI)
    #[arg(long)]
    once: bool,
    /// print one JSON snapshot and exit (implies --once)
    #[arg(long)]
    json: bool,
    /// show only this GPU (TUI default: all; one-shot default: 0)
    #[arg(long)]
    gpu: Option<usize>,
    /// list selectable GPU adapters and exit
    #[arg(long)]
    list_gpus: bool,
    /// refresh seconds (default 1)
    #[arg(short, long, default_value_t = 1.0)]
    interval: f64,
}

#[derive(Subcommand)]
enum Command {
    /// Compare two --json snapshots, for example before and after loading a model.
    Diff {
        /// earlier --json snapshot
        before: PathBuf,
        /// later --json snapshot
        after: PathBuf,
        /// print the comparison as JSON
        #[arg(long)]
        json: bool,
    },
}

/// An argparse-style usage error: message on stderr, exit status 2.
fn usage_error(message: &str) -> i32 {
    Cli::command()
        .error(ErrorKind::ValueValidation, message)
        .exit()
}

/// Writes `text` and a newline to stdout. A closed pipe (`gputaskman --json | head -1`) is a clean
/// exit (0); any other write error is reported on stderr (1). Never panics.
fn emit(text: &str) -> i32 {
    let mut out = std::io::stdout().lock();
    match writeln!(out, "{text}").and_then(|()| out.flush()) {
        Ok(()) => 0,
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => 0,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "gputaskman: cannot write to stdout: {e}");
            1
        }
    }
}

pub fn run() -> i32 {
    let cli = Cli::parse();
    if let Some(Command::Diff {
        before,
        after,
        json,
    }) = cli.command
    {
        return diff_main(&before, &after, json);
    }
    if cli.interval.is_nan() || cli.interval.is_infinite() || cli.interval <= 0.0 {
        return usage_error("--interval must be a finite number greater than zero");
    }

    let root = |var: &str, default: &str| {
        std::env::var_os(var).map_or_else(|| PathBuf::from(default), PathBuf::from)
    };
    let proc_root = root("GPUTASKMAN_PROC_ROOT", "/proc");
    let sys_root = root("GPUTASKMAN_SYS_ROOT", "/sys");
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let (backends, notes) = linux_backends_at(&sys_root, &proc_root, clock.clone());

    if cli.list_gpus {
        let lines: Vec<String> = list_adapters(&backends)
            .iter()
            .enumerate()
            .map(|(i, a)| format!("{i}: {} ({}, key 0x{:016x})", a.name, a.vendor(), a.key))
            .collect();
        if lines.is_empty() {
            return 0;
        }
        return emit(&lines.join("\n"));
    }

    if !(cli.once || cli.json) {
        drop(backends);
        return tui_main(&cli, &sys_root, &proc_root);
    }

    let deps = CollectorDeps {
        backends,
        procs: Box::new(LinuxProcs::new(proc_root)),
        http: Box::new(UreqGet::default()),
        clock,
        paths: Paths::from_env(),
        notes,
    };
    // One-shot: identify everything now.
    let mut collector = match Collector::new(deps, cli.gpu.unwrap_or(0), ScanPolicy::Unbounded) {
        Ok(c) => c,
        Err(e) => return usage_error(&e.to_string()),
    };
    std::thread::sleep(Duration::from_secs(1)); // engine utilisation is a rate; needs two samples
    let snapshot = collector.sample();
    if cli.json {
        emit(&schema::v2::to_json(&snapshot))
    } else {
        emit(&report::to_text(&snapshot, 30))
    }
}

fn diff_main(before: &std::path::Path, after: &std::path::Path, json: bool) -> i32 {
    let result = schema::load(before)
        .and_then(|b| schema::load(after).map(|a| (b, a)))
        .and_then(|(b, a)| diff::diff(&b, &a));
    match result {
        Ok(d) if json => match serde_json::to_string_pretty(&d) {
            Ok(s) => emit(&s),
            Err(e) => usage_error(&e.to_string()),
        },
        Ok(d) => emit(&diff::to_text(&d)),
        Err(e) => usage_error(&e),
    }
}

/// The live TUI: one collector per GPU (or just `--gpu N`), sharing the scan budget.
fn tui_main(cli: &Cli, sys_root: &std::path::Path, proc_root: &std::path::Path) -> i32 {
    if !std::io::stdout().is_terminal() {
        eprintln!("the TUI needs an interactive terminal; use --once or --json");
        return 2;
    }
    let indices: Vec<usize> = match cli.gpu {
        Some(i) => vec![i],
        None => {
            let clock: Arc<dyn Clock> = Arc::new(SystemClock);
            let (backends, _) = linux_backends_at(sys_root, proc_root, clock);
            (0..list_adapters(&backends).len().max(1)).collect()
        }
    };
    let budget = SCAN_BUDGET / u32::try_from(indices.len()).unwrap_or(1).max(1);
    let mut sources: Vec<Box<dyn Source>> = Vec::new();
    for index in indices {
        let clock: Arc<dyn Clock> = Arc::new(SystemClock);
        let (backends, notes) = linux_backends_at(sys_root, proc_root, clock.clone());
        let deps = CollectorDeps {
            backends,
            procs: Box::new(LinuxProcs::new(proc_root.to_path_buf())),
            http: Box::new(UreqGet::default()),
            clock,
            paths: Paths::from_env(),
            notes,
        };
        match Collector::new(deps, index, ScanPolicy::Budgeted(budget)) {
            Ok(c) => sources.push(Box::new(c)),
            Err(e) => return usage_error(&e.to_string()),
        }
    }
    let config = Config {
        interval: Duration::from_secs_f64(cli.interval),
    };
    match gputaskman_tui::run(sources, config) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("gputaskman: {e}");
            1
        }
    }
}
