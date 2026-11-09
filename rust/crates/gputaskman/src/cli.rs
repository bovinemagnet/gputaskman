//! Argument handling and the one-shot flow, mirroring the Python `cli.py`.
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, Subcommand};
use gputaskman_core::collector::{Collector, CollectorDeps, list_adapters};
use gputaskman_core::env::Paths;
use gputaskman_core::http::UreqGet;
use gputaskman_core::traits::{Clock, ScanPolicy, SystemClock};
use gputaskman_core::{diff, report, schema};
use gputaskman_linux::{linux_backends_at, procs::LinuxProcs};

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
        for (i, a) in list_adapters(&backends).iter().enumerate() {
            println!("{i}: {} ({}, key 0x{:016x})", a.name, a.vendor(), a.key);
        }
        return 0;
    }

    if !(cli.once || cli.json) {
        eprintln!("the TUI is not available yet in the Rust build; use --once or --json");
        return 2;
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
        println!("{}", schema::v2::to_json(&snapshot));
    } else {
        println!("{}", report::to_text(&snapshot, 30));
    }
    0
}

fn diff_main(before: &std::path::Path, after: &std::path::Path, json: bool) -> i32 {
    let result = schema::load(before)
        .and_then(|b| schema::load(after).map(|a| (b, a)))
        .and_then(|(b, a)| diff::diff(&b, &a));
    match result {
        Ok(d) if json => match serde_json::to_string_pretty(&d) {
            Ok(s) => {
                println!("{s}");
                0
            }
            Err(e) => usage_error(&e.to_string()),
        },
        Ok(d) => {
            println!("{}", diff::to_text(&d));
            0
        }
        Err(e) => usage_error(&e),
    }
}
