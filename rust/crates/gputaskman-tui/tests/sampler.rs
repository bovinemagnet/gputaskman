#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use gputaskman_core::collector::Snapshot;
use gputaskman_tui::fixtures::snapshot;
use gputaskman_tui::sampler::{Sampler, Source};

struct Fake {
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
    panic: bool,
}

impl Source for Fake {
    fn sample(&mut self) -> Snapshot {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic, "collector failed");
        snapshot()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn fakes(n: usize, panic: bool) -> (Vec<Box<dyn Source>>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let sources = (0..n)
        .map(|_| {
            Box::new(Fake {
                calls: Arc::clone(&calls),
                dropped: Arc::clone(&dropped),
                panic,
            }) as Box<dyn Source>
        })
        .collect();
    (sources, calls, dropped)
}

const TICK: Duration = Duration::from_millis(20);

#[test]
fn the_sampler_collects_every_gpu_and_drops_each_collector() {
    let (sources, _, dropped) = fakes(2, false);
    let (sampler, rx) = Sampler::spawn(
        sources,
        TICK,
        Duration::ZERO,
        Arc::new(AtomicBool::new(false)),
    );
    let first = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(first.len(), 2);
    let started = Instant::now();
    drop(sampler);
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
}

#[test]
fn nothing_is_sampled_while_paused() {
    let (sources, calls, _) = fakes(1, false);
    let paused = Arc::new(AtomicBool::new(true));
    let (sampler, rx) = Sampler::spawn(sources, TICK, Duration::ZERO, Arc::clone(&paused));
    std::thread::sleep(TICK * 5);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(rx.try_recv().is_err());
    paused.store(false, Ordering::SeqCst);
    assert!(rx.recv_timeout(Duration::from_secs(2)).is_ok());
    drop(sampler);
}

#[test]
fn the_first_sample_waits_for_the_warmup() {
    let (sources, calls, _) = fakes(1, false);
    let (sampler, _rx) = Sampler::spawn(
        sources,
        TICK,
        Duration::from_millis(300),
        Arc::new(AtomicBool::new(false)),
    );
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop(sampler);
}

#[test]
fn a_panicking_collector_disconnects_and_reports_its_message() {
    let (sources, _, _) = fakes(1, true);
    let (sampler, rx) = Sampler::spawn(
        sources,
        TICK,
        Duration::ZERO,
        Arc::new(AtomicBool::new(false)),
    );
    assert!(rx.recv_timeout(Duration::from_secs(2)).is_err());
    let err = sampler.join().unwrap_err();
    assert!(err.contains("collector failed"), "{err}");
}
