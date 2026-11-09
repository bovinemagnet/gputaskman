//! The sampling thread: owns the collectors and sends one snapshot per GPU every interval.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use gputaskman_core::collector::{Collector, Snapshot};

/// Something that produces a snapshot per call: a `Collector`, or a fake in tests.
pub trait Source: Send {
    fn sample(&mut self) -> Snapshot;
}

impl Source for Collector {
    fn sample(&mut self) -> Snapshot {
        Collector::sample(self)
    }
}

pub struct Sampler {
    stop: Option<Sender<()>>,
    handle: Option<JoinHandle<()>>,
}

/// Waits up to `d`; true when told to stop (a message or a dropped sender).
fn stop_requested(rx: &Receiver<()>, d: Duration) -> bool {
    !matches!(rx.recv_timeout(d), Err(RecvTimeoutError::Timeout))
}

impl Sampler {
    pub fn spawn(
        mut sources: Vec<Box<dyn Source>>,
        interval: Duration,
        warmup: Duration,
        paused: Arc<AtomicBool>,
    ) -> (Sampler, Receiver<Vec<Snapshot>>) {
        let (tx, rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            if stop_requested(&stop_rx, warmup) {
                return;
            }
            loop {
                if !paused.load(Ordering::Relaxed) {
                    let snaps: Vec<Snapshot> = sources.iter_mut().map(|s| s.sample()).collect();
                    if tx.send(snaps).is_err() {
                        return;
                    }
                }
                if stop_requested(&stop_rx, interval) {
                    return;
                }
            }
        });
        (
            Sampler {
                stop: Some(stop_tx),
                handle: Some(handle),
            },
            rx,
        )
    }

    /// Stops the thread and waits for it; `Err` carries a panic's message.
    pub fn join(mut self) -> Result<(), String> {
        self.stop.take();
        match self.handle.take().map(JoinHandle::join) {
            Some(Err(payload)) => Err(payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "the sampling thread stopped".to_string())),
            _ => Ok(()),
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
