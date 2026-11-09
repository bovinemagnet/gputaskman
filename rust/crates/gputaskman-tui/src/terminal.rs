//! Raw mode, the alternate screen and mouse capture, undone on every exit path.
use std::io::{self, Stdout};
use std::sync::Arc;
use std::thread::{self, ThreadId};

use crossterm::cursor::Show;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

pub fn restore() {
    let _ = execute!(
        io::stdout(),
        DisableMouseCapture,
        LeaveAlternateScreen,
        Show
    );
    let _ = disable_raw_mode();
}

type Hook = dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static;

/// Restores the terminal before a main-thread panic prints. A sampler-thread panic is reported by
/// `run` after the terminal is restored, so only that thread's default message is suppressed.
fn install_panic_hook(main: ThreadId) -> Arc<Hook> {
    let previous: Arc<Hook> = Arc::from(std::panic::take_hook());
    let chained = Arc::clone(&previous);
    std::panic::set_hook(Box::new(move |info| {
        if thread::current().name() == Some(crate::sampler::THREAD_NAME) {
            return;
        }
        if thread::current().id() == main {
            restore();
        }
        chained(info);
    }));
    previous
}

/// Reinstalls the hook that was set before `install_panic_hook`, as a thin wrapper.
fn uninstall_panic_hook(previous: Arc<Hook>) {
    drop(std::panic::take_hook());
    std::panic::set_hook(Box::new(move |info| previous(info)));
}

pub struct TerminalGuard {
    previous_hook: Option<Arc<Hook>>,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<(TerminalGuard, Tui)> {
        let previous_hook = Some(install_panic_hook(thread::current().id()));
        let guard = TerminalGuard { previous_hook }; // from here on, Drop restores
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        Ok((guard, terminal))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
        if let Some(previous) = self.previous_hook.take() {
            uninstall_panic_hook(previous);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn panic_in(name: &str) {
        let _ = thread::Builder::new()
            .name(name.into())
            .spawn(|| panic!("boom"))
            .map(|h| h.join());
    }

    #[test]
    fn only_the_sampler_threads_panics_are_suppressed_and_the_hook_is_restored() {
        let seen = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&seen);
        std::panic::set_hook(Box::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        }));
        let not_main = thread::spawn(|| thread::current().id())
            .join()
            .unwrap_or_else(|_| thread::current().id());
        let previous = install_panic_hook(not_main);
        panic_in(crate::sampler::THREAD_NAME);
        assert_eq!(seen.load(Ordering::SeqCst), 0, "sampler panic suppressed");
        panic_in("some-other-thread");
        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "other panics still reported"
        );
        uninstall_panic_hook(previous);
        panic_in(crate::sampler::THREAD_NAME);
        assert_eq!(seen.load(Ordering::SeqCst), 2, "the previous hook is back");
        let _ = std::panic::take_hook();
    }
}
