//! User-facing diagnostics: satkit's warnings and notices go through the
//! [`log`] facade, tagged with the emitting module as the target (e.g.
//! `satkit::earth_orientation_params`).
//!
//! With no logger installed (`log::max_level()` is `Off`, the state until an
//! application installs one) a record is printed to stderr instead, as
//! `Warning: <message>` or `satkit: <message>`, so nothing is lost in a
//! program that never sets up logging. Every diagnostic in the crate goes
//! through [`warn!`] or [`info!`]; nothing else prints to stderr.
//!
//! [`deferred`] holds back the records emitted on the current thread until a
//! closure returns. Code that holds a lock or runs a `Once`/`OnceLock`
//! initializer wraps that region in it: a logger may block (the Python
//! bindings' logger takes the GIL), and a thread holding the GIL could be
//! waiting on that same lock.

use std::cell::RefCell;
use std::fmt;

use log::{Level, LevelFilter};

/// Log a warning through [`emit`] with the calling module as the target.
macro_rules! diag_warn {
    ($($arg:tt)+) => {
        $crate::utils::diag::emit(::log::Level::Warn, module_path!(), format_args!($($arg)+))
    };
}

/// Log an informational notice through [`emit`] with the calling module as
/// the target.
macro_rules! diag_info {
    ($($arg:tt)+) => {
        $crate::utils::diag::emit(::log::Level::Info, module_path!(), format_args!($($arg)+))
    };
}

// Re-exported under short names; a `macro_rules! warn` could not be
// imported (it clashes with the built-in `#[warn]` attribute).
pub(crate) use {diag_info as info, diag_warn as warn};

/// A record held back by [`deferred`].
struct Pending {
    level: Level,
    target: &'static str,
    msg: String,
}

thread_local! {
    /// `Some` while this thread is inside a [`deferred`] region.
    static DEFERRED: RefCell<Option<Vec<Pending>>> = const { RefCell::new(None) };
}

/// Emit one record: held back inside a [`deferred`] region, otherwise sent
/// to the installed logger, or to stderr when there is none.
pub(crate) fn emit(level: Level, target: &'static str, args: fmt::Arguments) {
    let held = DEFERRED.with(|d| match d.borrow_mut().as_mut() {
        Some(pending) => {
            pending.push(Pending {
                level,
                target,
                msg: args.to_string(),
            });
            true
        }
        None => false,
    });
    if !held {
        dispatch(level, target, args);
    }
}

/// Where a record goes once it is released.
#[derive(Debug, PartialEq, Eq)]
enum Sink {
    Logger,
    Stderr,
}

fn sink() -> Sink {
    if log::max_level() == LevelFilter::Off {
        Sink::Stderr
    } else {
        Sink::Logger
    }
}

fn dispatch(level: Level, target: &str, args: fmt::Arguments) {
    match sink() {
        Sink::Logger => log::log!(target: target, level, "{args}"),
        Sink::Stderr => {
            let prefix = if level <= Level::Warn {
                "Warning"
            } else {
                "satkit"
            };
            eprintln!("{prefix}: {args}");
        }
    }
}

/// Run `f`, holding back the records it emits on this thread until it has
/// returned (and dropped any lock guards it held); see the module docs.
/// Nested regions release everything when the outermost one ends.
pub(crate) fn deferred<R>(f: impl FnOnce() -> R) -> R {
    let outermost = DEFERRED.with(|d| {
        let mut d = d.borrow_mut();
        d.is_none().then(|| *d = Some(Vec::new())).is_some()
    });
    if !outermost {
        return f();
    }
    // Released on drop, so an unwinding `f` does not leave the thread
    // deferring (and silently discarding) every later record.
    struct Release;
    impl Drop for Release {
        fn drop(&mut self) {
            let pending = DEFERRED.with(|d| d.borrow_mut().take());
            for p in pending.into_iter().flatten() {
                dispatch(p.level, p.target, format_args!("{}", p.msg));
            }
        }
    }
    let _release = Release;
    f()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Mutex, Once};

    /// A record seen by the [`Capture`] logger.
    #[derive(Clone, Debug)]
    pub(crate) struct Captured {
        pub level: Level,
        pub target: String,
        pub msg: String,
    }

    static CAPTURED: Mutex<Vec<Captured>> = Mutex::new(Vec::new());

    /// A test logger that records every satkit record it is given.
    struct Capture;

    impl log::Log for Capture {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.target().starts_with("satkit")
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                CAPTURED.lock().unwrap().push(Captured {
                    level: r.level(),
                    target: r.target().to_string(),
                    msg: r.args().to_string(),
                });
            }
        }
        fn flush(&self) {}
    }

    /// Install [`Capture`] as the process logger (once per test binary) and
    /// return the records captured so far whose message contains `needle`.
    pub(crate) fn captured(needle: &str) -> Vec<Captured> {
        static INSTALL: Once = Once::new();
        INSTALL.call_once(|| {
            log::set_logger(&Capture).expect("no other logger in the test binary");
            log::set_max_level(LevelFilter::Info);
        });
        CAPTURED
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.msg.contains(needle))
            .cloned()
            .collect()
    }

    /// With a logger installed, a warning reaches it with the calling
    /// module's target and the Warn level, without the stderr prefix.
    #[test]
    fn warning_reaches_installed_logger() {
        captured("");
        warn!("diag probe {}", "warn-logger");
        info!("diag probe {}", "info-logger");
        let w = captured("diag probe warn-logger");
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(w[0].level, Level::Warn);
        assert_eq!(w[0].target, "satkit::utils::diag::tests");
        assert_eq!(w[0].msg, "diag probe warn-logger");
        let i = captured("diag probe info-logger");
        assert_eq!(i.len(), 1, "{i:?}");
        assert_eq!(i[0].level, Level::Info);
    }

    /// Records emitted inside a deferred region arrive only once it ends,
    /// in order, including those from a nested region.
    #[test]
    fn deferred_records_arrive_after_the_region() {
        captured("");
        let r = deferred(|| {
            warn!("diag probe deferred-1");
            deferred(|| warn!("diag probe deferred-2"));
            assert!(captured("diag probe deferred-").is_empty());
            7
        });
        assert_eq!(r, 7);
        let got: Vec<String> = captured("diag probe deferred-")
            .into_iter()
            .map(|c| c.msg)
            .collect();
        assert_eq!(got, ["diag probe deferred-1", "diag probe deferred-2"]);
    }

    /// With no logger installed, records take the stderr path, with the
    /// prefix they always had. In its own process, since other tests in
    /// this binary install the capturing logger.
    #[test]
    fn no_logger_falls_back_to_stderr() {
        let Some(out) = crate::utils::download::own_process_output(
            module_path!(),
            "no_logger_falls_back_to_stderr",
        ) else {
            assert_eq!(log::max_level(), LevelFilter::Off);
            assert_eq!(sink(), Sink::Stderr);
            warn!("diag probe {}", "stderr-warn");
            info!("diag probe {}", "stderr-info");
            deferred(|| warn!("diag probe stderr-deferred"));
            return;
        };
        let stderr = String::from_utf8_lossy(&out.stderr);
        for line in [
            "Warning: diag probe stderr-warn",
            "satkit: diag probe stderr-info",
            "Warning: diag probe stderr-deferred",
        ] {
            assert!(stderr.contains(line), "missing {line:?} in\n{stderr}");
        }
    }
}
