//! Bridge from Rust `log` records to Python's `logging` module.
//!
//! Each record goes to the Python logger named after its target with `::`
//! replaced by `.` (`satkit::earth_orientation_params` →
//! `satkit.earth_orientation_params`), built with `makeRecord` so handlers
//! see the Rust file and line, and passed to `Logger.handle`.
//!
//! Only satkit's own records at `Info` or above are forwarded: every
//! forwarded record attaches to the interpreter, and the HTTP/TLS crates log
//! at debug level on every request, from `update_datafiles` worker threads
//! among others. Python loggers and levels are looked up per record rather
//! than cached: satkit logs rarely, and this way
//! `logging.getLogger("satkit").setLevel(...)` takes effect whenever it is
//! called, not only before the first record.

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

const TARGET: &str = "satkit";
const MAX_LEVEL: LevelFilter = LevelFilter::Info;

struct PyLogger;

/// Python `logging` level for a Rust level (`Trace` has no Python
/// equivalent; 5 sits below `DEBUG`).
fn py_level(level: Level) -> u32 {
    match level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    }
}

fn forward(py: Python<'_>, record: &Record) -> PyResult<()> {
    let name = record.target().replace("::", ".");
    let level = py_level(record.level());
    let logger = py.import("logging")?.call_method1("getLogger", (&name,))?;
    // `Logger.handle` does not check the level itself.
    if !logger.call_method1("isEnabledFor", (level,))?.is_truthy()? {
        return Ok(());
    }
    let pyrecord = logger.call_method1(
        "makeRecord",
        (
            name,
            level,
            record.file(),
            record.line().unwrap_or_default(),
            record.args().to_string(),
            PyTuple::empty(py), // args
            py.None(),          // exc_info
            py.None(),          // func
            py.None(),          // extra
        ),
    )?;
    logger.call_method1("handle", (pyrecord,))?;
    Ok(())
}

impl Log for PyLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        let target = metadata.target();
        metadata.level() <= MAX_LEVEL
            && target
                .strip_prefix(TARGET)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with("::"))
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        // `try_attach` returns None (and the record is dropped) while the
        // interpreter is shutting down, e.g. a worker thread still running.
        Python::try_attach(|py| {
            // Keep an exception already being raised by the caller, and drop
            // any raised while logging: `log` cannot report one, and leaving
            // it set would surface in unrelated code.
            let pending = PyErr::take(py);
            let _ = forward(py, record);
            if let Some(e) = pending {
                e.restore(py);
            }
        });
    }

    fn flush(&self) {}
}

/// Install the bridge as the global `log` logger.
///
/// Does nothing if a logger is already installed (the module initialized
/// twice in one process); the first one keeps working.
pub fn init() {
    static LOGGER: PyLogger = PyLogger;
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(MAX_LEVEL);
    }
}
