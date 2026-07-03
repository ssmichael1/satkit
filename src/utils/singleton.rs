//! A small helper for module-scope, refreshable, lazily-loaded singletons.
//!
//! The refreshable data subsystems (space weather, Earth-orientation
//! parameters, solar-cycle forecast) all share the same lifecycle: a global
//! `RwLock<Option<T>>` that is lazily populated on first read (best-effort,
//! at most once) and can be replaced at any time from bytes / a path / a fresh
//! download. Before this helper each module hand-rolled that scaffolding, with
//! subtly different failure policies (one retried the load on *every* read).
//!
//! Each module keeps its own lookup (`get`) logic; only the load/refresh
//! plumbing lives here.

use std::sync::{Once, RwLock, RwLockReadGuard};

/// A lazily-initialized, replaceable global value.
///
/// The default load runs at most once and is best-effort: if it fails the
/// singleton simply stays empty and the module's `get` falls through to its
/// "no data" branch. An explicit [`set`](Self::set) (from an `init_from_*` or
/// `update` call) replaces the contents and marks the default-load slot as
/// consumed, so a subsequent first read can never clobber explicitly-provided
/// data.
pub struct RefreshableSingleton<T> {
    data: RwLock<Option<T>>,
    default_load: Once,
}

impl<T> RefreshableSingleton<T> {
    /// Create an empty singleton (usable in a `static` initializer).
    pub const fn new() -> Self {
        Self {
            data: RwLock::new(None),
            default_load: Once::new(),
        }
    }

    /// Populate the default contents by running `loader` at most once across
    /// the lifetime of the program. `loader` returns `None` to indicate the
    /// data could not be loaded (leaving the singleton empty); the load is not
    /// retried on subsequent reads.
    pub fn ensure_default_loaded(&self, loader: impl FnOnce() -> Option<T>) {
        self.default_load.call_once(|| {
            if let Some(v) = loader() {
                *self.data.write().unwrap() = Some(v);
            }
        });
    }

    /// Replace the contents (used by `init_from_*` / `update`). Also consumes
    /// the default-load slot so a later first read can't overwrite this value.
    pub fn set(&self, value: T) {
        self.default_load.call_once(|| {});
        *self.data.write().unwrap() = Some(value);
    }

    /// Acquire a read guard on the current contents (`None` if unloaded).
    pub fn read(&self) -> RwLockReadGuard<'_, Option<T>> {
        self.data.read().unwrap()
    }
}

impl<T> Default for RefreshableSingleton<T> {
    fn default() -> Self {
        Self::new()
    }
}
