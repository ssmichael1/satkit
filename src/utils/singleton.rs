//! A small helper for module-scope, refreshable, lazily-loaded singletons.
//!
//! The refreshable data subsystems (space weather, Earth-orientation
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
    ///
    /// A panic inside `loader` is caught and treated like `None`: the panic
    /// message still reaches stderr through the panic hook, but the `Once`
    /// is not poisoned, so later reads fall through to the module's "no
    /// data" branch instead of panicking with "Once instance has previously
    /// been poisoned" in every data-dependent call for the rest of the
    /// process.
    pub fn ensure_default_loaded(&self, loader: impl FnOnce() -> Option<T>) {
        self.default_load.call_once(|| {
            let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(loader));
            if let Ok(Some(v)) = loaded {
                *self.data.write().unwrap_or_else(|e| e.into_inner()) = Some(v);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A loader that panics leaves the singleton empty; it does not poison
    /// it, so later reads, loads and sets still work.
    #[test]
    fn panicking_loader_does_not_poison() {
        let s: RefreshableSingleton<u32> = RefreshableSingleton::new();
        s.ensure_default_loaded(|| panic!("loader bug"));
        assert!(s.read().is_none());
        // The default load is not retried, and does not panic either.
        s.ensure_default_loaded(|| Some(1));
        assert!(s.read().is_none());
        s.set(2);
        assert_eq!(*s.read(), Some(2));
    }
}
