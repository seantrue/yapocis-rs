//! Device-buffer reuse across [`Program::call`](crate::Program::call)s.
//!
//! The Python device managers keep one device buffer per host array, keyed by
//! `id(array)`, and write the array into it on every call. This is the same
//! binding, keyed by the host array's address and byte length: a call with the
//! same host array gets the buffer it used last time, and otherwise any free
//! buffer of the same length, so steady-state calls allocate no device memory.
//! Contents are still uploaded on every call, as in Python; there is no change
//! detection. `resident` parameters do not go through this cache: they take a
//! device buffer and are never uploaded (see [`CallArg::Buffer`](crate::CallArg::Buffer)).
//!
//! A buffer is checked out of the cache for the duration of a call, so two
//! concurrent calls never share one. Asynchronous calls do not use the cache
//! (their buffers may still be in use when the call returns). The cache is
//! bounded: `YAPOCIS_BUFFER_CACHE_MB` (default 512, `0` disables it) and at
//! most [`MAX_ENTRIES`] buffers, least recently used evicted first.

use crate::raw::RawBuffer;
use std::sync::{Arc, Mutex};

pub const MAX_ENTRIES: usize = 64;
const DEFAULT_BUDGET_MB: usize = 512;

/// Counters for [`Device::buffer_cache_stats`](crate::Device::buffer_cache_stats).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BufferCacheStats {
    /// Buffers currently cached (not checked out).
    pub entries: usize,
    /// Bytes held by cached buffers.
    pub bytes: usize,
    /// Reuses of the buffer bound to the same host array.
    pub bound_hits: u64,
    /// Reuses of another free buffer of the same length.
    pub size_hits: u64,
    /// Requests that had to allocate.
    pub misses: u64,
    /// Buffers dropped to stay within the budget.
    pub evictions: u64,
}

struct Entry {
    raw: Arc<dyn RawBuffer>,
    len: usize,
    host: Option<usize>,
    tick: u64,
}

#[derive(Default)]
struct Inner {
    entries: Vec<Entry>,
    tick: u64,
    stats: BufferCacheStats,
}

pub(crate) struct BufferCache {
    budget: usize,
    inner: Mutex<Inner>,
}

impl BufferCache {
    pub(crate) fn from_env() -> BufferCache {
        let mb = std::env::var("YAPOCIS_BUFFER_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_BUDGET_MB);
        BufferCache::with_budget(mb << 20)
    }

    pub(crate) fn with_budget(budget: usize) -> BufferCache {
        BufferCache {
            budget,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Check out a buffer of exactly `len` bytes: the one last used with the
    /// host array at `host`, else the most recently used free one.
    pub(crate) fn take(&self, len: usize, host: Option<usize>) -> Option<Arc<dyn RawBuffer>> {
        if self.budget == 0 {
            return None;
        }
        let mut inner = self.inner.lock().unwrap();
        let bound = host.and_then(|h| {
            inner
                .entries
                .iter()
                .position(|e| e.len == len && e.host == Some(h))
        });
        let found = bound.or_else(|| {
            inner
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.len == len)
                .max_by_key(|(_, e)| e.tick)
                .map(|(i, _)| i)
        });
        match found {
            Some(i) => {
                if bound.is_some() {
                    inner.stats.bound_hits += 1;
                } else {
                    inner.stats.size_hits += 1;
                }
                let e = inner.entries.swap_remove(i);
                inner.stats.entries -= 1;
                inner.stats.bytes -= e.len;
                Some(e.raw)
            }
            None => {
                inner.stats.misses += 1;
                None
            }
        }
    }

    /// Return a buffer after a call, bound to the host array at `host` (if
    /// any). Buffers still referenced elsewhere are not cached.
    pub(crate) fn give(&self, raw: Arc<dyn RawBuffer>, host: Option<usize>) {
        let len = raw.len_bytes();
        if self.budget == 0 || len > self.budget || Arc::strong_count(&raw) != 1 {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;
        // One binding per host array: a stale entry for the same address
        // becomes a plain free buffer.
        if let Some(h) = host {
            for e in inner.entries.iter_mut().filter(|e| e.host == Some(h)) {
                e.host = None;
            }
        }
        inner.entries.push(Entry {
            raw,
            len,
            host,
            tick,
        });
        inner.stats.entries += 1;
        inner.stats.bytes += len;
        while inner.stats.bytes > self.budget || inner.entries.len() > MAX_ENTRIES {
            let oldest = inner
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.tick)
                .map(|(i, _)| i)
                .expect("non-empty while over budget");
            let e = inner.entries.swap_remove(oldest);
            inner.stats.entries -= 1;
            inner.stats.bytes -= e.len;
            inner.stats.evictions += 1;
        }
    }

    /// Drop every cached buffer (Python's `purge`).
    pub(crate) fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
        inner.stats.entries = 0;
        inner.stats.bytes = 0;
    }

    pub(crate) fn stats(&self) -> BufferCacheStats {
        self.inner.lock().unwrap().stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    struct Fake(usize);
    impl RawBuffer for Fake {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn len_bytes(&self) -> usize {
            self.0
        }
    }

    fn id(raw: &Arc<dyn RawBuffer>) -> usize {
        Arc::as_ptr(raw) as *const u8 as usize
    }

    #[test]
    fn bound_buffer_preferred_then_same_size() {
        let c = BufferCache::with_budget(1 << 20);
        let a: Arc<dyn RawBuffer> = Arc::new(Fake(100));
        let b: Arc<dyn RawBuffer> = Arc::new(Fake(100));
        let (ia, ib) = (id(&a), id(&b));
        c.give(a, Some(0x1000));
        c.give(b, Some(0x2000));
        assert_eq!(id(&c.take(100, Some(0x1000)).unwrap()), ia);
        // no binding for 0x3000: the most recent free buffer of that size
        assert_eq!(id(&c.take(100, Some(0x3000)).unwrap()), ib);
        assert!(c.take(100, None).is_none());
        assert!(c.take(50, None).is_none());
        let s = c.stats();
        assert_eq!((s.bound_hits, s.size_hits, s.misses), (1, 1, 2));
    }

    #[test]
    fn budget_and_sharing() {
        let c = BufferCache::with_budget(250);
        for _ in 0..3 {
            c.give(Arc::new(Fake(100)), None);
        }
        let s = c.stats();
        assert_eq!((s.entries, s.bytes, s.evictions), (2, 200, 1));
        let shared: Arc<dyn RawBuffer> = Arc::new(Fake(10));
        let keep = shared.clone();
        c.give(shared, None);
        assert_eq!(
            c.stats().entries,
            2,
            "a buffer still referenced is not cached"
        );
        drop(keep);
        c.clear();
        assert_eq!(c.stats().bytes, 0);
        assert!(BufferCache::with_budget(0).take(1, None).is_none());
    }
}
