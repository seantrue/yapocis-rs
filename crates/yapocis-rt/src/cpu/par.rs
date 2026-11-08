//! The two loop shapes CPU kernels run in parallel.
//!
//! Both hand each output element (or each fixed-size chunk of outputs) to
//! exactly one call of `f`, and `f` only reads shared inputs, so there is
//! nothing to synchronise: the result is the same as the sequential loop.
//! A kernel whose work items write anywhere else (a histogram bin, a scatter
//! write, an atomic) uses a plain `for` loop instead.
//!
//! The thread pool is rayon's global pool (`RAYON_NUM_THREADS` sets its size).

use rayon::prelude::*;

/// Below this many elements a loop runs on one thread.
const MIN_PER_THREAD: usize = 4096;

/// `out[i] = f(i)` for every `i`, in parallel.
pub fn map<T: Send>(out: &mut [T], f: impl Fn(usize) -> T + Sync) {
    out.par_iter_mut()
        .with_min_len(MIN_PER_THREAD)
        .enumerate()
        .for_each(|(i, o)| *o = f(i));
}

/// `f(k, chunk)` for the `k`th run of `n` elements of `out` (one pixel's
/// channels, one row, ...), in parallel. The last chunk may be shorter.
pub fn map_chunks<T: Send>(out: &mut [T], n: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let min = (MIN_PER_THREAD / n.max(1)).max(1);
    out.par_chunks_mut(n.max(1))
        .with_min_len(min)
        .enumerate()
        .for_each(|(k, c)| f(k, c));
}

/// `(a[i], b[i]) = f(i)` for every `i`, in parallel: one work item writing
/// one element of each of two outputs.
pub fn map2<A: Send, B: Send>(a: &mut [A], b: &mut [B], f: impl Fn(usize) -> (A, B) + Sync) {
    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .with_min_len(MIN_PER_THREAD)
        .enumerate()
        .for_each(|(i, (x, y))| (*x, *y) = f(i));
}
