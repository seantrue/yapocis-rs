---
title: "rustcpu vs numbax on yellowfin"
date: 2026-11-08
---

# Summary

**After the three fixes below (branch `rustcpu-perf`, not yet committed), rustcpu on yellowfin is 2.57× numbax with all 80 threads (1.38× before) and 0.74× on one thread (0.56× before).** The fixes made rustcpu itself 1.86× faster on 80 threads and 1.32× faster on one thread (geometric means). See "After the fixes" at the end; the sections before it describe the first run.

The benchmark from `PERF_NUMBAX_VS_CPU.md` (13 yapocis library operations, the same shapes and parameters on both sides) was repeated on yellowfin, a dual Xeon E5-2698 v4. rustcpu was the merged yapocis-rs `abdd32b`.

| Comparison | rustcpu vs numbax, 1 thread | rustcpu vs numbax, all threads |
|---|---:|---:|
| yellowfin (80 threads) | 0.56× | 1.38× |
| M4 Max (14 threads) | 0.82× | 2.77× |

- **rustcpu still wins overall with every thread in use, by less than on the M4.** Its biggest leads are where numbax drops to NumPy (mandelbrot 52×, FFT 5.8×) or runs a serial Numba loop (hsi 4.8×, gradient 1.9×, zcs 1.9×, anisodiff 1.7×).
- **numbax wins its own parallel Numba kernels on yellowfin:** simplex2d (rustcpu 0.60×) and simplex3d (0.83×). median3x3 and gaussian are close. Numba's `prange` reaches 23–31× on 80 threads; rustcpu reaches 10–24× on the same cases.
- **rustcpu's small memory-bound kernels are much slower than numbax's on this CPU:** add (0.13×), colormath (0.25×) and the histogram (0.21×), even with all threads.
- **On one thread the per-thread gap is wider than on the M4.** For example, add takes 11.8 ms here against 2.44 ms on the M4, and the histogram 8.1 ms against 1.0 ms. That is more than the clock difference explains (2.2 GHz Broadwell against the M4's performance cores).

# Setup

| Item | Value |
|---|---|
| Machine | yellowfin: 2 × Xeon E5-2698 v4 (20 cores, 40 threads each; 80 threads), 2.2 GHz, Linux |
| Load | Load average 3.0 before, 3.6 after; the production render workers rust-1 and rust-2 were running |
| numbax | Yapocis3 checkout on yellowfin (`29f8403` plus an unrelated uncommitted CUDA template change), Python 3.13.7, NumPy 2.3.3, Numba 0.62.1 |
| rustcpu | yapocis-rs `abdd32b` (merged PR #16), in a separate worktree, `--release`, rayon |
| Timing | 2 warm-up calls, then 5 to 30 timed calls within 2 s, the median call; each configuration run 3 times (interleaved) and the median of the three taken |
| Threads | numbax `NUMBA_NUM_THREADS` 1 or 80 (default); rustcpu `RAYON_NUM_THREADS` 1 or 80 (default) |

# yellowfin

Milliseconds per call, lower is better. The last two columns are numbax time divided by rustcpu time: above 1× means rustcpu is faster.

| Case | numbax runs | numbax, 1 thread (ms) | numbax, 80 threads (ms) | rustcpu, 1 thread (ms) | rustcpu, 80 threads (ms) | rustcpu vs numbax, 1 thread | rustcpu vs numbax, 80 threads |
|------------------------------------|--------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| operators.add 1024² | numba | 0.405 | 0.423 | 11.8 | 3.21 | 0.03× | 0.13× |
| gaussian.gauss_image 1024², scale 2 | numba, parallel | 233.7 | 18.8 | 108.0 | 10.4 | 2.16× | 1.80× |
| median3x3 1024², 5 iterations | numba, parallel | 356.1 | 11.5 | 102.8 | 10.3 | 3.46× | 1.12× |
| anisodiff_n 1024², 10 iterations | numba | 64.4 | 65.1 | 240.4 | 37.6 | 0.27× | 1.73× |
| zcs 1024² | numpy | 9.81 | 9.70 | 33.2 | 5.01 | 0.30× | 1.94× |
| gradient 1024², stride 3 | numba | 27.8 | 28.4 | 79.2 | 14.6 | 0.35× | 1.94× |
| hsi.rgb2hsi3 1024²×3 | numba + numpy | 38.9 | 38.4 | 33.1 | 8.04 | 1.18× | 4.78× |
| colormath srgb_d65.rgb2xyz 1024²×3 | numba | 1.91 | 1.89 | 16.6 | 7.58 | 0.11× | 0.25× |
| mandelbrot 1024², 100 iterations | numpy | 364.9 | 346.2 | 74.6 | 6.66 | 4.89× | 52.02× |
| simplex2d 1024², 4 octaves | numba, parallel | 128.9 | 5.54 | 154.1 | 9.24 | 0.84× | 0.60× |
| simplex3d 128³, 4 octaves | numba, parallel | 488.5 | 18.6 | 530.1 | 22.4 | 0.92× | 0.83× |
| fft.batch_forward 4096 × 256 | numpy (pocketfft) | 51.3 | 50.8 | 65.0 | 8.76 | 0.79× | 5.80× |
| imageops.histogram 1024², 256 bins | numba | 1.67 | 1.73 | 8.08 | 8.07 | 0.21× | 0.21× |
| **Geometric mean** | | | | | | **0.56×** | **1.38×** |

# M4 Max, for comparison

The same benchmark on the M4 Max, earlier on 2026-11-08 (`PERF_NUMBAX_VS_CPU.md`), relabelled with the backend's final name.

| Case | numbax runs | numbax, 1 thread (ms) | numbax, 14 threads (ms) | rustcpu, 1 thread (ms) | rustcpu, 14 threads (ms) | rustcpu vs numbax, 1 thread | rustcpu vs numbax, 14 threads |
|------------------------------------|--------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| operators.add 1024² | numba | 0.107 | 0.107 | 2.44 | 0.540 | 0.04× | 0.20× |
| gaussian.gauss_image 1024², scale 2 | numba, parallel | 42.7 | 7.62 | 23.5 | 2.85 | 1.82× | 2.67× |
| median3x3 1024², 5 iterations | numba, parallel | 167.8 | 20.1 | 39.1 | 4.50 | 4.30× | 4.48× |
| anisodiff_n 1024², 10 iterations | numba | 20.6 | 20.6 | 57.9 | 6.94 | 0.36× | 2.97× |
| zcs 1024² | numpy | 17.1 | 17.2 | 15.2 | 1.70 | 1.13× | 10.10× |
| gradient 1024², stride 3 | numba | 5.93 | 5.91 | 22.9 | 2.54 | 0.26× | 2.33× |
| hsi.rgb2hsi3 1024²×3 | numba + numpy | 6.16 | 6.14 | 4.27 | 0.952 | 1.44× | 6.45× |
| colormath srgb_d65.rgb2xyz 1024²×3 | numba | 0.601 | 0.603 | 2.56 | 0.748 | 0.23× | 0.81× |
| mandelbrot 1024², 100 iterations | numpy | 217.9 | 218.4 | 37.1 | 3.96 | 5.88× | 55.20× |
| simplex2d 1024², 4 octaves | numba, parallel | 32.1 | 4.22 | 31.8 | 3.42 | 1.01× | 1.23× |
| simplex3d 128³, 4 octaves | numba, parallel | 156.5 | 19.4 | 134.2 | 13.1 | 1.17× | 1.48× |
| fft.batch_forward 4096 × 256 | numpy (pocketfft) | 14.4 | 14.4 | 14.6 | 1.97 | 0.98× | 7.31× |
| imageops.histogram 1024², 256 bins | numba | 0.925 | 0.916 | 1.04 | 1.04 | 0.89× | 0.88× |
| **Geometric mean** | | | | | | **0.82×** | **2.77×** |

# Scaling and per-thread speed

The single-thread columns compare rustcpu's own speed on the two machines. The speedup columns are 1-thread time divided by all-threads time.

| Case | rustcpu M4, 1 thread (ms) | rustcpu yellowfin, 1 thread (ms) | rustcpu speedup M4, 14 threads | rustcpu speedup yellowfin, 80 threads | numbax speedup yellowfin, 80 threads |
|------------------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|
| operators.add 1024² | 2.44 | 11.8 | 4.5× | 3.7× | 1.0× |
| gaussian.gauss_image 1024², scale 2 | 23.5 | 108.0 | 8.2× | 10.4× | 12.4× |
| median3x3 1024², 5 iterations | 39.1 | 102.8 | 8.7× | 10.0× | 31.0× |
| anisodiff_n 1024², 10 iterations | 57.9 | 240.4 | 8.3× | 6.4× | 1.0× |
| zcs 1024² | 15.2 | 33.2 | 8.9× | 6.6× | 1.0× |
| gradient 1024², stride 3 | 22.9 | 79.2 | 9.0× | 5.4× | 1.0× |
| hsi.rgb2hsi3 1024²×3 | 4.27 | 33.1 | 4.5× | 4.1× | 1.0× |
| colormath srgb_d65.rgb2xyz 1024²×3 | 2.56 | 16.6 | 3.4× | 2.2× | 1.0× |
| mandelbrot 1024², 100 iterations | 37.1 | 74.6 | 9.4× | 11.2× | 1.1× |
| simplex2d 1024², 4 octaves | 31.8 | 154.1 | 9.3× | 16.7× | 23.3× |
| simplex3d 128³, 4 octaves | 134.2 | 530.1 | 10.3× | 23.7× | 26.3× |
| fft.batch_forward 4096 × 256 | 14.6 | 65.0 | 7.4× | 7.4× | 1.0× |
| imageops.histogram 1024², 256 bins | 1.04 | 8.08 | 1.0× | 1.0× | 1.0× |

# Why rustcpu scales and runs as it does

- **Serial work in every call.** `Program::call` uploads each input into a device buffer, zero-fills the output and copies it back. On the CPU backend those are single-threaded memory copies. At 1024², with little work per element, they bound the speedup (Amdahl's law): add and colormath reach only 2–4×, and the histogram, sequential by design, none.
- **Per-element cost of the argument views.** `Input`/`Output` re-check a byte-to-element cast on every element access, which also blocks vectorisation. On the M4, fixing it (one central change in `cpu/args.rs`), plus a generic closure in `operators.rs`, took `add` from 2.24 ms to 0.49 ms on one thread. The Xeon's slower per-access cost and narrower memory path make that worse here.
- **Compute-bound kernels scale as expected:** simplex3d 24×, simplex2d 17×, mandelbrot 11×. At this size rayon still loses some ground to Numba's `prange`.

# What would improve it, in order of effect per change

1. **Take each typed slice once in `cpu/args.rs`**, and use generic closures where ports take function pointers. This is one central change, with no port changes. It mainly helps one-thread and memory-bound cases.
2. **No host copies on the CPU backend.** Let `Program::call` hand host arrays to the kernels directly instead of uploading and downloading them. This is a larger change to the shared call path, and it raises the parallel ceiling of every kernel.
3. **A parallel histogram** from per-thread partial histograms added in a fixed order. The counts are integers, so the result stays exact.

All three were then made at the maintainer's request ("do all three fixes and rerun the affected code on yellowfin"); see below.

# Reproducing

The scripts are in `claude-debug/` (`perf_numbax_vs_cpu.py`, `perf_numbax_vs_cpu_rs/`, `perf_yellowfin_run.sh`, `perf_numbax_vs_cpu_table.py`). On yellowfin they ran from a detached worktree of `~/workspace/yapocis-rs` at `~/workspace/yapocis-rs-perf` (`origin/main`, `abdd32b`), building into the clone's existing `target`. The raw JSON is in `claude-debug/perf_yellowfin/`.

# After the fixes

## What changed (yapocis-rt, branch `rustcpu-perf`)

1. **Argument views take their slice once.** `Input` and `Output` hold the typed slice they serve, computed when they are created, and the buffer's lock guard. Element access is an ordinary slice index. The operator and graphics ports take generic closures instead of `fn` pointers.
2. **No host copies on the CPU backend.**
   - **The mechanism:** `RawDevice` gains `runs_on_host()` and `wrap_host()`. Both are off by default, so every GPU backend is unchanged.
   - **On the CPU:** for a synchronous call, `Program::call` wraps the caller's arrays in place, and writes each `out_like` result straight into the array it returns. Nothing is uploaded, zeroed twice or read back.
   - **An `in` array** is wrapped read-only. A kernel that writes an `in` parameter (yapocis-external's `fill`) gets a private copy, which is discarded, as the GPU's temporary buffer was.
   - **Fallback:** an array that isn't 8-byte aligned is copied as before.
3. **Parallel histogram.** Each 16K-element chunk is counted into its own partial histogram in parallel (`par::map_each`), and the partials are added in chunk order. The counts are integers, so the result is exact.

**Checks:**
- `cargo test --release -p yapocis --features rustcpu`: 72 passed.
- The workspace on OpenCL, Metal and rustcpu: all pass.
  - **One adjusted test:** `yapocis-rt`'s buffer-cache test now expects the CPU backend not to use the cache. Its checks on aliased arguments and in-place `inout` writes pass through the new path.
- impressionist-rs yapocis-external against this runtime: 47 passed.
- Clippy is clean, including the bindings.

## rustcpu before and after, on yellowfin

rustcpu rerun alone, three runs of each configuration. Load average 1.6 before and 4.4 after; most of that is the benchmark itself.

| Case | 1 thread before (ms) | 1 thread after (ms) | Gain | 80 threads before (ms) | 80 threads after (ms) | Gain |
|------------------------------------|-----------:|-----------:|-------:|-----------:|-----------:|-------:|
| operators.add 1024² | 11.8 | 2.05 | 5.77× | 3.21 | 1.32 | 2.44× |
| gaussian.gauss_image 1024², scale 2 | 108.0 | 108.6 | 0.99× | 10.4 | 9.93 | 1.05× |
| median3x3 1024², 5 iterations | 102.8 | 99.3 | 1.04× | 10.3 | 10.6 | 0.97× |
| anisodiff_n 1024², 10 iterations | 240.4 | 198.6 | 1.21× | 37.6 | 26.3 | 1.43× |
| zcs 1024² | 33.2 | 30.9 | 1.07× | 5.01 | 2.86 | 1.75× |
| gradient 1024², stride 3 | 79.2 | 70.7 | 1.12× | 14.6 | 5.09 | 2.88× |
| hsi.rgb2hsi3 1024²×3 | 33.1 | 25.6 | 1.29× | 8.04 | 3.79 | 2.12× |
| colormath srgb_d65.rgb2xyz 1024²×3 | 16.6 | 9.55 | 1.74× | 7.58 | 3.34 | 2.27× |
| mandelbrot 1024², 100 iterations | 74.6 | 68.1 | 1.09× | 6.66 | 4.52 | 1.47× |
| simplex2d 1024², 4 octaves | 154.1 | 153.5 | 1.00× | 9.24 | 7.12 | 1.30× |
| simplex3d 128³, 4 octaves | 530.1 | 525.9 | 1.01× | 22.4 | 19.3 | 1.16× |
| fft.batch_forward 4096 × 256 | 65.0 | 60.0 | 1.08× | 8.76 | 5.88 | 1.49× |
| imageops.histogram 1024², 256 bins | 8.08 | 5.28 | 1.53× | 8.07 | 0.703 | 11.49× |
| **Geometric mean** | | | **1.32×** | | | **1.86×** |

## Against numbax, after the fixes

numbax's times are from the first run (its code did not change).

| Case | numbax runs | numbax, 1 thread (ms) | numbax, 80 threads (ms) | rustcpu, 1 thread (ms) | rustcpu, 80 threads (ms) | rustcpu vs numbax, 1 thread | rustcpu vs numbax, 80 threads |
|------------------------------------|--------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| operators.add 1024² | numba | 0.405 | 0.423 | 2.05 | 1.32 | 0.20× | 0.32× |
| gaussian.gauss_image 1024², scale 2 | numba, parallel | 233.7 | 18.8 | 108.6 | 9.93 | 2.15× | 1.89× |
| median3x3 1024², 5 iterations | numba, parallel | 356.1 | 11.5 | 99.3 | 10.6 | 3.58× | 1.08× |
| anisodiff_n 1024², 10 iterations | numba | 64.4 | 65.1 | 198.6 | 26.3 | 0.32× | 2.48× |
| zcs 1024² | numpy | 9.81 | 9.70 | 30.9 | 2.86 | 0.32× | 3.39× |
| gradient 1024², stride 3 | numba | 27.8 | 28.4 | 70.7 | 5.09 | 0.39× | 5.58× |
| hsi.rgb2hsi3 1024²×3 | numba + numpy | 38.9 | 38.4 | 25.6 | 3.79 | 1.52× | 10.13× |
| colormath srgb_d65.rgb2xyz 1024²×3 | numba | 1.91 | 1.89 | 9.55 | 3.34 | 0.20× | 0.56× |
| mandelbrot 1024², 100 iterations | numpy | 364.9 | 346.2 | 68.1 | 4.52 | 5.36× | 76.61× |
| simplex2d 1024², 4 octaves | numba, parallel | 128.9 | 5.54 | 153.5 | 7.12 | 0.84× | 0.78× |
| simplex3d 128³, 4 octaves | numba, parallel | 488.5 | 18.6 | 525.9 | 19.3 | 0.93× | 0.96× |
| fft.batch_forward 4096 × 256 | numpy (pocketfft) | 51.3 | 50.8 | 60.0 | 5.88 | 0.85× | 8.63× |
| imageops.histogram 1024², 256 bins | numba | 1.67 | 1.73 | 5.28 | 0.703 | 0.32× | 2.46× |
| **Geometric mean** | | | | | | **0.74×** | **2.57×** |

## What is left

- **Compute-bound kernels did not move:** gaussian, median3x3, simplex. Numba's `prange` still scales better on them at this size: median3x3 is 31× on 80 threads against rustcpu's 9×, and simplex2d 23× against 22×.
- **The smallest kernels:** add (0.32×) and colormath (0.56×) still trail numbax. Each call still takes a lock, wraps buffers and starts a parallel loop for about a millisecond of memory traffic, and rayon's per-element closure blocks some of the vectorisation Numba gets.
