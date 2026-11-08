---
title: "numbax vs the Rust CPU backend"
date: 2026-11-08
---

Wall-clock time for 13 yapocis library operations, on the same shapes and parameters, run two ways:

- through Yapocis3's numbax backend (Python, Numba and NumPy);
- through yapocis-rs's new CPU backend (Rust ports of the OpenCL kernels, rayon for parallel loops).

## Summary

- With every core in use, the CPU backend is faster in 10 of 13 cases: **2.8× geometric mean**, from 1.2× to 55×.
- Its largest leads are where numbax drops to NumPy (mandelbrot 55×, zcs 10×, FFT 7×) or runs a serial Numba loop (hsi 6.5×, anisodiff 3.0×, gradient 2.3×).
- Against numbax kernels that are themselves parallel Numba (gaussian, median3x3, simplex), it is 1.2× to 4.5× faster.
- On one thread each, the CPU backend is **0.82× geometric mean**, which is slower.
  - Its compute-heavy kernels match or beat numbax.
  - Its small memory-bound kernels are 3× to 23× slower: add, colormath, gradient and anisodiff.
- Most of that single-thread gap is overhead in the CPU backend's argument views, not in the kernels' arithmetic (see Diagnosis). A central fix in `cpu/args.rs` would recover most of it.
- Three cases are slower on the CPU backend even with every core:
  - `operators.add`, 0.20×;
  - colormath, 0.81×;
  - the histogram, 0.88×, which is sequential by design.

## Setup

| Item | Value |
|---|---|
| Machine | Apple M4 Max, 14 cores (10 performance, 4 efficiency), 36 GB, macOS |
| numbax | Yapocis3 7e9e943, Python 3.13.7, NumPy 2.3.3, Numba 0.62.1 |
| CPU backend | yapocis-rs 351e7d2 plus the uncommitted `worktree-cpu-backend` changes, rustc 1.99.0, `--release`, rayon |
| Data | 1024 × 1024 float32 images (× 3 channels for hsi and colormath); mandelbrot on the same grid both sides; random inputs of the same shape and distribution |
| Timing | 2 warm-up calls (the first compiles the Numba kernel), then 5 to 30 timed calls within 2 s; the median call |
| Repeats | Each configuration run 3 times; the table shows the median of the three medians |
| Threads | numbax: `NUMBA_NUM_THREADS` 1 or 14 (default); CPU backend: `RAYON_NUM_THREADS` 1 or 14 (default) |

## Results

Times are in milliseconds per call, lower is better. The last two columns are numbax time divided by CPU time, so above 1× means the CPU backend is faster. "numbax runs" says what the numbax backend executes for that case.

| Case | numbax runs | numbax, 1 thread (ms) | numbax, 14 threads (ms) | CPU, 1 thread (ms) | CPU, 14 threads (ms) | CPU vs numbax, 1 thread | CPU vs numbax, 14 threads |
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

## Reading the results

- **numbax uses more than one core only where a kernel is parallel Numba.** That is gaussian, median3x3 and simplex. Its other cases take the same time at 1 and 14 threads, and the CPU backend's lead in those cases is mostly parallelism.
- **On one thread, the compute-bound kernels are even or ahead.** simplex, FFT and zcs are within 15% either way. The CPU backend is faster where numbax does extra work:
  - median3x3 4.3×: numbax sorts each neighbourhood, where the port does the kernel's partial selection;
  - mandelbrot 5.9×: NumPy keeps iterating points that have already escaped;
  - gaussian 1.8×: numbax's vertical pass transposes the image twice.
- **On one thread, the small memory-bound kernels are slower.** These are add, colormath, gradient and anisodiff. Each does a few operations per element, so the per-element overhead described below dominates.
- **The histogram is sequential on the CPU backend by design.** Work items share bins. It is within 12% of numbax's serial Numba loop.
- **The numbax kernels are not the same computation.** For example, numbax's 1-D convolution reflects at the edges where OpenCL writes zeros, and its division adds 1e-20. The table compares cost, not identical results.

## Diagnosis of the single-thread gap

I measured `operators.add` (1024², one thread) through three temporary changes, applied one after another and all reverted afterwards. The branch is unchanged.

| Variant | Time (ms) | Change |
|-------------------------------------------------|---------:|-----------------------------------------------|
| numbax `add` (serial Numba) | 0.107 | |
| CPU `add`, host arrays (as benchmarked) | 2.44 | |
| CPU `add_res`, resident buffers (no host copies) | 2.24 | the copies are about 0.2 ms on one thread |
| ... plus taking `&[f32]` from each `Input`/`Output` once | 0.78 | 2.9× faster |
| ... plus a generic closure instead of a `fn` pointer in `operators.rs` | 0.49 | 1.6× faster |
| ... plus chunked `par::map` instead of rayon's per-element iterator | 0.57 | no gain (kept out) |

**`Input`/`Output` views.**

- Their `Deref` recomputes a checked `bytemuck::cast_slice` on every element access.
- The kernels index the view directly (`input1[i]`), so each access pays that check, and the loop can't be vectorised.
- This affects every port, and it is most visible in kernels that do little work per element.
- Fix: one central change in `cpu/args.rs`. Compute the typed slice once when the view is created, and hand it out from `Deref`. No kernel changes are needed.

**`fn` pointers in operators.** `pixelwise` takes its operation as a `fn(f32, f32) -> f32`, which can't be inlined, so each element is an indirect call. Taking `impl Fn` fixes it.

**Host copies.**

- With 14 threads, the host-array add takes 0.54 ms and the resident-buffer add 0.28 ms. About half of a small kernel's time is spent:
  - uploading inputs into "device" buffers;
  - zero-filling the output buffer;
  - copying the output back.
- That is the GPU calling convention. On the CPU backend it is pure overhead.
- Letting `Program::call` bind host arrays directly when the backend is the CPU would remove it. That is a larger change to the shared call path.

**Histogram.**

- Each thread could fill its own partial histogram, and the partials could be added in a fixed order.
- The counts are integers, so the result would be exact and deterministic.
- This is parallelism that's easy to reason about, if the histogram matters.

## Caveats

- **Multi-thread timings vary.** A single run swung by up to 2× on a few cases (gradient, hsi, gaussian, median3x3), most likely from scheduling across performance and efficiency cores. Each configuration was therefore run 3 times and the median taken. Single-thread runs agreed within a few percent.
- **The numbax timings include the Python call path.** That is argument preparation and output allocation, as a Python caller would see it. The CPU timings include the Rust library call path, host copies included.
- **These figures hold for this machine.** An x86-64 host (other SIMD widths, no efficiency cores) may rank the cases differently.

## Reproducing

All files are in `claude-debug/` of the `cpu-backend` worktree.

```bash
# numbax (Yapocis3's venv); again with NUMBA_NUM_THREADS=1, writing ..._numbax_1.json
/path/to/yapocis3/.venv/bin/python -I perf_numbax_vs_cpu.py perf_numbax_vs_cpu_numbax.json

# CPU backend; again with RAYON_NUM_THREADS=1, writing ..._cpu_1.json
cd perf_numbax_vs_cpu_rs && CARGO_TARGET_DIR=../../target cargo build --release && cd ..
../target/release/perf_numbax_vs_cpu_rs perf_numbax_vs_cpu_cpu_all.json

# Repeat each as ..._run2.json and ..._run3.json, then:
python3 perf_numbax_vs_cpu_table.py
```
