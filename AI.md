# AI Contributions — yapocis-rs

*About AI slop.* Fair enough: nearly all of the code in this repository was written by a machine, and so was most of this file. Call it instant ramen. It is fast, it is cheap, it is genuinely edible, and nobody should pretend it simmered on a stove all afternoon. The honest answer is not a claim about quality but the record below: what the maintainer decided, what the model wrote, what the model got wrong, and which of those mistakes were caught by the parity tests replaying Python's recorded outputs, by running on real GPUs on two machines, or by the maintainer rather than by luck. Where AI-written code had a defect, it is written down here in the same detail as the wins. Bon appétit, but mind the sodium content.

Tracks what AI helpers contributed to yapocis-rs versus the human maintainer.

## Human maintainer

The maintainer (Sean True) owns yapocis, the Python package whose kernels this repository runs, and set the direction and the design constraints of the port. The Rust code itself was written by Claude from the start; there is no pre-AI code in this repository. The maintainer's decisions (2026-10-18, before any code was written):

- The Rust code lives in a separate repository, consuming a kernel bundle that the Python package exports at build time.
- Runtime-parameterised kernels (arbitrary convolution weights) are handled by enumerating variants at build time.
- The first release covers all interface kernels plus the library wrappers.
- CUDA goes through NVRTC via `cudarc`, pinned to the CUDA 12.x bindings so the Pascal P40 on the Linux host keeps working.

Later direction and corrections are credited in the entries below.

## Contributions

### 2026-11-09 — A software renderer (Windows' WARP) is no longer taken for a GPU (Claude Opus 5.5)

**What the maintainer asked for.** While virtimp_com_app's desktop app was being brought up on a Windows 11 ARM64 VM, the maintainer asked whether it was now "finding an OpenCL engine, and if so, if it's slower", then said "Yes" to fixing the choice here, on a branch, and testing it there.

**What happened.** The VM has no GPU driver. Microsoft's OpenCL, OpenGL and Vulkan Compatibility Pack (installed for the app's OpenGL window) exposes Windows' software rasterizer as an OpenCL device named "Microsoft Basic Render Driver" (WARP, via OpenCLOn12). `Device::default` takes the first backend that opens, and OpenCL comes before rustcpu, so it chose WARP. A Pen and Ink #1 render that rustcpu did in 35.7 s had not finished on WARP after about 8 minutes at 100% of 4 vCPUs; the app's window starved meanwhile.

**What the AI did** (~85 LOC of code, ~110 of tests):
- `RawDevice::is_software_renderer` (default false); the OpenCL device answers by name. The list of names is one constant in `device.rs`, holding WARP's.
- `Device::default` takes candidates through `first_hardware`: the first device that opens and isn't a software renderer wins; a software one is kept only as a last resort, so a build without rustcpu still renders (slowly) rather than finding nothing. Candidates are opened lazily, so nothing after the chosen device is opened. `YAPOCIS_BACKEND` and `YAPOCIS_DEVICE` still choose exactly what they name, WARP included.
- Within OpenCL, the preference order (ocl/device.py's) is now `preferred_device`, a pure function, and WARP is never one of its picks while another device exists. OpenCLOn12 on a machine that does have a GPU (a Snapdragon laptop, say) can list WARP beside the GPU's adapter, and the old "first non-CPU device" rule could have taken WARP there. Otherwise the order is unchanged, including "else the last device".
- 10 unit tests: the name match; `first_hardware` (hardware after software wins, software when nothing else opens, the last error when nothing opens, laziness); and `preferred_device` (vendor GPU first, an unnamed GPU over WARP listed first, a CPU device over WARP, WARP alone left to the last-device rule, the old last-device rule otherwise).

**Checked.**
- Mac: `cargo fmt`; clippy clean with `-D warnings` as CLAUDE.md gives it (`--features yapocis/ocl,yapocis/metal`) and for `yapocis-rt` with `ocl,rustcpu,metal`; `cargo test --release -p yapocis-rt --lib --features ocl,rustcpu,metal`: 13 passed (the 10 new ones among them).
- ARM64 VM, compatibility pack installed, no `YAPOCIS_BACKEND`: virtimp_com_app's desktop app built against this branch (all five yapocis-rs crates patched to a local copy) chose `rustcpu: cpu` and rendered Pen and Ink #1 in 4.9 s.
- Not checked: a machine with both a real GPU and WARP in its OpenCL list (covered only by the unit test), and the full GPU/parity suites, which this doesn't touch.

**Ruled out.** `cargo clippy -p yapocis-rt --all-targets` with no backend feature fails with 4 errors (unreachable statement, unused `mut` and `selector`, `select` unused). They are the same on `main` before this change; that configuration isn't one CLAUDE.md builds.

### 2026-11-08 — Python bindings: embeddable, and NumPy arrays used in place (Claude Opus 5.5)

The maintainer asked to "Bring ../impressionist forward to new crates", chose "Add the rustcpu engine", and then chose a wheel built from impressionist-rs that carries yapocis_external's ports too. Later: "Fix the GPU bugs, too".

**What the AI did** (~130 LOC changed):
- **Embeddable bindings.** `bindings/python` (`yapocis-py`) is also an rlib, with `add_to_module`. Its own module entry point is behind a default-on `module` feature. An extension module that also registers its own CPU ports can therefore build on it: impressionist-rs's `bindings/python` does, for yapocis_external. The two must share one yapocis-rt, so both come from the same yapocis-rs revision.
- **NumPy arrays used in place.** The bindings now treat a NumPy array as its own device buffer, as numbax does and as a GPU's buffer bound to the array does. Each array is wrapped in place, writable, through the new public `BufferAny::wrap_host`, whatever the parameter's role, so a kernel's write to an `in` parameter persists. An array passed twice is one buffer; read-only arrays are never written; an array the backend can't use in place is copied in and back.
- **Why:** yapocis_external's `label` interface declares `first`'s output as `in`, and Virtual Impressionist relies on the kernel's write to it persisting, as it does on a GPU and on numbax. Before this change, VI's two label tests produced empty pictures on rustcpu.
- **README:** how to embed the bindings.

**Verified (M4 MacBook):**
- **yapocis-rs:** workspace tests (OpenCL, Metal, rustcpu) without npexact, 90 passed; `-p npexact`, 28 passed.
  - In a combined run the five npexact fixture tests failed. Their test binaries had been built from another session's scratch copy of yapocis-rs (`/private/tmp/.../rsmerge/yapocis-rs`), which shares this target directory, and they looked for fixtures under that copy's path. Built here, they pass.
- **Yapocis3** with the rebuilt wheel, `tests/lib` and `tests/backend` on rustcpu: 442 passed, 44 skipped, 0 failed.
- **Virtual Impressionist** on the impressionist-rs wheel built from this: 699 passed, 0 failed (3 failed before).
- Clippy is clean, including the bindings with and without `module`.



The maintainer asked for a rustcpu performance test on yellowfin, then: "do all three fixes and rerun the affected code on yellowfin". The fixes were the three this ledger's previous entry and `claude-debug/PERF_RUSTCPU_YELLOWFIN.md` proposed.

**What the AI did** (yapocis-rt, ~350 LOC changed):
- **Argument views:** `cpu/args.rs`'s `Input` and `Output` hold the typed slice they serve, computed once, and the buffer's lock guard. Before, every element access re-checked a byte-to-element cast. The operator and graphics ports take generic closures instead of `fn` pointers.
- **No host copies:** `RawDevice` gains `runs_on_host()` and `wrap_host()`, both off by default, so every GPU backend is unchanged.
  - **On the CPU:** for synchronous calls, `Program::call` runs kernels on the caller's arrays in place, and writes each `out_like` result straight into the array it returns.
  - **The buffer:** `CpuBuffer` became a lock plus a byte pointer: its own aligned storage, or the caller's for one call. The two `unsafe` operations (forming the byte slice under the lock, and `wrap_host` itself) are confined to it, with their safety conditions written down.
  - **`in` arrays:** wrapped read-only; a kernel that writes an `in` parameter gets a private copy that is discarded, as on a GPU.
  - **Fallback:** an array that isn't 8-byte aligned is copied as before.
- **Parallel histogram:** partial histograms per 16K-element chunk, built in parallel by the new `par::map_each`, are added in chunk order. The integer counts are exact.
- **Test change:** `yapocis-rt`'s buffer-cache test now expects the CPU backend not to use the cache.

**Measured** (`claude-debug/PERF_RUSTCPU_YELLOWFIN.md`):
- **rustcpu itself, on yellowfin:** 1.86× faster on 80 threads and 1.32× faster on one thread (geometric means).
- **Against numbax, on yellowfin:** from 1.38× to 2.57× on all threads, and from 0.56× to 0.74× on one.
- **Largest gains:** the histogram (11.5×), gradient (2.9× on 80 threads), add (5.8× on one thread).
- **M4 Max, one thread:** `operators.add` went from 2.44 ms to 0.52 ms.
- **Unchanged:** the compute-bound kernels (gaussian, median3x3, simplex).

**Verified (M4 MacBook):**
- `cargo test --release -p yapocis --features rustcpu`: 72 passed.
- The workspace on OpenCL, Metal and rustcpu: all passed, after the buffer-cache test was adjusted.
- impressionist-rs's yapocis-external against this runtime: 47 passed.
- Clippy is clean, including the bindings.

**Defects and corrections:**
- **A deadlock in the first `CpuBuffer::read`:** it blocked on its lock. Where an output was taken before an aliased input, that would have deadlocked instead of panicking with "take inputs before outputs". I noticed it reviewing my own change before it ran, and replaced it with `try_read`.
- **The buffer-cache test failed** on the CPU backend, because the CPU backend no longer allocates device buffers. That was the intended effect of fix 2; the test was adjusted to assert it.

**Merged:** PR #17, as `570655a`, after CI passed: build, bundle-drift and `rustcpu` on Linux x86-64. impressionist-rs was then repinned to it (its PR #48). The working worktree was removed. Its untracked `claude-debug` material (the benchmark scripts and Rust benchmark crate, the raw M4 and yellowfin results, the report PDFs) was first moved into the main checkout's `claude-debug/` and checked byte for byte. yellowfin's benchmark worktree (`~/workspace/yapocis-rs-perf`) was removed too; its results had already been copied here.

### 2026-11-08 — The `rustcpu` backend: every kernel in plain Rust on the CPU, and Python bindings for Yapocis3 (Claude Opus 5.5)

The maintainer: "Start a worktree to provide a CPU only version of all kernels. Emphasize simplicity and readability of the generated code. The object is to provide a fallback for systems … that do not have any GPU (even a software OpenCL) implementation … Attempt parallelism only if it is easy to reason about." Later: "go ahead with rustcpu" (a Yapocis3 backend through Python bindings), and "backend name should be the same -- lets go with rustcpu for both."

**The backend (`yapocis-rt`, feature `rustcpu`, `Backend::RustCpu`).** It implements the existing `RawDevice` interface, so argument derivation, dispatch sizes, the buffer cache and every library port are shared with the GPU backends. Buffers are host memory. "Compiling" an interface looks up its Rust ports, and a launch calls the port with the arguments in the OpenCL kernel's order. The ports follow the bundle's OpenCL text, quirks included: boundary tests, integer types, and `double` literals computed in `f64`. The numbax kernels were only an outline, because they pad and guard differently. The convolution families read their weights back out of the OpenCL source, so weights generated at runtime work too. Parallelism (rayon) is used only where each work item writes just its own outputs, through two helpers (`par::map`, `par::map_chunks`). Histograms, atomics and scatter writes run sequentially. A registry (`cpu::kernels::register`) lets a crate with its own bundle add ports; impressionist-rs's yapocis-external uses it.

- `crates/yapocis-rt/src/cpu/`: the device, argument views and parallel helpers (~420 LOC), and 174 kernel symbols ported in `kernels/` (~3,000 LOC). One coordinating session wrote the plumbing and the convolution, operator and mandelbrot ports; five parallel sub-agents ported the other families.
- `Backend::source_backend()`: the CPU backend loads the OpenCL source and launches like OpenCL. `BufferAny::alloc`, `upload_bytes` and `download_bytes` are now public, for the bindings.
- Tests: the parity replay runs on the CPU against PoCL's CPU OpenCL fixtures (`ocl/cpu`, or `YAPOCIS_CPU_REFERENCE`). `tests/cpu_{filters,pixels,cnn,signal}.rs` compare the CPU with OpenCL and Metal for kernels the fixtures don't cover, and skip without a GPU (~750 LOC). CI gains a `rustcpu` job running the whole test suite on the CPU backend, the first kernel parity CI can run.
- `bindings/python` (~320 LOC): the PyO3 module `yapocis_rs`, outside the workspace, used by Yapocis3's new `rustcpu` backend (Yapocis3 branch `rustcpu-backend`, uncommitted there). Yapocis3's suite on numbax plus rustcpu: 958 passed, 33 skipped, 0 failed.
- README and CLAUDE.md sections. This commit also carries the earlier README update (the npzfile crate, complex64 multiply, Metal on iOS, and the quantile variants).

**Verified:** on the M4 MacBook, `cargo test --release --workspace` with OpenCL, Metal and rustcpu: 118 passed. `-p yapocis --features rustcpu` alone: 72 passed. Clippy (`-D warnings`) is clean, including the bindings. Not run: CUDA hosts (no change to that path), and Linux CI (it runs when this is pushed).

**Merged:** PR #16, as `abdd32b`, after CI passed: build, bundle-drift, and the new `rustcpu` job, which ran the whole yapocis test suite on the CPU backend on Linux x86-64. The Yapocis3 side of the bindings merged there as PR #77 (`9d4ec87`), and impressionist-rs's ports as PR #48 (`3bdcc2a`).

**Defects and corrections, and how they were caught:**
- **Missed source lookups:** the first wiring took kernel sources by the backend's own name in `lib::convolve` and `lib::graphics`, so the CPU backend reported "backend unavailable". The test suite caught it.
- **Hidden dependency:** `cpu/args.rs` used a bytemuck function that compiles only when another crate enables its `extern_crate_alloc` feature. A sub-agent's `cargo clippy -p yapocis-rt` caught it.
- **Bad backend design (maintainer):** I first proposed that Yapocis3 select the Rust runtime's backend through `YAPOCIS_DEVICE`. The maintainer asked whether `YAPOCIS_BACKEND=rust YAPOCIS_DEVICE=cuda` would then do something. It would have, which showed that I was overloading a device selector. The design became a single CPU-only `rustcpu` backend.
- **Two names (maintainer):** I named the Rust backend `cpu` while the Python one was `rustcpu`. The maintainer made it one name.
- **Upgraded NumPy:** installing the wheel into Yapocis3's venv upgraded its NumPy from 2.3.3 to 2.5.4. I caught it from the install output and restored 2.3.3; installs now use `--no-deps`.
- **Still present:** the per-element cost of `Input`/`Output` views. A benchmark against numbax (`claude-debug/PERF_NUMBAX_VS_CPU.md`) found that their `Deref` re-checks a byte-to-element cast on every access, which makes small memory-bound kernels 3 to 23× slower than numbax on one thread. A temporary experiment cut `add` from 2.24 ms to 0.49 ms, but no fix is applied.

**Found and recorded, not fixed here (upstream or fixtures):**
- **Stale fixture:** PoCL's `ocl/cpu` Mandelbrot fixture predates the first-escape change. It is skipped on the CPU by file hash until it is re-recorded on the Windows VM (the maintainer's call).
- **Yapocis3 bundle quirks the ports reproduce:**
  - vgg16 `matrix_multiply_simd` indexes its output in 1-D on Metal but 2-D in OpenCL.
  - vgg16 `batch_norm` computes only part of its output, because `depth_of` means one dimension in dispatch and another in argument derivation.
  - `zcs` reads its neighbours at channel 0 for every channel.
  - The 3-D simplex fixtures differ by about 0.49 between OpenCL and Metal/CUDA.
- **Reports:** `claude-debug/PERF_NUMBAX_VS_CPU.md` (the CPU backend is 2.8× faster than numbax on all cores, geometric mean, and 0.82× on one thread) and `claude-debug/MISSING_INTERFACES.md` (the 26 project interfaces in no yapocis-rs bundle).

### 2026-11-07 — CUDA simplex from Yapocis3 7e98bb0; the RTX 3080's re-recorded fixtures (Claude Opus 5.5)

- **The kernel:** `bundle/cuda/simplex.cu` is regenerated from Yapocis3 `7e98bb0` (its #76). CUDA's `simplex2d` now takes its coordinates as OpenCL and Metal do; before, its field was theirs transposed (the maintainer: "Make cuda noise match").
- **The fixtures, recorded on yellowfin's RTX 3080:**
  - the CUDA library fixture `simplex.all`;
  - the CUDA and OpenCL Mandelbrot fixtures, left over from the first-escape change (they were on yellowfin's clone, uncommitted).

**Verified:** `cargo test --release` on yellowfin (CUDA, OpenCL: 109 passed) and on the M4 MacBook (Metal, OpenCL: 109 passed).

**Still to do:** the Windows VM's CPU OpenCL Mandelbrot fixture still needs re-recording there.

### 2026-11-07 — Metal on iOS (experimental branch `ios-metal`) (Claude Opus 5.5)

The maintainer: "evaluate moving plumimpress to native Mac and iOS on swift", then "Try integrating into ../virtimp_com_app as a local alternative", and on hearing it was Mac-only: "Macos only is useless, almost. Why not iOS?".

`yapocis-rt` built its Metal backend only for `target_os = "macos"`. Four cfgs (`lib.rs`, `device.rs`, `Cargo.toml`) now also take `ios`; `metal.rs` needed no change, because `MTLCopyAllDevices` exists on iOS 18. A research agent found and tried the change on a scratch copy.

**Verified:** the VirtImp app's renderer renders through it on the iPhone simulator's GPU (`metal: Apple iOS simulator GPU`), and on the Mac. A build was installed on the maintainer's iPad Pro (M4) and launched.

**Not run:** this repo's own test suite for iOS targets (no iOS CI); a render on the iPad itself, which is still to be tried.

**Merged into `main` (2026-11-07).** The maintainer asked for it, so that impressionist-rs and the VirtImp app can pin it from GitHub rather than a sibling checkout. Before merging, `cargo test --release --features yapocis/ocl,yapocis/metal` passed on the M4 MacBook (109 passed) and clippy was clean. Not rerun on yellowfin: the change only adds `ios` to Metal's cfgs.

### 2026-11-07 — `npexact::order::quantile_linear_f32_f64q` (Claude Opus 5.5)

Moving impressionist-rs from yapocis-rs `1fbf919` to `783f2da` moved 26 of PlumImpress's 157 exactness digests, all of them style renders. The cause was 0.1.1's `quantile_linear_f32` (the PlumCraft ports, 49e357d). It now computes the virtual index and gamma in float32, as NumPy 2 does for a Python-float `q`. Virtual Impressionist's digests were recorded with the float64 index the function used before.

- **The fix:** both behaviours are kept. `quantile_linear_f32` stays NumPy 2's, which PlumCraft relies on through crates.io. The new `quantile_linear_f32_f64q` is the earlier float64 one, which impressionist-rs now calls.
- **The test:** on 39,402 cases it equals the float64 formula bit for bit, and it shows that the two functions do differ.
- **Confirmed:** with the old body restored locally, PlumImpress's gate passed all 157 workflows, each engine starting on an empty checkpoint store.
- **How the change slipped through:** the AI's first run against the local checkouts reported 156 of 157. The renderer's checkpoint cache is keyed by a fixed code version, so it served stages computed by the old code. The gate now gives each engine an empty store.
- **Not published:** this exists only on the git branch, not as a crates.io release. ~60 LOC.

### 2026-11-06 — `npzfile` (zip 9); Histogram and Mandelbrot kernels from Yapocis3 29f8403 (Claude Opus 5.5)

**The zip advisories.** The maintainer asked to "look at the dependabot alerts" on PlumImpress, then to "get started on the zip fix". All four alerts (one critical, two high, one moderate) were against `zip` 6.0.0. It reached PlumImpress only through `ndarray-npy` 0.10's `npz` feature, enabled here and in impressionist-rs, and 0.10 is the newest `ndarray-npy`. The new crate `crates/npzfile` (~190 LOC with its test, `publish = false`) is `ndarray-npy`'s `NpzReader` and `NpzWriter` API on `zip` 9.4 (deflate only), reusing `ndarray-npy`'s `.npy` reading and writing. `yapocis::lib::cnn`, and the tests of `yapocis` and `npexact`, use it. The workspace's `ndarray-npy` no longer turns on `npz`. `Cargo.lock` holds only `zip` 9.4.1. `npexact` itself gained no dependency, since PlumCraft takes it from crates.io.

**The kernels.** These were regenerated from Yapocis3 `29f8403` (`bundle/SOURCE`), whose AI.md has the details. They follow the maintainer's rulings on the array/Impressionist comparison: the histogram bins now span `low..high`, and Mandelbrot stops at the first escape. The Metal and OpenCL Mandelbrot parity fixtures were re-recorded on the M4 MacBook. New test `histogram_bins_span_low_to_high`.

**Verified:** `cargo test --release --features yapocis/ocl,yapocis/metal` and clippy (`-D warnings`) on the M4 MacBook.

**Not verified:** CUDA. Yellowfin's Mandelbrot fixtures (CUDA and OpenCL on the RTX 3080) and the Windows VM's CPU OpenCL fixtures are from the old kernels, so the Mandelbrot parity cases will fail there until they're re-recorded on those machines.

**Defect in AI-written code:** the first version of the new test failed clippy's `manual_repeat_n`, caught by clippy before the commit.

### 2026-11-05 — `npexact` 0.1.2 published (Claude Opus 5.5)

The maintainer: "release npexact 0.1.2". It is an addition, `npexact::complex` (#11), so a patch version, as the maintainer named it. `yapocis` depends on 0.1.2, and the crate description now mentions the complex64 multiply. Before uploading, `cargo test -p npexact --release` passed (27 tests) and `cargo publish --dry-run` built the crate from its packaged form. Tagged `npexact-v0.1.2`, the first tag for an npexact release (0.1.1 had none). PlumCraft can then call `Mul::array_default()` instead of its own copy.

### 2026-11-05 — `npexact::complex`: numpy's complex64 multiply, fused where numpy's loop fuses (Claude Opus 5.5)

**What the maintainer asked:** they put `fix approved` on #10 and said "fix approved bugs". #10 was filed from PlumCraft, whose Mandelbrot kernel borrowed `random::Contraction::host_default()` for a complex multiply and so got x86-64 wrong.

**What the AI did:** it measured first, then ported.

- **The measurement:** a probe (PlumCraft's `claude-debug/complex64_mul_model_probe.py`, numpy 2.3.3 on both machines) matched numpy's products against five candidate models.
  - **Arrays:** on the M4 Max and on yellowfin's Xeon (FMA3, AVX2), every product of a contiguous array of every length, a strided view and an array times a scalar is `re = fma(ar, br, -[ai*bi])`, `im = fma(ar, bi, [ai*br])`, the bracketed product rounded first.
  - **Scalars:** numpy scalars go through compiled C and are fused on the Mac but plain on the Xeon. So the issue's ask, a runtime rule, was half the answer.
- **`src/complex.rs` (~138 lines):** `Mul { Plain, Fused }`.
  - `array_default()` decides at run time: fused on aarch64, and on x86-64 when `is_x86_feature_detected!` finds `fma` and `avx2`.
  - `scalar_default()` decides at build time, as the other policies do.
  - `array_for(machine, fma3_and_avx2)` and `scalar_for(machine)` serve replay, and `mul_c64` and `multiply_c64` are the arithmetic.
  - The crate docs name `Mul` as the one policy that asks the CPU.
- **Fixtures:** a new `complex` case for both hosts, recorded with Yapocis3's new section (branch `complex64-fixtures`). Only `complex.npz` and its index entry were added, since everything else was kept as committed.
- **`tests/complex.rs` (~163 lines):** it replays both hosts anywhere, under each host's rules. That's 52,651 products per host, bit for bit with any NaN matching any NaN, specials included. It also asserts that the other rule misses numpy on each host, so the fixtures really tell the rules apart.
- **Checked:** npexact's whole suite passes, clippy is clean on arm64 and for `x86_64-apple-darwin`, and fmt passes.

**Not covered, and said so in the docs:** x86-64 without AVX2 (assumed plain), and AVX-512 hosts (assumed fused). The issue's second ask, to check which other ported loops depend on numpy's runtime dispatch, wasn't done. `MinMax` and `Quadrant` are still decided at build time, which the two hosts' fixtures prove for those two CPUs only.

**Defect in AI-written work:** the first unit test meant to show the rules differ used (1+ε)² − (1−ε)², which is 4ε under both. It was caught before commit by working the arithmetic, and replaced with a deterministic search.

### 2026-11-03 — `npexact` 0.1.1 published (Claude Opus 5.5)

The maintainer: "Publish npexact". The AI gave npexact its own version, 0.1.1, instead of the workspace's 0.1.0, because the changes since the last release are additions plus two bug fixes. `yapocis` depends on the new version, and the crate description now mentions elementwise ops. Before uploading, `cargo test -p npexact --release` passed (24 tests) and a dry run built the crate from its packaged form.

### 2026-11-03 — `npexact`: PlumCraft's hand-ported numpy behaviours, and x86-64 fixtures (Claude Opus 5.5)

The maintainer asked first to "assess the gap between npexact and all of numpy", then: "Move the hand-ported behaviours into npexact, record fixtures on Linux and test."

**What the AI did.** It moved the numpy behaviours that PlumCraft's array engine had written itself into npexact. Each one is held to fixtures numpy recorded through yapocis3's `host_fixtures.py`.

- `elementwise.rs` (new, ~215 LOC): `maximum`, `minimum`, `clip_min`, `sign`, `mod` and `nan_to_num`.
- `extrema.rs` (new, ~230 LOC): `max`, `min`, `nanmax`, `nanmin` and int64 `cumsum`.
- `fft.rs` (new, ~50 LOC): `fftshift_2d` and `ifftshift_2d`.
- `order.rs` (+185/−9): `sort_f32`, `searchsorted`, `lexsort_f32`, `nanpercentile_linear_f32` and `unique_inverse_i64`.
- `hist.rs` (+9): `linspace_f64_as_f32`.
- `simd_math.rs` (+63/−6): `Quadrant`.
- `tests/ops.rs` (new, ~330 LOC): replays every host's fixtures under that host's policies.

It also recorded a full fixture set on yellowfin (Xeon E5-2698 v4, AVX2, no AVX-512; numpy 2.3.3, as on the Mac), which adds 31 MB.

**Defects in earlier AI-written npexact code, caught by the new fixtures:**

- `quantile_linear_f32` was 1 ulp off on both hosts, for example at `q = 0.9` with 2 or 5 values. With a Python-float `q`, NumPy 2 does the virtual index, gamma and the interpolation in float32, but npexact did them in float64. It now works in float32 throughout.
- `sin_f32`/`cos_f32` were 1 ulp off in 470 of 355,342 lanes on x86-64. GCC, which builds numpy's Linux wheels, fuses `x * 2/π + magic` into one FMA; clang on macOS doesn't. The fix is `Quadrant::{Separate, Fused}`, defaulting by architecture, and each model matches every lane on its host. The AI found this with Python lane models (`claude-debug/npexact/sincos_x86_*.py` in PlumCraft).
- `tests/simd_math.rs` assumed numpy's float32 `exp`/`log` are libm calls. That holds only on arm64, so the expectation now follows the architecture.

**A difference between the CPUs, not a bug.** numpy's `maximum`/`minimum` follow IEEE on arm64 (`+0` beats `-0`), while on x86-64 `maxps` gives a tie to the second operand. That makes x86's `max`/`min`/`nanmax`/`nanmin` of an array depend on numpy's AVX2 reduction order. The AI modelled that order in Python (`avx2_reduce_sim.py`), then ported it from `loops_minmax.dispatch.c.src` into `MinMax::X86`. All 2,054 `extrema` checks match on both hosts.

**Not reproduced:** where `np.sort` puts `-0` and `+0` depends on its sort library (Highway or x86-simd-sort). This is documented, and the test ignores zero signs. AVX-512 loops are unproven.

**Tests.** `cargo test -p npexact --release` passes on the Mac and on yellowfin. Both hosts' fixtures for the new sections replay on either machine. Workspace clippy (`ocl`, `metal`, `-D warnings`), fmt and `cargo doc` are clean. Not yet published: the crate's version still comes from the workspace (0.1.0).

### 2026-11-02 — `npexact` ready for crates.io (Claude Opus 5.5)

The maintainer: "Publish npexact". PlumCraft's array engine depends on it, and crates.io refuses git dependencies. The AI removed `publish = false`, added the licence to the crate, and excluded `fixtures` (22 MB, over the 10 MB limit) and the tests that replay them, which stay in this repository. All 16 of the crate's tests passed here first, and a dry run built the crate from its packaged form.

### 2026-10-30 — `extent_freetype`, and fixtures re-recorded on Apple OpenCL and the RTX 3080 (Claude Opus 5.5)

The maintainer: "port extent_freetype; restart the python fleet; update the fixtures, you can use yellowfin for the 3080."

**What the AI did.**

- `typeset::extent_freetype`, sharing its box calculation with `render_freetype`. `typeset.rs` +33/−10.
- Fixtures re-recorded for Metal and OpenCL on the M4 Max, and for CUDA and OpenCL on the RTX 3080 on yellowfin, in a temporary directory there, with the render workers left running.
- Extents replayed in `tests/graphics.rs` (+12), and `spec_batch` compared numerically on every backend but PoCL (`tests/processing.rs` +4/−4).

**What the re-recording showed.** Existing fixtures held up: on the RTX 3080's CUDA set, everything but the index matched. RTX OpenCL's truetrace set was stale from the initial import, and Apple OpenCL's `spec_batch` predated the race fix. The HSI plane kernels `rgb2hsi` and `hsi2rgb` now have real fixtures on all four devices, which they never had; the generator was fixed in Yapocis3 #71.

**Tests.** With `yapocis/metal,yapocis/ocl` on the M4 Max: 95 passed, 0 failed. With `yapocis/ocl,yapocis/cuda` on the RTX 3080: 95 passed, 0 failed, with 97 kernel cases matched on each backend, checked so the run couldn't pass by looping over no devices. Clippy and `cargo fmt --check` clean on the M4 Max.

**Not done.** The PoCL fixtures (`ocl/cpu`) are from the Windows VM and weren't re-recorded. `coherence` and `custom_weights_match_cpu_and_bundled_variant` still fail under `YAPOCIS_PARITY_EXACT=1`.

### 2026-10-30 — `svg_regions` no longer reproduces the outer-ring bug (Claude Opus 5.5)

The maintainer: "fix the svg bug". Yapocis3's `write_svg_regions` wrote outer rings as polygons with `curve="cubic"`, and the port reproduced that deliberately. Both are fixed together, and `truetrace.regions.svg` is regenerated for all five device folders; it comes from CPU code, so it's one file. The RTX 3080 OpenCL copy had been stale since the initial import, predating the nesting fix, unnoticed because that test reads only the first backend a host offers.

`truetrace.rs` +7/−12. With `yapocis/metal,yapocis/ocl`: 95 passed, 0 failed; clippy clean.

### 2026-10-30 — Bundle from Yapocis3 `ea57832`, Metal fixtures re-recorded, `trace_regions` and `regions_svg` (Claude Opus 5.5)

The maintainer asked "What is missing for parity between yapocis-rs and yapocis3?", then "Go for 1-3" from the answer's list.

**What the AI did.**

- **Bundle.** Only `metal/truetrace.metal` changed (Yapocis3 `7d25604`, the lambda became an inline function), with its hash and `SOURCE`. A fresh export shows no drift.
- **Metal fixtures.** Re-recorded on the M4 MacBook. Every kernel fixture and all but one library data fixture came out unchanged. New: `regions_svg` text fixtures and `models.resnet50` and `models.vgg16`.
- **Port.** `trace_regions` is split out of `trace_mask`, and `regions_svg` writes the outlines as SVG, ported byte for byte. `truetrace.rs` +105/−9, `tests/processing.rs` +43/−5, kernel +11/−5.

**A gap closed.** `spec_batch` in `fft.analysis` was the one data fixture that changed: up to 36.8 on 0.82% of elements. The test only checked its shape, because the batch path used to vary between runs. Forty runs in Python were now identical on Metal and on OpenCL, and Rust matches the new fixture, so the Metal check is numeric. The old fixture most likely came from before the butterfly race fix; that's an inference, not something reproduced.

**Defects.** Writing the port through a shell heredoc stripped its string escapes. The compiler caught that before any test ran. `svg_regions` still reproduces `write_svg_regions`' outer-ring bug on purpose.

**Tests.** With `yapocis/metal,yapocis/ocl`: 95 passed, 0 failed; clippy and `cargo fmt --check` clean. ImpCraft's `impcraft-local` tests, exact suite included, pass against this tree. `YAPOCIS_PARITY_EXACT=1` fails `coherence` and `custom_weights_match_cpu_and_bundled_variant` with last-bit differences, identically on `543e0c4` without this work, so those failures aren't from here and aren't fixed.

**Not done.** The OpenCL and CUDA fixtures are still from `2909778`, without `regions_svg` cases and with the shape-only FFT check.

### 2026-10-26 — Parity fixtures for CPU OpenCL (PoCL 7.2), recorded on the Windows VM (Claude Opus 5.5)

**What the maintainer asked for.** Eight tests failed on the Windows VM because no fixtures existed for its only OpenCL device, PoCL on the CPU. The maintainer: "You may be able to record PoCL fixtures on yellowfin", then "Go ahead and record the fixtures after the build".

**Where they were recorded, and why not on the Linux host.** The Linux host's PoCLs are 1.8 (LLVM 11, the system ICD) and 3.0-rc2 (LLVM 14, PyPI's `pocl-binary-distribution`), and both name the device `pthread-Intel(R) Xeon(R) …`. The VM's is PoCL 7.2 (LLVM 22) named `cpu`. The tests look fixtures up by device slug, and a compiler eleven major versions older need not reproduce PoCL 7.2's results, so the fixtures were recorded on the VM with the yapocis3 commit `SOURCE` names: `yapocis-tools bundle --parity ocl --parity-lib ocl`.

**What the AI did.** `bundle/parity/ocl/cpu/` (95 kernel cases) and `bundle/parity/lib/ocl/cpu/` (67 library cases), 166 files, 3.1 MB; a `parity (cpu)` line in `SOURCE`. Skipped by the recorder: the HSI pair (not reproducible) and the two CNN models (no weights on the VM). The recording first failed on yapocis3's default-encoding file writes (cp1252 on Windows); it was done with `PYTHONUTF8=1`, and that bug is fixed in yapocis3 separately. The two `index.json` files were written with `\r\n` and are committed with `\n` (`.gitattributes`); the `.npz` files are byte-identical to the recording.

**Testing.** On the VM, dev `36e9b4f` with these fixtures: `cargo test --release --workspace --features yapocis/ocl,yapocis/text`: 95 passed, 0 failed, including the eight that had no fixtures (`parity_with_python_outputs`, `cnn_kernels_match_python`, `truetrace_gpu_segments_match_python` and the five graphics parity tests). These tests fail when they find no cases, so they compared PoCL results. They cover PoCL 7.2 named `cpu` only; the Linux host's PoCL still has none.

### 2026-10-26 — main merged into dev; bundle from yapocis3 dev `2909778` (Claude Opus 5.5)

**What the maintainer asked for.** "Move the pin" in impcraft and impressionist-rs to the commit with the Windows fixes. Those repositories pin `dev`, which had diverged from `main` here and in yapocis3; the maintainer chose to merge main into dev in both.

**Defect: work aimed at the wrong branch.** The Windows fixes (#1) went to `main` while the downstream projects build from `dev`, and the FFT race they carried was already fixed on dev (yapocis3 `80884c3`, patched into dev's bundle by hand on 2026-10-22). Claude had not looked at dev; the mismatch surfaced only when `npexact`, which exists only on dev, failed to resolve at the main commit.

**What the AI did.** Merged main into dev (#2). Code (`typeset.rs`, `typed.rs`, `.gitattributes`, the freetype-sys patch) merged without conflicts. The bundle was not hand-merged: it was exported again from yapocis3 dev after that repository's own main-into-dev merge, and differs from dev's previous bundle only in the FFT sources and their hashes and the two HSI dispatch entries (checked structurally). `SOURCE` now names that commit, so `bundle-drift` can rebuild it. AI.md keeps both sides' entries; this entry followed in a second commit because the edit adding it failed and the merge was committed without it.

**Verification (yellowfin).** `cargo test --workspace --features yapocis/ocl,yapocis/cuda` on the RTX 3080 (CUDA and NVIDIA OpenCL): 95 passed, 0 failed, including `parity_with_python_outputs`. `scripts/check-bundle.sh` against yapocis3 `2909778`: up to date; CI `build` and `bundle-drift` pass. Windows and Metal were not rerun for the merge.

### 2026-10-26 — Windows test failures fixed: bundle from yapocis3 #60, LF checkouts, `div` tolerance (Claude Opus 5.5)

**What the maintainer asked for.** "address the windows failures", which are the ones listed at the end of the previous entry. For the heap corruption the maintainer asked for the valgrind command line, ran it, and had the log read. Offered a test-side fix or a fix at the source, they chose "Fix at the source. Then address FFT, div test." A PoCL link failure that the first fixes exposed got "Apply workaround, test in isolation". Then CPMD.

**What the AI did.**

- **`.gitattributes` (`* text=auto eol=lf`).** Git for Windows defaults to `core.autocrlf=true`, so the bundle checked out with CRLF. `build.rs` embeds it with `include_str!`, and the generators slice it on newline-sensitive markers, so the source was cut short after its first kernel. This fixes the `lib::convolve` bundled-text failures.
- **Bundle re-exported from yapocis3 `5453d0b`** (the fixes themselves are recorded in yapocis3's AI.md). `SOURCE` records the provenance. Changes:
  - The FFT templates no longer race two work-items on one radix-2 butterfly. This caused the wrong `fft_roundtrip_and_peak` values.
  - `rgb2hsi3`/`hsi2rgb3` declare a per-pixel `dispatch_size`. The raw interface launched one work-item per float and wrote 3x past the output buffer: that was `STATUS_HEAP_CORRUPTION` in `tests/typed.rs`, and a PoCL abort under valgrind on Linux.
  - OpenCL `window`/`power_spectrum` no longer call `get_global_size()`, which PoCL 7.2 on Windows cannot link under `-cl-opt-disable`.
  - Parity fixtures were not re-recorded.
- **`tests/typed.rs` (~8 LOC).** `operators` is built with `-cl-fast-relaxed-math`, which allows `x / y` to be off by 2.5 ulp, and CPU OpenCL returns 21.499998 for 43 / 2. The `div` check now allows 3 ulp instead of requiring exact equality.

**Defects in AI-written code and testing.**

- Three of Claude's test runs on the Linux host left out `--features ocl`. Every test looped over zero devices and passed, and that included a `div` "pass" Claude nearly reported. Claude caught it from the 0.00 s run times. The figures below come from runs with the backend enabled.
- The tests open each backend's first device and ignore device selection. On the Linux host that is the P40, or PoCL when it is installed, and neither has fixtures. Parity was replayed by pinning to the RTX 3080 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=…`, and `OCL_ICD_FILENAMES` with an empty `OCL_ICD_VENDORS` directory). This limitation remains.

**Testing.**

- `cargo fmt --check` and `cargo clippy --workspace --all-targets --features yapocis/cuda,yapocis/ocl,yapocis/text -- -D warnings` are clean.
- RTX 3080 (`--features ocl,cuda,text`): all 56 tests pass. Parity replays 95 cases on CUDA and 95 on OpenCL, with no mismatches.
- PoCL on Linux and on the Windows VM (`--features ocl,text`): everything passes except the 8 tests that have no fixtures for a CPU device (`cnn_kernels_match_python`, five graphics parity tests, `parity_with_python_outputs`, `truetrace_gpu_segments_match_python`).

### 2026-10-25 — `text` builds on Windows: patched freetype-sys, C `long` widths in typesetting (Claude Opus 5.5)

**What the maintainer asked for.** A Windows 11 development VM for porting this stack, then, of the fixes offered for the first Windows build failures: "1,2,4", meaning fork freetype-sys with the fix, consume the fork through `[patch]`, and fix the type errors here. They ruled out proposing the change upstream: "many repos don't want AI proposing changes."

**What the AI did.**

- **`Cargo.toml` (~6 LOC):** `[patch.crates-io] freetype-sys` points at the maintainer's fork (`seantrue/freetype-sys`, rev `900ebc5`). The published 0.23.0 compiles its bundled libpng with `-I libz-sys/src/zlib`, a path that exists only in a git checkout, so MSVC stops on `zlib.h`; the fork takes zlib's headers from libz-sys (`DEP_Z_INCLUDE`). The fork carries its own AI.md entry.
- **`typeset.rs` (~16 LOC changed):** FreeType's `FT_Fixed`/`FT_Pos` are C `long`, 64-bit on Linux and macOS but 32-bit on Windows. The rotation matrix now casts to `FT_Fixed`, and kerning/advance values widen to `i64` through a small `wide()` helper.

**Defects in AI-written code, caught by clippy before commit.** The first version used `i64::from(...)` inline and was described to the maintainer as compiling cleanly everywhere; on Linux, where the value is already `i64`, clippy's `useless_conversion` rejects it under `-D warnings`. The second attempt, `as i64`, failed the same way (`unnecessary_cast`). The `wide()` helper scopes one `#[allow(clippy::useless_conversion)]` with the reason.

**Testing.** On the Linux host: `cargo fmt --check` and `cargo clippy --workspace --all-targets --features yapocis/cuda,yapocis/ocl,yapocis/text -- -D warnings` are clean, and `cargo test --release` with the same features passes (72 tests), including `typeset_parity` ("12 text cases match FreeType output"). On the Windows VM (PoCL CPU OpenCL), everything now compiles and links; `typeset_parity` cannot run there because no parity fixtures exist for the `ocl/cpu` device, so the Windows typesetting path is compile-checked only.

**Found on Windows and not fixed here** (`cargo test --release --features yapocis/ocl,yapocis/text --no-fail-fast`, 13 failures plus one crash): the bundled-text comparisons in `lib::convolve` see the bundled source cut short after its first kernel; `fft_roundtrip_and_peak` gets wrong values (`0.5878 vs -42.44+168.97i`); `tests/typed.rs` dies with `STATUS_HEAP_CORRUPTION`; the remaining parity tests stop on missing `ocl/cpu` fixtures. These are the first items of the Windows port.

### 2026-10-23 — Pillow's resize, exact CNN preprocessing, numpy's complex abs, and the FFT race (Claude Opus 5.5)

**What the maintainer asked for.** "Port all the nodes": ImpCraft's local engine on impressionist-rs, with every node bit-exact against its Python engine.

**What the AI did.**
- **`lib/pil.rs` (355 lines, new).** Pillow's resize, made exact for Pillow's two paths:
  - 8-bit images use its fixed-point BILINEAR and LANCZOS resampler (`resize`).
  - Float images use `resize_f32`. Pillow fuses the multiply-add in the vertical pass but not the horizontal one, and the port does the same.
  - It also has `to_luma` and `Filter`.
- **`lib/cnn.rs`, `resnet50.rs`, `vgg16.rs`.** ImageNet preprocessing exact with yapocis3's (Pillow resize and the same float order), plus `predict_array` for ImpCraft's Classify node.
- **npexact.** `cabs_f32` matches numpy's SIMD complex `abs`, which uses a fused multiply-add. Also a `hist.rs` helper and a fix to `random` seeding.
- **The FFT bundle kernels** carry yapocis3's fix (80884c3 on dev): one work-item per radix-2 butterfly (two raced on one pair, so batch FFTs varied run to run), and the radix-2 path only for power-of-two lengths. `bundle/SOURCE` records the patch.
- **`truetrace.rs`.** `off_field_iso` uses `contains`, for this toolchain's clippy. The behaviour is the same.

**Testing.**
- `cargo test --release --workspace --features yapocis/ocl,yapocis/metal` passes on the M4, and clippy is clean.
- ImpCraft's exact cases cover resize, Classify preprocessing, FFT magnitude and inverse FFT against its Python engine.
- The CUDA FFT kernel has not yet been run on CUDA.

**Defect record.**
- The first float resize fused both passes, and the ImpCraft cases caught the mismatch.
- The FFT magnitude first used a plain `hypot`; the exact digests showed numpy's FMA.
- The FFT race itself predates this work. A bit-exact digest that changed between runs exposed it.

**LOC (measured):** `pil.rs` +355, `cnn.rs` +31/−14, `imageops.rs` +17/−12, `resnet50.rs` and `vgg16.rs` +14 each, npexact +53/−2, bundle kernels +14/−14.

### 2026-10-22 — truetrace on CUDA no longer transposed: vendored kernel and RTX 3080 fixtures (Claude Opus 5.5)

Yapocis3's CUDA truetrace kernel addressed `x` as the fast axis while its wrapper passes `(W, H)` arrays with `x` slow, so on CUDA every traced shape came out transposed (recorded in Yapocis3's AI.md, fixed on its `dev` branch in `a42bfab`). This repository vendors that kernel, so the Rust port had the same bug. It showed when the Rust worker's CUDA dry run on yellowfin was compared with OpenCL.

- `crates/yapocis/bundle/cuda/truetrace.cu` and its sha256 in `manifest.json` are re-exported from Yapocis3 `dev`; no other bundle file changed.
- The RTX 3080's truetrace library fixtures (`parity/lib/cuda/nvidia-geforce-rtx-3080/truetrace.*` and their `index.json` entry, other generators untouched) were re-recorded on yellowfin with `yapocis-tools bundle --parity-lib cuda --generators truetrace`. The old ones had recorded the transposed output, and also predated the scanline fill and exact chaining.

**Defect record.** The CUDA fixtures were recorded on 2026-10-19 from the transposing kernel, so a CUDA parity replay would have enforced the bug. They had not been replayed on yellowfin since the truetrace changes.

### 2026-10-21 — truetrace: exact chaining and `off_field_iso` (Claude Opus 5.5)

**What the maintainer asked for.** "Defaut to .75, fix in yapicis-rs": mirror Yapocis3's fix for truetrace above iso 0.5, where 0.25 px endpoint snapping fused the two outlines of 1 px strokes into black wedges.

**What the AI did.**

- `lib/truetrace.rs`: endpoints chained by exact keys, `TraceParams::default()` with `chain_tol_px: 0.0`, and `off_field_iso`, all as in Yapocis3 (on its `dev` branch).
- The truetrace parity fixtures (OpenCL and Metal on the M4 Max) were regenerated from Yapocis3 with 1 px stroke cases at iso 0.75 and 1.0. `tests/processing.rs` and `tests/truetrace_stages.rs` follow (iso 0.75, tolerance 0).

**Verification.** On commit (2026-10-22) the suite on OpenCL (`--features yapocis/ocl`) passed: 92 tests, none failed. `convolve` and `fft_analysis` fail only under `YAPOCIS_PARITY_EXACT=1` (last-bit differences, also without this change) and pass at the normal tolerance.

### 2026-10-21 — npexact, bundle codegen as a library, truetrace scanline fill, image helpers (Claude Fable 5.1)

**What the maintainer asked for.** The Rust port of Virtual Impressionist (impressionist-rs), with the decision that numpy-exact numerics live here, proven by fixtures numpy itself records.

**What the AI did.** This entry is reconstructed from that session's report (`../impressionist/claude-debug/rustport_status_20261021.md`, `rustport_spikes_20261021.md`); it was written when the work was first committed on 2026-10-22.

- **`crates/npexact`** (new, ~4,800 lines with tests) ports numpy 2.5.3's host numerics bit for bit:
  - `SeedSequence`, `PCG64` and the `Generator` draws, the ziggurat included;
  - pairwise `sum`/`mean`/`var`/`std` and `reduceat`;
  - quantiles and medians;
  - `histogram`, `linspace`, `interp` and `cumsum`;
  - numpy's own float32 SIMD `sin`/`cos` and the `tanh`s;
  - `math.fsum`.
  
  The host's fused-multiply-add policy is a parameter. It is proven by fixtures from Yapocis3's `bundle --host-fixtures`.
- **`yapocis-bundle::codegen`** (new, ~460 lines): the typed-interface generator moved out of `yapocis/build.rs` (409 lines fewer there), so impressionist-rs generates its own bundle's API the same way. `Interface::manifest()` lets each bundle carry its own manifest.
- **`lib/truetrace.rs`:** float64 `fsum` geometry and the even-odd scanline fill (`fill_polygons`), matching Yapocis3. The earcut and graphics-kernel path was dropped because the C++ and Rust earcut libraries triangulate differently. New fixtures: `truetrace.fill.npz`.
- **`lib/image.rs`:** `normalize` and `align_image`, as `yapocis.utils.basic` has them.
- **Tests:** `cnn`, `convolve`, `graphics`, `processing` and `tests/common` adjusted for per-device fixtures and the new cases.

**Defects.** That session's report does not itemise its defects.

### 2026-10-20 — Machine names replaced with generic descriptions (Claude Opus 5.5)

The maintainer asked for the names of their two machines to be replaced in the repository: one becomes "the Linux host" and the other "the M4 MacBook". The change covers 16 lines in `AI.md`, `CLAUDE.md`, `bundle/SOURCE`, the CI workflow and two comments in `yapocis-rt`. The README never named them. Two matches were left alone, because they are an ImageNet class label in the VGG16 and ResNet50 `classes.txt` files, not a machine. Only text and comments changed.

### 2026-10-19 — Repository prepared for its first commit (Claude Opus 5.5)

**What the maintainer asked for.** "Best way to make yapocis-rs ready for checkin as a separate project and git repo (private)?", then, on Claude's seven-step answer: "do 1-5. For 6, do not enable MacOS CI at all. Then 7."

**What the AI did.**

- **Bundle provenance:** the exporter was committed to Yapocis3 first (`d0fc1ef`, PR #54; that repository's AI.md has the entry). The bundle was regenerated from that commit, and `crates/yapocis/bundle/SOURCE` pins it by SHA, which CI's drift job installs. No tag was made. Before merging, the PR was rebased onto `main` to resolve an `AI.md` conflict, which rewrote the commit as `a9aba38`. Claude had told the maintainer the rebase would keep the pinned SHA, which was wrong; the maintainer then said "Merge and fix", and `SOURCE` now pins `a9aba38`, which is on Yapocis3's `main` through merge commit `2940fee`. The rebase changed only Yapocis3's `AI.md`, not the exporter. The regenerated manifest, sources and derivation fixtures were byte-identical to the vendored ones, so only the parity fixtures changed. Exports on the M4 MacBook (macOS) and the Linux host were also identical except for `yapocis_version`, which is read from the installed package metadata; the Linux host's venv has an old 0.3.8 install, so the vendored manifest is the M4 MacBook's (1.0.0).
- **Per-device parity fixtures:** fixtures moved from `parity/<backend>/` to `parity/<backend>/<device>/` (and the same under `parity/lib/`). The test helpers open the backend's default device, take its name as a directory slug (`device_slug`, matching the Python exporter's rule, with a unit test), and skip with a message when nothing was recorded on that device. Sets are vendored for the Apple M4 Max (Metal, OpenCL) and the RTX 3080 (CUDA, NVIDIA OpenCL).
- **Cleanup:** the throwaway Phase 0 `spike/` crate left the workspace (archived in Yapocis3's untracked `claude-debug/rust_port_spike/`). Added `publish = false` on the three crates, `.idea/` in `.gitignore`, `cargo fmt` across the workspace (2 files were unformatted), this ledger and `CLAUDE.md`.
- **CI actions on Node 24** (maintainer: "bump the actions to Node 24 versions", after GitHub warned that `checkout@v4` and `setup-python@v5` target the deprecated Node 20): now `actions/checkout@v7` and `actions/setup-python@v7`, checked to declare `using: node24`. `Swatinem/rust-cache@v2` already runs on Node 24, and `dtolnay/rust-toolchain` is a composite action.
- **CI:** Linux only, as the maintainer specified. It builds the OpenCL and CUDA backends, runs clippy with `-D warnings`, rustfmt and the device-free tests, and checks the bundle for drift. It installs `ocl-icd-opencl-dev` because `opencl3` links `libOpenCL` at build time.

**Verification.**

- 72 of 72 tests pass on the M4 MacBook (Metal, OpenCL) and on the Linux host (CUDA, NVIDIA OpenCL) against the vendored per-device fixtures. That includes the slug test, and the two cross-device failures below no longer occur.
- `scripts/check-bundle.sh` reports the bundle up to date against `d0fc1ef`; after the re-pin, CI's drift job checks it against `a9aba38`.
- On the Linux host, `cargo fmt --check`, clippy with `-D warnings` (OpenCL, CUDA), the device-free tests and `cargo doc` all pass with the CI commands. `cargo doc` had 2 warnings (an unescaped `[2]` and a redundant link target), both fixed. The workflow itself had not run on GitHub at the time of writing.

**Defects in AI-written code.**

- The parity fixtures were keyed by backend only. OpenCL fixtures recorded on the Apple M4 Max failed two tests on NVIDIA's OpenCL (`parity_with_python_outputs`: one Mandelbrot pixel at 6 vs 7 iterations; `simplex_matches_python`: `noise3d` element 223). This was caught by running the suite on the Linux host. With fixtures recorded on the RTX 3080 both tests pass, so it was a cross-device comparison rather than a runtime bug. The per-device layout above fixes it.
- The Python parity exporter recorded the CUDA device as `<bound method name of <pycuda._driver.Device ...>>` (pycuda's `Device.name` is a method). This was caught by reading `index.json` while designing the per-device layout, and fixed in Yapocis3.
- The `text` feature's comment in `crates/yapocis/Cargo.toml` said it needed the system FreeType library, but the dependency builds FreeType from source (`bundled`). This was caught while writing the CI install step, and the comment was corrected.
- The drift check compared exported files byte for byte, which made it flaky. On one hosted runner, the same Yapocis3 commit, Ubuntu image, CPython 3.13.16 and numpy 2.5.4 rendered some Gaussian weights one digit apart in the 17th significant digit (`0.0028602900165330333` vs `…338`). The weights come from numpy `exp`/`sum`, whose SIMD paths depend on the CPU. A rerun on the same commit passed; the runner's CPU was not recorded. It was caught by CI on the action-bump commit. The maintainer chose "Tolerate drift" over making the Python weights deterministic. `scripts/compare-bundle.py` now rounds float literals to 12 significant digits before comparing (the kernels compute in float32) and compares everything else exactly. The first version of that fix still failed in CI, because the manifest's per-source SHA-256 changes with the same noise. The scratch test had edited a source without updating its hash, which a real export never does. Hashes are now left out of the comparison, and each vendored hash is checked against its vendored file instead. The scratch test now updates the hash with every source edit. It checks that the CI last-digit change passes, and that a 6th-digit weight change, an integer index change, a text change, an extra file, a missing file and a stale vendored hash each fail.
- The new workflow's first run on GitHub failed before starting. The drift job's `run:` line was a plain YAML scalar containing `ref: `, which YAML reads as a mapping. It was caught by GitHub ("workflow file issue") and confirmed with a YAML parser. The command is now a block scalar, and the file is parsed before each push.
- The removed CI workflow had never run. It listed a macOS job and the `spike` crate's features, and pinned the drift job to Yapocis3 `main`, which did not yet have `yapocis-tools bundle`, so the drift job would have failed on the first push.

### 2026-10-19 — Benchmarks against Python, buffer reuse, host-output copy, resident arguments, `set_margin` (Claude Opus 5.5)

**What the maintainer asked for.** In order: "Do timing (performance) comparisons between pure rust+yapocis-rs and python+yapocis"; "Fix Large arrays on Linux are slower in Rust; address GPU buffer reuse (binding array<->buffer)"; "We shouldn't be writing when a "resident" marked argument is used"; "Rerun the timing comparison with the resident change"; "Benchmark a more complex case where residence should make a difference."

**What the AI did.** Matched Rust and Python benchmark harnesses and the reports live in Yapocis3's `claude-debug/` and are not part of this repository. Changes here:

- **`HostData`** (`yapocis-rt/src/host_data.rs`, new): `out_like` results are downloaded once into a `Vec` of the element type, and the typed layer takes that `Vec` without copying it. Large outputs come from `alloc_zeroed`. A transparent-huge-page hint is available behind `YAPOCIS_HUGE_PAGES=1` and is off by default.
- **`BufferCache`** (`yapocis-rt/src/buffer_cache.rs`, new): each device keeps a cache of device buffers keyed by host address and length, plus same-length reuse. Buffers are checked out for the duration of a call, asynchronous calls bypass the cache, the cache is bounded by `YAPOCIS_BUFFER_CACHE_MB` and 64 entries, and `Device::purge_buffers` / `buffer_cache_stats` expose it.
- **Resident parameters** accept only device buffers. A host array is an `ArgKind` error instead of an upload on every call; this is the Rust counterpart of Python's `id(array)` binding without the risk of a reused address.
- **OpenCL device enumeration** is serialised by a process-wide lock.
- **`set_margin`** fills the four border slices instead of visiting every pixel.
- **Tests:** two `BufferCache` unit tests; `buffer_cache_on_every_available_backend`, `resident_parameters_are_never_uploaded` and `set_margin_touches_exactly_the_border`.

**Results.** Measured against Python on the same devices, Rust is as fast or faster in every steady-state case except CUDA `operators.add` 2048², which is about 6% slower (9.4 ms vs 8.9 ms over 8 repeats; cause not found). ResNet50 runs 7.4x faster on CUDA and 8.8x on OpenCL. Ten resident operator stages run 12x to 18x faster than the same stages with host arrays on the RTX 3080.

**Defects in AI-written code, and how each was caught.**

- **Double host copy on outputs.** Every `out_like` output was downloaded into a `Vec<u8>` and then copied into a `Vec<T>`. Writing 16 MB into fresh pages costs about 11 ms in page faults, so Rust was 3x slower than Python on CUDA for large arrays (`operators.add` 2048²: 28.4 ms vs 9.4 ms). Caught by the first benchmark run.
- **Huge-page hint enabled by default.** The first version of the fix turned the hint on. On the Linux host (`defrag=madvise`) it made some processes stall in synchronous compaction: 220 ms instead of 43 ms per 64 MB output, with 377 `compact_stall`s per run. Caught by bimodal benchmark runs; the hint is now opt-in.
- **pocl segfault.** Running the tests with `ocl` on the Linux host segfaulted about 1 run in 7, inside pocl's `clGetDeviceInfo`, while parallel test threads enumerated devices. Caught on the first OpenCL run there; the open lock fixed it (0 failures in 30 runs).
- **Resident parameters written on every call.** The untyped API uploaded a host array passed to a `resident` parameter on every call. Caught by the maintainer.
- **`set_margin` visited every pixel.** It visited every pixel with a bounds-checked write: 12.9 ms at 2048² on the Linux host and 4.1 ms on the M4 MacBook, now 0.66 ms and 0.15 ms. Caught by a per-stage timing of `zcsdog`.

**Found and not changed (library behaviour shared with Python).**

- `Gaussian::zcsdog`'s host and resident forms give different edge maps for about 0.45% of pixels. The two blur kernel families treat the image border differently, and the cleared margin is sized for the smaller kernel only.
- Python documents a 256-point limit for `fft.batch_forward` but does not enforce it; Rust rejects larger sizes.
- The first CUDA call after load takes 10.9 ms in Rust and 1.1 ms in Python. The likely cause is the driver compiling PTX on first use; this is not verified.

**Verification.** 71 of 71 tests pass on the M4 MacBook (Metal, OpenCL) and on the Linux host (CUDA, NVIDIA OpenCL, with OpenCL fixtures recorded there). Clippy is clean on both.

### 2026-10-18 — The port: bundle crate, runtime, generated API and library ports (Claude Opus 5.5)

**What the maintainer asked for.** An evaluation of how to run the yapocis kernels from Rust with no Python at runtime, then the implementation of the plan that came out of it. The evaluation report is in Yapocis3, `claude-debug/rust_port_evaluation_2026-10-18.md`.

**What the AI did.** Lines measured on 2026-10-19 over the whole repository (both sessions' work), excluding the vendored bundle and fixture data:

- `yapocis-bundle`: manifest schema, derivation and dispatch evaluation (~1,010 LOC; ~250 LOC of tests against Python's derivation fixtures).
- `yapocis-rt`: the OpenCL (`opencl3`), Metal (`objc2-metal`) and CUDA (`cudarc`, NVRTC) backends, the program cache and launches (~2,490 LOC; ~340 LOC of tests).
- `yapocis`: `build.rs` generating one typed struct per interface (~420 LOC), the library ports under `lib::*` (~7,350 LOC), integration and parity tests (~2,630 LOC), and examples mirroring the Python ones (~3,730 LOC).

**Defects.** Defects found and fixed during the initial port were not itemised at the time and are not reconstructed here.
