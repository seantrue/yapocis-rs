# AI Contributions — yapocis-rs

*About AI slop.* Fair enough: nearly all of the code in this repository was written by a machine, and so was most of this file. Call it instant ramen. It is fast, it is cheap, it is genuinely edible, and nobody should pretend it simmered on a stove all afternoon. The honest answer is not a claim about quality but the record below: what the maintainer decided, what the model wrote, what the model got wrong, and which of those mistakes were caught by the parity tests replaying Python's recorded outputs, by running on real GPUs on two machines, or by the maintainer rather than by luck. Where AI-written code had a defect, it is written down here in the same detail as the wins. Bon appétit, but mind the sodium content.

Tracks what AI helpers contributed to yapocis-rs versus the human maintainer.

## Human maintainer

The maintainer (Sean True) owns yapocis, the Python package whose kernels this repository runs, and set the direction and the design constraints of the port. The Rust code itself was written by Claude from the start; there is no pre-AI code in this repository. The maintainer's decisions (2026-10-18, before any code was written):

- The Rust code lives in a separate repository, consuming a kernel bundle that the Python package exports at build time.
- Runtime-parameterised kernels (arbitrary convolution weights) are handled by enumerating variants at build time.
- The first release covers all interface kernels plus the library wrappers.
- CUDA goes through NVRTC via `cudarc`, pinned to the CUDA 12.x bindings so the Pascal P40 on yellowfin keeps working.

Later direction and corrections are credited in the entries below.

## Contributions

### 2026-10-19 — Repository prepared for its first commit (Claude Opus 5.5)

**What the maintainer asked for.** "Best way to make yapocis-rs ready for checkin as a separate project and git repo (private)?", then, on Claude's seven-step answer: "do 1-5. For 6, do not enable MacOS CI at all. Then 7."

**What the AI did.**

- **Bundle provenance:** the exporter was committed to Yapocis3 first (`d0fc1ef`, PR #54; that repository's AI.md has the entry). The bundle was regenerated from that commit, and `crates/yapocis/bundle/SOURCE` pins it by SHA, which CI's drift job installs. No tag was made: the commit is reachable on its branch now and from `main` once the PR is merged. The regenerated manifest, sources and derivation fixtures were byte-identical to the vendored ones, so only the parity fixtures changed. Exports on thresher (macOS) and yellowfin (Linux) were also identical except for `yapocis_version`, which is read from the installed package metadata; yellowfin's venv has an old 0.3.8 install, so the vendored manifest is thresher's (1.0.0).
- **Per-device parity fixtures:** fixtures moved from `parity/<backend>/` to `parity/<backend>/<device>/` (and the same under `parity/lib/`). The test helpers open the backend's default device, take its name as a directory slug (`device_slug`, matching the Python exporter's rule, with a unit test), and skip with a message when nothing was recorded on that device. Sets are vendored for the Apple M4 Max (Metal, OpenCL) and the RTX 3080 (CUDA, NVIDIA OpenCL).
- **Cleanup:** the throwaway Phase 0 `spike/` crate left the workspace (archived in Yapocis3's untracked `claude-debug/rust_port_spike/`). Added `publish = false` on the three crates, `.idea/` in `.gitignore`, `cargo fmt` across the workspace (2 files were unformatted), this ledger and `CLAUDE.md`.
- **CI actions on Node 24** (maintainer: "bump the actions to Node 24 versions", after GitHub warned that `checkout@v4` and `setup-python@v5` target the deprecated Node 20): now `actions/checkout@v7` and `actions/setup-python@v7`, checked to declare `using: node24`. `Swatinem/rust-cache@v2` already runs on Node 24, and `dtolnay/rust-toolchain` is a composite action.
- **CI:** Linux only, as the maintainer specified. It builds the OpenCL and CUDA backends, runs clippy with `-D warnings`, rustfmt and the device-free tests, and checks the bundle for drift. It installs `ocl-icd-opencl-dev` because `opencl3` links `libOpenCL` at build time.

**Verification.**

- 72 of 72 tests pass on thresher (Metal, OpenCL) and on yellowfin (CUDA, NVIDIA OpenCL) against the vendored per-device fixtures. That includes the slug test, and the two cross-device failures below no longer occur.
- `scripts/check-bundle.sh` reports the bundle up to date against `d0fc1ef`.
- On yellowfin (Linux), `cargo fmt --check`, clippy with `-D warnings` (OpenCL, CUDA), the device-free tests and `cargo doc` all pass with the CI commands. `cargo doc` had 2 warnings (an unescaped `[2]` and a redundant link target), both fixed. The workflow itself had not run on GitHub at the time of writing.

**Defects in AI-written code.**

- The parity fixtures were keyed by backend only. OpenCL fixtures recorded on the Apple M4 Max failed two tests on NVIDIA's OpenCL (`parity_with_python_outputs`: one Mandelbrot pixel at 6 vs 7 iterations; `simplex_matches_python`: `noise3d` element 223). This was caught by running the suite on yellowfin. With fixtures recorded on the RTX 3080 both tests pass, so it was a cross-device comparison rather than a runtime bug. The per-device layout above fixes it.
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
- **Huge-page hint enabled by default.** The first version of the fix turned the hint on. On yellowfin (`defrag=madvise`) it made some processes stall in synchronous compaction: 220 ms instead of 43 ms per 64 MB output, with 377 `compact_stall`s per run. Caught by bimodal benchmark runs; the hint is now opt-in.
- **pocl segfault.** Running the tests with `ocl` on yellowfin segfaulted about 1 run in 7, inside pocl's `clGetDeviceInfo`, while parallel test threads enumerated devices. Caught on the first OpenCL run there; the open lock fixed it (0 failures in 30 runs).
- **Resident parameters written on every call.** The untyped API uploaded a host array passed to a `resident` parameter on every call. Caught by the maintainer.
- **`set_margin` visited every pixel.** It visited every pixel with a bounds-checked write: 12.9 ms at 2048² on yellowfin and 4.1 ms on thresher, now 0.66 ms and 0.15 ms. Caught by a per-stage timing of `zcsdog`.

**Found and not changed (library behaviour shared with Python).**

- `Gaussian::zcsdog`'s host and resident forms give different edge maps for about 0.45% of pixels. The two blur kernel families treat the image border differently, and the cleared margin is sized for the smaller kernel only.
- Python documents a 256-point limit for `fft.batch_forward` but does not enforce it; Rust rejects larger sizes.
- The first CUDA call after load takes 10.9 ms in Rust and 1.1 ms in Python. The likely cause is the driver compiling PTX on first use; this is not verified.

**Verification.** 71 of 71 tests pass on thresher (Metal, OpenCL) and on yellowfin (CUDA, NVIDIA OpenCL, with OpenCL fixtures recorded there). Clippy is clean on both.

### 2026-10-18 — The port: bundle crate, runtime, generated API and library ports (Claude Opus 5.5)

**What the maintainer asked for.** An evaluation of how to run the yapocis kernels from Rust with no Python at runtime, then the implementation of the plan that came out of it. The evaluation report is in Yapocis3, `claude-debug/rust_port_evaluation_2026-10-18.md`.

**What the AI did.** Lines measured on 2026-10-19 over the whole repository (both sessions' work), excluding the vendored bundle and fixture data:

- `yapocis-bundle`: manifest schema, derivation and dispatch evaluation (~1,010 LOC; ~250 LOC of tests against Python's derivation fixtures).
- `yapocis-rt`: the OpenCL (`opencl3`), Metal (`objc2-metal`) and CUDA (`cudarc`, NVRTC) backends, the program cache and launches (~2,490 LOC; ~340 LOC of tests).
- `yapocis`: `build.rs` generating one typed struct per interface (~420 LOC), the library ports under `lib::*` (~7,350 LOC), integration and parity tests (~2,630 LOC), and examples mirroring the Python ones (~3,730 LOC).

**Defects.** Defects found and fixed during the initial port were not itemised at the time and are not reconstructed here.
