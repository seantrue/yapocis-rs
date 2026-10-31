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
