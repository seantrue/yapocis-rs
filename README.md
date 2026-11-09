# yapocis-rs

*yapocis-rs* provides platform independent Rust access to GPU kernels running in OpenCL, Metal, and Cuda, that is based on the Python package [yapocis](https://github.com/seantrue/Yapocis3) GPU image-processing kernels. There is _no_ requirement for Python at Rust run time.

The Python package stays the single source of truth for the kernels: its
interface definitions, Mako templates and YAML contexts are rendered by
`yapocis-tools bundle` into a kernel bundle (`manifest.json` plus the rendered
OpenCL C, MSL and CUDA C sources) that is vendored here. A `build.rs` generates
one typed Rust struct per interface from the manifest, and the runtime compiles
the unchanged kernel sources on the device through `opencl3`, `objc2-metal` and
`cudarc` (NVRTC).

```rust
use ndarray::Array2;
use yapocis::{Device, interfaces::gradient::Gradient, lib::Median};

let device = Device::default()?;                 // YAPOCIS_BACKEND / YAPOCIS_DEVICE
let image = Array2::<f32>::zeros((640, 480)).into_dyn();

let gradient = Gradient::load(&device)?;         // generated typed API
let (grad, theta) = gradient.gradient(&image.view(), 3)?;

let median = Median::load(&device)?;             // library port
let smoothed = median.median3x3(&image.view(), 5)?;
# Ok::<(), yapocis::Error>(())
```

## Crates

| Crate | Contents |
|---|---|
| `yapocis` | Public API: generated interface structs, library ports (`lib::*`), the vendored bundle |
| `yapocis-rt` | Runtime: devices, buffers, program compilation and cache, launches; features `ocl`, `metal`, `cuda`, and `rustcpu` (Rust ports of every kernel, for machines with no GPU) |
| `yapocis-bundle` | Manifest schema, argument derivation and dispatch evaluation, and `codegen`, the typed-API generator any crate's `build.rs` can run on a bundle; no GPU dependencies |
| `npexact` | Bit-exact ports of numpy host numerics: `SeedSequence`/`PCG64`/`Generator` draws, pairwise `sum`/`mean`/`var`/`std`, numpy's own float32 `sin`/`cos` and `tanh`, `histogram`/`linspace`/`interp`, `minimum`/`maximum`/`clip`/`sign`/`mod`/`nan_to_num`, `min`/`max`/`nanmin`/`nanmax`, `sort`/`searchsorted`/`lexsort`/`quantile`/`median`/`nanpercentile`/`unique`, `fftshift`, complex64 multiply; proven by per-host fixtures (`yapocis-tools bundle --host-fixtures`) compared on bit patterns, recorded on arm64 (M4 Max) and x86-64 (Xeon E5-2698 v4, AVX2) |
| `npzfile` | numpy `.npz` reading and writing for ndarray, with ndarray-npy's `NpzReader`/`NpzWriter` API on `zip` 9 (ndarray-npy's own `npz` feature pins `zip` 6, which has open advisories); used for weights and fixtures |

Enable backends with Cargo features: `yapocis = { version = "0.1", features = ["metal"] }`.
`metal` builds on macOS and (experimentally) iOS; `cuda` loads `libcuda` and `libnvrtc` at runtime and is
pinned to the CUDA 12.x bindings so Pascal cards still work.

## The CPU backend (`rustcpu`)

The `rustcpu` feature adds a fallback for machines with no GPU and no OpenCL
driver (not even a software one such as PoCL): every bundled kernel ported to
plain Rust, under `yapocis-rt/src/cpu/kernels/`, one module per interface and
one function per OpenCL kernel, with the same arguments in the same order.
Without `YAPOCIS_BACKEND`, it is chosen only when no GPU backend opens a
device; `YAPOCIS_BACKEND=rustcpu` selects it explicitly.

- The ports follow the OpenCL sources, quirks included, not the numbax
  kernels (which pad and guard differently); the parity fixtures and library
  ports were made against OpenCL. Only the CPU backend's bottom layer is new:
  argument derivation, dispatch sizes, the buffer cache and the library ports
  are shared with the GPU backends.
- It loads each interface's OpenCL text, from which the convolution families
  read their rendered weights, so runtime weights (`Convolve1D`,
  `Gaussian::with_weights`) work too.
- Work items run in parallel (rayon; `RAYON_NUM_THREADS`) only where each
  writes just its own outputs; histograms, scatter writes and atomics run in
  order.
- It cannot compile kernel source given at run time (inline kernels need a
  GPU backend).
- A crate with its own bundle (see "A second bundle") adds CPU ports for its
  interfaces with `yapocis_rt::cpu::kernels::register(name, provider)` before
  loading them; impressionist-rs's yapocis-external does this from its `host::load`
  (generated with `yapocis_path: "crate"`).

### From Python: Yapocis3's `rustcpu` backend

`bindings/python` builds a Python extension module, `yapocis_rs` (PyO3,
maturin), that Yapocis3 uses as its `rustcpu` backend
(`YAPOCIS_BACKEND=rustcpu`). Yapocis3 renders each interface's manifest entry and
OpenCL source as `yapocis-tools bundle` would; the module loads them on the CPU
device and calls kernels with NumPy arrays, with argument derivation and dispatch
done here. It is outside the Cargo workspace (an extension module cannot link as
a test binary):

```bash
cd bindings/python
uvx maturin build --release -i /path/to/venv/bin/python --out dist
uv pip install --python /path/to/venv/bin/python --no-deps dist/yapocis_rs-*.whl
```

Install with `--no-deps`: the wheel needs NumPy, and letting the installer pick
one can replace the version the venv's Numba and fixtures were made with.
Interfaces outside this bundle (project interfaces) have no CPU port and fail to
load with "has no CPU port".

A crate with its own bundle and ports builds its own `yapocis_rs` module instead:
it depends on `yapocis-py` (this crate, with `default-features = false` to leave out
the module entry point), registers its ports, and calls
`yapocis_rs::add_to_module`. Ports and bindings must share one yapocis-rt, so both
come from the same yapocis-rs revision. impressionist-rs's `bindings/python` does
this for yapocis_external (distribution `yapocis-rs-external`); install that wheel
or this one, not both.

## Calling conventions

The generated methods follow the Python runtime:

- `in` parameters take `&ArrayViewD<T>` (copied to standard layout if needed);
- `inout` and `out` take `&mut ArrayViewMutD<T>` and are written in place;
- `resident` take `&Buffer<T>`, a device buffer you keep across calls;
- scalars take their element type; derived dimensions (`width_of a`, ...) are never passed;
- `out_like` outputs are returned as new `ArrayD<T>`.

Every kernel also has a `<name>_with(&CallOptions, ...)` form for explicit global
and local sizes or asynchronous launches, and `Program::call` takes untyped
arguments for the library ports and advanced use.

## Testing

```bash
cargo test -p yapocis-bundle                                  # no GPU needed
cargo test --workspace --features yapocis/metal,yapocis/ocl   # macOS
cargo test --workspace --features yapocis/cuda,yapocis/ocl    # Linux + NVIDIA
cargo test --release -p yapocis --features rustcpu                # anywhere: the CPU backend
```

The tests replay two kinds of fixtures generated by the Python package:
`bundle/fixtures/derivation.json` (what Python derives from arguments) and
recorded kernel and library outputs under `bundle/parity/<backend>/<device>/`
and `bundle/parity/lib/<backend>/<device>/`, so drift between the two runtimes
fails a test. Results can differ between devices in the last bit, so a test
compares only against fixtures recorded on the device it runs on (the directory
is the device name, lowercased, with runs of other characters as `-`), and
skips with a message when there are none.

The CPU backend has no Python recorder, so it replays the fixtures recorded on
a CPU OpenCL device, PoCL's (`ocl/cpu`), or the set `YAPOCIS_CPU_REFERENCE`
names (e.g. `ocl/apple-m4-max`). A fixture known to predate a kernel change is
listed, with its SHA-256, in `STALE_CPU_FIXTURES` in `tests/parity.rs` and
skipped until it is re-recorded. The `tests/cpu_*.rs` tests compare the CPU
backend with OpenCL and Metal on the same machine for kernels the fixtures do
not cover, and skip when neither is built.

CI (Linux, no GPU) builds the OpenCL, CUDA and CPU backends, runs clippy, rustfmt,
the device-free tests and every kernel and library test on the CPU backend, and
checks the vendored bundle for drift. Metal and the GPU tests run locally.

Setting `YAPOCIS_PARITY_EXACT=1` runs the library parity tests with zero
tolerance, to see which ports match Python bit for bit on this device (on the
M4 Max, every one that Virtual Impressionist uses; FFT coherence differs).

`npexact`'s tests compare against `crates/npexact/fixtures/host/<host slug>/`,
recorded by `yapocis-tools bundle --host-fixtures crates/npexact/fixtures/host`
on each host (numpy's SIMD dispatch, libm and FMA contraction differ by CPU, and
the index records which applies), and skip with a message on a host without them.
Where a difference is a policy rather than hardware (`Contraction`, the
`MinMax` tie between signed zeros, the `Quadrant` rounding of `sin`/`cos`), the
function takes it as a parameter with a `host_default()`, and the `ops` tests
replay every recorded host's `elementwise`, `extrema`, `order` and `fft` cases on
any machine. The complex64 multiply's fusing (`complex::Mul`) is the exception:
numpy picks its SIMD ufunc loop when it runs, so `Mul::array_default()` asks the
CPU (fused on arm64, and on x86-64 with FMA3 and AVX2), while
`Mul::scalar_default()` follows the build like the other policies.
`quantile_linear_f32` follows NumPy 2 (a float32 index for a Python-float `q`);
`quantile_linear_f32_f64q` keeps the float64 index for results recorded that way.

## A second bundle

Another project's kernels (found by the Python runtime through
`YAPOCIS_PROJECT`) can be vendored by another crate: export them with
`yapocis-tools bundle --project-only`, and have that crate's `build.rs` call
`yapocis_bundle::codegen::generate` on the bundle, defining its own `MANIFEST`
static. impressionist-rs does this for yapocis_external.

## Regenerating the bundle

`crates/yapocis/bundle/SOURCE` records the Python commit the bundle was
generated from; the CI drift job installs that ref. From a checkout of the
Python package at that ref:

```bash
yapocis-tools bundle --out crates/yapocis/bundle --fixtures
```

Parity fixtures need a device, so record them on each machine for its own
devices, e.g. `--parity metal,ocl --parity-lib metal,ocl` on a Mac and
`--parity cuda,ocl --parity-lib cuda,ocl` on an NVIDIA machine.

`scripts/check-bundle.sh` regenerates into a temporary directory and compares
the manifest, sources and derivation fixtures with the vendored copy
(`scripts/compare-bundle.py`). Float literals are compared to 12 significant
digits, because numpy can round the rendered Gaussian weights differently in the
last bit on different CPUs; everything else must match exactly.

## Scope

Every bundled interface kernel is callable through the generated API, and the
whole Python library layer is ported under `lib::*`:

| Module | Python source | Notes |
|---|---|---|
| `operators`, `gradient`, `hsi`, `colormath`, `zcs`, `median`, `mandelbrot` | the thin wrappers | per-depth kernel selection, explicit global sizes, device ping-pong |
| `anisodiff` | `anisodiff.py` | `anisodiff_n`, the iterator, and the RGB intensity/saturation variants |
| `gaussian`, `convolve` | `gaussian.py`, `convolve.py` | default weight ladder from the bundle; arbitrary runtime weights through a Rust port of the `convolve` and `convolvesep` templates |
| `fft` | `fft.py` | FFTs, batches, windows, power spectra, `analyze_signal`, `spectrogram`, coherence/energy/RMS/SNR |
| `imageops`, `simplex` | `imageops.py`, `simplex.py` | including host-side `histeq`/`histmatch` and the turbulence/ridged noise variants |
| `truetrace` | `truetrace.py` | GPU jump-flood + marching squares and the full CPU pipeline (Otsu, EDT, polylines, Douglas-Peucker, holes, Béziers, SVG) |
| `graphics`, `kerneldraw`, `typeset`, `paths` | `abstract_graphics.py`, `graphics.py`, `graphics4.py`, `kerneldraw*.py`, `typeset.py`, `mplbackend.py` | display lists, sprites and composites, FreeType text (bundled FreeType, feature `text`), filled-path rendering with earcut; the matplotlib `RendererBase` plugin itself stays in Python |
| `cnn`, `vgg16`, `resnet50` | `vgg16.py`, `resnet50.py` | weights from `YAPOCIS_MODELS/<model>/weights.npz`, preprocessing, labels, top-k |

Not portable by nature: the numba (Python) backend and the matplotlib backend plugin.
