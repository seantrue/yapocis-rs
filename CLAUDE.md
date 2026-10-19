# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

Rust runtime for the [yapocis](https://github.com/seantrue/Yapocis3) GPU kernels (OpenCL, Metal, CUDA), with no Python at runtime. The Python package is the single source of truth: `yapocis-tools bundle` renders its interfaces and templates into `crates/yapocis/bundle/`, which is vendored here and never edited by hand.

- `crates/yapocis-bundle`: manifest schema, argument derivation, dispatch sizing (no GPU).
- `crates/yapocis-rt`: devices, buffers, compilation cache, launches, the per-device buffer cache; features `ocl`, `metal`, `cuda`.
- `crates/yapocis`: generated typed API (`build.rs` reads the manifest), library ports under `lib::*` (each cites the Python file it mirrors), examples.

## Commands

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --features yapocis/ocl,yapocis/metal -- -D warnings   # macOS
cargo test --release --features yapocis/ocl,yapocis/metal                                     # thresher (this Mac)
cargo test -p yapocis-bundle                                                                  # no GPU
```

CUDA and NVIDIA OpenCL run on yellowfin (Linux, RTX 3080 default, plus a 3060 and a P40), from a clone of this repository:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64 YAPOCIS_MODELS=/data/models/yapocis3
cargo test --release --features yapocis/cuda,yapocis/ocl
```

## Bundle

- `crates/yapocis/bundle/SOURCE` names the Yapocis3 ref the bundle came from; CI's drift job installs it and runs `scripts/check-bundle.sh`.
- To change a kernel, change it in Yapocis3, then regenerate here from that commit and update `SOURCE`.
- Parity fixtures are recorded per device under `bundle/parity/<backend>/<device>/` and `bundle/parity/lib/<backend>/<device>/`. Record them on the machine that has the device (`--parity`, `--parity-lib`); tests compare only against fixtures from their own device.

## Constraints

- Do not fix tests by switching to another backend or device. The bug is still a bug.
- Do not "fix" CUDA orientation or indexing: row/column major lives in the kernel templates (`helpers.mako`), and x = column, y = row. Ask first.
- Derivation and dispatch semantics must match Python exactly, quirks included; parity tests are the drift detector.
- CI is Linux only (no macOS runners). Metal is checked locally.
- Create commits only when asked. Update `AI.md` (the AI contributions ledger) with every commit.
