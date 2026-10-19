//! The yapocis kernel bundle: what a runtime needs to call yapocis kernels
//! without Python.
//!
//! A bundle is produced by `yapocis-tools bundle` from the Python package. It
//! holds a `manifest.json` (this crate's [`Manifest`]) plus the rendered OpenCL
//! C, Metal and CUDA sources per interface. This crate has no GPU dependencies:
//! it parses the manifest and reproduces the host-side rules the Python backends
//! apply at call time:
//!
//! * [`derive`]: turning user arguments into the kernel argument list
//!   (`width_of`/`height_of`/... values, `out_like` allocations, argument order),
//!   as `yapocis.core.kernel_base` does;
//! * [`dispatch`]: evaluating `dispatch_size` trait expressions, as
//!   `yapocis.core.dispatch_expr` does (including its quirks);
//! * [`launch`]: per-backend default global sizes and the CUDA grid/block and
//!   Metal grid/threadgroup geometry.
//!
//! The Python package also exports `fixtures/derivation.json`, which the tests
//! replay to prove these rules match the reference implementation.

pub mod derive;
pub mod dispatch;
pub mod launch;
pub mod manifest;

pub use derive::{derive_args, ArgShape, DeriveError, KernelArg, ScalarValue};
pub use dispatch::{dispatch_global_size, DispatchError};
pub use launch::{cuda_grid_block, default_global_size, metal_grid_tgs, Backend, Launch};
pub use manifest::{
    DType, Dim, DispatchNode, InterfaceEntry, KernelEntry, Manifest, ManifestError, ParamEntry,
    ParamKind, Role, SourceEntry, SCHEMA_VERSION,
};
