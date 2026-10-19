//! Runtime for yapocis kernel bundles.
//!
//! A [`Device`] wraps one OpenCL, Metal or CUDA device (behind the `ocl`,
//! `metal` and `cuda` Cargo features). A [`Program`] is one interface variant
//! from a bundle compiled for that device; [`Program::call`] turns user
//! arguments into the kernel argument list exactly as the Python runtime does
//! (through `yapocis-bundle`), uploads, launches, and reads outputs back.
//!
//! The contract matches `yapocis.core.kernel_base` and the backend `kernel.py`
//! modules. Device memory can be explicit (a [`Buffer`] is a handle you keep,
//! and `resident` parameters take buffers), and host-array arguments are bound
//! to cached device buffers across calls as the Python device managers do
//! (see [`buffer_cache`]).

pub mod buffer;
pub mod buffer_cache;
pub mod cache;
pub mod device;
pub mod error;
pub mod host_data;
pub mod program;
pub mod raw;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(feature = "ocl")]
pub mod ocl;

pub use buffer::{Buffer, BufferAny, Element};
pub use buffer_cache::BufferCacheStats;
pub use device::{available_backends, Device};
pub use error::{Error, Result};
pub use host_data::HostData;
pub use program::{
    CallArg, CallOptions, CallOutput, Event, HostView, HostViewMut, Output, OutputValue, Program,
};
pub use yapocis_bundle::{self as bundle, Backend, DType, InterfaceEntry, KernelEntry, Manifest};
