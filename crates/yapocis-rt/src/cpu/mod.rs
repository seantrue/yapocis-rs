//! The CPU backend: the bundle's kernels as plain Rust, run on the host.
//!
//! This is a fallback for machines with no GPU and no OpenCL driver (not even
//! a software one such as PoCL). Everything above [`RawDevice`] (argument
//! derivation, dispatch sizes, the buffer cache, the library ports) is shared
//! with the GPU backends; only the bottom layer changes:
//!
//! * a buffer is host memory ([`CpuBuffer`]);
//! * "compiling" an interface looks its kernels up in [`kernels`], where each
//!   one is a Rust port of the bundle's OpenCL kernel of the same symbol;
//! * a launch calls that function synchronously with the arguments and the
//!   global size, and the returned event is already complete.
//!
//! The ports follow the OpenCL sources, quirks included (boundary tests,
//! integer types, the order of floating-point operations where it is visible),
//! because the parity fixtures and the library ports were made against them.
//! The OpenCL source is what [`Backend::RustCpu`] loads (see
//! [`Backend::source_backend`]): the convolution families read their weights
//! from it.
//!
//! A kernel's work items run in parallel only where each one writes its own
//! outputs and nothing else ([`par`]); a kernel whose work items share
//! outputs (histograms, atomics, scatter writes) runs them in order.

pub mod args;
pub mod kernels;
pub mod par;

use crate::error::{Error, Result};
use crate::raw::{RawArg, RawBuffer, RawDevice, RawEvent, RawProgram};
use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::RwLock;
use yapocis_bundle::Backend;

pub use args::Args;
pub use kernels::Kernel;

/// The device name: what `YAPOCIS_DEVICE` matches and what the parity tests slug.
pub const DEVICE_NAME: &str = "cpu";

/// A buffer in host memory. The bytes live in `u64` words so any element
/// type up to 8-byte alignment can be viewed in place.
pub struct CpuBuffer {
    pub(crate) words: RwLock<Vec<u64>>,
    pub(crate) len: usize,
}

impl CpuBuffer {
    fn new(len_bytes: usize) -> CpuBuffer {
        CpuBuffer {
            words: RwLock::new(vec![0u64; len_bytes.div_ceil(8)]),
            len: len_bytes,
        }
    }
}

impl RawBuffer for CpuBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn len_bytes(&self) -> usize {
        self.len
    }
}

/// One interface's kernels, by symbol.
pub struct CpuProgram {
    interface: String,
    kernels: HashMap<String, Kernel>,
}

impl RawProgram for CpuProgram {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Launches run to completion before `launch` returns.
struct Done;

impl RawEvent for Done {
    fn wait(&self) -> Result<()> {
        Ok(())
    }
}

/// The host CPU as a device.
pub struct CpuDevice;

impl CpuDevice {
    /// There is one CPU device; `selector` may be `None`, `"0"` or its name.
    pub fn open(selector: Option<&str>) -> Result<CpuDevice> {
        let names = [DEVICE_NAME.to_string()];
        crate::device::select(Backend::RustCpu, &names, selector, |_| Some(0))?;
        Ok(CpuDevice)
    }
}

fn bytes(words: &[u64], len: usize) -> &[u8] {
    &bytemuck::cast_slice(words)[..len]
}

fn bytes_mut(words: &mut [u64], len: usize) -> &mut [u8] {
    &mut bytemuck::cast_slice_mut(words)[..len]
}

fn cpu_buffer(b: &dyn RawBuffer) -> Result<&CpuBuffer> {
    b.as_any()
        .downcast_ref::<CpuBuffer>()
        .ok_or_else(|| Error::backend(Backend::RustCpu, "buffer", "not a CPU buffer"))
}

impl RawDevice for CpuDevice {
    fn backend(&self) -> Backend {
        Backend::RustCpu
    }

    fn name(&self) -> String {
        DEVICE_NAME.to_string()
    }

    fn identity(&self) -> String {
        format!("yapocis-rt cpu {}", env!("CARGO_PKG_VERSION"))
    }

    fn compile(
        &self,
        interface: &str,
        source: &str,
        _options: &[String],
        _cache_key: &str,
    ) -> Result<Box<dyn RawProgram>> {
        let kernels = kernels::for_interface(interface, source).map_err(|log| Error::Compile {
            backend: Backend::RustCpu,
            interface: interface.to_string(),
            log,
        })?;
        Ok(Box::new(CpuProgram {
            interface: interface.to_string(),
            kernels: kernels.into_iter().collect(),
        }))
    }

    fn alloc(&self, len_bytes: usize) -> Result<Box<dyn RawBuffer>> {
        Ok(Box::new(CpuBuffer::new(len_bytes)))
    }

    fn upload(&self, buffer: &dyn RawBuffer, data: &[u8]) -> Result<()> {
        let b = cpu_buffer(buffer)?;
        let mut words = b.words.write().unwrap_or_else(|e| e.into_inner());
        bytes_mut(&mut words, b.len)[..data.len()].copy_from_slice(data);
        Ok(())
    }

    fn download(&self, buffer: &dyn RawBuffer, out: &mut [u8]) -> Result<()> {
        let b = cpu_buffer(buffer)?;
        let words = b.words.read().unwrap_or_else(|e| e.into_inner());
        let n = out.len();
        out.copy_from_slice(&bytes(&words, b.len)[..n]);
        Ok(())
    }

    fn launch(
        &self,
        program: &dyn RawProgram,
        symbol: &str,
        args: &[RawArg<'_>],
        global: &[usize],
        local: Option<&[usize]>,
    ) -> Result<Box<dyn RawEvent>> {
        let program = program
            .as_any()
            .downcast_ref::<CpuProgram>()
            .ok_or_else(|| Error::backend(Backend::RustCpu, "launch", "not a CPU program"))?;
        let kernel = program.kernels.get(symbol).ok_or_else(|| {
            Error::backend(
                Backend::RustCpu,
                "launch",
                format!("{}: kernel {symbol:?} has no CPU port", program.interface),
            )
        })?;
        let args = Args::new(args, global, local);
        // A port that indexes out of bounds panics; report it as a launch error.
        catch_unwind(AssertUnwindSafe(|| kernel(&args))).map_err(|panic| {
            let detail = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "kernel panicked".to_string());
            Error::backend(
                Backend::RustCpu,
                "launch",
                format!("{}.{symbol}: {detail}", program.interface),
            )
        })?;
        Ok(Box::new(Done))
    }

    fn wait_all(&self) -> Result<()> {
        Ok(())
    }
}
