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
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use yapocis_bundle::Backend;

pub use args::Args;
pub use kernels::Kernel;

/// The device name: what `YAPOCIS_DEVICE` matches and what the parity tests slug.
pub const DEVICE_NAME: &str = "cpu";

/// A buffer in host memory: its own bytes, or (from [`RawDevice::wrap_host`],
/// for the length of one call) the caller's array, used in place so a call
/// copies nothing. Either way the bytes start 8-byte aligned, so any element
/// type up to that alignment can be viewed in place.
pub struct CpuBuffer {
    /// Readers and the one writer of the bytes during a launch. The bytes are
    /// not inside it because a wrapped buffer's bytes belong to the caller.
    lock: RwLock<()>,
    ptr: *mut u8,
    pub(crate) len: usize,
    /// Whether kernels may write the bytes (false for an `in` argument used in place).
    pub(crate) writable: bool,
    /// An owned buffer's storage (empty for a wrapped one); `ptr` points into it.
    _words: Vec<u64>,
}

// SAFETY: the bytes are only reached through `lock` (shared for reading,
// exclusive for writing), and an owned buffer's storage moves with it.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    fn new(len_bytes: usize) -> CpuBuffer {
        let mut words = vec![0u64; len_bytes.div_ceil(8)];
        CpuBuffer {
            lock: RwLock::new(()),
            ptr: words.as_mut_ptr() as *mut u8,
            len: len_bytes,
            writable: true,
            _words: words,
        }
    }

    /// The bytes, for as long as the read guard is held; `None` if a writer
    /// holds the buffer (a port took an output before an input bound to it).
    pub(crate) fn try_read(&self) -> Option<(RwLockReadGuard<'_, ()>, &[u8])> {
        let guard = self.lock.try_read().ok()?;
        // SAFETY: `ptr` is valid for `len` bytes while the buffer lives (its own
        // storage, or caller memory kept alive and unwritten for the call, see
        // `wrap_host`), and the read guard excludes the writer.
        Some((guard, unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }))
    }

    /// The bytes, writable, for as long as the write guard is held; `None` if
    /// the guard is already taken (the buffer is bound to two arguments).
    // `&mut` from `&self` is the point: the write guard returned with it makes
    // it the only access, as `RwLock::write` does for its own contents.
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn try_write(&self) -> Option<(RwLockWriteGuard<'_, ()>, &mut [u8])> {
        let guard = self.lock.try_write().ok()?;
        // SAFETY: as in `read`; the write guard makes this the only access.
        Some((guard, unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }))
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
        let (_guard, bytes) = b
            .try_write()
            .ok_or_else(|| Error::backend(Backend::RustCpu, "upload", "buffer in use"))?;
        bytes[..data.len()].copy_from_slice(data);
        Ok(())
    }

    fn download(&self, buffer: &dyn RawBuffer, out: &mut [u8]) -> Result<()> {
        let b = cpu_buffer(buffer)?;
        let (_guard, bytes) = b
            .try_read()
            .ok_or_else(|| Error::backend(Backend::RustCpu, "download", "buffer in use"))?;
        let n = out.len();
        out.copy_from_slice(&bytes[..n]);
        Ok(())
    }

    fn runs_on_host(&self) -> bool {
        true
    }

    /// Run kernels on host memory in place (8-byte aligned; anything else is
    /// copied as on a GPU).
    unsafe fn wrap_host(
        &self,
        ptr: *mut u8,
        len: usize,
        writable: bool,
    ) -> Option<Box<dyn RawBuffer>> {
        if len == 0 || !(ptr as usize).is_multiple_of(8) {
            return None;
        }
        Some(Box::new(CpuBuffer {
            lock: RwLock::new(()),
            ptr,
            len,
            writable,
            _words: Vec::new(),
        }))
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
