//! The backend abstraction: untyped bytes in, untyped bytes out.
//!
//! Each backend module implements [`RawDevice`]; everything above it (buffers,
//! programs, argument derivation) is backend-agnostic.

use crate::error::Result;
use std::any::Any;
use yapocis_bundle::Backend;

/// One kernel argument as the backend sees it.
pub enum RawArg<'a> {
    /// A device buffer (OpenCL `cl_mem`, Metal `MTLBuffer`, CUDA device pointer).
    Buffer(&'a dyn RawBuffer),
    /// A scalar passed by value (OpenCL `clSetKernelArg`, Metal `setBytes`,
    /// CUDA kernel parameter). The bytes are the value's native representation.
    Bytes(&'a [u8]),
}

pub trait RawBuffer: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn len_bytes(&self) -> usize;
}

pub trait RawProgram: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

/// Something to wait on after a launch.
pub trait RawEvent: Send {
    fn wait(&self) -> Result<()>;
}

pub trait RawDevice: Send + Sync {
    fn backend(&self) -> Backend;
    /// Human-readable device name (what `YAPOCIS_DEVICE` matches against).
    fn name(&self) -> String;
    /// Vendor, driver and platform versions: part of the compiled-program cache key.
    fn identity(&self) -> String;
    /// Compile a rendered source with the given build options. `cache_key` is a
    /// stable hash of source + options + device the backend may use to store a
    /// binary under [`crate::cache`].
    fn compile(
        &self,
        interface: &str,
        source: &str,
        options: &[String],
        cache_key: &str,
    ) -> Result<Box<dyn RawProgram>>;
    fn alloc(&self, len_bytes: usize) -> Result<Box<dyn RawBuffer>>;
    /// Whether [`RawDevice::wrap_host`] may return buffers: kernels run on
    /// host memory (the CPU backend). False for every GPU backend.
    fn runs_on_host(&self) -> bool {
        false
    }
    /// A software stand-in for a GPU: Windows' WARP ("Microsoft Basic Render
    /// Driver"), which the OpenCL, OpenGL and Vulkan Compatibility Pack offers
    /// as an OpenCL device on a machine with no GPU driver. It works, but far
    /// slower than the CPU backend, so [`crate::Device::default`] takes it only
    /// when no other backend opens a device.
    fn is_software_renderer(&self) -> bool {
        false
    }
    /// A buffer over `len` bytes of host memory at `ptr`, used in place by
    /// backends whose kernels can read host memory (the CPU backend), so a
    /// call copies nothing; `None` (the default, every GPU backend) means
    /// upload and download as usual.
    ///
    /// # Safety
    ///
    /// The memory must stay valid, and nothing else may write it (nor read it,
    /// if `writable`), until the returned buffer is dropped. With
    /// `writable == false` the backend never writes it.
    unsafe fn wrap_host(
        &self,
        _ptr: *mut u8,
        _len: usize,
        _writable: bool,
    ) -> Option<Box<dyn RawBuffer>> {
        None
    }
    fn upload(&self, buffer: &dyn RawBuffer, data: &[u8]) -> Result<()>;
    fn download(&self, buffer: &dyn RawBuffer, out: &mut [u8]) -> Result<()>;
    /// Enqueue one launch. `global` and `local` are already resolved (1 to 3 axes).
    fn launch(
        &self,
        program: &dyn RawProgram,
        symbol: &str,
        args: &[RawArg<'_>],
        global: &[usize],
        local: Option<&[usize]>,
    ) -> Result<Box<dyn RawEvent>>;
    /// Block until every launch so far has completed.
    fn wait_all(&self) -> Result<()>;
}
