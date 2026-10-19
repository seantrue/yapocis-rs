//! CUDA backend on `cudarc` (driver API + NVRTC), mirroring
//! `yapocis/cu/{device,device_manager,kernel}.py` with NVRTC in place of nvcc.

use crate::cache;
use crate::device::select;
use crate::error::{Error, Result};
use crate::raw::{RawArg, RawBuffer, RawDevice, RawEvent, RawProgram};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use yapocis_bundle::Backend;

const B: Backend = Backend::Cuda;

fn err(op: &'static str, e: impl std::fmt::Debug) -> Error {
    Error::backend(B, op, format!("{e:?}"))
}

pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    /// One stream per device manager, as in `cu/device_manager.py`.
    stream: Arc<CudaStream>,
    name: String,
    arch: String,
}

impl CudaDevice {
    pub fn open(selector: Option<&str>) -> Result<CudaDevice> {
        let count = CudaContext::device_count().map_err(|e| err("device_count", e))? as usize;
        let mut names = Vec::with_capacity(count);
        for i in 0..count {
            // Opening a context just to read the name is what pycuda's Device(i).name() avoids,
            // but cudarc exposes names through the context only.
            let ctx = CudaContext::new(i).map_err(|e| err("context", e))?;
            names.push(ctx.name().unwrap_or_else(|_| format!("cuda:{i}")));
        }
        // cu/device.py: the default is device 0
        let index = select(B, &names, selector, |_| Some(0))?;
        let ctx = CudaContext::new(index).map_err(|e| err("context", e))?;
        let stream = ctx.new_stream().map_err(|e| err("new_stream", e))?;
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| err("attribute", e))?;
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| err("attribute", e))?;
        Ok(CudaDevice {
            ctx,
            stream,
            name: names[index].clone(),
            arch: format!("compute_{major}{minor}"),
        })
    }
}

struct CudaProgram {
    module: Arc<CudaModule>,
    functions: Mutex<HashMap<String, CudaFunction>>,
}

impl RawProgram for CudaProgram {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct CudaBuffer {
    slice: Mutex<CudaSlice<u8>>,
    len: usize,
}

impl RawBuffer for CudaBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn len_bytes(&self) -> usize {
        self.len
    }
}

struct CudaRawEvent(CudaEvent);

impl RawEvent for CudaRawEvent {
    fn wait(&self) -> Result<()> {
        self.0
            .synchronize()
            .map_err(|e| err("event synchronize", e))
    }
}

/// Scalar kernel parameters of a given byte size. cuLaunchKernel copies
/// `size_of::<T>()` bytes from each parameter pointer, so only the size matters.
#[repr(C)]
#[derive(Clone, Copy)]
struct B1([u8; 1]);
#[repr(C)]
#[derive(Clone, Copy)]
struct B2([u8; 2]);
#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct B4([u8; 4]);
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct B8([u8; 8]);
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct B16([u8; 16]);
unsafe impl DeviceRepr for B1 {}
unsafe impl DeviceRepr for B2 {}
unsafe impl DeviceRepr for B4 {}
unsafe impl DeviceRepr for B8 {}
unsafe impl DeviceRepr for B16 {}

enum Scalar {
    B1(B1),
    B2(B2),
    B4(B4),
    B8(B8),
    B16(B16),
}

impl Scalar {
    fn new(bytes: &[u8]) -> Result<Scalar> {
        Ok(match bytes.len() {
            1 => Scalar::B1(B1(bytes.try_into().unwrap())),
            2 => Scalar::B2(B2(bytes.try_into().unwrap())),
            4 => Scalar::B4(B4(bytes.try_into().unwrap())),
            8 => Scalar::B8(B8(bytes.try_into().unwrap())),
            16 => Scalar::B16(B16(bytes.try_into().unwrap())),
            n => return Err(err("scalar arg", format!("unsupported scalar size {n}"))),
        })
    }
}

impl RawDevice for CudaDevice {
    fn backend(&self) -> Backend {
        B
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn identity(&self) -> String {
        format!("{} | {}", self.name, self.arch)
    }

    fn compile(
        &self,
        interface: &str,
        source: &str,
        options: &[String],
        cache_key: &str,
    ) -> Result<Box<dyn RawProgram>> {
        let ptx = match cache::read(B, cache_key, "ptx") {
            Some(text) => {
                tracing::debug!(interface, "PTX from cache");
                Ptx::from_src(String::from_utf8_lossy(&text).into_owned())
            }
            None => {
                let mut opts: Vec<String> = options.to_vec();
                opts.push(format!("--gpu-architecture={}", self.arch));
                let ptx = compile_ptx_with_opts(
                    source,
                    CompileOptions {
                        options: opts,
                        ..Default::default()
                    },
                )
                .map_err(|e| Error::Compile {
                    backend: B,
                    interface: interface.to_string(),
                    log: format!("{e:?}"),
                })?;
                cache::write(B, cache_key, "ptx", ptx.to_src().as_bytes());
                ptx
            }
        };
        let module = self
            .ctx
            .load_module(ptx)
            .map_err(|e| err("load_module", e))?;
        Ok(Box::new(CudaProgram {
            module,
            functions: Mutex::new(HashMap::new()),
        }))
    }

    fn alloc(&self, len_bytes: usize) -> Result<Box<dyn RawBuffer>> {
        let slice = self
            .stream
            .alloc_zeros::<u8>(len_bytes.max(1))
            .map_err(|e| err("alloc", e))?;
        Ok(Box::new(CudaBuffer {
            slice: Mutex::new(slice),
            len: len_bytes,
        }))
    }

    fn upload(&self, buffer: &dyn RawBuffer, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let b = buffer
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or(Error::ForeignBuffer)?;
        let mut guard = b.slice.lock().unwrap();
        let mut view = guard.slice_mut(0..data.len());
        self.stream
            .memcpy_htod(data, &mut view)
            .map_err(|e| err("memcpy_htod", e))?;
        // cu/device_manager.py synchronises after each write unless async_write
        self.stream.synchronize().map_err(|e| err("synchronize", e))
    }

    fn download(&self, buffer: &dyn RawBuffer, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let b = buffer
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or(Error::ForeignBuffer)?;
        let guard = b.slice.lock().unwrap();
        let view = guard.slice(0..out.len());
        self.stream
            .memcpy_dtoh(&view, out)
            .map_err(|e| err("memcpy_dtoh", e))?;
        self.stream.synchronize().map_err(|e| err("synchronize", e))
    }

    fn launch(
        &self,
        program: &dyn RawProgram,
        symbol: &str,
        args: &[RawArg<'_>],
        global: &[usize],
        local: Option<&[usize]>,
    ) -> Result<Box<dyn RawEvent>> {
        let p = program
            .as_any()
            .downcast_ref::<CudaProgram>()
            .ok_or(Error::ForeignBuffer)?;
        let func = {
            let mut functions = p.functions.lock().unwrap();
            if !functions.contains_key(symbol) {
                let f = p
                    .module
                    .load_function(symbol)
                    .map_err(|e| err("load_function", e))?;
                functions.insert(symbol.to_string(), f);
            }
            functions[symbol].clone()
        };
        let (grid, block) = yapocis_bundle::cuda_grid_block(global, local);
        let cfg = LaunchConfig {
            grid_dim: (grid[0] as u32, grid[1] as u32, grid[2] as u32),
            block_dim: (block[0] as u32, block[1] as u32, block[2] as u32),
            shared_mem_bytes: 0,
        };
        // Materialise scalars and buffer locks first so references stay valid
        // for the whole builder. A buffer may appear more than once in a call,
        // so each distinct buffer is locked once.
        let mut scalars: Vec<Option<Scalar>> = Vec::with_capacity(args.len());
        let mut lock_index: Vec<Option<usize>> = Vec::with_capacity(args.len());
        let mut locked: Vec<(*const CudaBuffer, std::sync::MutexGuard<'_, CudaSlice<u8>>)> =
            Vec::new();
        for arg in args {
            match arg {
                RawArg::Bytes(bytes) => {
                    scalars.push(Some(Scalar::new(bytes)?));
                    lock_index.push(None);
                }
                RawArg::Buffer(b) => {
                    let cb = b
                        .as_any()
                        .downcast_ref::<CudaBuffer>()
                        .ok_or(Error::ForeignBuffer)?;
                    let key = cb as *const CudaBuffer;
                    let idx = match locked.iter().position(|(k, _)| *k == key) {
                        Some(i) => i,
                        None => {
                            locked.push((key, cb.slice.lock().unwrap()));
                            locked.len() - 1
                        }
                    };
                    scalars.push(None);
                    lock_index.push(Some(idx));
                }
            }
        }
        let mut builder = self.stream.launch_builder(&func);
        for i in 0..args.len() {
            match (&scalars[i], lock_index[i]) {
                (Some(Scalar::B1(v)), _) => {
                    builder.arg(v);
                }
                (Some(Scalar::B2(v)), _) => {
                    builder.arg(v);
                }
                (Some(Scalar::B4(v)), _) => {
                    builder.arg(v);
                }
                (Some(Scalar::B8(v)), _) => {
                    builder.arg(v);
                }
                (Some(Scalar::B16(v)), _) => {
                    builder.arg(v);
                }
                (None, Some(idx)) => {
                    builder.arg(&*locked[idx].1);
                }
                (None, None) => unreachable!(),
            }
        }
        unsafe { builder.launch(cfg) }.map_err(|e| err("launch", e))?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|e| err("record_event", e))?;
        Ok(Box::new(CudaRawEvent(event)))
    }

    fn wait_all(&self) -> Result<()> {
        self.stream.synchronize().map_err(|e| err("synchronize", e))
    }
}
