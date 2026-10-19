//! Metal backend on `objc2-metal`, mirroring pymetallic's `Kernel.__call__`
//! and `yapocis/metallic/{device,device_manager,kernel}.py`.
//!
//! Differences from pymetallic: the library is compiled once per interface
//! (not once per kernel name), scalars go through `setBytes` instead of a
//! fresh `MTLBuffer` per call, and launches are asynchronous with a real
//! completion handle.

use crate::device::{prefer, select};
use crate::error::{Error, Result};
use crate::raw::{RawArg, RawBuffer, RawDevice, RawEvent, RawProgram};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;
use yapocis_bundle::Backend;

const B: Backend = Backend::Metal;

fn err(op: &'static str, e: impl ToString) -> Error {
    Error::backend(B, op, e)
}

type DeviceObj = ProtocolObject<dyn MTLDevice>;
type QueueObj = ProtocolObject<dyn MTLCommandQueue>;
type BufferObj = ProtocolObject<dyn MTLBuffer>;
type PsoObj = ProtocolObject<dyn MTLComputePipelineState>;
type CmdObj = ProtocolObject<dyn MTLCommandBuffer>;

/// `MTLBuffer` and `MTLCommandBuffer` are not marked thread-safe by objc2-metal.
/// Apple documents Metal resources as safe to use from any thread, and we only
/// wait on or inspect a committed command buffer from other threads.
struct SendRetained<T>(Retained<T>);
unsafe impl<T> Send for SendRetained<T> {}
unsafe impl<T> Sync for SendRetained<T> {}

pub struct MetalDevice {
    device: Retained<DeviceObj>,
    queue: Retained<QueueObj>,
    name: String,
    /// Committed command buffers not yet known to be complete (for `wait_all`).
    pending: Mutex<Vec<SendRetained<CmdObj>>>,
}

impl MetalDevice {
    pub fn open(selector: Option<&str>) -> Result<MetalDevice> {
        let all = MTLCopyAllDevices();
        let devices: Vec<Retained<DeviceObj>> = all.to_vec();
        let names: Vec<String> = devices.iter().map(|d| d.name().to_string()).collect();
        // metallic/device.py: prefer radeon, amd, apple, gpu; else the last device
        let index = select(B, &names, selector, |names| {
            prefer(names, &["radeon", "amd", "apple", "gpu"])
        })?;
        let device = devices[index].clone();
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| err("newCommandQueue", "returned nil"))?;
        Ok(MetalDevice {
            device,
            queue,
            name: names[index].clone(),
            pending: Mutex::new(Vec::new()),
        })
    }

    fn prune_pending(&self) {
        let mut pending = self.pending.lock().unwrap();
        pending.retain(|c| {
            !matches!(
                c.0.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        });
    }
}

struct MetalProgram {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: Mutex<HashMap<String, Retained<PsoObj>>>,
}

impl RawProgram for MetalProgram {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct MetalBuffer {
    buffer: SendRetained<BufferObj>,
    len: usize,
}

impl RawBuffer for MetalBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn len_bytes(&self) -> usize {
        self.len
    }
}

struct MetalEvent(SendRetained<CmdObj>);

impl RawEvent for MetalEvent {
    fn wait(&self) -> Result<()> {
        self.0 .0.waitUntilCompleted();
        if let Some(e) = self.0 .0.error() {
            return Err(err("command buffer", e.localizedDescription()));
        }
        Ok(())
    }
}

impl RawDevice for MetalDevice {
    fn backend(&self) -> Backend {
        B
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn identity(&self) -> String {
        format!("{} | registry {}", self.name, self.device.registryID())
    }

    fn compile(
        &self,
        interface: &str,
        source: &str,
        _options: &[String],
        _cache_key: &str,
    ) -> Result<Box<dyn RawProgram>> {
        // pymetallic: makeLibrary(source:, options: nil); Metal keeps its own compiler cache.
        let library = autoreleasepool(|_| {
            self.device
                .newLibraryWithSource_options_error(&NSString::from_str(source), None)
                .map_err(|e| Error::Compile {
                    backend: B,
                    interface: interface.to_string(),
                    log: e.localizedDescription().to_string(),
                })
        })?;
        Ok(Box::new(MetalProgram {
            library,
            pipelines: Mutex::new(HashMap::new()),
        }))
    }

    fn alloc(&self, len_bytes: usize) -> Result<Box<dyn RawBuffer>> {
        // pymetallic Buffer.from_numpy: shared storage mode
        let buffer = self
            .device
            .newBufferWithLength_options(len_bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| err("newBufferWithLength", "returned nil"))?;
        Ok(Box::new(MetalBuffer {
            buffer: SendRetained(buffer),
            len: len_bytes,
        }))
    }

    fn upload(&self, buffer: &dyn RawBuffer, data: &[u8]) -> Result<()> {
        let b = buffer
            .as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or(Error::ForeignBuffer)?;
        let dst = b.buffer.0.contents().as_ptr() as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        Ok(())
    }

    fn download(&self, buffer: &dyn RawBuffer, out: &mut [u8]) -> Result<()> {
        let b = buffer
            .as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or(Error::ForeignBuffer)?;
        let src = b.buffer.0.contents().as_ptr() as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len()) };
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
        let p = program
            .as_any()
            .downcast_ref::<MetalProgram>()
            .ok_or(Error::ForeignBuffer)?;
        let pso = {
            let mut pipelines = p.pipelines.lock().unwrap();
            if let Some(pso) = pipelines.get(symbol) {
                pso.clone()
            } else {
                let function = p
                    .library
                    .newFunctionWithName(&NSString::from_str(symbol))
                    .ok_or_else(|| Error::NoSuchKernel(symbol.to_string()))?;
                let pso = self
                    .device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| err("pipeline state", e.localizedDescription()))?;
                pipelines.insert(symbol.to_string(), pso.clone());
                pso
            }
        };
        self.prune_pending();
        let (grid, tgs) = yapocis_bundle::metal_grid_tgs(global, local);
        let cmd = autoreleasepool(|_| -> Result<Retained<CmdObj>> {
            let cmd = self
                .queue
                .commandBuffer()
                .ok_or_else(|| err("commandBuffer", "returned nil"))?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or_else(|| err("computeCommandEncoder", "returned nil"))?;
            enc.setComputePipelineState(&pso);
            for (i, arg) in args.iter().enumerate() {
                match arg {
                    RawArg::Bytes(bytes) => unsafe {
                        enc.setBytes_length_atIndex(
                            NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                            bytes.len(),
                            i,
                        )
                    },
                    RawArg::Buffer(b) => {
                        let mb = b
                            .as_any()
                            .downcast_ref::<MetalBuffer>()
                            .ok_or(Error::ForeignBuffer)?;
                        unsafe { enc.setBuffer_offset_atIndex(Some(&mb.buffer.0), 0, i) }
                    }
                }
            }
            let grid = MTLSize {
                width: grid[0],
                height: grid[1],
                depth: grid[2],
            };
            let tgs = MTLSize {
                width: tgs[0],
                height: tgs[1],
                depth: tgs[2],
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tgs);
            enc.endEncoding();
            cmd.commit();
            Ok(cmd)
        })?;
        self.pending.lock().unwrap().push(SendRetained(cmd.clone()));
        Ok(Box::new(MetalEvent(SendRetained(cmd))))
    }

    fn wait_all(&self) -> Result<()> {
        let pending: Vec<_> = std::mem::take(&mut *self.pending.lock().unwrap());
        for cmd in pending {
            cmd.0.waitUntilCompleted();
            if let Some(e) = cmd.0.error() {
                return Err(err("command buffer", e.localizedDescription()));
            }
        }
        Ok(())
    }
}
