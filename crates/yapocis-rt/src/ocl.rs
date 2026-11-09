//! OpenCL backend on `opencl3`, mirroring `yapocis/ocl/{device,device_manager,kernel}.py`.

use crate::cache;
use crate::device::{is_software_renderer_name, prefer, select};
use crate::error::{Error, Result};
use crate::raw::{RawArg, RawBuffer, RawDevice, RawEvent, RawProgram};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU};
use opencl3::event::Event;
use opencl3::kernel::{set_kernel_arg, Kernel};
use opencl3::memory::{Buffer, ClMem, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_mem, CL_BLOCKING};
use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use yapocis_bundle::Backend;

const B: Backend = Backend::Ocl;

fn err(op: &'static str, e: impl ToString) -> Error {
    Error::backend(B, op, e)
}

pub struct OclDevice {
    device: Device,
    context: Context,
    /// One in-order queue per device, as in `ocl/device_manager.py`.
    queue: CommandQueue,
    name: String,
    identity: String,
}

// OpenCL objects are thread-safe (the spec requires it for everything but
// clSetKernelArg on a shared kernel, which `OclProgram` serialises).
unsafe impl Send for OclDevice {}
unsafe impl Sync for OclDevice {}

/// Serialises device enumeration and context creation. pocl (the CPU
/// platform installed next to NVIDIA's on the Linux host) crashes in
/// `clGetDeviceInfo` when several threads enumerate devices at once, which
/// parallel tests that each open a device do.
static OPEN_LOCK: Mutex<()> = Mutex::new(());

impl OclDevice {
    pub fn open(selector: Option<&str>) -> Result<OclDevice> {
        let _open = OPEN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ids = get_all_devices(CL_DEVICE_TYPE_ALL).map_err(|e| err("get_all_devices", e))?;
        let devices: Vec<Device> = ids.iter().map(|&id| Device::new(id)).collect();
        let names: Vec<String> = devices
            .iter()
            .map(|d| d.name().unwrap_or_else(|_| "?".into()))
            .collect();
        let is_cpu: Vec<bool> = devices
            .iter()
            .map(|d| {
                d.dev_type()
                    .map(|t| t & CL_DEVICE_TYPE_CPU != 0)
                    .unwrap_or(false)
            })
            .collect();
        let index = select(B, &names, selector, |names| {
            preferred_device(names, &is_cpu)
        })?;
        let device = Device::new(ids[index]);
        let context = Context::from_device(&device).map_err(|e| err("create context", e))?;
        // Apple's OpenCL is 1.2: clCreateCommandQueueWithProperties is absent.
        let queue =
            CommandQueue::create_default(&context, 0).map_err(|e| err("create queue", e))?;
        let identity = format!(
            "{} | {} | {} | {}",
            device.vendor().unwrap_or_default(),
            names[index],
            device.version().unwrap_or_default(),
            device.driver_version().unwrap_or_default()
        );
        Ok(OclDevice {
            device,
            context,
            queue,
            name: names[index].clone(),
            identity,
        })
    }

    fn device_id(&self) -> cl_device_id {
        self.device.id()
    }
}

/// The device to use when none is asked for. As ocl/device.py: non-CPU nvidia,
/// amd, intel; then CPU in the same order; then the first other non-CPU
/// device; else the last device. A software renderer (WARP, see
/// [`is_software_renderer_name`]) is never one of those picks while another
/// device exists: on a machine with a GPU and the compatibility pack,
/// OpenCLOn12 can list WARP beside the GPU's own adapter.
fn preferred_device(names: &[String], is_cpu: &[bool]) -> Option<usize> {
    let soft: Vec<bool> = names.iter().map(|n| is_software_renderer_name(n)).collect();
    for pass in [false, true] {
        for key in ["nvidia", "amd", "intel"] {
            if let Some(i) = (0..names.len()).find(|&i| {
                is_cpu[i] == pass && !soft[i] && names[i].to_ascii_lowercase().contains(key)
            }) {
                return Some(i);
            }
        }
    }
    if let Some(i) = (0..names.len()).find(|&i| !is_cpu[i] && !soft[i]) {
        return Some(i);
    }
    let _ = prefer; // apple: a single GPU device, handled by the non-CPU rule above
                    // "else the last device", but the last that isn't a software renderer
    if let Some(i) = (0..names.len()).rev().find(|&i| !soft[i]) {
        return Some(i);
    }
    None
}

struct OclProgram {
    program: Program,
    /// One cl_kernel per symbol; argument setting and enqueue happen under the lock.
    kernels: Mutex<HashMap<String, Kernel>>,
}
unsafe impl Send for OclProgram {}
unsafe impl Sync for OclProgram {}

impl RawProgram for OclProgram {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct OclBuffer {
    buffer: Mutex<Buffer<u8>>,
    len: usize,
}
unsafe impl Send for OclBuffer {}
unsafe impl Sync for OclBuffer {}

impl RawBuffer for OclBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn len_bytes(&self) -> usize {
        self.len
    }
}

struct OclEvent(Event);
unsafe impl Send for OclEvent {}

impl RawEvent for OclEvent {
    fn wait(&self) -> Result<()> {
        self.0.wait().map_err(|e| err("wait", e))
    }
}

impl RawDevice for OclDevice {
    fn backend(&self) -> Backend {
        B
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn identity(&self) -> String {
        self.identity.clone()
    }

    fn is_software_renderer(&self) -> bool {
        is_software_renderer_name(&self.name)
    }

    fn compile(
        &self,
        interface: &str,
        source: &str,
        options: &[String],
        cache_key: &str,
    ) -> Result<Box<dyn RawProgram>> {
        let opts = options.join(" ");
        let devices = [self.device_id()];
        let mut program = None;
        if let Some(bin) = cache::read(B, cache_key, "bin") {
            // A stale or foreign binary fails to load or build: fall through to source.
            if let Ok(mut p) =
                unsafe { Program::create_from_binary(&self.context, &devices, &[&bin]) }
            {
                if p.build(&devices, &opts).is_ok() {
                    tracing::debug!(interface, "OpenCL program from cache");
                    program = Some(p);
                }
            }
        }
        let program = match program {
            Some(p) => p,
            None => {
                let p = Program::create_and_build_from_source(&self.context, source, &opts)
                    .map_err(|log| Error::Compile {
                        backend: B,
                        interface: interface.to_string(),
                        log,
                    })?;
                if let Ok(bins) = p.get_binaries() {
                    if let Some(bin) = bins.first() {
                        if !bin.is_empty() {
                            cache::write(B, cache_key, "bin", bin);
                        }
                    }
                }
                p
            }
        };
        Ok(Box::new(OclProgram {
            program,
            kernels: Mutex::new(HashMap::new()),
        }))
    }

    fn alloc(&self, len_bytes: usize) -> Result<Box<dyn RawBuffer>> {
        let buffer = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                len_bytes.max(1),
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| err("create buffer", e))?;
        Ok(Box::new(OclBuffer {
            buffer: Mutex::new(buffer),
            len: len_bytes,
        }))
    }

    fn upload(&self, buffer: &dyn RawBuffer, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let b = buffer
            .as_any()
            .downcast_ref::<OclBuffer>()
            .ok_or(Error::ForeignBuffer)?;
        let mut guard = b.buffer.lock().unwrap();
        // ocl/device_manager.py writes synchronously unless async_write
        let evt = unsafe {
            self.queue
                .enqueue_write_buffer(&mut guard, CL_BLOCKING, 0, data, &[])
        }
        .map_err(|e| err("write buffer", e))?;
        evt.wait().map_err(|e| err("write wait", e))
    }

    fn download(&self, buffer: &dyn RawBuffer, out: &mut [u8]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        let b = buffer
            .as_any()
            .downcast_ref::<OclBuffer>()
            .ok_or(Error::ForeignBuffer)?;
        let guard = b.buffer.lock().unwrap();
        let evt = unsafe {
            self.queue
                .enqueue_read_buffer(&guard, CL_BLOCKING, 0, out, &[])
        }
        .map_err(|e| err("read buffer", e))?;
        evt.wait().map_err(|e| err("read wait", e))
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
            .downcast_ref::<OclProgram>()
            .ok_or(Error::ForeignBuffer)?;
        let mut kernels = p.kernels.lock().unwrap();
        if !kernels.contains_key(symbol) {
            let k = Kernel::create(&p.program, symbol).map_err(|e| err("create kernel", e))?;
            kernels.insert(symbol.to_string(), k);
        }
        let kernel = &kernels[symbol];
        // cl_mem handles stay valid without holding the buffer lock (the same
        // buffer may appear twice in one call; a held lock would deadlock).
        for (i, arg) in args.iter().enumerate() {
            match arg {
                RawArg::Bytes(bytes) => unsafe {
                    set_kernel_arg(
                        kernel.get(),
                        i as u32,
                        bytes.len(),
                        bytes.as_ptr() as *const c_void,
                    )
                }
                .map_err(|e| err("set scalar arg", e))?,
                RawArg::Buffer(b) => {
                    let ob = b
                        .as_any()
                        .downcast_ref::<OclBuffer>()
                        .ok_or(Error::ForeignBuffer)?;
                    let mem: cl_mem = ob.buffer.lock().unwrap().get();
                    unsafe {
                        set_kernel_arg(
                            kernel.get(),
                            i as u32,
                            std::mem::size_of::<cl_mem>(),
                            &mem as *const cl_mem as *const c_void,
                        )
                    }
                    .map_err(|e| err("set buffer arg", e))?;
                }
            }
        }
        let dims = global.len().clamp(1, 3) as u32;
        let local_ptr = match local {
            Some(l) if !l.is_empty() => {
                // pad to the global dimensionality with 1s
                let mut v = l.to_vec();
                v.resize(dims as usize, 1);
                Some(v)
            }
            _ => None,
        };
        let evt = unsafe {
            self.queue.enqueue_nd_range_kernel(
                kernel.get(),
                dims,
                std::ptr::null(),
                global.as_ptr(),
                local_ptr
                    .as_ref()
                    .map(|v| v.as_ptr())
                    .unwrap_or(std::ptr::null()),
                &[],
            )
        }
        .map_err(|e| err("enqueue kernel", e))?;
        Ok(Box::new(OclEvent(evt)))
    }

    fn wait_all(&self) -> Result<()> {
        self.queue.finish().map_err(|e| err("finish", e))
    }
}

#[cfg(test)]
mod tests {
    use super::preferred_device;

    const WARP: &str = "Microsoft Basic Render Driver";

    fn pick(devices: &[(&str, bool)]) -> Option<usize> {
        let names: Vec<String> = devices.iter().map(|(n, _)| n.to_string()).collect();
        let is_cpu: Vec<bool> = devices.iter().map(|&(_, c)| c).collect();
        preferred_device(&names, &is_cpu)
    }

    #[test]
    fn a_vendor_gpu_still_comes_first() {
        let devices = [
            ("pthread-haswell", true),
            ("NVIDIA GeForce RTX 3080", false),
        ];
        assert_eq!(pick(&devices), Some(1));
    }

    #[test]
    fn an_unnamed_gpu_wins_over_warp_listed_first() {
        // OpenCLOn12 on a Snapdragon laptop with the compatibility pack
        let devices = [(WARP, false), ("Qualcomm(R) Adreno(TM) X1-85 GPU", false)];
        assert_eq!(pick(&devices), Some(1));
    }

    #[test]
    fn a_cpu_device_wins_over_warp() {
        let devices = [("pthread-haswell", true), (WARP, false)];
        assert_eq!(pick(&devices), Some(0));
    }

    #[test]
    fn warp_alone_is_left_to_the_last_device_rule() {
        // None: select() then takes the last device, WARP, so asking for
        // OpenCL explicitly still works on such a machine.
        assert_eq!(pick(&[(WARP, false)]), None);
    }

    #[test]
    fn otherwise_the_last_device_as_before() {
        let devices = [("pthread-a", true), ("pthread-b", true)];
        assert_eq!(pick(&devices), Some(1));
    }
}
