//! Compiled programs and kernel calls.

use crate::buffer::BufferAny;
use crate::cache;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::host_data::HostData;
use crate::raw::{RawArg, RawEvent, RawProgram};
use std::collections::HashMap;
use std::sync::Arc;
use yapocis_bundle::{
    derive_args, dispatch_global_size, ArgShape, Backend, DType, InterfaceEntry, KernelArg,
    KernelEntry, Role, ScalarValue,
};

/// A read-only host array for a call: contiguous row-major bytes plus shape and dtype.
#[derive(Clone, Copy)]
pub struct HostView<'a> {
    pub dtype: DType,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

/// A host array the call writes back into (for `inout` and `out` parameters).
pub struct HostViewMut<'a> {
    pub dtype: DType,
    pub shape: &'a [usize],
    pub bytes: &'a mut [u8],
}

impl<'a> HostView<'a> {
    pub fn of<T: crate::Element>(data: &'a [T], shape: &'a [usize]) -> HostView<'a> {
        HostView {
            dtype: T::DTYPE,
            shape,
            bytes: bytemuck::cast_slice(data),
        }
    }
}

impl<'a> HostViewMut<'a> {
    pub fn of<T: crate::Element>(data: &'a mut [T], shape: &'a [usize]) -> HostViewMut<'a> {
        HostViewMut {
            dtype: T::DTYPE,
            shape,
            bytes: bytemuck::cast_slice_mut(data),
        }
    }
}

/// One user argument of a kernel call (one per `KernelEntry::user_args`).
pub enum CallArg<'a> {
    /// Host array for an `in` parameter: uploaded on every call, never written back.
    In(HostView<'a>),
    /// Host array for an `in`, `inout` or `out` parameter: uploaded, and read
    /// back after the launch for `inout` and `out`.
    InOut(HostViewMut<'a>),
    /// A device buffer, used in place: never uploaded and never read back
    /// (`resident` semantics). Any parameter accepts one, which is also the
    /// counterpart of Python's call-level `resident=(...)`; `resident`
    /// parameters accept nothing else.
    Buffer(&'a BufferAny),
    /// A scalar, cast to the parameter's dtype.
    Scalar(f64),
}

impl<'a> CallArg<'a> {
    fn shape(&self) -> Option<&[usize]> {
        match self {
            CallArg::In(v) => Some(v.shape),
            CallArg::InOut(v) => Some(v.shape),
            CallArg::Buffer(b) => Some(b.shape()),
            CallArg::Scalar(_) => None,
        }
    }

    fn dtype(&self) -> Option<DType> {
        match self {
            CallArg::In(v) => Some(v.dtype),
            CallArg::InOut(v) => Some(v.dtype),
            CallArg::Buffer(b) => Some(b.dtype()),
            CallArg::Scalar(_) => None,
        }
    }

    fn arg_shape(&self) -> ArgShape {
        match self {
            CallArg::Scalar(v) => ArgShape::Scalar(*v),
            other => ArgShape::Array {
                shape: other.shape().unwrap().to_vec(),
                dtype: other.dtype().unwrap(),
            },
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CallOptions {
    /// Explicit global size; overrides `dispatch_size` and the backend default.
    pub global: Option<Vec<usize>>,
    /// Explicit local size (OpenCL local, CUDA block, Metal threadgroup).
    pub local: Option<Vec<usize>>,
    /// Enqueue and return without waiting or reading anything back. Outputs
    /// that would have been read back are returned as device buffers instead.
    pub async_exec: bool,
}

/// What a call produced for one returned parameter, in `KernelEntry::returns` order.
#[derive(Debug)]
pub enum OutputValue {
    /// The caller's `InOut` array was written in place.
    Written,
    /// A new host array (`out_like`, synchronous calls), stored as its
    /// element type: [`HostData::into_vec`] takes it without a copy.
    Host {
        shape: Vec<usize>,
        dtype: DType,
        bytes: HostData,
    },
    /// A device buffer: `out_like` in async calls, or a `Buffer` argument.
    Device(BufferAny),
}

#[derive(Debug)]
pub struct Output {
    pub param: String,
    pub value: OutputValue,
}

/// Handle to one launch.
pub struct Event {
    raw: Box<dyn RawEvent>,
}

impl Event {
    pub fn wait(&self) -> Result<()> {
        self.raw.wait()
    }
}

pub struct CallOutput {
    pub outputs: Vec<Output>,
    /// Present for async calls.
    pub event: Option<Event>,
}

impl std::fmt::Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Event")
    }
}

impl std::fmt::Debug for CallOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallOutput")
            .field("outputs", &self.outputs)
            .field("event", &self.event)
            .finish()
    }
}

impl CallOutput {
    /// The single output of a one-output kernel, as host bytes.
    pub fn into_host(mut self) -> Option<(Vec<usize>, DType, HostData)> {
        if self.outputs.len() != 1 {
            return None;
        }
        match self.outputs.pop()?.value {
            OutputValue::Host {
                shape,
                dtype,
                bytes,
            } => Some((shape, dtype, bytes)),
            _ => None,
        }
    }
}

/// One interface variant compiled for one device.
pub struct Program {
    device: Device,
    entry: Arc<InterfaceEntry>,
    raw: Box<dyn RawProgram>,
    kernels: HashMap<String, usize>,
}

impl Program {
    pub(crate) fn load(device: Device, entry: &InterfaceEntry, source: &str) -> Result<Program> {
        let backend = device.backend();
        let src = entry
            .sources
            .get(backend.name())
            .ok_or(Error::BackendUnavailable(backend))?;
        let key = cache::cache_key(backend, &device.raw.identity(), source, &src.build_options);
        let _span = tracing::info_span!("compile", interface = %entry.id, backend = backend.name())
            .entered();
        let raw = device
            .raw
            .compile(&entry.id, source, &src.build_options, &key)?;
        let kernels = entry
            .kernels
            .iter()
            .enumerate()
            .map(|(i, k)| (k.name.clone(), i))
            .collect();
        Ok(Program {
            device,
            entry: Arc::new(entry.clone()),
            raw,
            kernels,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn entry(&self) -> &InterfaceEntry {
        &self.entry
    }

    pub fn kernel(&self, name: &str) -> Result<&KernelEntry> {
        self.kernels
            .get(name)
            .map(|&i| &self.entry.kernels[i])
            .ok_or_else(|| Error::NoSuchKernel(name.to_string()))
    }

    /// Launch a kernel by symbol with already-prepared raw arguments.
    pub fn launch_raw(
        &self,
        symbol: &str,
        args: &[RawArg<'_>],
        global: &[usize],
        local: Option<&[usize]>,
    ) -> Result<Event> {
        let raw = self
            .device
            .raw
            .launch(self.raw.as_ref(), symbol, args, global, local)?;
        Ok(Event { raw })
    }

    /// A device buffer for one call: from the device's buffer cache when
    /// `cached` (preferring the one bound to the host array at `host`), else
    /// newly allocated.
    fn temp(
        &self,
        shape: &[usize],
        dtype: DType,
        host: Option<usize>,
        cached: bool,
    ) -> Result<BufferAny> {
        let len = (shape.iter().product::<usize>() * dtype.size_of()).max(1);
        if cached {
            if let Some(raw) = self.device.buffers.take(len, host) {
                return Ok(BufferAny::from_raw(self.device.clone(), raw, shape, dtype));
            }
        }
        BufferAny::alloc(self.device.clone(), shape, dtype)
    }

    /// Call a kernel the way the Python runtime does: derive dimensions and
    /// `out_like` outputs from the arguments, upload, launch, and (unless
    /// `async_exec`) wait and read the outputs back.
    pub fn call(
        &self,
        name: &str,
        args: &mut [CallArg<'_>],
        opts: &CallOptions,
    ) -> Result<CallOutput> {
        let kernel = self.kernel(name)?;
        let backend = self.device.backend();
        let shapes: Vec<ArgShape> = args.iter().map(CallArg::arg_shape).collect();
        let derived = derive_args(kernel, &shapes)?;

        // Check dtypes of array arguments against the parameters.
        for (ka, p) in derived.iter().zip(&kernel.params) {
            let arg = match ka {
                KernelArg::Buffer { arg, .. } => *arg,
                _ => continue,
            };
            if let Some(got) = args[arg].dtype() {
                if got != p.dtype {
                    return Err(Error::DType {
                        param: p.name.clone(),
                        expected: p.dtype,
                        got,
                    });
                }
            }
        }

        // Global size: explicit, then dispatch_size, then the backend default.
        let global = match &opts.global {
            Some(g) => g.clone(),
            None => match dispatch_global_size(kernel.dispatch_size.as_deref(), &shapes)? {
                Some(g) => g,
                None => yapocis_bundle::default_global_size(backend, &shapes)
                    .ok_or(Error::NoGlobalSize)?,
            },
        };

        // Materialise device buffers for every array parameter. Temporaries
        // come from the buffer cache for synchronous calls (an async call's
        // buffers may still be in use when it returns) and remember the host
        // array they were bound to.
        enum Slot {
            Temp(BufferAny, Option<usize>),
            Borrowed(usize),
        }
        let cached = !opts.async_exec;
        let mut slots: Vec<Option<Slot>> = Vec::with_capacity(derived.len());
        let mut readbacks: Vec<(usize, usize)> = Vec::new(); // (slot index, user arg index)
        let mut out_likes: Vec<(usize, String)> = Vec::new(); // (slot index, param name)
        let mut scalar_bytes: Vec<Vec<u8>> = Vec::with_capacity(derived.len());
        for (i, ka) in derived.iter().enumerate() {
            match ka {
                KernelArg::Scalar { value, dtype, .. } => {
                    scalar_bytes.push(scalar_to_bytes(*dtype, value));
                    slots.push(None);
                }
                KernelArg::Dim { value, .. } => {
                    scalar_bytes.push(value.to_ne_bytes().to_vec());
                    slots.push(None);
                }
                KernelArg::OutLike { shape, dtype, .. } => {
                    scalar_bytes.push(Vec::new());
                    let buf = self.temp(shape, *dtype, None, cached)?;
                    out_likes.push((i, ka_param(ka).to_string()));
                    slots.push(Some(Slot::Temp(buf, None)));
                }
                KernelArg::Buffer {
                    role,
                    arg,
                    dtype,
                    param,
                } => {
                    scalar_bytes.push(Vec::new());
                    let slot = match &args[*arg] {
                        CallArg::Buffer(b) => {
                            if !b.device().same_as(&self.device) {
                                return Err(Error::ForeignBuffer);
                            }
                            Slot::Borrowed(*arg)
                        }
                        // A resident argument is never written. Python looks up the
                        // buffer bound to the array object (its cache holds the
                        // array, so the id stays unique); a host address here could
                        // belong to a new array, so the caller passes the buffer.
                        CallArg::In(_) | CallArg::InOut(_) if *role == Role::Resident => {
                            return Err(Error::ArgKind {
                                param: param.clone(),
                                role: "resident".to_string(),
                                needs:
                                    "a device buffer (CallArg::Buffer, e.g. from Device::upload)",
                            });
                        }
                        CallArg::In(v) => {
                            if matches!(role, Role::Inout | Role::Out) {
                                return Err(Error::ArgKind {
                                    param: param.clone(),
                                    role: format!("{role:?}").to_lowercase(),
                                    needs:
                                        "a mutable host array (CallArg::InOut) or a device buffer",
                                });
                            }
                            let host = Some(v.bytes.as_ptr() as usize);
                            let buf = self.temp(v.shape, *dtype, host, cached)?;
                            buf.upload_bytes(v.bytes)?;
                            Slot::Temp(buf, host)
                        }
                        CallArg::InOut(v) => {
                            let host = Some(v.bytes.as_ptr() as usize);
                            let buf = self.temp(v.shape, *dtype, host, cached)?;
                            buf.upload_bytes(v.bytes)?;
                            if matches!(role, Role::Inout | Role::Out) {
                                readbacks.push((i, *arg));
                            }
                            Slot::Temp(buf, host)
                        }
                        CallArg::Scalar(_) => {
                            return Err(Error::ArgKind {
                                param: param.clone(),
                                role: format!("{role:?}").to_lowercase(),
                                needs: "an array",
                            })
                        }
                    };
                    slots.push(Some(slot));
                }
            }
        }

        // Hidden CUDA global-size argument.
        let total: u64 = global.iter().product::<usize>() as u64;
        let hidden = total.to_ne_bytes();

        let mut raw_args: Vec<RawArg<'_>> = Vec::with_capacity(derived.len() + 1);
        for (i, slot) in slots.iter().enumerate() {
            match slot {
                None => raw_args.push(RawArg::Bytes(&scalar_bytes[i])),
                Some(Slot::Temp(b, _)) => raw_args.push(RawArg::Buffer(b.raw.as_ref())),
                Some(Slot::Borrowed(a)) => match &args[*a] {
                    CallArg::Buffer(b) => raw_args.push(RawArg::Buffer(b.raw.as_ref())),
                    _ => unreachable!(),
                },
            }
        }
        if backend == Backend::Cuda && kernel.cuda.takes_global_size.unwrap_or(false) {
            raw_args.push(RawArg::Bytes(&hidden));
        }

        let _span = tracing::debug_span!("launch", kernel = %kernel.symbol, ?global).entered();
        let event = self.launch_raw(&kernel.symbol, &raw_args, &global, opts.local.as_deref())?;

        if !opts.async_exec {
            event.wait()?;
        }
        // Outputs in declaration order: inout/out buffers and out_like results.
        // Positions, not names, because an interface may declare the same
        // out_like source twice (gradient: `out_like input, out_like input`).
        let mut outputs = Vec::new();
        for (i, ka) in derived.iter().enumerate() {
            let value = match (ka, &slots[i]) {
                (KernelArg::OutLike { .. }, Some(Slot::Temp(b, _))) => {
                    if opts.async_exec {
                        OutputValue::Device(b.clone())
                    } else {
                        let mut data = HostData::zeroed(b.dtype(), b.len());
                        b.download_bytes(data.as_bytes_mut())?;
                        OutputValue::Host {
                            shape: b.shape().to_vec(),
                            dtype: b.dtype(),
                            bytes: data,
                        }
                    }
                }
                (
                    KernelArg::Buffer {
                        role: Role::Inout | Role::Out,
                        arg,
                        ..
                    },
                    Some(Slot::Temp(b, _)),
                ) => {
                    if !opts.async_exec {
                        if let CallArg::InOut(v) = &mut args[*arg] {
                            b.download_bytes(v.bytes)?;
                        }
                    }
                    OutputValue::Written
                }
                (
                    KernelArg::Buffer {
                        role: Role::Inout | Role::Out,
                        arg,
                        ..
                    },
                    Some(Slot::Borrowed(_)),
                ) => match &args[*arg] {
                    CallArg::Buffer(b) => OutputValue::Device((*b).clone()),
                    _ => OutputValue::Written,
                },
                _ => continue,
            };
            outputs.push(Output {
                param: ka_param(ka).to_string(),
                value,
            });
        }
        let _ = (&readbacks, &out_likes);
        drop(raw_args);
        if cached {
            for slot in slots.into_iter().flatten() {
                if let Slot::Temp(b, host) = slot {
                    self.device.buffers.give(b.raw, host);
                }
            }
        }
        Ok(CallOutput {
            outputs,
            event: if opts.async_exec { Some(event) } else { None },
        })
    }
}

fn ka_param(ka: &KernelArg) -> &str {
    match ka {
        KernelArg::Scalar { param, .. }
        | KernelArg::Dim { param, .. }
        | KernelArg::Buffer { param, .. }
        | KernelArg::OutLike { param, .. } => param,
    }
}

fn scalar_to_bytes(dtype: DType, value: &ScalarValue) -> Vec<u8> {
    let f = value.as_f64();
    match dtype {
        DType::Bool => vec![(f != 0.0) as u8],
        DType::Int8 => (f as i8).to_ne_bytes().to_vec(),
        DType::Int16 => (f as i16).to_ne_bytes().to_vec(),
        DType::Int32 => (f as i32).to_ne_bytes().to_vec(),
        DType::Int64 => (f as i64).to_ne_bytes().to_vec(),
        DType::Uint8 => (f as u8).to_ne_bytes().to_vec(),
        DType::Uint16 => (f as u16).to_ne_bytes().to_vec(),
        DType::Uint32 => (f as u32).to_ne_bytes().to_vec(),
        DType::Uint64 => (f as u64).to_ne_bytes().to_vec(),
        DType::Float16 => half::f16::from_f64(f).to_ne_bytes().to_vec(),
        DType::Float32 => (f as f32).to_ne_bytes().to_vec(),
        DType::Float64 => f.to_ne_bytes().to_vec(),
        DType::Complex64 => {
            let mut v = (f as f32).to_ne_bytes().to_vec();
            v.extend_from_slice(&0f32.to_ne_bytes());
            v
        }
        DType::Complex128 => {
            let mut v = f.to_ne_bytes().to_vec();
            v.extend_from_slice(&0f64.to_ne_bytes());
            v
        }
    }
}
