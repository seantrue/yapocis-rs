//! `yapocis_rs`: yapocis-rs's CPU backend as a Python module, for Yapocis3's
//! `rustcpu` backend (`YAPOCIS_BACKEND=rustcpu`).
//!
//! Yapocis3 renders an interface (its manifest entry and OpenCL text, as
//! `yapocis-tools bundle` would), and this module loads it on the CPU device
//! and calls its kernels with NumPy arrays. Argument derivation, dispatch
//! sizes and the kernels themselves are yapocis-rs's, as in Rust.
//!
//! ```python
//! import yapocis_rs
//! device = yapocis_rs.Device()
//! program = device.load(entry_json, ocl_source)
//! outputs = program.call("add", [a, b], global_size=None, local_size=None)
//! ```
//!
//! Arguments are user arguments in the kernel's order: NumPy arrays (already
//! C-contiguous and of the parameter's dtype; the Python side converts) and
//! numbers. A NumPy array plays the part of a device buffer, as on numbax and
//! as a GPU's buffer bound to the array does: the kernel runs on it in place,
//! whatever the parameter's role, so a kernel that writes an `in` parameter
//! (yapocis_external's `label` `first` alias does) leaves the result in the
//! array for the next call. An array passed twice is one buffer. Read-only
//! arrays are never written; an array the backend cannot use in place (not
//! 8-byte aligned) is copied in and back.

use numpy::{
    Complex32, Complex64, PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use yapocis_rt::bundle::ParamKind;
use yapocis_rt::{
    Backend, BufferAny, CallArg, CallOptions, DType, Device, HostData, InterfaceEntry, OutputValue,
    Program,
};

fn err(e: yapocis_rt::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// The CPU device.
#[pyclass(name = "Device", frozen)]
struct PyDevice(Device);

#[pymethods]
impl PyDevice {
    /// Open the CPU device (`selector`: `None`, `"0"` or `"cpu"`).
    #[new]
    #[pyo3(signature = (selector = None))]
    fn new(selector: Option<&str>) -> PyResult<Self> {
        Ok(PyDevice(
            Device::open(Backend::RustCpu, selector).map_err(err)?,
        ))
    }

    #[getter]
    fn name(&self) -> String {
        self.0.name()
    }

    /// Load an interface from its manifest entry (JSON) and OpenCL source.
    fn load(&self, entry_json: &str, source: &str) -> PyResult<PyProgram> {
        let entry: InterfaceEntry = serde_json::from_str(entry_json)
            .map_err(|e| PyValueError::new_err(format!("manifest entry: {e}")))?;
        Ok(PyProgram(self.0.load(&entry, source).map_err(err)?))
    }
}

/// One interface loaded on the CPU device.
#[pyclass(name = "Program", frozen)]
struct PyProgram(Program);

/// A NumPy array's element type, shape and bytes.
struct Array {
    dtype: DType,
    shape: Vec<usize>,
    data: *mut u8,
    len: usize,
    writeable: bool,
}

fn dtype_of(kind: u8, size: usize) -> Option<DType> {
    Some(match (kind, size) {
        (b'b', 1) => DType::Bool,
        (b'i', 1) => DType::Int8,
        (b'i', 2) => DType::Int16,
        (b'i', 4) => DType::Int32,
        (b'i', 8) => DType::Int64,
        (b'u', 1) => DType::Uint8,
        (b'u', 2) => DType::Uint16,
        (b'u', 4) => DType::Uint32,
        (b'u', 8) => DType::Uint64,
        (b'f', 2) => DType::Float16,
        (b'f', 4) => DType::Float32,
        (b'f', 8) => DType::Float64,
        (b'c', 8) => DType::Complex64,
        (b'c', 16) => DType::Complex128,
        _ => return None,
    })
}

fn array(a: &Bound<'_, PyUntypedArray>, param: &str) -> PyResult<Array> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{param}: array must be C-contiguous"
        )));
    }
    let descr = a.dtype();
    let dtype = dtype_of(descr.kind(), descr.itemsize())
        .ok_or_else(|| PyTypeError::new_err(format!("{param}: unsupported dtype")))?;
    // SAFETY: a NumPy array's `data` points at `len() * itemsize` bytes,
    // contiguous (checked above), alive while `a` is.
    let data = unsafe { (*a.as_array_ptr()).fields.data as *mut u8 };
    let writeable = a
        .getattr("flags")?
        .getattr("writeable")?
        .extract::<bool>()?;
    Ok(Array {
        dtype,
        shape: a.shape().to_vec(),
        data,
        len: a.len() * descr.itemsize(),
        writeable,
    })
}

/// A host output as a new NumPy array.
fn to_numpy(py: Python<'_>, shape: &[usize], dtype: DType, data: HostData) -> PyResult<Py<PyAny>> {
    macro_rules! typed {
        ($t:ty) => {{
            let v: Vec<$t> = data
                .into_vec::<$t>()
                .unwrap_or_else(|b| bytemuck_cast(b.as_bytes()));
            Ok(PyArray1::from_vec(py, v)
                .reshape(shape.to_vec())?
                .into_any()
                .unbind())
        }};
    }
    match dtype {
        DType::Float32 => typed!(f32),
        DType::Float64 => typed!(f64),
        DType::Float16 => typed!(half::f16),
        DType::Int8 => typed!(i8),
        DType::Int16 => typed!(i16),
        DType::Int32 => typed!(i32),
        DType::Int64 => typed!(i64),
        DType::Uint8 => typed!(u8),
        DType::Uint16 => typed!(u16),
        DType::Uint32 => typed!(u32),
        DType::Uint64 => typed!(u64),
        DType::Complex64 => typed!(Complex32),
        DType::Complex128 => typed!(Complex64),
        DType::Bool => Err(PyTypeError::new_err("bool outputs are not supported")),
    }
}

fn bytemuck_cast<T: Copy + Default>(bytes: &[u8]) -> Vec<T> {
    let n = bytes.len() / std::mem::size_of::<T>();
    let mut v = vec![T::default(); n];
    // SAFETY: `v` has room for exactly `n * size_of::<T>()` bytes, and every
    // type passed here is plain old data.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            v.as_mut_ptr() as *mut u8,
            n * std::mem::size_of::<T>(),
        )
    };
    v
}

#[pymethods]
impl PyProgram {
    /// Call kernel `name` with user arguments `args`. Returns, for each of the
    /// kernel's outputs in order, `(param, array)`: a new array for `out_like`,
    /// `None` for an output written into its argument.
    #[pyo3(signature = (name, args, global_size = None, local_size = None))]
    fn call(
        &self,
        py: Python<'_>,
        name: &str,
        args: Vec<Bound<'_, PyAny>>,
        global_size: Option<Vec<usize>>,
        local_size: Option<Vec<usize>>,
    ) -> PyResult<Vec<(String, Option<Py<PyAny>>)>> {
        let kernel = self.0.kernel(name).map_err(err)?;
        if args.len() != kernel.user_args.len() {
            return Err(PyTypeError::new_err(format!(
                "{name} takes {} arguments ({}), got {}",
                kernel.user_args.len(),
                kernel.user_args.join(", "),
                args.len()
            )));
        }
        let device = self.0.device();

        // One buffer per distinct array (by address): the array itself, used in
        // place, or (if the backend cannot) a copy written back after the launch.
        let mut buffers: Vec<(BufferAny, Array, bool)> = Vec::new(); // (buffer, array, copied)
        let mut slot: Vec<Option<usize>> = Vec::new(); // argument -> buffer
        let mut scalars: Vec<f64> = Vec::new();
        for (arg, param) in args.iter().zip(&kernel.user_args) {
            let entry = kernel
                .params
                .iter()
                .find(|p| {
                    &p.name == param && matches!(p.kind, ParamKind::Buffer | ParamKind::Scalar)
                })
                .ok_or_else(|| PyTypeError::new_err(format!("{name}: no parameter {param}")))?;
            if entry.kind == ParamKind::Scalar {
                scalars.push(arg.extract::<f64>()?);
                slot.push(None);
                continue;
            }
            scalars.push(0.0);
            let a = array(arg.cast::<PyUntypedArray>()?, param)?;
            if let Some(k) = buffers
                .iter()
                .position(|(_, b, _)| b.data == a.data && b.len == a.len)
            {
                slot.push(Some(k)); // the same array again: the same buffer
                continue;
            }
            // SAFETY: the array is alive and unchanged by Python for the whole
            // call (we hold it and the GIL), and the buffer is dropped before
            // `call` returns.
            let wrapped = unsafe {
                BufferAny::wrap_host(device, a.data, a.len, a.writeable, &a.shape, a.dtype)
            };
            let (buf, copied) = match wrapped {
                Some(buf) => (buf, false),
                None => {
                    let buf = BufferAny::alloc(device.clone(), &a.shape, a.dtype).map_err(err)?;
                    // SAFETY: see `array`; only read here.
                    let bytes = unsafe { std::slice::from_raw_parts(a.data, a.len) };
                    buf.upload_bytes(bytes).map_err(err)?;
                    (buf, true)
                }
            };
            slot.push(Some(buffers.len()));
            buffers.push((buf, a, copied));
        }

        let opts = CallOptions {
            global: global_size,
            local: local_size,
            async_exec: false,
        };
        let outputs = {
            let mut call_args: Vec<CallArg<'_>> = Vec::with_capacity(args.len());
            for (i, s) in slot.iter().enumerate() {
                call_args.push(match s {
                    Some(k) => CallArg::Buffer(&buffers[*k].0),
                    None => CallArg::Scalar(scalars[i]),
                });
            }
            self.0
                .call(name, &mut call_args, &opts)
                .map_err(err)?
                .outputs
        };

        // Write copied arrays back (those used in place already hold the result),
        // except read-only ones, which kernels may not change.
        for (buf, a, copied) in &buffers {
            if *copied && a.writeable {
                // SAFETY: see `array`; no other view of these bytes is alive now.
                let bytes = unsafe { std::slice::from_raw_parts_mut(a.data, a.len) };
                buf.download_bytes(bytes).map_err(err)?;
            }
        }
        drop(buffers);

        let mut result = Vec::with_capacity(outputs.len());
        for out in outputs {
            let array = match out.value {
                OutputValue::Host {
                    shape,
                    dtype,
                    bytes,
                } => Some(to_numpy(py, &shape, dtype, bytes)?),
                OutputValue::Written | OutputValue::Device(_) => None,
            };
            result.push((out.param, array));
        }
        Ok(result)
    }
}

/// Add the module's classes (`Device`, `Program`) and `__version__` to `m`.
/// Another crate's extension module embeds the bindings with this, after
/// registering its own CPU ports (`yapocis_rt::cpu::kernels::register`).
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyProgram>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(feature = "module")]
#[pymodule]
fn yapocis_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_to_module(m)
}
