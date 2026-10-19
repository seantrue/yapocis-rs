//! Host-side glue between `ndarray` and the runtime, used by the generated
//! interface structs.

use crate::rt::{self, CallOutput, Device, Element, Output, OutputValue, Program};
use crate::{Error, Result, MANIFEST};
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, CowArray, IxDyn};

/// Implemented by every generated interface struct.
pub trait Interface: Sized {
    const ID: &'static str;
    fn from_program(program: Program) -> Self;
    fn program(&self) -> &Program;
    fn source(backend: &str) -> Option<&'static str>;
}

/// Compile a generated interface for `device`.
pub fn load<I: Interface>(device: &Device) -> Result<I> {
    let entry = MANIFEST
        .interface(I::ID)
        .unwrap_or_else(|| panic!("interface {} missing from the bundled manifest", I::ID));
    let backend = device.backend();
    let source = I::source(backend.name()).ok_or(Error::BackendUnavailable(backend))?;
    Ok(I::from_program(device.load(entry, source)?))
}

/// A C-contiguous copy-on-write view of an input array.
pub fn standard<'a, T: Clone>(a: &'a ArrayViewD<'_, T>) -> CowArray<'a, T, IxDyn> {
    a.as_standard_layout()
}

/// The contiguous slice behind an in/out array (which must be C-contiguous).
pub fn slice_mut<'a, T>(a: &'a mut ArrayViewMutD<'_, T>) -> Result<&'a mut [T]> {
    a.as_slice_mut().ok_or(Error::NotContiguous)
}

/// Turn a host output into a typed array.
pub fn to_array<T: Element>(out: Output) -> Result<ArrayD<T>> {
    match out.value {
        OutputValue::Host {
            shape,
            dtype,
            bytes,
        } => {
            if dtype != T::DTYPE {
                return Err(Error::Rt(rt::Error::DType {
                    param: out.param,
                    expected: T::DTYPE,
                    got: dtype,
                }));
            }
            // The runtime downloaded into a Vec<T> already: take it, no copy.
            let data: Vec<T> = bytes
                .into_vec()
                .unwrap_or_else(|b| bytemuck::cast_slice(&b).to_vec());
            Ok(ArrayD::from_shape_vec(IxDyn(&shape), data).expect("shape matches byte length"))
        }
        OutputValue::Device(buf) => {
            let typed = buf.typed::<T>().map_err(|b| {
                Error::Rt(rt::Error::DType {
                    param: out.param.clone(),
                    expected: T::DTYPE,
                    got: b.dtype(),
                })
            })?;
            let shape = typed.shape().to_vec();
            Ok(ArrayD::from_shape_vec(IxDyn(&shape), typed.to_vec()?).expect("shape matches"))
        }
        OutputValue::Written => Err(Error::NotAnOutLike(out.param)),
    }
}

/// Only the `out_like` outputs of a call (in-place `inout`/`out` ones are dropped).
pub fn out_like_outputs(out: CallOutput) -> Vec<Output> {
    out.outputs
        .into_iter()
        .filter(|o| !matches!(o.value, OutputValue::Written))
        .collect()
}

/// The single `out_like` output of a call.
pub fn one_output<T: Element>(out: CallOutput) -> Result<ArrayD<T>> {
    let mut outs = out_like_outputs(out);
    match outs.len() {
        1 => to_array(outs.pop().unwrap()),
        n => Err(Error::OutputCount {
            expected: 1,
            got: n,
        }),
    }
}
