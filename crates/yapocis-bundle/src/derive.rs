//! Turning user arguments into the kernel argument list, as
//! `prepare_arguments` in the Python backends does (`yapocis/ocl/kernel.py`
//! and friends, with the dimension handlers from `yapocis/core/kernel_base.py`).

use crate::manifest::{DType, Dim, KernelEntry, ParamKind, Role};

/// What the derivation needs to know about a user argument.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgShape {
    Array { shape: Vec<usize>, dtype: DType },
    Scalar(f64),
}

impl ArgShape {
    pub fn array(shape: &[usize], dtype: DType) -> ArgShape {
        ArgShape::Array {
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            ArgShape::Array { shape, .. } => Some(shape),
            ArgShape::Scalar(_) => None,
        }
    }

    pub fn size(&self) -> Option<usize> {
        self.shape().map(|s| s.iter().product())
    }
}

/// A scalar cast to the parameter's element type (Python: `dtype(arg)`).
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Int(i64),
    Float(f64),
}

impl ScalarValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            ScalarValue::Int(v) => *v as f64,
            ScalarValue::Float(v) => *v,
        }
    }
}

/// One entry of the kernel argument list, in kernel-argument order.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelArg {
    /// A scalar parameter cast to its element type.
    Scalar {
        param: String,
        dtype: DType,
        value: ScalarValue,
    },
    /// A derived dimension (always int32).
    Dim {
        param: String,
        dim: Dim,
        /// User argument the value comes from.
        arg: usize,
        value: i32,
    },
    /// A user-supplied array.
    Buffer {
        param: String,
        role: Role,
        dtype: DType,
        arg: usize,
    },
    /// An output allocated like a user argument (`np.empty_like(arg, dtype)`).
    OutLike {
        param: String,
        arg: usize,
        shape: Vec<usize>,
        dtype: DType,
    },
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DeriveError {
    #[error("kernel {kernel:?} takes {expected} arguments, got {got}")]
    ArgumentCount {
        kernel: String,
        expected: usize,
        got: usize,
    },
    #[error("parameter {param:?} expects an array")]
    ExpectedArray { param: String },
    #[error("parameter {param:?} expects a scalar")]
    ExpectedScalar { param: String },
    #[error("{dim} of {param:?}: array has no dimension {index} (shape={shape:?})")]
    NoSuchDimension {
        dim: String,
        param: String,
        index: usize,
        shape: Vec<usize>,
    },
    #[error("parameter {param:?} is derived from unknown parameter {of:?}")]
    UnknownSource { param: String, of: String },
}

/// Dimension value with the semantics of `KernelBase.handle_*` (not `dispatch_expr`).
pub fn dim_value(dim: Dim, shape: &[usize], param: &str) -> Result<usize, DeriveError> {
    let at = |i: usize| {
        shape
            .get(i)
            .copied()
            .ok_or_else(|| DeriveError::NoSuchDimension {
                dim: dim.name(),
                param: param.to_string(),
                index: i,
                shape: shape.to_vec(),
            })
    };
    Ok(match dim {
        Dim::SizeOf => shape.iter().product(),
        Dim::LengthOf | Dim::WidthOf => at(0)?,
        Dim::HeightOf => {
            if shape.len() > 1 {
                shape[1]
            } else {
                1
            }
        }
        Dim::DepthOf => at(2)?,
        Dim::DimN(n) => at(n)?,
    })
}

fn cast_scalar(dtype: DType, value: f64) -> ScalarValue {
    match dtype {
        DType::Float16 | DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128 => {
            ScalarValue::Float(value)
        }
        _ => ScalarValue::Int(value.trunc() as i64),
    }
}

/// Build the kernel argument list for a call with `args` (one per `user_args` entry).
pub fn derive_args(kernel: &KernelEntry, args: &[ArgShape]) -> Result<Vec<KernelArg>, DeriveError> {
    if args.len() != kernel.user_args.len() {
        return Err(DeriveError::ArgumentCount {
            kernel: kernel.name.clone(),
            expected: kernel.user_args.len(),
            got: args.len(),
        });
    }
    let mut out = Vec::with_capacity(kernel.params.len());
    for p in &kernel.params {
        let source = p.of.as_deref().unwrap_or(&p.name);
        let arg = kernel
            .user_arg_index(source)
            .ok_or_else(|| DeriveError::UnknownSource {
                param: p.name.clone(),
                of: source.to_string(),
            })?;
        let value = &args[arg];
        match p.kind {
            ParamKind::Scalar => {
                let ArgShape::Scalar(v) = value else {
                    return Err(DeriveError::ExpectedScalar {
                        param: p.name.clone(),
                    });
                };
                out.push(KernelArg::Scalar {
                    param: p.name.clone(),
                    dtype: p.dtype,
                    value: cast_scalar(p.dtype, *v),
                });
            }
            ParamKind::Dim => {
                let shape = value.shape().ok_or_else(|| DeriveError::ExpectedArray {
                    param: source.to_string(),
                })?;
                let dim = p.dim.unwrap_or(Dim::SizeOf);
                out.push(KernelArg::Dim {
                    param: p.name.clone(),
                    dim,
                    arg,
                    value: dim_value(dim, shape, source)? as i32,
                });
            }
            ParamKind::OutLike => {
                let shape = value.shape().ok_or_else(|| DeriveError::ExpectedArray {
                    param: source.to_string(),
                })?;
                out.push(KernelArg::OutLike {
                    param: p.name.clone(),
                    arg,
                    shape: shape.to_vec(),
                    dtype: p.dtype,
                });
            }
            ParamKind::Buffer => {
                if value.shape().is_none() {
                    return Err(DeriveError::ExpectedArray {
                        param: p.name.clone(),
                    });
                }
                out.push(KernelArg::Buffer {
                    param: p.name.clone(),
                    role: p.role.unwrap_or(Role::In),
                    dtype: p.dtype,
                    arg,
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_values_follow_kernel_base() {
        let s = [7usize, 5, 3];
        assert_eq!(dim_value(Dim::SizeOf, &s, "a").unwrap(), 105);
        assert_eq!(dim_value(Dim::WidthOf, &s, "a").unwrap(), 7);
        assert_eq!(dim_value(Dim::HeightOf, &s, "a").unwrap(), 5);
        assert_eq!(dim_value(Dim::DepthOf, &s, "a").unwrap(), 3);
        assert_eq!(dim_value(Dim::DimN(2), &s, "a").unwrap(), 3);
        assert_eq!(dim_value(Dim::HeightOf, &[11], "a").unwrap(), 1);
        assert_eq!(dim_value(Dim::LengthOf, &[11], "a").unwrap(), 11);
        assert!(matches!(
            dim_value(Dim::DepthOf, &[7, 5], "a"),
            Err(DeriveError::NoSuchDimension { index: 2, .. })
        ));
    }
}
