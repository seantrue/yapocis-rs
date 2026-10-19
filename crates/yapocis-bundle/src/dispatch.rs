//! `dispatch_size` evaluation with the semantics of `yapocis.core.dispatch_expr`.
//!
//! Two quirks are reproduced on purpose, because the Python runtime has them:
//!
//! * dimension words mean something different here than for derived parameters:
//!   `width_of` is `shape[2]` for 3-D (or more) arrays and `shape[0]` otherwise,
//!   `depth_of` is `shape[0]`;
//! * a reference's `param_index` is the parameter's position in the *full*
//!   parameter list (derived parameters included) but indexes the *user*
//!   argument tuple.
//!
//! Division is true division and the final value is truncated toward zero
//! (Python `int()`), `//` is floor division and `%` is Python's modulo.

use crate::derive::ArgShape;
use crate::manifest::DispatchNode;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DispatchError {
    #[error("cannot find parameter {name:?} (index {index}) in {nargs} arguments")]
    MissingParameter {
        name: String,
        index: usize,
        nargs: usize,
    },
    #[error("parameter {name:?} must be an array for {dim}")]
    NotAnArray { name: String, dim: String },
    #[error("array {name:?} does not have dimension {n} (shape={shape:?})")]
    NoSuchDimension {
        name: String,
        n: usize,
        shape: Vec<usize>,
    },
    #[error("unknown dimension type {0:?}")]
    UnknownDim(String),
    #[error("unknown operator {0:?}")]
    UnknownOp(String),
}

/// Evaluate one expression to its (untruncated) numeric value.
pub fn evaluate(node: &DispatchNode, args: &[ArgShape]) -> Result<f64, DispatchError> {
    match node {
        DispatchNode::Lit(v) => Ok(*v),
        DispatchNode::Dim {
            dim,
            of,
            param_index,
        } => {
            let arg = args
                .get(*param_index)
                .ok_or_else(|| DispatchError::MissingParameter {
                    name: of.clone(),
                    index: *param_index,
                    nargs: args.len(),
                })?;
            let shape = match arg {
                ArgShape::Array { shape, .. } => shape,
                ArgShape::Scalar(_) => {
                    return Err(DispatchError::NotAnArray {
                        name: of.clone(),
                        dim: dim.clone(),
                    })
                }
            };
            let value =
                if let Some(n) = dim.strip_prefix("dim_").and_then(|s| s.strip_suffix("_of")) {
                    let n: usize = n
                        .parse()
                        .map_err(|_| DispatchError::UnknownDim(dim.clone()))?;
                    *shape.get(n).ok_or_else(|| DispatchError::NoSuchDimension {
                        name: of.clone(),
                        n,
                        shape: shape.clone(),
                    })?
                } else {
                    match dim.as_str() {
                        "width_of" => {
                            if shape.len() > 2 {
                                shape[2]
                            } else {
                                shape[0]
                            }
                        }
                        "height_of" => {
                            if shape.len() > 1 {
                                shape[1]
                            } else {
                                1
                            }
                        }
                        "depth_of" => shape[0],
                        "sizeof" => shape.iter().product(),
                        other => return Err(DispatchError::UnknownDim(other.to_string())),
                    }
                };
            Ok(value as f64)
        }
        DispatchNode::Neg(arg) => Ok(-evaluate(arg, args)?),
        DispatchNode::Binary { op, lhs, rhs } => {
            let a = evaluate(lhs, args)?;
            let b = evaluate(rhs, args)?;
            Ok(match op.as_str() {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => a / b,
                "//" => (a / b).floor(),
                "%" => a - b * (a / b).floor(),
                other => return Err(DispatchError::UnknownOp(other.to_string())),
            })
        }
    }
}

/// Evaluate a kernel's `dispatch_size` to a global size, `None` when it has none.
pub fn dispatch_global_size(
    dispatch: Option<&[DispatchNode]>,
    args: &[ArgShape],
) -> Result<Option<Vec<usize>>, DispatchError> {
    let Some(nodes) = dispatch else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        // Python: int(result), truncation toward zero
        out.push(evaluate(node, args)?.trunc() as i64 as usize);
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::DType;

    fn arr(shape: &[usize]) -> ArgShape {
        ArgShape::Array {
            shape: shape.to_vec(),
            dtype: DType::Float32,
        }
    }

    fn dim(dim: &str, index: usize) -> DispatchNode {
        DispatchNode::Dim {
            dim: dim.into(),
            of: "x".into(),
            param_index: index,
        }
    }

    #[test]
    fn dimension_semantics_follow_dispatch_expr() {
        let args = [arr(&[3, 7, 5]), arr(&[9])];
        assert_eq!(evaluate(&dim("width_of", 0), &args).unwrap(), 5.0);
        assert_eq!(evaluate(&dim("width_of", 1), &args).unwrap(), 9.0);
        assert_eq!(evaluate(&dim("height_of", 0), &args).unwrap(), 7.0);
        assert_eq!(evaluate(&dim("height_of", 1), &args).unwrap(), 1.0);
        assert_eq!(evaluate(&dim("depth_of", 0), &args).unwrap(), 3.0);
        assert_eq!(evaluate(&dim("sizeof", 0), &args).unwrap(), 105.0);
        assert_eq!(evaluate(&dim("dim_1_of", 0), &args).unwrap(), 7.0);
        assert!(matches!(
            evaluate(&dim("dim_1_of", 1), &args),
            Err(DispatchError::NoSuchDimension { n: 1, .. })
        ));
        assert!(matches!(
            evaluate(&dim("dim_0_of", 5), &args),
            Err(DispatchError::MissingParameter { index: 5, .. })
        ));
    }

    #[test]
    fn arithmetic_matches_python() {
        let args = [arr(&[3, 7, 5])];
        let half = DispatchNode::Binary {
            op: "/".into(),
            lhs: Box::new(dim("dim_1_of", 0)),
            rhs: Box::new(DispatchNode::Lit(2.0)),
        };
        assert_eq!(
            dispatch_global_size(Some(&[half]), &args).unwrap(),
            Some(vec![3])
        );
        let floor = DispatchNode::Binary {
            op: "//".into(),
            lhs: Box::new(dim("dim_1_of", 0)),
            rhs: Box::new(DispatchNode::Lit(3.0)),
        };
        let modulo = DispatchNode::Binary {
            op: "%".into(),
            lhs: Box::new(dim("dim_1_of", 0)),
            rhs: Box::new(DispatchNode::Lit(4.0)),
        };
        assert_eq!(
            dispatch_global_size(Some(&[floor, modulo, DispatchNode::Lit(64.0)]), &args).unwrap(),
            Some(vec![2, 3, 64])
        );
        assert_eq!(dispatch_global_size(None, &args).unwrap(), None);
    }
}
