//! numpy's float32 `min`, `max`, `nanmin` and `nanmax` reductions, and `cumsum`
//! of int64.
//!
//! Under [`MinMax::Ieee`] (arm64) the reductions fold with IEEE
//! `maximum`/`minimum`, which are associative and order the signed zeros, so
//! numpy's lane layout doesn't show in the result. Under [`MinMax::X86`] a tie
//! between `-0` and `+0` goes to the second operand, so the result depends on
//! the order numpy combines the elements, and the reduction here follows numpy
//! 2.3's AVX2 loop (`loops_minmax.dispatch.c.src`, `simd/avx2/math.h`) step by
//! step. Where numpy raises (an empty array) these return `None`. Proven by the
//! `extrema` host fixtures.

use crate::elementwise::{max_ps, maximum_f32_with, min_ps, minimum_f32_with, MinMax};

/// The four reductions, as numpy's `maximum`, `minimum`, `fmax` and `fmin`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Max,
    Min,
    FMax,
    FMin,
}

/// numpy's AVX2 float32 reduction of `first` and `rest` (the ufunc's reduce puts
/// the first element in the output and loops over the others): eight lanes all
/// starting at `first`, 64 elements a step combined as a tree, then single
/// vectors, the lanes folded across (`npyv_reduce_*`), and the tail one by one.
fn reduce_x86(kind: Kind, first: f32, rest: &[f32]) -> f32 {
    let ps: fn(f32, f32) -> f32 = match kind {
        Kind::Max | Kind::FMax => max_ps,
        Kind::Min | Kind::FMin => min_ps,
    };
    // the vector step: `maxn` keeps a NaN in a; `maxp` keeps a when b is NaN
    let v = |a: f32, b: f32| -> f32 {
        match kind {
            Kind::Max | Kind::Min => {
                if a.is_nan() {
                    a
                } else {
                    ps(a, b)
                }
            }
            Kind::FMax | Kind::FMin => {
                if b.is_nan() {
                    a
                } else {
                    ps(a, b)
                }
            }
        }
    };
    let vv = |a: &[f32; 8], b: &[f32; 8]| -> [f32; 8] { std::array::from_fn(|i| v(a[i], b[i])) };
    let mut acc = [first; 8];
    let (blocks, rest) = rest.as_chunks::<64>();
    for block in blocks {
        let (r, _) = block.as_chunks::<8>();
        let (r01, r23) = (vv(&r[0], &r[1]), vv(&r[2], &r[3]));
        let (r45, r67) = (vv(&r[4], &r[5]), vv(&r[6], &r[7]));
        acc = vv(&acc, &vv(&vv(&r01, &r23), &vv(&r45, &r67)));
    }
    let (singles, tail) = rest.as_chunks::<8>();
    for chunk in singles {
        acc = vv(&acc, chunk);
    }
    // npyv_reduce_max_f32: the two halves, then lanes (0, 2) and (1, 3), then (0, 1)
    let across = |a: [f32; 8]| -> f32 {
        let v128: [f32; 4] = std::array::from_fn(|i| ps(a[i], a[i + 4]));
        let v64 = [ps(v128[0], v128[2]), ps(v128[1], v128[3])];
        ps(v64[0], v64[1])
    };
    let mut r = match kind {
        Kind::Max | Kind::Min => {
            if acc.iter().any(|x| x.is_nan()) {
                f32::NAN
            } else {
                across(acc)
            }
        }
        Kind::FMax | Kind::FMin => {
            if acc.iter().all(|x| x.is_nan()) {
                acc[0]
            } else {
                let fill = if kind == Kind::FMax {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
                across(acc.map(|x| if x.is_nan() { fill } else { x }))
            }
        }
    };
    for &x in tail {
        r = match kind {
            // scalar_max_f / scalar_min_f: maxss / minss, a NaN in r kept
            Kind::Max | Kind::Min => {
                if r.is_nan() {
                    r
                } else {
                    ps(r, x)
                }
            }
            // glibc's fmaxf / fminf: the number beside a NaN, else a tie gives r
            Kind::FMax => {
                if r.is_nan() {
                    x
                } else if x.is_nan() || r >= x {
                    r
                } else {
                    x
                }
            }
            Kind::FMin => {
                if r.is_nan() {
                    x
                } else if x.is_nan() || r <= x {
                    r
                } else {
                    x
                }
            }
        };
    }
    r
}

fn reduce(policy: MinMax, kind: Kind, a: &[f32]) -> Option<f32> {
    let (&first, rest) = a.split_first()?;
    Some(match policy {
        MinMax::X86 => reduce_x86(kind, first, rest),
        MinMax::Ieee => match kind {
            Kind::Max => rest
                .iter()
                .fold(first, |m, &v| maximum_f32_with(policy, m, v)),
            Kind::Min => rest
                .iter()
                .fold(first, |m, &v| minimum_f32_with(policy, m, v)),
            Kind::FMax => a
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .reduce(|m, v| maximum_f32_with(policy, m, v))
                .unwrap_or(f32::NAN),
            Kind::FMin => a
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .reduce(|m, v| minimum_f32_with(policy, m, v))
                .unwrap_or(f32::NAN),
        },
    })
}

/// `np.max(a)` of float32 on this host: NaN if any element is NaN; `None` when empty.
pub fn max_f32(a: &[f32]) -> Option<f32> {
    max_f32_with(MinMax::host_default(), a)
}

/// `np.min(a)` of float32 on this host: NaN if any element is NaN; `None` when empty.
pub fn min_f32(a: &[f32]) -> Option<f32> {
    min_f32_with(MinMax::host_default(), a)
}

/// `np.nanmax(a)` of float32 on this host: NaN left out; NaN when every element
/// is NaN (numpy warns); `None` when empty.
pub fn nanmax_f32(a: &[f32]) -> Option<f32> {
    nanmax_f32_with(MinMax::host_default(), a)
}

/// `np.nanmin(a)` of float32 on this host: NaN left out; NaN when every element
/// is NaN (numpy warns); `None` when empty.
pub fn nanmin_f32(a: &[f32]) -> Option<f32> {
    nanmin_f32_with(MinMax::host_default(), a)
}

/// [`max_f32`] under `policy`.
pub fn max_f32_with(policy: MinMax, a: &[f32]) -> Option<f32> {
    reduce(policy, Kind::Max, a)
}

/// [`min_f32`] under `policy`.
pub fn min_f32_with(policy: MinMax, a: &[f32]) -> Option<f32> {
    reduce(policy, Kind::Min, a)
}

/// [`nanmax_f32`] under `policy` (`np.fmax.reduce`).
pub fn nanmax_f32_with(policy: MinMax, a: &[f32]) -> Option<f32> {
    reduce(policy, Kind::FMax, a)
}

/// [`nanmin_f32`] under `policy` (`np.fmin.reduce`).
pub fn nanmin_f32_with(policy: MinMax, a: &[f32]) -> Option<f32> {
    reduce(policy, Kind::FMin, a)
}

/// `np.cumsum` of int64: sequential, wrapping on overflow as numpy's integer
/// arithmetic does.
pub fn cumsum_i64(a: &[i64]) -> Vec<i64> {
    let mut acc = 0i64;
    a.iter()
        .map(|&v| {
            acc = acc.wrapping_add(v);
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_nan() {
        for p in [MinMax::Ieee, MinMax::X86] {
            assert_eq!(max_f32_with(p, &[]), None);
            let a = [1.0, f32::NAN, 3.0];
            assert!(max_f32_with(p, &a).is_some_and(f32::is_nan));
            assert_eq!(nanmax_f32_with(p, &a), Some(3.0));
            assert_eq!(nanmin_f32_with(p, &a), Some(1.0));
            assert!(nanmin_f32_with(p, &[f32::NAN]).is_some_and(f32::is_nan));
            let long: Vec<f32> = (0..200).map(|i| i as f32).collect();
            assert_eq!(max_f32_with(p, &long), Some(199.0));
            assert_eq!(min_f32_with(p, &long), Some(0.0));
        }
        assert!(max_f32_with(MinMax::Ieee, &[-0.0, 0.0, -0.0]).is_some_and(f32::is_sign_positive));
        // x86 below a vector's worth: maxss, the later zero wins
        assert!(max_f32_with(MinMax::X86, &[0.0, 0.0, -0.0]).is_some_and(f32::is_sign_negative));
        assert_eq!(cumsum_i64(&[1, 2, 3]), vec![1, 3, 6]);
    }
}
