//! numpy's elementwise float32 ufuncs where their results are more than IEEE
//! arithmetic: `minimum` and `maximum` (NaN propagates; how a tie between `-0`
//! and `+0` resolves depends on the CPU, see [`MinMax`]), `clip` with only a
//! lower bound, `sign`, `mod` (Python's modulo: the divisor's sign) and
//! `nan_to_num`.
//!
//! Proven by the `elementwise` host fixtures over every pair of specials and over
//! long arrays with NaN and signed zeros sprinkled in. Nothing here panics.

/// How numpy's `maximum` and `minimum` (and the `max`, `min`, `nanmax` and
/// `nanmin` reductions built on them) resolve a tie between `-0` and `+0`.
/// numpy's loops use the CPU's own instructions, which differ:
///
/// * `Ieee`: arm64 (`fmax`/`fmin`, NEON). IEEE 754-2019 `maximum`/`minimum`:
///   `+0` is the larger zero, so the result doesn't depend on the order.
/// * `X86`: x86-64 (`maxss`/`maxps` under SSE and AVX2). `a > b ? a : b`, so a
///   tie gives the second operand, and a reduction's result depends on numpy's
///   lane layout, reproduced for AVX2 ([`crate::extrema`]).
///
/// Proven by fixtures recorded on an Apple M4 Max (`Ieee`) and a Xeon E5-2698 v4,
/// AVX2 without AVX-512 (`X86`). numpy picks its AVX-512 loops on a CPU that
/// has them; their reductions are not reproduced or proven here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinMax {
    Ieee,
    X86,
}

impl MinMax {
    /// The policy of the CPU this code was built for.
    pub const fn host_default() -> MinMax {
        if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
            MinMax::X86
        } else {
            MinMax::Ieee
        }
    }

    /// The policy of a machine named as Python's `platform.machine()` does
    /// (what host fixtures record).
    pub fn for_machine(machine: &str) -> MinMax {
        match machine.to_ascii_lowercase().as_str() {
            "x86_64" | "amd64" | "i386" | "i686" | "x86" => MinMax::X86,
            _ => MinMax::Ieee,
        }
    }
}

/// x86's `maxps(a, b)`: `a > b ? a : b` (a tie or a NaN gives `b`).
pub(crate) fn max_ps(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// x86's `minps(a, b)`: `a < b ? a : b`.
pub(crate) fn min_ps(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

/// `np.maximum(a, b)` for float32 on this host ([`MinMax::host_default`]).
pub fn maximum_f32(a: f32, b: f32) -> f32 {
    maximum_f32_with(MinMax::host_default(), a, b)
}

/// `np.minimum(a, b)` for float32 on this host.
pub fn minimum_f32(a: f32, b: f32) -> f32 {
    minimum_f32_with(MinMax::host_default(), a, b)
}

/// `np.maximum(a, b)` for float32 under `policy`: NaN if either is NaN; a tie
/// between `-0` and `+0` gives `+0` (`Ieee`) or `b` (`X86`).
pub fn maximum_f32_with(policy: MinMax, a: f32, b: f32) -> f32 {
    match policy {
        MinMax::X86 => {
            // npyv_maxn_f32 / scalar_max_f: blendv(a, maxps(a, b), ordered(a))
            if a.is_nan() {
                a
            } else {
                max_ps(a, b)
            }
        }
        MinMax::Ieee => {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else if a == b {
                if a.is_sign_positive() {
                    a
                } else {
                    b
                }
            } else {
                max_ps(a, b)
            }
        }
    }
}

/// `np.minimum(a, b)` for float32 under `policy`: NaN if either is NaN; a tie
/// between `-0` and `+0` gives `-0` (`Ieee`) or `b` (`X86`).
pub fn minimum_f32_with(policy: MinMax, a: f32, b: f32) -> f32 {
    match policy {
        MinMax::X86 => {
            if a.is_nan() {
                a
            } else {
                min_ps(a, b)
            }
        }
        MinMax::Ieee => {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else if a == b {
                if a.is_sign_negative() {
                    a
                } else {
                    b
                }
            } else {
                min_ps(a, b)
            }
        }
    }
}

/// `np.clip(v, low, None)` for float32 on this host: numpy clips with a lower
/// bound only through `maximum(v, low)`, so NaN stays NaN and `-0` clipped at
/// `0` is `+0` (on either policy).
pub fn clip_min_f32(v: f32, low: f32) -> f32 {
    maximum_f32(v, low)
}

/// [`clip_min_f32`] under `policy`.
pub fn clip_min_f32_with(policy: MinMax, v: f32, low: f32) -> f32 {
    maximum_f32_with(policy, v, low)
}

/// `np.sign` of a float32: `-1`, `+0` (for either zero) or `1`; NaN for NaN.
pub fn sign_f32(v: f32) -> f32 {
    if v.is_nan() {
        f32::NAN
    } else if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `np.mod(a, b)` (`np.remainder`) for float32, numpy's `npy_divmodf`: `fmod`,
/// moved to the divisor's sign when the signs differ; an exact zero takes the
/// divisor's sign; a zero divisor gives NaN. With NumPy 2's weak scalars, a
/// Python-float divisor is cast to float32 first.
pub fn mod_f32(a: f32, b: f32) -> f32 {
    let mut m = a % b;
    if b == 0.0 {
        return m;
    }
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        m = 0.0f32.copysign(b);
    }
    m
}

/// `np.nan_to_num` of a float32: NaN → `0`, `±inf` → the largest finite float32
/// of that sign.
pub fn nan_to_num_f32(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else if v == f32::INFINITY {
        f32::MAX
    } else if v == f32::NEG_INFINITY {
        f32::MIN
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_zeros_and_nan() {
        let ieee = MinMax::Ieee;
        assert!(maximum_f32_with(ieee, -0.0, 0.0).is_sign_positive());
        assert!(maximum_f32_with(ieee, 0.0, -0.0).is_sign_positive());
        assert!(minimum_f32_with(ieee, 0.0, -0.0).is_sign_negative());
        let x86 = MinMax::X86;
        assert!(maximum_f32_with(x86, 0.0, -0.0).is_sign_negative());
        assert!(minimum_f32_with(x86, -0.0, 0.0).is_sign_positive());
        for p in [ieee, x86] {
            assert!(maximum_f32_with(p, f32::NAN, 1.0).is_nan());
            assert!(maximum_f32_with(p, 1.0, f32::NAN).is_nan());
            assert!(minimum_f32_with(p, 1.0, f32::NAN).is_nan());
            assert!(clip_min_f32_with(p, -0.0, 0.0).is_sign_positive());
        }
        assert!(sign_f32(-0.0).is_sign_positive());
        assert_eq!(mod_f32(-1.0, 360.0), 359.0);
        assert!(mod_f32(-0.0, 3.0).is_sign_positive());
        assert!(mod_f32(1.0, 0.0).is_nan());
        assert_eq!(mod_f32(-5.0, f32::INFINITY), f32::INFINITY);
    }
}
