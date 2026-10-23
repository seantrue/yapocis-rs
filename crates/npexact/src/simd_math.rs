//! numpy's own SIMD implementations of transcendental ufuncs, as scalar code.
//!
//! On every host with FMA, numpy computes float32 `sin`/`cos` with a Cody-Waite
//! reduction and short polynomials (`loops_trigonometric.dispatch.cpp`), and
//! float32/float64 `tanh` with a table-driven polynomial
//! (`loops_hyperbolic.dispatch.cpp.src`), both lane-independent fused
//! multiply-add arithmetic. Those results differ from libm by one or two ulp, so
//! a bit-exact port cannot call `f32::sin`. These functions reproduce the lane
//! arithmetic exactly, including the libm fallback for arguments outside the
//! reduction range, where numpy calls `sinf`/`cosf` (and so does Rust).
//!
//! Float64 `sin`/`cos`/`exp`/`log` and float32 `exp`/`log` go to libm on arm64
//! hosts (numpy disabled its float64 SIMD trig; its float32 exp/log SIMD exists
//! only for AVX2/AVX512), so the Rust `f64::sin` etc. match there. The host
//! fixtures (`yapocis-tools bundle --host-fixtures`) prove which is which on
//! each machine.

mod tables {
    #![allow(clippy::unreadable_literal)]
    include!("simd_math_tables.rs");
}

/// Round to nearest even through numpy's magic-number trick
/// (`x + 0x1.8p23 - 0x1.8p23`), valid for |x| < 2^22.
#[inline]
fn rint_magic_f32(x: f32) -> f32 {
    let magic = f32::from_bits(0x4b40_0000); // 0x1.800000p+23f
    (x + magic) - magic
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigOp {
    Sin,
    Cos,
}

/// numpy's `simd_sincos_f32` for one lane.
fn sincos_f32(x_in: f32, op: TrigOp) -> f32 {
    // numpy's hex-float constants, as bit patterns (checked against
    // numpy.float32(float.fromhex(...)))
    let two_over_pi = f32::from_bits(0x3f22_f983); // 0x1.45f306p-1f
    let codyw_pio2_high = f32::from_bits(0xbfc9_0fd8); // -0x1.921fb0p+00f
    let codyw_pio2_med = f32::from_bits(0xb4a8_885a); // -0x1.5110b4p-22f
    let codyw_pio2_low = f32::from_bits(0xa7c2_34c4); // -0x1.846988p-48f
    if x_in.is_nan() {
        return f32::NAN;
    }
    // 117435.992f and 71476.0625f in numpy's source, as bit patterns
    let max_cody = match op {
        TrigOp::Sin => f32::from_bits(0x47e55dff),
        TrigOp::Cos => f32::from_bits(0x478b9a08),
    };
    // Outside Cody-Waite's range numpy calls libm per element (the NaN case
    // returned above, so a plain comparison is enough)
    if x_in.abs() > max_cody {
        return match op {
            TrigOp::Sin => x_in.sin(),
            TrigOp::Cos => x_in.cos(),
        };
    }
    let x = x_in;
    let quadrant = rint_magic_f32(x * two_over_pi);
    let mut reduced = quadrant.mul_add(codyw_pio2_high, x);
    reduced = quadrant.mul_add(codyw_pio2_med, reduced);
    reduced = quadrant.mul_add(codyw_pio2_low, reduced);
    let x2 = reduced * reduced;
    // cosine polynomial
    let cos = {
        let invf8 = f32::from_bits(0x37cc_730b); // 0x1.98e616p-16f
        let invf6 = f32::from_bits(0xbab6_036e); // -0x1.6c06dcp-10f
        let invf4 = f32::from_bits(0x3d2a_aa9e); // 0x1.55553cp-05f
        let invf2 = -0.5f32;
        let invf0 = 1.0f32;
        let mut r = invf8.mul_add(x2, invf6);
        r = r.mul_add(x2, invf4);
        r = r.mul_add(x2, invf2);
        r.mul_add(x2, invf0)
    };
    // sine polynomial
    let sin = {
        let invf9 = f32::from_bits(0x363e_9dde); // 0x1.7d3bbcp-19f
        let invf7 = f32::from_bits(0xb950_35dd); // -0x1.a06bbap-13f
        let invf5 = f32::from_bits(0x3c08_88cd); // 0x1.11119ap-07f
        let invf3 = f32::from_bits(0xbe2a_aaab); // -0x1.555556p-03f
        let mut r = invf9.mul_add(x2, invf7);
        r = r.mul_add(x2, invf5);
        r = r.mul_add(x2, invf3);
        r = r.mul_add(x2, 0.0);
        r.mul_add(reduced, reduced)
    };
    // NearestInt(quadrant): quadrant is already integral
    let mut iquadrant = quadrant.round_ties_even() as i32;
    if op == TrigOp::Cos {
        iquadrant = iquadrant.wrapping_add(1);
    }
    let mut v = if iquadrant & 1 == 0 { sin } else { cos };
    if iquadrant & 2 == 2 {
        v = 0.0 - v;
    }
    v
}

/// `np.abs` of a complex64 (`loops_unary_complex.dispatch.c.src`): numpy's SIMD lanes
/// compute `larger * sqrt(fma(r, r, 1))` with `r = smaller / larger`, not `hypotf`, so the
/// two differ by an ulp on about one value in three hundred. An infinite part gives
/// infinity even beside a NaN; otherwise a NaN part gives NaN.
pub fn cabs_f32(re: f32, im: f32) -> f32 {
    let (mut re, mut im) = (re.abs(), im.abs());
    if re == f32::INFINITY {
        im = f32::INFINITY;
    }
    if im == f32::INFINITY {
        re = f32::INFINITY;
    }
    if re.is_nan() {
        im = f32::NAN;
    }
    if im.is_nan() {
        re = f32::NAN;
    }
    if re.is_nan() {
        return f32::NAN;
    }
    let (larger, smaller) = if re >= im { (re, im) } else { (im, re) };
    let ratio = if larger == 0.0 || smaller == f32::INFINITY {
        0.0
    } else {
        smaller / larger
    };
    ratio.mul_add(ratio, 1.0).sqrt() * larger
}

/// `np.sin` on a float32 array, one element.
pub fn sin_f32(x: f32) -> f32 {
    sincos_f32(x, TrigOp::Sin)
}

/// `np.cos` on a float32 array, one element.
pub fn cos_f32(x: f32) -> f32 {
    sincos_f32(x, TrigOp::Cos)
}

/// `np.tanh` on a float32 array, one element (`simd_tanh_f32`).
pub fn tanh_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let bits = x.to_bits();
    let ndnan = (bits & 0x7fe0_0000) as i32;
    let special = ndnan <= 0x7f00_0000;
    let idx = (ndnan.wrapping_sub(0x3d40_0000)).clamp(0, 0x3e0_0000) as u32 >> 21;
    let t = &tables::TANH_F32_LUT32X8;
    let coef = |row: usize| f32::from_bits(t[row * 32 + idx as usize]);
    let (b, c0, c1, c2, c3, c4, c5, c6) = (
        coef(0),
        coef(1),
        coef(2),
        coef(3),
        coef(4),
        coef(5),
        coef(6),
        coef(7),
    );
    let sign = bits & 0x8000_0000;
    let y = x.abs() - b;
    let mut r = c6.mul_add(y, c5);
    r = r.mul_add(y, c4);
    r = r.mul_add(y, c3);
    r = r.mul_add(y, c2);
    r = r.mul_add(y, c1);
    r = r.mul_add(y, c0);
    if !special {
        r = 1.0;
    }
    f32::from_bits(r.to_bits() | sign)
}

/// `np.tanh` on a float64 array, one element (`simd_tanh_f64`).
pub fn tanh_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let bits = x.to_bits();
    let ndnan = (bits & 0x7ff8_0000_0000_0000) as i64;
    let special = ndnan <= 0x7fe0_0000_0000_0000;
    // (ndnan - 0x3fc0...) as two 32-bit lanes: the low lane stays 0, the high
    // lane is clamped to 0 ..= 0x780000, then the 64-bit value shifts right by 51
    let high = ((ndnan.wrapping_sub(0x3fc0_0000_0000_0000) >> 32) as i32).clamp(0, 0x78_0000);
    let idx = (high as u32 >> 19) as usize;
    let t = &tables::TANH_F64_LUT16X18;
    let coef = |row: usize| f64::from_bits(t[row * 16 + idx]);
    let b = coef(0);
    let sign = bits & 0x8000_0000_0000_0000;
    let y = x.abs() - b;
    let mut r = coef(17).mul_add(y, coef(16));
    for row in (1..=15).rev() {
        r = r.mul_add(y, coef(row));
    }
    if !special {
        r = 1.0;
    }
    f64::from_bits(r.to_bits() | sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_small_and_large() {
        assert_eq!(tanh_f32(0.0), 0.0);
        assert_eq!(tanh_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(tanh_f32(100.0), 1.0);
        assert_eq!(tanh_f32(-100.0), -1.0);
        assert_eq!(tanh_f64(50.0), 1.0);
        assert!(tanh_f32(f32::NAN).is_nan());
        assert!((tanh_f64(0.5) - 0.5f64.tanh()).abs() < 1e-15);
        assert!((tanh_f32(0.5) - 0.5f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn sincos_near_libm() {
        for i in -1000..1000 {
            let x = i as f32 * 0.1;
            assert!((sin_f32(x) - x.sin()).abs() < 1e-6, "sin {x}");
            assert!((cos_f32(x) - x.cos()).abs() < 1e-6, "cos {x}");
        }
        // beyond the reduction range numpy and this port both call libm
        assert_eq!(sin_f32(200_000.0), 200_000.0f32.sin());
        assert!(sin_f32(f32::NAN).is_nan());
    }
}

#[cfg(test)]
mod cabs_tests {
    use super::cabs_f32;

    #[test]
    fn special_values() {
        assert_eq!(cabs_f32(3.0, -4.0), 5.0);
        assert_eq!(cabs_f32(0.0, 0.0), 0.0);
        assert_eq!(cabs_f32(f32::NEG_INFINITY, f32::NAN), f32::INFINITY);
        assert!(cabs_f32(f32::NAN, 1.0).is_nan());
        assert_eq!(cabs_f32(1.0, f32::INFINITY), f32::INFINITY);
    }
}
