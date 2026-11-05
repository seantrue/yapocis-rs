//! numpy's complex64 multiply, `a * b = (ar·br − ai·bi) + i(ar·bi + ai·br)`.
//!
//! Its result depends on whether the products and sums are fused into multiply-adds,
//! and numpy fuses in one place and not another:
//!
//! * **Arrays** go through the ufunc's SIMD loop (`loops_arithm_fp.dispatch.c.src`),
//!   which numpy picks when it runs, by CPU. On arm64 (ASIMD) and on x86-64 with FMA3
//!   and AVX2 it computes `re = fma(ar, br, −[ai·bi])` and `im = fma(ar, bi, [ai·br])`,
//!   the bracketed product rounded first: [`Mul::Fused`]. This holds for contiguous
//!   arrays of every length (body and tail), strided views and an array times a scalar.
//! * **numpy scalars** (`np.complex64` times `np.complex64`) go through compiled C,
//!   which follows the compiler that built the wheel, as [`crate::random::Contraction`]
//!   does: fused on arm64 (clang), plain on x86-64 (GCC without `-mfma`).
//!
//! So on an x86-64 host the same product of two numbers can differ by an ulp between
//! an array and a scalar. Proven by the `complex` host fixtures from an Apple M4 Max
//! and a Xeon E5-2698 v4 (FMA3, AVX2, no AVX-512), numpy 2.3.3. Not covered: x86-64
//! without AVX2 (numpy's baseline loop, assumed [`Mul::Plain`]) and AVX-512 hosts
//! (assumed [`Mul::Fused`], as numpy's AVX-512 loop uses the same multiply-adds).

/// Whether a complex multiply fuses its multiply-adds (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mul {
    /// Four products rounded, then a subtraction and an addition rounded.
    Plain,
    /// `re = fma(ar, br, −[ai·bi])`, `im = fma(ar, bi, [ai·br])`.
    Fused,
}

impl Mul {
    /// numpy's ufunc loop over arrays on this machine, decided at run time: fused on
    /// aarch64, and on x86-64 when the CPU has FMA3 and AVX2; plain elsewhere.
    pub fn array_default() -> Mul {
        #[cfg(target_arch = "aarch64")]
        {
            Mul::Fused
        }
        #[cfg(target_arch = "x86_64")]
        {
            Mul::array_for(
                "x86_64",
                std::arch::is_x86_feature_detected!("fma")
                    && std::arch::is_x86_feature_detected!("avx2"),
            )
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Mul::Plain
        }
    }

    /// numpy's ufunc loop for a machine named as Python's `platform.machine()` does, with
    /// whether its CPU has FMA3 and AVX2 (numpy's `__cpu_features__`; ignored on arm64).
    pub fn array_for(machine: &str, fma3_and_avx2: bool) -> Mul {
        match machine.to_ascii_lowercase().as_str() {
            "arm64" | "aarch64" => Mul::Fused,
            "x86_64" | "amd64" if fma3_and_avx2 => Mul::Fused,
            _ => Mul::Plain,
        }
    }

    /// numpy's scalar path (compiled C) for the architecture this code was built for:
    /// fused on aarch64, plain elsewhere. A numpy built some other way may differ; its
    /// host fixtures will say.
    pub const fn scalar_default() -> Mul {
        if cfg!(target_arch = "aarch64") {
            Mul::Fused
        } else {
            Mul::Plain
        }
    }

    /// numpy's scalar path for a machine named as `platform.machine()` does.
    pub fn scalar_for(machine: &str) -> Mul {
        match machine.to_ascii_lowercase().as_str() {
            "arm64" | "aarch64" => Mul::Fused,
            _ => Mul::Plain,
        }
    }
}

/// One complex64 product, `[re, im]`, as numpy computes it under `m`.
pub fn mul_c64(m: Mul, a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let ([ar, ai], [br, bi]) = (a, b);
    match m {
        Mul::Plain => [ar * br - ai * bi, ar * bi + ai * br],
        Mul::Fused => [ar.mul_add(br, -(ai * bi)), ar.mul_add(bi, ai * br)],
    }
}

/// `a * b` elementwise, as numpy's ufunc computes it under `m` (use
/// [`Mul::array_default`] to match this machine's numpy). The shorter length wins.
pub fn multiply_c64(m: Mul, a: &[[f32; 2]], b: &[[f32; 2]]) -> Vec<[f32; 2]> {
    a.iter().zip(b).map(|(&x, &y)| mul_c64(m, x, y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rules_by_machine() {
        assert_eq!(Mul::array_for("arm64", false), Mul::Fused);
        assert_eq!(Mul::array_for("x86_64", true), Mul::Fused);
        assert_eq!(Mul::array_for("AMD64", false), Mul::Plain);
        assert_eq!(Mul::array_for("riscv64", true), Mul::Plain);
        assert_eq!(Mul::scalar_for("arm64"), Mul::Fused);
        assert_eq!(Mul::scalar_for("x86_64"), Mul::Plain);
        if cfg!(target_arch = "aarch64") {
            assert_eq!(
                (Mul::array_default(), Mul::scalar_default()),
                (Mul::Fused, Mul::Fused)
            );
        }
    }

    #[test]
    fn fused_and_plain_agree_when_exact_and_differ_somewhere() {
        // pseudo-random products: about a quarter of them round differently (the host
        // fixtures pin which rule is numpy's)
        let mut s = 0x9e37_79b9_u32;
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1 << 24) as f32 * 4.0 - 2.0
        };
        let differ = (0..1000).filter(|_| {
            let (a, b) = ([next(), next()], [next(), next()]);
            mul_c64(Mul::Plain, a, b) != mul_c64(Mul::Fused, a, b)
        });
        assert!(differ.count() > 100);
        assert_eq!(mul_c64(Mul::Plain, [2.0, 3.0], [4.0, 5.0]), [-7.0, 22.0]);
        assert_eq!(mul_c64(Mul::Fused, [2.0, 3.0], [4.0, 5.0]), [-7.0, 22.0]);
        assert_eq!(
            multiply_c64(Mul::Fused, &[[1.0, 0.0]; 3], &[[0.0, 1.0]; 2]),
            vec![[0.0, 1.0]; 2]
        );
    }
}
