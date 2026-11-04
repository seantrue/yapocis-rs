//! Bit-exact ports of numpy host numerics.
//!
//! Virtual Impressionist's Python renderer is the reference for its Rust port,
//! and much of what it computes on the host goes through numpy: random draws,
//! sums and means, quantiles, sorts, and `np.sin`/`np.cos` on float32 arrays
//! (numpy's own SIMD polynomials, not libm). Reproducing those results bit for
//! bit means reproducing numpy's algorithms, not just its mathematics. Each
//! module here ports one family from numpy 2.5.3's sources, cites the file it
//! mirrors, and is proven by fixtures numpy itself recorded
//! (`yapocis-tools bundle --host-fixtures`), compared on bit patterns with no
//! tolerance.
//!
//! * [`random`]: `SeedSequence`, `PCG64` and the `Generator` draws
//!   (`random`, `integers`, `geometric`, `standard_normal`, `normal`, `uniform`,
//!   `exponential`, `choice(replace=False)`, `permutation`, `shuffle`).
//! * [`hist`]: `histogram` (uniform bins, density), `linspace`, `interp`, `cumsum`.
//! * [`reduce`]: pairwise `sum`, `mean`, `var` and `std`.
//! * [`simd_math`]: numpy's own float32 `sin`/`cos` and float32/float64 `tanh`.
//! * [`elementwise`]: float32 `minimum`, `maximum`, `clip` (lower bound),
//!   `sign`, `mod`, `nan_to_num`.
//! * [`extrema`]: float32 `min`, `max`, `nanmin`, `nanmax`; int64 `cumsum`.
//! * [`order`]: `sort`, `searchsorted`, `lexsort`, `quantile`, `median`,
//!   `nanpercentile`, `unique` with its inverse.
//! * [`fft`]: `fftshift` and `ifftshift` of a 2-D plane.
//!
//! Where numpy's result depends on how it was built or which CPU runs it, the
//! difference is a policy with a `host_default()`: [`random::Contraction`] (FMA
//! in the generator's arithmetic), [`elementwise::MinMax`] (how `maximum` and
//! `minimum` break a tie between `-0` and `+0`, and the AVX2 reduction order)
//! and [`simd_math::Quadrant`] (whether `sin`/`cos` round their quadrant through
//! a fused multiply-add). Fixtures from an Apple M4 Max and an x86-64 Xeon with
//! AVX2 prove both settings of each; an AVX-512 host is not covered.

pub mod elementwise;
pub mod extrema;
pub mod fft;
pub mod fsum;
pub mod hist;
pub mod order;
pub mod random;
pub mod reduce;
pub mod simd_math;
