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

pub mod fsum;
pub mod hist;
pub mod order;
pub mod random;
pub mod reduce;
pub mod simd_math;
