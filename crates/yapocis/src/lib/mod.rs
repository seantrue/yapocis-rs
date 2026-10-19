//! Ports of the thin Python library wrappers (`yapocis.lib`).
//!
//! Each module mirrors one Python module and names the file it follows. Host
//! logic is kept to what the Python wrappers do: kernel selection by channel
//! count, explicit global sizes, ping-pong iteration on device buffers, and
//! small numpy pre/post-processing.

pub mod anisodiff;
pub mod cnn;
pub mod colormath;
pub mod convolve;
pub mod fft;
pub mod gaussian;
pub mod gradient;
pub mod graphics;
pub mod hsi;
pub mod image;
pub mod imageops;
pub mod kerneldraw;
pub mod mandelbrot;
pub mod median;
pub mod operators;
pub mod paths;
pub mod resnet50;
pub mod simplex;
pub mod truetrace;
#[cfg(feature = "text")]
pub mod typeset;
pub mod vgg16;
pub mod zcs;

pub use anisodiff::Anisodiff;
pub use colormath::Colormath;
pub use convolve::Convolve1D;
pub use fft::Fft;
pub use gaussian::{gaussian_weight, Gaussian};
pub use gradient::Gradient;
pub use graphics::{Graphics1, Graphics4};
pub use hsi::Hsi;
pub use imageops::Imageops;
pub use kerneldraw::{KernelDraw1, KernelDraw3, KernelDraw4};
pub use mandelbrot::Mandelbrot;
pub use median::Median;
pub use operators::Operators;
pub use paths::PathRenderer;
pub use resnet50::ResNet50;
pub use simplex::Simplex;
pub use truetrace::TrueTrace;
pub use vgg16::Vgg16;
pub use zcs::Zcs;
