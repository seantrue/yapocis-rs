//! The CPU ports of the bundle's kernels, one module per interface.
//!
//! Each module ports `bundle/ocl/<interface>.cl`: one Rust function per
//! OpenCL kernel, named after its symbol, taking the same arguments in the
//! same order (see [`Args`]). Aliases (`add_res`, `gauss9_res`, ...) share a
//! symbol and therefore a port. The OpenCL text is the reference; the numbax
//! kernels in Yapocis3 (`src/yapocis/kernels/numbax/`) were a starting point,
//! but they differ from OpenCL in places (padding, epsilons) and the ports
//! follow OpenCL.
//!
//! Conventions for a port:
//!
//! * a work item is one iteration of a loop over `0..a.global_size()` (or a
//!   loop nest over `a.global_dim(axis)`), with the kernel's own bounds tests;
//! * `size_t` arithmetic is `usize`, `int` is `i32`, and where OpenCL mixes
//!   them the port says which wins;
//! * a float literal without `f` is a `double` in OpenCL C, so an expression
//!   that uses one is computed in `f64` and rounded where OpenCL assigns it to
//!   a `float`;
//! * work items that write only their own outputs run through [`super::par`];
//!   anything else is a plain sequential loop.

pub mod anisodiff;
pub mod audio;
pub mod cnn;
pub mod colormath;
pub mod convolve;
pub mod demo;
pub mod fft;
pub mod gradient;
pub mod graphics;
pub mod graphics4;
pub mod hsi;
pub mod imageops;
pub mod mandelbrot;
pub mod median3x3;
pub mod operators;
pub mod simplex;
pub mod truetrace;
pub mod vgg16;
pub mod zcs;

use super::Args;
use std::sync::RwLock;

/// A ported kernel.
pub type Kernel = Box<dyn Fn(&Args) + Send + Sync>;

/// A module's kernels, by symbol.
pub type Table = &'static [(&'static str, fn(&Args))];

/// A [`Table`] as the boxed kernels a [`Provider`] returns.
pub fn table(t: Table) -> Vec<(String, Kernel)> {
    t.iter()
        .map(|&(name, f)| (name.to_string(), Box::new(f) as Kernel))
        .collect()
}

/// Makes the ports of interface `name` (without its variant) from its OpenCL
/// source (see [`for_interface`]).
pub type Provider = fn(name: &str, source: &str) -> Result<Vec<(String, Kernel)>, String>;

/// Interfaces other crates have ported, by name (see [`register`]).
static REGISTERED: RwLock<Vec<(String, Provider)>> = RwLock::new(Vec::new());

/// Add the ports of interface `name` from another crate's bundle (a second
/// bundle such as impressionist-rs's yapocis_external). A registered name is
/// looked up before this crate's own, and registering it again replaces it.
pub fn register(name: &str, provider: Provider) {
    let mut registered = REGISTERED.write().unwrap_or_else(|e| e.into_inner());
    registered.retain(|(n, _)| n != name);
    registered.push((name.to_string(), provider));
}

/// The ports for interface `id` (`"operators"`, `"convolve.box3"`, ...):
/// a registered interface's, else this crate's own. `source` is the
/// interface's OpenCL text; only the convolution families, whose weights are
/// rendered into it, read it.
pub fn for_interface(id: &str, source: &str) -> Result<Vec<(String, Kernel)>, String> {
    let name = id.split('.').next().unwrap_or(id);
    let registered = REGISTERED.read().unwrap_or_else(|e| e.into_inner());
    if let Some((_, provider)) = registered.iter().find(|(n, _)| n == name) {
        return provider(name, source);
    }
    drop(registered);
    Ok(match name {
        "anisodiff" => table(anisodiff::KERNELS),
        "audio" => table(audio::KERNELS),
        "cnn" => table(cnn::KERNELS),
        "colormath" => table(colormath::KERNELS),
        "convolve" | "convolves" => convolve::one_dimensional(source)?,
        "convolvesep" => convolve::separable(source)?,
        "demo" => table(demo::KERNELS),
        "fft" => table(fft::KERNELS),
        "gradient" => table(gradient::KERNELS),
        "graphics" => table(graphics::KERNELS),
        "graphics4" => table(graphics4::KERNELS),
        "hsi" => table(hsi::KERNELS),
        "imageops" => table(imageops::KERNELS),
        "mandelbrot" => table(mandelbrot::KERNELS),
        "median3x3" => table(median3x3::KERNELS),
        "operators" => table(operators::KERNELS),
        "simplex" => table(simplex::KERNELS),
        "truetrace" => table(truetrace::KERNELS),
        "vgg16" => table(vgg16::KERNELS),
        "zcs" => table(zcs::KERNELS),
        other => return Err(format!("interface {other:?} has no CPU port")),
    })
}
