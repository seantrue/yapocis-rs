//! `yapocis.lib.mandelbrot`: escape-time iteration counts over a complex grid.

use crate::interfaces::mandelbrot;
use crate::{Device, Result};
use ndarray::{Array2, ArrayD, ArrayViewD};
use num_complex::Complex;

pub struct Mandelbrot {
    pub kernels: mandelbrot::Mandelbrot,
}

impl Mandelbrot {
    pub fn load(device: &Device) -> Result<Mandelbrot> {
        Ok(Mandelbrot {
            kernels: mandelbrot::Mandelbrot::load(device)?,
        })
    }

    /// Iteration counts for a flat array of complex coordinates.
    pub fn calc(&self, q: &ArrayViewD<'_, Complex<f32>>, maxiter: i32) -> Result<ArrayD<i16>> {
        self.kernels.mandelbrot(q, maxiter)
    }

    /// A `w × h` grid over `[x0, x1] × [y0, y1]` (like `examples/mandelbrot.py`).
    pub fn grid(w: usize, h: usize, x0: f32, x1: f32, y0: f32, y1: f32) -> Array2<Complex<f32>> {
        Array2::from_shape_fn((h, w), |(i, j)| {
            Complex::new(
                x0 + (x1 - x0) * j as f32 / w as f32,
                y0 + (y1 - y0) * i as f32 / h as f32,
            )
        })
    }
}
