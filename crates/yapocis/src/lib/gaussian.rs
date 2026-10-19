//! `yapocis.lib.gaussian`: separable Gaussian blur at a ladder of scales,
//! difference-of-Gaussians zero crossings (`zcsdog`), and the weight generator.
//!
//! The kernels come from the bundled default `convolvesep` variant, i.e. the
//! weights `GaussianWeights()` produces from `convolvesep.yml` (max_width 150,
//! scale 1.6, limit 0.001). The scale ladder is recovered from the kernel
//! names (`gauss9`, `gauss13`, ...): scale `i` is the i-th narrowest kernel.

use super::gradient::Gradient;
use super::image::{as_2d, set_frame, set_margin};
use super::operators::Operators;
use super::zcs::Zcs;
use crate::interfaces::convolvesep;
use crate::rt::{CallArg, CallOptions};
use crate::{Device, Error, Program, Result};
use ndarray::{Array2, ArrayD, ArrayViewD, IxDyn};

/// `GaussianWeights.compute_gaussian_weight(sigma, dt=1.0, limit, normalize=True)`.
pub fn gaussian_weight(sigma: f64, dt: f64, limit: f64, normalize: bool) -> Vec<f64> {
    let mut tail = Vec::new();
    let mut x = 0.0f64;
    let mut val = 1.0f64;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let scale_factor = 1.0 / (two_sigma_sq * std::f64::consts::PI).sqrt();
    while val > limit {
        val = (-(x * x) / two_sigma_sq).exp() * scale_factor;
        tail.push(val);
        x += dt;
    }
    let mut k: Vec<f64> = tail.iter().rev().cloned().collect();
    k.extend_from_slice(&tail[1..]);
    if normalize {
        let sum: f64 = k.iter().sum();
        for v in &mut k {
            *v /= sum / dt;
        }
    }
    k
}

/// The default scale ladder (`GaussianWeights()` with the YAML parameters).
pub fn default_weights() -> Vec<Vec<f64>> {
    let (max_width, scale, limit) = (150usize, 1.6f64, 0.001f64);
    let mut sigma = 1.0;
    let mut out = Vec::new();
    loop {
        let w = gaussian_weight(sigma, 1.0, limit, true);
        if w.len() > max_width {
            break;
        }
        out.push(w);
        sigma *= scale;
    }
    out
}

pub struct Gaussian {
    /// The `convolvesep` program: the bundled default ladder, or one generated
    /// for custom weights by [`Gaussian::with_weights`].
    pub program: Program,
    pub zc: Zcs,
    pub grad: Gradient,
    pub ops: Operators,
    /// Kernel widths, narrowest first; `widths[scale]` names `gauss<width>`.
    widths: Vec<usize>,
    pub threshold: f32,
}

impl Gaussian {
    /// The bundled default weight ladder (`Gaussian()` in Python).
    pub fn load(device: &Device) -> Result<Gaussian> {
        Self::from_program(
            device,
            convolvesep::Convolvesep::load(device)?.into_program(),
        )
    }

    /// `Gaussian(gaussian_weights=...)`: compile kernels for arbitrary weight
    /// sets (see [`crate::lib::convolve::load_convolvesep`]).
    pub fn with_weights(device: &Device, weights: &[Vec<f64>]) -> Result<Gaussian> {
        Self::from_program(device, super::convolve::load_convolvesep(device, weights)?)
    }

    fn from_program(device: &Device, program: Program) -> Result<Gaussian> {
        let mut widths: Vec<usize> = program
            .entry()
            .kernels
            .iter()
            .filter_map(|k| {
                k.name
                    .strip_prefix("gauss")
                    .and_then(|w| w.parse::<usize>().ok())
            })
            .collect();
        widths.sort_unstable();
        widths.dedup();
        Ok(Gaussian {
            program,
            zc: Zcs::load(device)?,
            grad: Gradient::load(device)?,
            ops: Operators::load(device)?,
            widths,
            threshold: 7.5e-06,
        })
    }

    pub fn scales(&self) -> usize {
        self.widths.len()
    }

    /// `GaussianWeights.get_gaussian_width(scale)`.
    pub fn width(&self, scale: usize) -> Result<usize> {
        self.widths.get(scale).copied().ok_or_else(|| {
            Error::Invalid(format!(
                "scale {scale} out of range (0..{})",
                self.widths.len()
            ))
        })
    }

    fn run(
        &self,
        name: &str,
        input: &crate::Buffer<f32>,
        output: &crate::Buffer<f32>,
        vertical: Option<i32>,
    ) -> Result<()> {
        let opts = CallOptions {
            async_exec: true,
            ..Default::default()
        };
        match vertical {
            Some(v) => {
                let mut args = [
                    CallArg::Scalar(v as f64),
                    CallArg::Buffer(input.as_any()),
                    CallArg::Buffer(output.as_any()),
                ];
                self.program.call(name, &mut args, &opts)?;
            }
            None => {
                let mut args = [
                    CallArg::Buffer(input.as_any()),
                    CallArg::Buffer(output.as_any()),
                ];
                self.program.call(name, &mut args, &opts)?;
            }
        }
        Ok(())
    }

    /// `Gaussian.gauss_image(a, scale)`: horizontal then vertical pass on the device.
    pub fn gauss_image(&self, a: &ArrayViewD<'_, f32>, scale: usize) -> Result<ArrayD<f32>> {
        let w = self.width(scale)?;
        let device = self.program.device().clone();
        let std = a.as_standard_layout();
        let aa = device.upload(std.as_slice().unwrap(), a.shape())?;
        let b = device.alloc::<f32>(a.shape())?;
        self.run(&format!("gauss{w}_h"), &aa, &b, None)?;
        self.run(&format!("gauss{w}_v"), &b, &aa, None)?;
        device.wait_all()?;
        Ok(ArrayD::from_shape_vec(IxDyn(a.shape()), aa.to_vec()?)?)
    }

    /// `Gaussian.zcsdog(a, scale, threshold, clear_margin=True, frame=True)`:
    /// zero crossings of the difference of Gaussians at `scale` and `scale+1`,
    /// plus the gradient of the smaller blur. Returns `(zca, grad, theta)`.
    pub fn zcsdog(
        &self,
        a: &ArrayViewD<'_, f32>,
        scale: usize,
        threshold: Option<f32>,
        clear_margin: bool,
        frame: bool,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        if a.ndim() != 2 {
            return Err(Error::Invalid(format!(
                "zcsdog needs a 2-D image, got {:?}",
                a.shape()
            )));
        }
        let threshold = threshold.unwrap_or(self.threshold);
        let w = self.width(scale)?;
        let w1 = self.width(scale + 1)?;
        let device = self.program.device().clone();
        let std = a.as_standard_layout();
        let src = device.upload(std.as_slice().unwrap(), a.shape())?;
        let smaller = device.alloc::<f32>(a.shape())?;
        let larger = device.alloc::<f32>(a.shape())?;
        let tmp = device.alloc::<f32>(a.shape())?;
        let zca = device.alloc::<f32>(a.shape())?;
        self.run(&format!("gauss{w}_res"), &src, &tmp, Some(0))?;
        self.run(&format!("gauss{w}_res"), &tmp, &smaller, Some(1))?;
        self.run(&format!("gauss{w1}_res"), &src, &tmp, Some(0))?;
        self.run(&format!("gauss{w1}_res"), &tmp, &larger, Some(1))?;
        self.ops.sub_res(&larger, &smaller, &tmp)?;
        self.zc.zcs_res(&tmp, &zca, threshold)?;
        device.wait_all()?;
        let mut zca = as_2d(ArrayD::from_shape_vec(IxDyn(a.shape()), zca.to_vec()?)?)?;
        let smaller = ArrayD::from_shape_vec(IxDyn(a.shape()), smaller.to_vec()?)?;
        let (grad, theta) = self.grad.gradient(&smaller.view(), 3)?;
        let mut grad = as_2d(grad)?;
        let mut theta = as_2d(theta)?;
        let margin = w + 1;
        if clear_margin {
            set_margin(&mut zca, margin, 0.0);
            set_margin(&mut grad, margin, 0.0);
            set_margin(&mut theta, margin, 0.0);
        }
        if frame {
            set_frame(&mut zca, margin, 1.0, 0.0);
        }
        Ok((zca, grad, theta))
    }

    /// `Gaussian.zcs(image, threshold)`.
    pub fn zcs(&self, image: &ArrayViewD<'_, f32>, threshold: Option<f32>) -> Result<ArrayD<f32>> {
        self.zc.zcs(image, threshold.unwrap_or(self.threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_match_python_generator() {
        // GaussianWeights.compute_gaussian_weight(1.0, limit=0.001): symmetric, sums to 1
        let w = gaussian_weight(1.0, 1.0, 0.001, true);
        assert_eq!(w.len(), 9);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(w[0], w[8]);
        let ladder = default_weights();
        let widths: Vec<usize> = ladder.iter().map(|w| w.len()).collect();
        assert_eq!(widths[0], 9);
        assert!(widths.windows(2).all(|p| p[0] < p[1]));
        assert!(*widths.last().unwrap() <= 150);
    }
}
