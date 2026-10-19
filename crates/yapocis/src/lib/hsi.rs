//! `yapocis.lib.hsi`: RGB <-> hue/saturation/intensity.

use crate::host;
use crate::interfaces::hsi;
use crate::{CallOptions, Device, Result};
use ndarray::{ArrayD, ArrayViewD};

pub struct Hsi {
    pub kernels: hsi::Hsi,
}

impl Hsi {
    pub fn load(device: &Device) -> Result<Hsi> {
        Ok(Hsi {
            kernels: hsi::Hsi::load(device)?,
        })
    }

    fn per_pixel(rgb: &ArrayViewD<'_, f32>) -> CallOptions {
        // HSI.rgb2hsi3: global_size = rgb.size // rgb.shape[-1]
        let depth = *rgb.shape().last().unwrap_or(&1);
        CallOptions {
            global: Some(vec![rgb.len() / depth.max(1)]),
            ..Default::default()
        }
    }

    /// W×H×3 RGB to W×H×3 HSI.
    pub fn rgb2hsi3(&self, rgb: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        host::one_output(self.kernels.rgb2hsi3_with(&Self::per_pixel(rgb), rgb)?)
    }

    /// W×H×3 HSI to W×H×3 RGB.
    pub fn hsi2rgb3(&self, hsi: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        host::one_output(self.kernels.hsi2rgb3_with(&Self::per_pixel(hsi), hsi)?)
    }

    /// Separate planes: `(h, s, i)` from `r, g, b`.
    pub fn rgb2hsi(
        &self,
        r: &ArrayViewD<'_, f32>,
        g: &ArrayViewD<'_, f32>,
        b: &ArrayViewD<'_, f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        self.kernels.rgb2hsi(r, g, b)
    }

    pub fn hsi2rgb(
        &self,
        h: &ArrayViewD<'_, f32>,
        s: &ArrayViewD<'_, f32>,
        i: &ArrayViewD<'_, f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        self.kernels.hsi2rgb(h, s, i)
    }
}
