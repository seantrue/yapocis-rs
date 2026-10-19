//! `yapocis.lib.gradient`: 2-D gradient magnitude and direction.

use crate::interfaces::gradient;
use crate::{Buffer, Device, Result};
use ndarray::{ArrayD, ArrayViewD};

pub struct Gradient {
    pub kernels: gradient::Gradient,
}

impl Gradient {
    pub fn load(device: &Device) -> Result<Gradient> {
        Ok(Gradient {
            kernels: gradient::Gradient::load(device)?,
        })
    }

    /// `Gradient.gradient(a, stride=3)`: returns `(grad, theta)`.
    pub fn gradient(
        &self,
        a: &ArrayViewD<'_, f32>,
        stride: i32,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        self.kernels.gradient(a, stride)
    }

    /// Resident form: `grad` and `theta` are written on the device.
    pub fn gradient_res(
        &self,
        input: &Buffer<f32>,
        stride: i32,
        grad: &Buffer<f32>,
        theta: &Buffer<f32>,
    ) -> Result<()> {
        self.kernels.gradient_res(input, stride, grad, theta)
    }
}
