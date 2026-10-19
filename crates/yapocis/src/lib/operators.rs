//! `yapocis.lib.operators`: elementwise c = a op b, from host arrays or
//! resident buffers.

use crate::interfaces::operators;
use crate::{Buffer, Device, Result};
use ndarray::{ArrayD, ArrayViewD};

pub struct Operators {
    pub kernels: operators::Operators,
}

impl Operators {
    pub fn load(device: &Device) -> Result<Operators> {
        Ok(Operators {
            kernels: operators::Operators::load(device)?,
        })
    }

    pub fn add(&self, a: &ArrayViewD<'_, f32>, b: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        self.kernels.add(a, b)
    }

    pub fn sub(&self, a: &ArrayViewD<'_, f32>, b: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        self.kernels.sub(a, b)
    }

    pub fn mul(&self, a: &ArrayViewD<'_, f32>, b: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        self.kernels.mul(a, b)
    }

    pub fn div(&self, a: &ArrayViewD<'_, f32>, b: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        self.kernels.div(a, b)
    }

    /// `output = a - b` on resident buffers (`sub_res`).
    pub fn sub_res(&self, a: &Buffer<f32>, b: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.kernels.sub_res(a, b, output)
    }

    pub fn add_res(&self, a: &Buffer<f32>, b: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.kernels.add_res(a, b, output)
    }

    pub fn mul_res(&self, a: &Buffer<f32>, b: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.kernels.mul_res(a, b, output)
    }

    pub fn div_res(&self, a: &Buffer<f32>, b: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.kernels.div_res(a, b, output)
    }
}
