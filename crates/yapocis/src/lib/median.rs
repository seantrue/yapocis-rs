//! `yapocis.lib.median`: iterated 3×3 median filter with device ping-pong.

use crate::interfaces::median3x3;
use crate::rt::{CallArg, CallOptions};
use crate::{Device, Result};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

pub struct Median {
    pub kernels: median3x3::Median3x3,
}

impl Median {
    pub fn load(device: &Device) -> Result<Median> {
        Ok(Median {
            kernels: median3x3::Median3x3::load(device)?,
        })
    }

    /// `Median.median3x3(image, iterations)`: ping-pong between two device
    /// buffers, reading back once (`median3x3fast`).
    pub fn median3x3(&self, image: &ArrayViewD<'_, f32>, iterations: usize) -> Result<ArrayD<f32>> {
        if iterations == 0 {
            return Ok(image.to_owned());
        }
        let device = self.kernels.device().clone();
        let std = image.as_standard_layout();
        let mut input = device.upload(std.as_slice().unwrap(), image.shape())?;
        let mut output = device.alloc::<f32>(image.shape())?;
        let opts = CallOptions {
            async_exec: true,
            ..Default::default()
        };
        for _ in 0..iterations {
            let mut args = [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(output.as_any()),
            ];
            self.kernels
                .program()
                .call("median3x3_res", &mut args, &opts)?;
            std::mem::swap(&mut input, &mut output);
        }
        device.wait_all()?;
        // after the final swap `input` holds the last result
        Ok(ArrayD::from_shape_vec(
            IxDyn(image.shape()),
            input.to_vec()?,
        )?)
    }

    /// One pass through the plain kernel (`median3x3slow` with one iteration).
    pub fn median3x3_once(&self, image: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        self.kernels.median3x3(image)
    }
}
