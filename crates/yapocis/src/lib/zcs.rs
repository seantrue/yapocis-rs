//! `yapocis.lib.zc`: zero-crossing detector, kernel chosen by channel count.

use super::image::image_depth;
use crate::interfaces::zcs;
use crate::rt::{CallArg, CallOptions};
use crate::{host, Buffer, Device, Result};
use ndarray::{ArrayD, ArrayViewD};

pub const DEFAULT_THRESHOLD: f32 = 0.00001;

pub struct Zcs {
    pub kernels: zcs::Zcs,
}

impl Zcs {
    pub fn load(device: &Device) -> Result<Zcs> {
        Ok(Zcs {
            kernels: zcs::Zcs::load(device)?,
        })
    }

    fn select(shape: &[usize]) -> Result<(&'static str, usize)> {
        let depth = image_depth(shape)?;
        let name = ["zcs1", "zcs2", "zcs3", "zcs4"][depth - 1];
        let size: usize = shape.iter().product();
        Ok((name, size / depth))
    }

    /// `ZC.zcs(image, threshold)`.
    pub fn zcs(&self, image: &ArrayViewD<'_, f32>, threshold: f32) -> Result<ArrayD<f32>> {
        let (name, global) = Self::select(image.shape())?;
        let opts = CallOptions {
            global: Some(vec![global]),
            ..Default::default()
        };
        let std = host::standard(image);
        let mut args = [
            CallArg::Scalar(threshold as f64),
            CallArg::In(crate::rt::HostView::of(
                std.as_slice().unwrap(),
                image.shape(),
            )),
        ];
        host::one_output(self.kernels.program().call(name, &mut args, &opts)?)
    }

    /// `ZC.zcs_res(image, zcs, threshold)`: both arrays resident.
    pub fn zcs_res(&self, image: &Buffer<f32>, zcs: &Buffer<f32>, threshold: f32) -> Result<()> {
        let (name, global) = Self::select(image.shape())?;
        let name = format!("{name}_res");
        let opts = CallOptions {
            global: Some(vec![global]),
            ..Default::default()
        };
        let mut args = [
            CallArg::Scalar(threshold as f64),
            CallArg::Buffer(image.as_any()),
            CallArg::Buffer(zcs.as_any()),
        ];
        self.kernels.program().call(&name, &mut args, &opts)?;
        Ok(())
    }
}
