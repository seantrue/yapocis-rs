//! `yapocis.lib.anisodiff`: Perona–Malik anisotropic diffusion.

use super::hsi::Hsi;
use super::image::{as_2d, image_depth, join_channels, pad_planes, split_channels, unpad_planes};
use crate::interfaces::anisodiff;
use crate::rt::{CallArg, CallOptions};
use crate::{Device, Error, Result};
use ndarray::{Array2, Array3, ArrayD, ArrayView2, ArrayViewD, Axis, IxDyn};

pub struct Anisodiff {
    pub kernels: anisodiff::Anisodiff,
    /// `Anisodiff.__init__` also builds an `HSI` for the RGB variants.
    pub hsi: Hsi,
}

const FILTER_BY_DEPTH: [&str; 4] = [
    "filterImage",
    "filterImage2",
    "filterImage3",
    "filterImage4",
];

impl Anisodiff {
    pub fn load(device: &Device) -> Result<Anisodiff> {
        Ok(Anisodiff {
            kernels: anisodiff::Anisodiff::load(device)?,
            hsi: Hsi::load(device)?,
        })
    }

    /// `Anisodiff.anisodiff_n(image, n, scale=3.0, step_size=0.2)`.
    ///
    /// Images with a maximum at or below 1.0 are scaled by 256 for the kernel
    /// and the result is scaled back; 3-plane images run through the 4-plane
    /// kernel with a zero padding plane.
    pub fn anisodiff_n(
        &self,
        image: &ArrayViewD<'_, f32>,
        n: usize,
        scale: f32,
        step_size: f32,
    ) -> Result<ArrayD<f32>> {
        if step_size > 0.25 {
            return Err(Error::Invalid(
                "step_size must be <= 0.25 for numerical stability".into(),
            ));
        }
        let max = image.iter().cloned().fold(f32::MIN, f32::max);
        let scaled: ArrayD<f32> = if max > 1.0 {
            image.to_owned()
        } else {
            image.mapv(|v| v * 256.0)
        };
        let (padded, padding_required) = pad_planes(&scaled.view());
        let depth = image_depth(padded.shape())?;
        let kernel = format!("{}_res", FILTER_BY_DEPTH[depth - 1]);
        let global = padded.len() / depth;

        let device = self.kernels.device().clone();
        let mut input = device.upload(padded.as_slice().unwrap(), padded.shape())?;
        let mut output = device.alloc::<f32>(padded.shape())?;
        let opts = CallOptions {
            global: Some(vec![global]),
            async_exec: true,
            ..Default::default()
        };
        for _ in 0..n {
            let mut args = [
                CallArg::Buffer(input.as_any()),
                CallArg::Scalar(scale as f64),
                CallArg::Scalar(step_size as f64),
                CallArg::Buffer(output.as_any()),
            ];
            self.kernels.program().call(&kernel, &mut args, &opts)?;
            std::mem::swap(&mut input, &mut output);
        }
        device.wait_all()?;
        let mut result = ArrayD::from_shape_vec(IxDyn(padded.shape()), input.to_vec()?)?;
        if padding_required {
            result = unpad_planes(result);
        }
        Ok(result.mapv(|v| v / 256.0))
    }

    /// `Anisodiff.anisodiff(image, scale=3.0, step_size=0.2)`: an iterator that
    /// yields the image after each diffusion step (Python's generator).
    pub fn anisodiff<'a>(
        &'a self,
        image: &ArrayViewD<'_, f32>,
        scale: f32,
        step_size: f32,
    ) -> Result<AnisodiffIter<'a>> {
        if step_size > 0.25 {
            return Err(Error::Invalid(
                "step_size must be <= 0.25 for numerical stability".into(),
            ));
        }
        let max = image.iter().cloned().fold(f32::MIN, f32::max);
        // Python: `if image.max() <= 1.0: image *= 256.0`
        let scaled: ArrayD<f32> = if max <= 1.0 {
            image.mapv(|v| v * 256.0)
        } else {
            image.to_owned()
        };
        let (padded, padded_flag) = pad_planes(&scaled.view());
        let depth = image_depth(padded.shape())?;
        let kernel = format!("{}_res", FILTER_BY_DEPTH[depth - 1]);
        let global = padded.len() / depth;
        let device = self.kernels.device().clone();
        let input = device.upload(padded.as_slice().unwrap(), padded.shape())?;
        let output = device.alloc::<f32>(padded.shape())?;
        Ok(AnisodiffIter {
            lib: self,
            device,
            kernel,
            global,
            shape: padded.shape().to_vec(),
            padded: padded_flag,
            input,
            output,
            scale,
            step_size,
        })
    }

    /// `Anisodiff.anisodiff_rgbI_n(rgb, n, ...)`: diffuse the intensity plane only.
    pub fn anisodiff_rgb_i_n(
        &self,
        rgb: &ArrayViewD<'_, f32>,
        n: usize,
        scale: f32,
        step_size: f32,
    ) -> Result<ArrayD<f32>> {
        let [r, g, b] = split_channels(rgb)?;
        let (h, s, i) = self.hsi.rgb2hsi(
            &r.view().into_dyn(),
            &g.view().into_dyn(),
            &b.view().into_dyn(),
        )?;
        let fi = self.anisodiff_n(&i.view(), n, scale, step_size)?;
        let (r, g, b) = self.hsi.hsi2rgb(&h.view(), &s.view(), &fi.view())?;
        Ok(join_channels(
            as_2d(r)?.view(),
            as_2d(g)?.view(),
            as_2d(b)?.view(),
        ))
    }

    /// `Anisodiff.anisodiff_rgbI(rgb, ...)`: frames with the intensity plane diffused step by step.
    pub fn anisodiff_rgb_i<'a>(
        &'a self,
        rgb: &ArrayViewD<'_, f32>,
        scale: f32,
        step_size: f32,
    ) -> Result<RgbIter<'a>> {
        let [r, g, b] = split_channels(rgb)?;
        let (h, s, i) = self.hsi.rgb2hsi(
            &r.view().into_dyn(),
            &g.view().into_dyn(),
            &b.view().into_dyn(),
        )?;
        let inner = self.anisodiff(&i.view(), scale, step_size)?;
        Ok(RgbIter {
            lib: self,
            inner,
            h: as_2d(h)?,
            s: Some(as_2d(s)?),
        })
    }

    /// `Anisodiff.anisodiff_rgbIS_n(rgb, n, ...)`: diffuse saturation and intensity together.
    pub fn anisodiff_rgb_is_n(
        &self,
        rgb: &ArrayViewD<'_, f32>,
        n: usize,
        scale: f32,
        step_size: f32,
    ) -> Result<ArrayD<f32>> {
        let [r, g, b] = split_channels(rgb)?;
        let (h, s, i) = self.hsi.rgb2hsi(
            &r.view().into_dyn(),
            &g.view().into_dyn(),
            &b.view().into_dyn(),
        )?;
        let si = join_planes(&[as_2d(s)?.view(), as_2d(i)?.view()]);
        let fsi = self.anisodiff_n(&si.view(), n, scale, step_size)?;
        let [fs, fi] = split_two_planes(&fsi.view())?;
        let (r, g, b) =
            self.hsi
                .hsi2rgb(&h.view(), &fs.view().into_dyn(), &fi.view().into_dyn())?;
        Ok(join_channels(
            as_2d(r)?.view(),
            as_2d(g)?.view(),
            as_2d(b)?.view(),
        ))
    }

    /// `Anisodiff.anisodiff_rgbIS(rgb, ...)`: frames with saturation and intensity diffused.
    pub fn anisodiff_rgb_is<'a>(
        &'a self,
        rgb: &ArrayViewD<'_, f32>,
        scale: f32,
        step_size: f32,
    ) -> Result<RgbIter<'a>> {
        let [r, g, b] = split_channels(rgb)?;
        let (h, s, i) = self.hsi.rgb2hsi(
            &r.view().into_dyn(),
            &g.view().into_dyn(),
            &b.view().into_dyn(),
        )?;
        let si = join_planes(&[as_2d(s)?.view(), as_2d(i)?.view()]);
        let inner = self.anisodiff(&si.view(), scale, step_size)?;
        Ok(RgbIter {
            lib: self,
            inner,
            h: as_2d(h)?,
            s: None,
        })
    }
}

/// `join_planes(*planes)`: stack W×H planes into W×H×k.
pub fn join_planes(planes: &[ArrayView2<'_, f32>]) -> ArrayD<f32> {
    let (w, h) = planes[0].dim();
    let mut out = Array3::<f32>::zeros((w, h, planes.len()));
    for (k, p) in planes.iter().enumerate() {
        out.index_axis_mut(Axis(2), k).assign(p);
    }
    out.into_dyn()
}

fn split_two_planes(image: &ArrayViewD<'_, f32>) -> Result<[Array2<f32>; 2]> {
    if image.ndim() != 3 || image.shape()[2] != 2 {
        return Err(Error::Invalid(format!(
            "expected W×H×2, got {:?}",
            image.shape()
        )));
    }
    let plane = |k: usize| -> Array2<f32> {
        image
            .index_axis(Axis(2), k)
            .to_owned()
            .into_dimensionality()
            .expect("2-D")
    };
    Ok([plane(0), plane(1)])
}

/// Step-by-step Perona–Malik diffusion on resident device buffers.
pub struct AnisodiffIter<'a> {
    lib: &'a Anisodiff,
    device: Device,
    kernel: String,
    global: usize,
    shape: Vec<usize>,
    padded: bool,
    input: crate::Buffer<f32>,
    output: crate::Buffer<f32>,
    scale: f32,
    step_size: f32,
}

impl AnisodiffIter<'_> {
    /// One diffusion step; returns the image after it (scaled back by 1/256).
    pub fn step(&mut self) -> Result<ArrayD<f32>> {
        let opts = CallOptions {
            global: Some(vec![self.global]),
            ..Default::default()
        };
        let mut args = [
            CallArg::Buffer(self.input.as_any()),
            CallArg::Scalar(self.scale as f64),
            CallArg::Scalar(self.step_size as f64),
            CallArg::Buffer(self.output.as_any()),
        ];
        self.lib
            .kernels
            .program()
            .call(&self.kernel, &mut args, &opts)?;
        self.device.wait_all()?;
        let mut frame = ArrayD::from_shape_vec(IxDyn(&self.shape), self.output.to_vec()?)?;
        if self.padded {
            frame = unpad_planes(frame);
        }
        std::mem::swap(&mut self.input, &mut self.output);
        Ok(frame.mapv(|v| v / 256.0))
    }
}

impl Iterator for AnisodiffIter<'_> {
    type Item = Result<ArrayD<f32>>;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step())
    }
}

/// Frames of `anisodiff_rgb_i` / `anisodiff_rgb_is`.
pub struct RgbIter<'a> {
    lib: &'a Anisodiff,
    inner: AnisodiffIter<'a>,
    h: Array2<f32>,
    /// Fixed saturation for the intensity-only variant; `None` when S is diffused too.
    s: Option<Array2<f32>>,
}

impl RgbIter<'_> {
    pub fn step(&mut self) -> Result<ArrayD<f32>> {
        let frame = self.inner.step()?;
        let (fs, fi) = match &self.s {
            Some(s) => (s.clone(), as_2d(frame)?),
            None => {
                let [fs, fi] = split_two_planes(&frame.view())?;
                (fs, fi)
            }
        };
        let (r, g, b) = self.lib.hsi.hsi2rgb(
            &self.h.view().into_dyn(),
            &fs.view().into_dyn(),
            &fi.view().into_dyn(),
        )?;
        Ok(join_channels(
            as_2d(r)?.view(),
            as_2d(g)?.view(),
            as_2d(b)?.view(),
        ))
    }
}

impl Iterator for RgbIter<'_> {
    type Item = Result<ArrayD<f32>>;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.step())
    }
}
