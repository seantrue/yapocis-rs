//! `yapocis.lib.imageops`: clamp, stretch, binarize, histograms, and the
//! host-side histogram equalisation and matching.

use crate::interfaces::imageops;
use crate::{Buffer, Device, Error, Result};
use ndarray::{Array1, ArrayD, ArrayViewD, IxDyn};

pub struct Imageops {
    pub kernels: imageops::Imageops,
}

impl Imageops {
    pub fn load(device: &Device) -> Result<Imageops> {
        Ok(Imageops {
            kernels: imageops::Imageops::load(device)?,
        })
    }

    /// `Imageops.clamp_pixels(input, low=0.0, high=1.0)`.
    pub fn clamp_pixels(
        &self,
        input: &ArrayViewD<'_, f32>,
        low: f32,
        high: f32,
    ) -> Result<ArrayD<f32>> {
        self.kernels.clamp_pixels(input, low, high)
    }

    /// `Imageops.stretch(input, input_min, input_max, low=0.0, high=1.0)`.
    pub fn stretch(
        &self,
        input: &ArrayViewD<'_, f32>,
        input_min: f32,
        input_max: f32,
        low: f32,
        high: f32,
    ) -> Result<ArrayD<f32>> {
        self.kernels.stretch(input, input_min, input_max, low, high)
    }

    /// `Imageops.binarize(input, threshold=0.5)`.
    pub fn binarize(&self, input: &ArrayViewD<'_, f32>, threshold: f32) -> Result<ArrayD<f32>> {
        self.kernels.binarize(input, threshold)
    }

    /// `Imageops.histogram(input, n_bins=256, low=0.0, high=1.0)`: counts per bin.
    pub fn histogram(
        &self,
        input: &ArrayViewD<'_, f32>,
        n_bins: usize,
        low: f32,
        high: f32,
    ) -> Result<Array1<i32>> {
        let mut hst = Array1::<i32>::zeros(n_bins).into_dyn();
        self.kernels
            .histogram(input, low, high, &mut hst.view_mut())?;
        Ok(hst.into_dimensionality()?)
    }

    /// `Imageops.histogram_res(input, hst, low, high)`: both on the device.
    pub fn histogram_res(
        &self,
        input: &Buffer<f32>,
        hst: &Buffer<i32>,
        low: f32,
        high: f32,
    ) -> Result<()> {
        self.kernels.histogram_res(input, low, high, hst)
    }

    /// `Imageops.histeq(im, nbr_bins=2**16)`.
    ///
    /// Reproduces the numpy arithmetic exactly, including the int32 wrap-around
    /// of `(nbr_bins - 1) * cdf` for images with more than 32768 pixels.
    pub fn histeq(&self, im: &ArrayViewD<'_, f32>, nbr_bins: usize) -> Result<ArrayD<f32>> {
        if nbr_bins < 2 {
            return Err(Error::Invalid("histeq needs at least 2 bins".into()));
        }
        let mut imhist = self.histogram(im, nbr_bins, 0.0, 1.0)?;
        imhist[0] = 0;
        let cdf_i32 = cumsum_i32(imhist.as_slice().unwrap());
        let last = *cdf_i32.last().unwrap() as f64;
        let scale = (nbr_bins - 1) as i32;
        // numpy: int * int32 array stays int32 (wrapping), then true division
        let cdf: Vec<f64> = cdf_i32
            .iter()
            .map(|&c| (scale.wrapping_mul(c) as f64 / last) / nbr_bins as f64)
            .collect();
        let bins = linspace(0.0, 1.0, nbr_bins);
        let flat = im.as_standard_layout();
        let out: Vec<f32> = flat
            .iter()
            .map(|&v| interp(v as f64, &bins, &cdf) as f32)
            .collect();
        Ok(ArrayD::from_shape_vec(IxDyn(im.shape()), out)?)
    }

    /// `Imageops.histmatch(im, target_hist, nbr_bins=None)`.
    pub fn histmatch(
        &self,
        im: &ArrayViewD<'_, f32>,
        target_hist: &[f64],
        nbr_bins: Option<usize>,
    ) -> Result<ArrayD<f32>> {
        let nbr_bins = nbr_bins.unwrap_or(target_hist.len());
        if nbr_bins < 2 || target_hist.is_empty() {
            return Err(Error::Invalid("histmatch needs at least 2 bins".into()));
        }
        let mut src_hist = self.histogram(im, nbr_bins, 0.0, 1.0)?;
        src_hist[0] = 0;
        let src_cdf_i = cumsum_i32(src_hist.as_slice().unwrap());
        let src_last = *src_cdf_i.last().unwrap() as f64;
        let src_cdf: Vec<f64> = src_cdf_i.iter().map(|&c| c as f64 / src_last).collect();

        let mut target = target_hist.to_vec();
        target[0] = 0.0;
        let mut target_cdf = Vec::with_capacity(target.len());
        let mut acc = 0.0f64;
        for t in &target {
            acc += t;
            target_cdf.push(acc);
        }
        let t_last = *target_cdf.last().unwrap();
        for t in &mut target_cdf {
            *t /= t_last;
        }

        let bins = linspace(0.0, 1.0, nbr_bins);
        // np.searchsorted(target_cdf, src_cdf) (side='left'), clipped to the bins
        let mapped: Vec<f64> = src_cdf
            .iter()
            .map(|&s| {
                let idx = target_cdf.partition_point(|&t| t < s);
                bins[idx.min(nbr_bins - 1)]
            })
            .collect();
        let flat = im.as_standard_layout();
        let out: Vec<f32> = flat
            .iter()
            .map(|&v| interp(v as f64, &bins, &mapped) as f32)
            .collect();
        Ok(ArrayD::from_shape_vec(IxDyn(im.shape()), out)?)
    }
}

/// `np.cumsum` of an int32 array: int32 with wrap-around.
pub fn cumsum_i32(values: &[i32]) -> Vec<i32> {
    let mut acc = 0i32;
    values
        .iter()
        .map(|&v| {
            acc = acc.wrapping_add(v);
            acc
        })
        .collect()
}

/// `np.linspace(start, stop, n)`.
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i == n - 1 {
                stop
            } else {
                start + step * i as f64
            }
        })
        .collect()
}

/// `np.interp(x, xp, fp)` for increasing `xp`: linear in between, clamped outside.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // first index with xp[j] > x
    let j = xp.partition_point(|&v| v <= x);
    let (x0, x1) = (xp[j - 1], xp[j]);
    let (y0, y1) = (fp[j - 1], fp[j]);
    let slope = (y1 - y0) / (x1 - x0);
    slope * (x - x0) + y0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_matches_numpy() {
        let xp = linspace(0.0, 1.0, 5);
        let fp = vec![0.0, 1.0, 1.0, 2.0, 4.0];
        assert_eq!(interp(0.125, &xp, &fp), 0.5);
        assert_eq!(interp(-1.0, &xp, &fp), 0.0);
        assert_eq!(interp(2.0, &xp, &fp), 4.0);
        assert_eq!(interp(0.75, &xp, &fp), 2.0);
        assert_eq!(linspace(0.0, 1.0, 3), vec![0.0, 0.5, 1.0]);
        assert_eq!(cumsum_i32(&[i32::MAX, 1]), vec![i32::MAX, i32::MIN]);
    }
}
