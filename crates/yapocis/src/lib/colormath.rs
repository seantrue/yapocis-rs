//! `yapocis.lib.colormath`: RGB/XYZ/xyY transforms with named matrices.

use crate::host;
use crate::interfaces::colormath;
use crate::{CallOptions, Device, Error, Result};
use ndarray::{ArrayD, ArrayViewD};

/// Transform names in kernel order (`transform_names` in the Python module).
pub const TRANSFORM_NAMES: &[&str] = &[
    "adobe1998_d65.rgb2xyz",
    "adobe1998_d65.xyz2rgb",
    "apple_d65.rgb2xyz",
    "apple_d65.xyz2rgb",
    "best_d50.rgb2xyz",
    "best_d50.xyz2rgb",
    "beta_d50.rgb2xyz",
    "beta_d50.xyz2rgb",
    "bruce_d65.rgb2xyz",
    "bruce_d65.xyz2rgb",
    "cie_e.rgb2xyz",
    "cie_e.xyz2rgb",
    "colormatch_d50.rgb2xyz",
    "colormatch_d50.xyz2rgb",
    "donrgb3_d50.rgb2xyz",
    "donrgb3_d50.xyz2rgb",
    "eci_d50.rgb2xyz",
    "eci_d50.xyz2rgb",
    "ektaspaceps5_d50.rgb2xyz",
    "ektaspaceps5_d50.xyz2rgb",
    "ntsc_c.rgb2xyz",
    "ntsc_c.xyz2rgb",
    "palsecam_d65.rgb2xyz",
    "palsecam_d65.xyz2rgb",
    "prophoto_d50.rgb2xyz",
    "prophoto_d50.xyz2rgb",
    "smpte-c_d65.rgb2xyz",
    "smpte-c_d65.xyz2rgb",
    "srgb_d65.rgb2xyz",
    "srgb_d65.xyz2rgb",
    "wide_d50.rgb2xyz",
    "wide_d50.xyz2rgb",
    "adobe_d50.rgb2xyz",
    "adobe_d50.xyz2rgb",
    "applergb_d50.rgb2xyz",
    "applergb_d50.xyz2rgb",
    "bruce_d50.rgb2xyz",
    "bruce_d50.xyz2rgb",
    "cie_d50.rgb2xyz",
    "cie_d50.xyz2rgb",
    "ntsc_d50.rgb2xyz",
    "ntsc_d50.xyz2rgb",
    "palsecam_d50.rgb2xyz",
    "palsecam_d50.xyz2rgb",
    "smptec_d50.rgb2xyz",
    "smptec_d50.xyz2rgb",
    "srgb_d50.rgb2xyz",
    "srgb_d50.xyz2rgb",
    "unity",
];

pub fn transform_id(name: &str) -> Option<i32> {
    TRANSFORM_NAMES
        .iter()
        .position(|n| *n == name)
        .map(|i| i as i32)
}

pub struct Colormath {
    pub kernels: colormath::Colormath,
}

impl Colormath {
    pub fn load(device: &Device) -> Result<Colormath> {
        Ok(Colormath {
            kernels: colormath::Colormath::load(device)?,
        })
    }

    fn per_pixel(a: &ArrayViewD<'_, f32>) -> CallOptions {
        CallOptions {
            global: Some(vec![a.len() / 3]),
            ..Default::default()
        }
    }

    /// `transform_named(name, array)` on a W×H×3 array.
    pub fn transform_named(&self, name: &str, array: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        let id = transform_id(name)
            .ok_or_else(|| Error::Invalid(format!("unknown transform {name:?}")))?;
        self.transform(array, id)
    }

    pub fn transform(&self, array: &ArrayViewD<'_, f32>, id: i32) -> Result<ArrayD<f32>> {
        host::one_output(
            self.kernels
                .transform_with(&Self::per_pixel(array), array, id)?,
        )
    }

    pub fn xyz2xyy(&self, input: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        host::one_output(self.kernels.xyz2xyY_with(&Self::per_pixel(input), input)?)
    }

    pub fn xyy2xyz(&self, input: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        host::one_output(self.kernels.xyY2xyz_with(&Self::per_pixel(input), input)?)
    }
}
