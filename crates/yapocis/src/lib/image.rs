//! Small array helpers shared by the library ports (`yapocis.utils.basic`,
//! `yapocis.lib.channels`).

use crate::{Error, Result};
use ndarray::{s, Array2, Array3, ArrayD, ArrayView2, ArrayViewD, Axis};

/// Channel count of a 2-D (1) or 3-D (`shape[2]`, 1..=4) image (`get_image_depth`).
pub fn image_depth(shape: &[usize]) -> Result<usize> {
    match shape {
        [_, _] => Ok(1),
        [_, _, d] if (1..=4).contains(d) => Ok(*d),
        other => Err(Error::Invalid(format!(
            "array shape {other:?} is not an image"
        ))),
    }
}

/// `set_margin`: set a border of `margin` pixels to `value`. Only the border
/// is touched, as Python's four slice assignments do.
pub fn set_margin(a: &mut Array2<f32>, margin: usize, value: f32) {
    let (w, h) = a.dim();
    let (mw, mh) = (margin.min(w), margin.min(h));
    a.slice_mut(s![..mw, ..]).fill(value);
    a.slice_mut(s![w - mw.., ..]).fill(value);
    a.slice_mut(s![.., ..mh]).fill(value);
    a.slice_mut(s![.., h - mh..]).fill(value);
}

/// `set_frame`: clear the margin, then set the outermost pixels to `frame`.
pub fn set_frame(a: &mut Array2<f32>, width: usize, frame: f32, margin: f32) {
    set_margin(a, width, margin);
    let (w, h) = a.dim();
    if w == 0 || h == 0 {
        return;
    }
    for x in 0..w {
        a[[x, 0]] = frame;
        a[[x, h - 1]] = frame;
    }
    for y in 0..h {
        a[[0, y]] = frame;
        a[[w - 1, y]] = frame;
    }
}

/// `pad_planes`: a 3-plane image becomes 4 planes (float3 is 16 bytes on
/// OpenCL and Metal but 12 on CUDA, so 3-plane buffers are never used).
pub fn pad_planes(image: &ArrayViewD<'_, f32>) -> (ArrayD<f32>, bool) {
    if image.ndim() == 3 && image.shape()[2] == 3 {
        let (w, h) = (image.shape()[0], image.shape()[1]);
        let mut out = Array3::<f32>::zeros((w, h, 4));
        out.slice_mut(ndarray::s![.., .., ..3]).assign(image);
        (out.into_dyn(), true)
    } else {
        (image.to_owned(), false)
    }
}

/// Drop the padding plane added by [`pad_planes`].
pub fn unpad_planes(image: ArrayD<f32>) -> ArrayD<f32> {
    image.slice(ndarray::s![.., .., ..3]).to_owned().into_dyn()
}

/// `split_channels` of a W×H×3 image.
pub fn split_channels(rgb: &ArrayViewD<'_, f32>) -> Result<[Array2<f32>; 3]> {
    if rgb.ndim() != 3 || rgb.shape()[2] != 3 {
        return Err(Error::Invalid(format!(
            "expected a W×H×3 image, got {:?}",
            rgb.shape()
        )));
    }
    let plane = |c: usize| -> Array2<f32> {
        rgb.index_axis(Axis(2), c)
            .to_owned()
            .into_dimensionality()
            .expect("2-D")
    };
    Ok([plane(0), plane(1), plane(2)])
}

/// `join_channels`: three W×H planes into a W×H×3 image.
pub fn join_channels(
    r: ArrayView2<'_, f32>,
    g: ArrayView2<'_, f32>,
    b: ArrayView2<'_, f32>,
) -> ArrayD<f32> {
    let (w, h) = r.dim();
    let mut out = Array3::<f32>::zeros((w, h, 3));
    out.index_axis_mut(Axis(2), 0).assign(&r);
    out.index_axis_mut(Axis(2), 1).assign(&g);
    out.index_axis_mut(Axis(2), 2).assign(&b);
    out.into_dyn()
}

pub(crate) fn as_2d(a: ArrayD<f32>) -> Result<Array2<f32>> {
    a.into_dimensionality::<ndarray::Ix2>().map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_margin_touches_exactly_the_border() {
        for (w, h) in [(7, 5), (1, 4), (6, 6)] {
            for margin in 0..9 {
                let mut a = Array2::<f32>::ones((w, h));
                set_margin(&mut a, margin, 0.0);
                for ((x, y), v) in a.indexed_iter() {
                    let border = x < margin || x + margin >= w || y < margin || y + margin >= h;
                    assert_eq!(
                        *v,
                        if border { 0.0 } else { 1.0 },
                        "{w}x{h} m{margin} at {x},{y}"
                    );
                }
            }
        }
    }
}
