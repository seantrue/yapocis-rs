//! Pillow's `Image.resize` (`libImaging/Resample.c`), exactly, for 8-bit RGB and 32-bit float
//! (`F`) images with the BILINEAR and LANCZOS filters: the filter's coefficients in float64,
//! normalised per output pixel, a horizontal pass over only the rows the vertical pass reads,
//! then the vertical pass. 8-bit images round the coefficients to 22-bit fixed point and each
//! sum by its half-unit start, clipped; float images sum in float64 (the vertical pass with fused
//! multiply-adds, as clang compiles Pillow on arm64) and store float32. Also
//! `convert("L")` of RGB.
//!
//! The `image` crate's `Triangle` filter is the same kernel but keeps float coefficients, so
//! some pixels differ from Pillow's by one 8-bit step; the CNN preprocessing uses this.

use image::RgbImage;
use ndarray::Array2;

/// A resampling filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Bilinear,
    Lanczos,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Lanczos => 3.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Filter::Bilinear => triangle(x),
            Filter::Lanczos => lanczos(x),
        }
    }
}

/// `sinc_filter`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

/// `lanczos_filter`: the sinc truncated to 3 lobes.
fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// `PRECISION_BITS = 32 - 8 - 2`.
const PRECISION_BITS: u32 = 22;

/// `bilinear_filter`.
fn triangle(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

/// `precompute_coeffs`: per output pixel, the first input pixel, the count, and the weights.
fn coeffs_f64(
    in_size: usize,
    out_size: usize,
    filter: Filter,
) -> (Vec<(usize, usize)>, Vec<Vec<f64>>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // (int) truncates toward zero
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64).max(0) as usize;
        let count = xmax.saturating_sub(xmin);
        let mut k: Vec<f64> = (0..count)
            .map(|x| filter.weight((x as f64 + xmin as f64 - center + 0.5) * ss))
            .collect();
        // ww accumulates in order, as the C loop does
        let mut ww = 0.0;
        for w in &k {
            ww += w;
        }
        if ww != 0.0 {
            for w in k.iter_mut() {
                *w /= ww;
            }
        }
        weights.push(k);
        bounds.push((xmin, count));
    }
    (bounds, weights)
}

/// `normalize_coeffs_8bpc`: the weights in 22-bit fixed point.
fn coeffs(in_size: usize, out_size: usize, filter: Filter) -> (Vec<(usize, usize)>, Vec<Vec<i32>>) {
    let (bounds, weights) = coeffs_f64(in_size, out_size, filter);
    let unit = (1u32 << PRECISION_BITS) as f64;
    let fixed = weights
        .iter()
        .map(|k| {
            k.iter()
                .map(|&w| {
                    if w < 0.0 {
                        (-0.5 + w * unit) as i32
                    } else {
                        (0.5 + w * unit) as i32
                    }
                })
                .collect()
        })
        .collect();
    (bounds, fixed)
}

/// `clip8`: the fixed-point sum's integer part, clamped to 0..=255.
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `Image.resize((width, height), Image.BILINEAR)` of an RGB image.
pub fn resize_bilinear(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    resize(img, width, height, Filter::Bilinear)
}

/// `Image.resize((width, height), filter)` of an RGB image.
pub fn resize(img: &RgbImage, width: u32, height: u32, filter: Filter) -> RgbImage {
    let (in_w, in_h) = (img.width() as usize, img.height() as usize);
    let (out_w, out_h) = (width as usize, height as usize);
    if (in_w, in_h) == (out_w, out_h) || out_w == 0 || out_h == 0 || in_w == 0 || in_h == 0 {
        // Pillow returns a copy when the size is unchanged
        return if (in_w, in_h) == (out_w, out_h) {
            img.clone()
        } else {
            RgbImage::new(width, height)
        };
    }
    let (hb, hk) = coeffs(in_w, out_w, filter);
    let (vb, vk) = coeffs(in_h, out_h, filter);
    let half = 1i32 << (PRECISION_BITS - 1);
    let px =
        |x: usize, y: usize, c: usize| -> i32 { img.get_pixel(x as u32, y as u32).0[c] as i32 };

    // rows the vertical pass reads: ybox_first ..= ybox_last
    let first = vb.first().map(|b| b.0).unwrap_or(0);
    let last = vb.last().map(|b| b.0 + b.1).unwrap_or(in_h);
    let need_h = out_w != in_w;
    let need_v = out_h != in_h;

    // horizontal pass: (rows, out_w) of RGB, or the input rows unchanged
    let rows = if need_h { last - first } else { in_h };
    let offset = if need_h { first } else { 0 };
    let mut temp = vec![[0u8; 3]; rows * if need_h { out_w } else { in_w }];
    let temp_w = if need_h { out_w } else { in_w };
    for y in 0..rows {
        for xx in 0..temp_w {
            temp[y * temp_w + xx] = if need_h {
                let (xmin, n) = hb[xx];
                let k = &hk[xx];
                let mut out = [0u8; 3];
                for (c, o) in out.iter_mut().enumerate() {
                    let mut ss = half;
                    for (x, w) in k.iter().enumerate().take(n) {
                        ss += px(x + xmin, y + offset, c) * w;
                    }
                    *o = clip8(ss);
                }
                out
            } else {
                [px(xx, y, 0) as u8, px(xx, y, 1) as u8, px(xx, y, 2) as u8]
            };
        }
    }
    if !need_v {
        let mut out = RgbImage::new(width, height);
        for (x, y, p) in out.enumerate_pixels_mut() {
            p.0 = temp[y as usize * temp_w + x as usize];
        }
        return out;
    }
    // vertical pass, the bounds shifted by the first row the horizontal pass kept
    let mut out = RgbImage::new(width, height);
    for yy in 0..out_h {
        let (ymin, n) = vb[yy];
        let ymin = ymin - offset;
        let k = &vk[yy];
        for xx in 0..out_w {
            let mut v = [0u8; 3];
            for (c, o) in v.iter_mut().enumerate() {
                let mut ss = half;
                for (y, w) in k.iter().enumerate().take(n) {
                    ss += temp[(y + ymin) * temp_w + xx][c] as i32 * w;
                }
                *o = clip8(ss);
            }
            out.put_pixel(xx as u32, yy as u32, image::Rgb(v));
        }
    }
    out
}

/// `Image.fromarray(a).resize((width, height), filter)` of a float32 (`F`) image, as rows ×
/// columns.
pub fn resize_f32(img: &Array2<f32>, width: usize, height: usize, filter: Filter) -> Array2<f32> {
    let (in_h, in_w) = img.dim();
    if (in_w, in_h) == (width, height) || width == 0 || height == 0 || in_w == 0 || in_h == 0 {
        return if (in_w, in_h) == (width, height) {
            img.clone()
        } else {
            Array2::zeros((height, width))
        };
    }
    let (hb, hk) = coeffs_f64(in_w, width, filter);
    let (vb, vk) = coeffs_f64(in_h, height, filter);
    let first = vb.first().map(|b| b.0).unwrap_or(0);
    let last = vb.last().map(|b| b.0 + b.1).unwrap_or(in_h);
    let need_h = width != in_w;
    let (offset, temp) = if need_h {
        let rows = last - first;
        let temp = Array2::from_shape_fn((rows, width), |(y, xx)| {
            let (xmin, n) = hb[xx];
            let mut ss = 0.0f64;
            for (x, w) in hk[xx].iter().enumerate().take(n) {
                ss += img[[y + first, x + xmin]] as f64 * w;
            }
            ss as f32
        });
        (first, temp)
    } else {
        (0, img.clone())
    };
    if height == in_h {
        return temp;
    }
    Array2::from_shape_fn((height, width), |(yy, xx)| {
        let (ymin, n) = vb[yy];
        let ymin = ymin - offset;
        let mut ss = 0.0f64;
        for (y, w) in vk[yy].iter().enumerate().take(n) {
            // clang contracts this loop's `ss += pixel * k[y]` into a fused multiply-add on
            // arm64, but not the horizontal one's (claude-debug/standalone/pil_probe in impcraft)
            ss = (temp[[y + ymin, xx]] as f64).mul_add(*w, ss);
        }
        ss as f32
    })
}

/// `convert("L")` of RGB: ITU-R 601-2 luma in 16-bit fixed point (`L24`).
pub fn to_luma(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 0x8000) >> 16) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checksums from Pillow 12.3.0
    /// (impcraft `claude-debug/standalone/pil_probe/pil_resize_ref.py`).
    #[test]
    fn matches_pillow() {
        type Case = ((u32, u32), (u32, u32), u64);
        let cases: [Case; 6] = [
            ((640, 480), (341, 256), 4372343677851),
            ((812, 808), (257, 256), 2483529939340),
            ((300, 200), (224, 224), 1444619307042),
            ((100, 150), (224, 224), 1444699326410),
            ((256, 300), (256, 300), 3385058102628),
            ((97, 41), (256, 108), 439639791757),
        ];
        for ((w, h), (nw, nh), want) in cases {
            let img = RgbImage::from_fn(w, h, |x, y| {
                let v = |c: u32| ((x * 31 + y * 17 + c * 7 + (x * y) % 13) % 256) as u8;
                image::Rgb([v(0), v(1), v(2)])
            });
            let out = resize_bilinear(&img, nw, nh);
            let m: u128 = (1u128 << 61) - 1;
            let sum = out
                .as_raw()
                .iter()
                .enumerate()
                .fold(0u128, |s, (i, &v)| (s + (i as u128 + 1) * v as u128) % m);
            assert_eq!(sum as u64, want, "{w}x{h} -> {nw}x{nh}");
        }
    }

    fn formula(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let v = |c: u32| ((x * 31 + y * 17 + c * 7 + (x * y) % 13) % 256) as u8;
            image::Rgb([v(0), v(1), v(2)])
        })
    }

    fn checksum(values: impl Iterator<Item = u64>) -> u64 {
        let m: u128 = (1u128 << 61) - 1;
        values
            .enumerate()
            .fold(0u128, |s, (i, v)| (s + (i as u128 + 1) * v as u128) % m) as u64
    }

    /// Checksums from Pillow 12.3.0 (impcraft
    /// `claude-debug/standalone/pil_probe/pil_resize_ref2.py`).
    #[test]
    fn lanczos_float_and_luma_match_pillow() {
        for ((w, h), (nw, nh), want) in [
            ((640, 480), (341, 256), 4372422375610u64),
            ((97, 41), (256, 108), 439702899475),
            ((300, 200), (224, 224), 1444423588211),
        ] {
            let out = resize(&formula(w, h), nw, nh, Filter::Lanczos);
            assert_eq!(
                checksum(out.as_raw().iter().map(|&v| v as u64)),
                want,
                "lanczos {w}x{h}"
            );
        }
        for ((w, h), (nw, nh), want) in [
            (
                (640usize, 480usize),
                (341usize, 256usize),
                241589814486094740u64,
            ),
            ((97, 41), (256, 108), 563312997931405113),
            ((300, 200), (300, 200), 544994184510528300),
        ] {
            let f = Array2::from_shape_fn((h, w), |(y, x)| {
                (((x * 31 + y * 17 + (x * y) % 13) % 256) as f64 / 255.0 - 0.25) as f32
            });
            let out = resize_f32(&f, nw, nh, Filter::Bilinear);
            assert_eq!(
                checksum(out.iter().map(|v| v.to_bits() as u64)),
                want,
                "float {w}x{h}"
            );
        }
        let grey = formula(97, 41);
        assert_eq!(
            checksum(
                grey.pixels()
                    .map(|p| to_luma(p.0[0], p.0[1], p.0[2]) as u64)
            ),
            1012039993
        );
    }
}
