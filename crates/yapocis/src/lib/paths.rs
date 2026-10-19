//! The device-independent rendering of `yapocis.lib.mplbackend`: stroking
//! polylines with dash patterns, filling closed polygons through ear-cut
//! triangulation drawn as mesh triangles, compositing images and text
//! coverage bitmaps, all on a `KernelDraw4` canvas.
//!
//! The matplotlib `RendererBase`/`FigureCanvas` plugin itself (which turns
//! matplotlib paths, transforms, fonts and graphics contexts into these calls)
//! stays in Python; this module is what a Rust program uses to draw the same
//! kind of vector content, and [`flatten_cubic`] / [`flatten_quadratic`] stand
//! in for matplotlib's `Path.to_polygons`.

use super::kerneldraw::{Item, KernelDraw4};
use crate::{Device, Error, Result};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};

/// A canvas with matplotlib's conventions: `(height, width, 4)` RGBA, white
/// and opaque to start, `y` down in canvas space.
pub struct PathRenderer {
    pub kd: KernelDraw4,
    pub width: usize,
    pub height: usize,
}

/// `RendererTemplate._kernel_line_width`: the lines kernel covers one pixel
/// more than its width, so pass `width - 1` for a stroke `width` pixels wide.
pub fn kernel_line_width(width_pixels: f32) -> f32 {
    (width_pixels - 1.0).max(0.0)
}

impl PathRenderer {
    /// `RendererTemplate(dpi, figsize)` with the canvas in pixels.
    pub fn new(device: &Device, width: usize, height: usize) -> Result<PathRenderer> {
        let canvas = Array3::<f32>::ones((height, width, 4)).into_dyn();
        Ok(PathRenderer {
            kd: KernelDraw4::new(
                device,
                &canvas.view(),
                Some(super::kerneldraw::DEFAULT_CHUNK),
            )?,
            width,
            height,
        })
    }

    /// `points_to_pixels(points)` at `dpi`.
    pub fn points_to_pixels(points: f64, dpi: f64) -> f64 {
        points / 72.0 * dpi
    }

    /// `_draw_dashed_line`: queue the dashes of one segment (canvas coordinates).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_dashed_line(
        &mut self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        color: [f32; 4],
        linewidth: f32,
        dash_pattern: &[f64],
        dash_offset: f64,
    ) {
        let (dx, dy) = (x2 - x1, y2 - y1);
        let length = (dx * dx + dy * dy).sqrt();
        if length < 1e-10 || dash_pattern.is_empty() {
            return;
        }
        let (ux, uy) = (dx / length, dy / length);
        let mut pos = 0.0;
        let mut dash_index = 0usize;
        let mut drawing = true;
        let mut offset_remaining = dash_offset;
        while offset_remaining > 0.0 && dash_index < dash_pattern.len() {
            if offset_remaining >= dash_pattern[dash_index] {
                offset_remaining -= dash_pattern[dash_index];
                dash_index = (dash_index + 1) % dash_pattern.len();
                drawing = !drawing;
            } else {
                pos = offset_remaining;
                offset_remaining = 0.0;
            }
        }
        while pos < length {
            let segment_length = dash_pattern[dash_index];
            if drawing {
                let end_pos = (pos + segment_length).min(length);
                let line = [
                    (x1 + ux * pos) as f32,
                    (y1 + uy * pos) as f32,
                    (x1 + ux * end_pos) as f32,
                    (y1 + uy * end_pos) as f32,
                ];
                self.kd.line(
                    line,
                    Item {
                        color: Some(color),
                        width: Some(linewidth),
                        aa_width: Some(0.5),
                        ..Default::default()
                    },
                );
                pos = end_pos;
            } else {
                pos = (pos + segment_length).min(length);
            }
            dash_index = (dash_index + 1) % dash_pattern.len();
            drawing = !drawing;
        }
    }

    /// `draw_path` from already-flattened polygons in matplotlib display
    /// coordinates (`y` up; flipped here). `linewidth_px` is the stroke width in
    /// pixels (a zero width draws no outline); `dashes` is `(offset, pattern)` in
    /// pixels. Closed polygons (first point equal to the last) are filled with
    /// `face`, the first as the outer ring and the rest as holes, exactly as the
    /// Python backend feeds them to earcut.
    pub fn draw_path(
        &mut self,
        polygons: &[Vec<[f64; 2]>],
        edge: [f32; 4],
        face: Option<[f32; 4]>,
        linewidth_px: f64,
        dashes: Option<(f64, Vec<f64>)>,
    ) -> Result<()> {
        let height = self.height as f64;
        let mut points: Vec<Vec<[f64; 2]>> = Vec::with_capacity(polygons.len());
        let mut closed: Vec<Vec<[f64; 2]>> = Vec::new();
        for polygon in polygons {
            let p: Vec<[f64; 2]> = polygon.iter().map(|[x, y]| [*x, height - *y]).collect();
            if let (Some(first), Some(last)) = (p.first(), p.last()) {
                // Python: (points[0] - points[-1]).sum() == 0.0
                if (first[0] - last[0]) + (first[1] - last[1]) == 0.0 {
                    closed.push(p.clone());
                }
            }
            points.push(p);
        }
        let linewidth_pixels = kernel_line_width(linewidth_px as f32);
        if linewidth_px > 0.0 {
            for p in &points {
                for w in p.windows(2) {
                    match &dashes {
                        Some((offset, pattern)) if !pattern.is_empty() => {
                            self.draw_dashed_line(
                                w[0][0],
                                w[0][1],
                                w[1][0],
                                w[1][1],
                                edge,
                                linewidth_pixels,
                                pattern,
                                *offset,
                            );
                        }
                        _ => self.kd.line(
                            [
                                w[0][0] as f32,
                                w[0][1] as f32,
                                w[1][0] as f32,
                                w[1][1] as f32,
                            ],
                            Item {
                                color: Some(edge),
                                width: Some(linewidth_pixels),
                                aa_width: Some(0.5),
                                ..Default::default()
                            },
                        ),
                    }
                }
            }
        }
        if let (false, Some(face)) = (closed.is_empty(), face) {
            if face[3] > 0.0 {
                let mut flat: Vec<f64> = Vec::new();
                let mut holes: Vec<usize> = Vec::new();
                for (i, ring) in closed.iter().enumerate() {
                    if i > 0 {
                        holes.push(flat.len() / 2);
                    }
                    for p in ring {
                        flat.push(p[0]);
                        flat.push(p[1]);
                    }
                }
                let simplices = earcutr::earcut(&flat, &holes, 2)
                    .map_err(|e| Error::Invalid(format!("earcut: {e:?}")))?;
                for tri in simplices.chunks(3) {
                    if tri.len() < 3 {
                        continue;
                    }
                    let v = |i: usize| [flat[2 * tri[i]] as f32, flat[2 * tri[i] + 1] as f32];
                    let (a, b, c) = (v(0), v(1), v(2));
                    self.kd.mesh_triangle(
                        [a[0], a[1], b[0], b[1], c[0], c[1]],
                        Item {
                            color: Some(face),
                            aa_width: Some(0.0),
                            ..Default::default()
                        },
                    );
                }
            }
        }
        self.kd.render()
    }

    /// `draw_image`: composite an RGBA (or RGB, or grey) image whose
    /// bottom-left corner is at matplotlib display coordinates `(x, y)`.
    pub fn draw_image(&mut self, x: f64, y: f64, im: ArrayView3<'_, f32>) -> Result<()> {
        let (rows, cols, depth) = im.dim();
        let rgba = match depth {
            4 => im.to_owned(),
            3 => Array3::from_shape_fn(
                (rows, cols, 4),
                |(r, c, k)| if k == 3 { 1.0 } else { im[[r, c, k]] },
            ),
            _ => return Err(Error::Invalid("draw_image needs (h, w, 3|4)".into())),
        };
        let canvas_x = x as i64;
        let canvas_y = (self.height as f64 - y - rows as f64) as i64;
        // np.flipud
        let flipped =
            Array3::from_shape_fn((rows, cols, 4), |(r, c, k)| rgba[[rows - 1 - r, c, k]]);
        self.kd
            .composite(&[(canvas_x as f64, canvas_y as f64)], &[flipped.view()])
    }

    /// Grey-scale images for [`draw_image`](Self::draw_image).
    pub fn draw_gray_image(&mut self, x: f64, y: f64, im: ArrayView2<'_, f32>) -> Result<()> {
        let (rows, cols) = im.dim();
        let rgba = Array3::from_shape_fn(
            (rows, cols, 4),
            |(r, c, k)| if k == 3 { 1.0 } else { im[[r, c]] },
        );
        self.draw_image(x, y, rgba.view())
    }

    /// `_draw_text_image`: composite a coverage bitmap whose bottom-left
    /// corner is at canvas `(x, y)`, rotated `angle` degrees counterclockwise
    /// about that corner, with straight colour `rgba` scaled by coverage.
    /// Rotation uses bilinear resampling with an expanded frame, as PIL's
    /// `rotate(expand=True)` does; the two resamplers are not bit-identical.
    pub fn draw_text_image(
        &mut self,
        coverage: ArrayView2<'_, f32>,
        x: i64,
        y: i64,
        angle: f64,
        rgba: [f32; 4],
    ) -> Result<()> {
        let (rows, cols) = coverage.dim();
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let theta = angle.to_radians();
        let (coverage, top, left): (Array2<f32>, i64, i64) = if angle.rem_euclid(360.0) == 0.0 {
            (coverage.to_owned(), y - rows as i64, x)
        } else {
            let rotated = rotate_bilinear_expand(coverage, angle);
            let (rh, rw) = rotated.dim();
            let (dx, dy) = (-(cols as f64) / 2.0, rows as f64 / 2.0);
            let (cx, cy) = (rw as f64 / 2.0, rh as f64 / 2.0);
            let corner_x = cx + dx * theta.cos() + dy * theta.sin();
            let corner_y = cy - dx * theta.sin() + dy * theta.cos();
            (
                rotated,
                (y as f64 - corner_y).round() as i64,
                (x as f64 - corner_x).round() as i64,
            )
        };
        let (rows, cols) = coverage.dim();
        let block = Array3::from_shape_fn((rows, cols, 4), |(r, c, k)| {
            if k == 3 {
                coverage[[r, c]] * rgba[3]
            } else {
                rgba[k]
            }
        });
        self.kd
            .composite(&[(left as f64, top as f64)], &[block.view()])
    }

    /// `render` then `read`, with the alpha channel forced opaque.
    pub fn read(&mut self) -> Result<Array3<f32>> {
        let a = self.kd.read()?;
        let mut a = a.into_dimensionality::<ndarray::Ix3>()?;
        a.index_axis_mut(Axis(2), 3).fill(1.0);
        Ok(a)
    }
}

/// Rotate a coverage image counterclockwise by `angle` degrees about its
/// centre into an expanded frame, bilinear, zero outside (PIL `rotate`).
pub fn rotate_bilinear_expand(src: ArrayView2<'_, f32>, angle: f64) -> Array2<f32> {
    let (h, w) = src.dim();
    let theta = angle.rem_euclid(360.0).to_radians();
    // PIL rounds the matrix to 15 decimals, so right angles are exact
    let round15 = |v: f64| (v * 1e15).round() / 1e15;
    let (cos, sin) = (round15(theta.cos()), round15(theta.sin()));
    // expanded size from the rotated corners (PIL: ceil(max) - floor(min))
    let corners = [
        (0.0, 0.0),
        (w as f64, 0.0),
        (w as f64, h as f64),
        (0.0, h as f64),
    ];
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y) in corners {
        let (dx, dy) = (x - cx, y - cy);
        xs.push(dx * cos + dy * sin + cx);
        ys.push(-dx * sin + dy * cos + cy);
    }
    let minmax = |v: &[f64]| {
        (
            v.iter().cloned().fold(f64::INFINITY, f64::min),
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let (x0, x1) = minmax(&xs);
    let (y0, y1) = minmax(&ys);
    let nw = (x1.ceil() - x0.floor()) as usize;
    let nh = (y1.ceil() - y0.floor()) as usize;
    let (ncx, ncy) = (nw as f64 / 2.0, nh as f64 / 2.0);
    let mut out = Array2::<f32>::zeros((nh, nw));
    for r in 0..nh {
        for c in 0..nw {
            // inverse map the output pixel centre into the source
            let (dx, dy) = (c as f64 + 0.5 - ncx, r as f64 + 0.5 - ncy);
            let sx = dx * cos - dy * sin + cx - 0.5;
            let sy = dx * sin + dy * cos + cy - 0.5;
            let (x_floor, y_floor) = (sx.floor(), sy.floor());
            let (fx, fy) = ((sx - x_floor) as f32, (sy - y_floor) as f32);
            let sample = |xx: i64, yy: i64| -> f32 {
                if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                    0.0
                } else {
                    src[[yy as usize, xx as usize]]
                }
            };
            let (xi, yi) = (x_floor as i64, y_floor as i64);
            let v = sample(xi, yi) * (1.0 - fx) * (1.0 - fy)
                + sample(xi + 1, yi) * fx * (1.0 - fy)
                + sample(xi, yi + 1) * (1.0 - fx) * fy
                + sample(xi + 1, yi + 1) * fx * fy;
            out[[r, c]] = v;
        }
    }
    out
}

/// Flatten a cubic Bézier into `segments` line segments (for building the
/// polygons `draw_path` takes without matplotlib).
pub fn flatten_cubic(
    p0: [f64; 2],
    p1: [f64; 2],
    p2: [f64; 2],
    p3: [f64; 2],
    segments: usize,
) -> Vec<[f64; 2]> {
    (0..=segments.max(1))
        .map(|i| {
            let t = i as f64 / segments.max(1) as f64;
            let mt = 1.0 - t;
            let b = |a: f64, b: f64, c: f64, d: f64| {
                mt * mt * mt * a + 3.0 * mt * mt * t * b + 3.0 * mt * t * t * c + t * t * t * d
            };
            [b(p0[0], p1[0], p2[0], p3[0]), b(p0[1], p1[1], p2[1], p3[1])]
        })
        .collect()
}

/// Flatten a quadratic Bézier into `segments` line segments.
pub fn flatten_quadratic(
    p0: [f64; 2],
    p1: [f64; 2],
    p2: [f64; 2],
    segments: usize,
) -> Vec<[f64; 2]> {
    (0..=segments.max(1))
        .map(|i| {
            let t = i as f64 / segments.max(1) as f64;
            let mt = 1.0 - t;
            let b = |a: f64, b: f64, c: f64| mt * mt * a + 2.0 * mt * t * b + t * t * c;
            [b(p0[0], p1[0], p2[0]), b(p0[1], p1[1], p2[1])]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_and_flattening() {
        assert_eq!(kernel_line_width(2.5), 1.5);
        assert_eq!(kernel_line_width(0.5), 0.0);
        let pts = flatten_cubic([0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], 4);
        assert_eq!(pts.len(), 5);
        assert_eq!(pts[0], [0.0, 0.0]);
        assert_eq!(pts[4], [1.0, 0.0]);
        let q = flatten_quadratic([0.0, 0.0], [1.0, 1.0], [2.0, 0.0], 2);
        assert_eq!(q[1], [1.0, 0.5]);
    }

    #[test]
    fn rotation_expands() {
        let src = Array2::<f32>::ones((4, 10));
        let r = rotate_bilinear_expand(src.view(), 90.0);
        assert_eq!(r.dim(), (10, 4));
        assert!(r[[5, 2]] > 0.9);
        let r0 = rotate_bilinear_expand(src.view(), 0.0);
        assert_eq!(r0.dim(), (4, 10));
    }
}
