//! A small plotting layer for the examples, drawn on the GPU with
//! `yapocis::lib::paths` (strokes, earcut fills, image compositing) and
//! `yapocis::lib::typeset` (FreeType text).
//!
//! This stands in for the matplotlib figures the Python examples save. It
//! reproduces their visible content (axes frames, ticks and labels, line
//! series, dashed markers, bars, filled regions, scatter markers, heatmaps,
//! legends, titles), not matplotlib's layout engine, so spacing and styling
//! differ from the Python PNGs.
#![allow(dead_code)]

use ndarray::{Array2, Array3};
use std::path::PathBuf;
use yapocis::lib::paths::PathRenderer;
use yapocis::lib::typeset::render_text;
use yapocis::{Device, Result};

pub type Rgba = [f32; 4];

pub const BLACK: Rgba = [0.0, 0.0, 0.0, 1.0];
pub const GREY: Rgba = [0.5, 0.5, 0.5, 1.0];
pub const LIGHT: Rgba = [0.88, 0.88, 0.88, 1.0];
/// Grid lines: translucent, because grids are drawn over the plotted data.
pub const GRID: Rgba = [0.5, 0.5, 0.5, 0.22];
pub const RED: Rgba = [0.84, 0.15, 0.16, 1.0];
pub const BLUE: Rgba = [0.12, 0.47, 0.71, 1.0];
pub const GREEN: Rgba = [0.17, 0.63, 0.17, 1.0];
pub const ORANGE: Rgba = [1.0, 0.5, 0.05, 1.0];
pub const PURPLE: Rgba = [0.58, 0.40, 0.74, 1.0];
pub const BROWN: Rgba = [0.55, 0.34, 0.29, 1.0];
pub const PINK: Rgba = [0.89, 0.47, 0.76, 1.0];
pub const OLIVE: Rgba = [0.74, 0.74, 0.13, 1.0];
pub const CYAN: Rgba = [0.09, 0.75, 0.81, 1.0];

/// matplotlib's default colour cycle (tab10), for series without a colour.
pub const CYCLE: [Rgba; 10] = [
    BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, GREY, OLIVE, CYAN,
];

pub fn with_alpha(c: Rgba, a: f32) -> Rgba {
    [c[0], c[1], c[2], a]
}

/// A figure: a white RGBA canvas in pixels, `(0, 0)` at the bottom left.
pub struct Figure {
    pub r: PathRenderer,
    pub width: usize,
    pub height: usize,
}

/// A line style for [`Axes::line`].
#[derive(Clone)]
pub struct Line {
    pub color: Rgba,
    pub width: f64,
    pub dashes: Option<Vec<f64>>,
}

impl Line {
    pub fn solid(color: Rgba, width: f64) -> Line {
        Line {
            color,
            width,
            dashes: None,
        }
    }
    pub fn dashed(color: Rgba, width: f64) -> Line {
        Line {
            color,
            width,
            dashes: Some(vec![6.0, 4.0]),
        }
    }
    pub fn dotted(color: Rgba, width: f64) -> Line {
        Line {
            color,
            width,
            dashes: Some(vec![2.0, 3.0]),
        }
    }
}

impl Figure {
    pub fn new(device: &Device, width: usize, height: usize) -> Result<Figure> {
        Ok(Figure {
            r: PathRenderer::new(device, width, height)?,
            width,
            height,
        })
    }

    /// Text with its bottom-left corner at display `(x, y)` (y up). `anchor`
    /// is 0 = left, 0.5 = centred, 1 = right; `valign` likewise from bottom.
    #[allow(clippy::too_many_arguments)]
    pub fn text(
        &mut self,
        x: f64,
        y: f64,
        s: &str,
        size: f64,
        color: Rgba,
        anchor: f64,
        valign: f64,
    ) -> Result<()> {
        self.text_rot(x, y, s, size, color, anchor, valign, 0.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn text_rot(
        &mut self,
        x: f64,
        y: f64,
        s: &str,
        size: f64,
        color: Rgba,
        anchor: f64,
        valign: f64,
        angle: f64,
    ) -> Result<()> {
        if s.trim().is_empty() {
            return Ok(());
        }
        let t = render_text(s, "default", Some(size), None)?;
        let (rows, cols) = t.block.dim();
        let (w, h) = (cols as f64, rows as f64);
        let (bx, by) = if angle == 90.0 {
            // rotated a quarter turn counter-clockwise: width runs up the page
            (x + h * (1.0 - valign) - h, y - w * anchor)
        } else {
            (x - w * anchor, y - h * valign)
        };
        let canvas_y = self.height as f64 - by;
        self.r.draw_text_image(
            t.block.view(),
            bx.round() as i64,
            canvas_y.round() as i64,
            angle,
            color,
        )
    }

    /// Composite an RGB image with its bottom-left corner at display `(x, y)`.
    pub fn image(&mut self, x: f64, y: f64, rgb: &Array3<f32>) -> Result<()> {
        self.r.draw_image(x, y, rgb.view())
    }

    pub fn polyline(&mut self, pts: &[[f64; 2]], style: &Line) -> Result<()> {
        if pts.len() < 2 {
            return Ok(());
        }
        let dashes = style.dashes.clone().map(|d| (0.0, d));
        self.r
            .draw_path(&[pts.to_vec()], style.color, None, style.width, dashes)
    }

    /// Fill a polygon (closed automatically); `edge` draws an outline.
    pub fn fill(&mut self, pts: &[[f64; 2]], face: Rgba, edge: Option<(Rgba, f64)>) -> Result<()> {
        let mut ring = pts.to_vec();
        if ring.first() != ring.last() {
            ring.push(ring[0]);
        }
        let (ec, ew) = edge.unwrap_or(([0.0; 4], 0.0));
        self.r.draw_path(&[ring], ec, Some(face), ew, None)
    }

    pub fn save(&mut self, path: &PathBuf) -> Result<()> {
        let a = self.r.read()?;
        let (h, w, _) = a.dim();
        let img = image::RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let p = |c: usize| (a[[y as usize, x as usize, c]].clamp(0.0, 1.0) * 255.0) as u8;
            image::Rgb([p(0), p(1), p(2)])
        });
        img.save(path).expect("save plot");
        println!("Plot saved to {}", path.display());
        Ok(())
    }
}

/// One set of axes inside a [`Figure`], in display pixels.
pub struct Axes {
    pub x0: f64,
    pub y0: f64,
    pub w: f64,
    pub h: f64,
    pub xlim: (f64, f64),
    pub ylim: (f64, f64),
    pub log_y: bool,
    legend: Vec<(String, Rgba)>,
}

fn nice_step(span: f64, target: usize) -> f64 {
    let raw = span / target.max(1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let nice = if norm < 1.5 {
        1.0
    } else if norm < 3.0 {
        2.0
    } else if norm < 7.0 {
        5.0
    } else {
        10.0
    };
    nice * mag
}

fn fmt_tick(v: f64, step: f64) -> String {
    if v.abs() < step * 1e-6 {
        return "0".into();
    }
    if step >= 1.0 && v.fract().abs() < 1e-9 {
        format!("{v:.0}")
    } else {
        let digits = (-step.log10().floor()).max(0.0) as usize;
        format!("{v:.digits$}")
    }
}

impl Axes {
    /// Axes occupying the fraction `[fx, fy, fw, fh]` of the figure (matplotlib's
    /// `add_axes` rectangle, origin bottom left).
    pub fn new(fig: &Figure, rect: [f64; 4], xlim: (f64, f64), ylim: (f64, f64)) -> Axes {
        let (fw, fh) = (fig.width as f64, fig.height as f64);
        Axes {
            x0: rect[0] * fw,
            y0: rect[1] * fh,
            w: rect[2] * fw,
            h: rect[3] * fh,
            xlim,
            ylim,
            log_y: false,
            legend: Vec::new(),
        }
    }

    /// A grid cell of `rows × cols` subplots (row 0 at the top), with margins
    /// for tick labels and titles.
    pub fn subplot(
        fig: &Figure,
        rows: usize,
        cols: usize,
        index: usize,
        xlim: (f64, f64),
        ylim: (f64, f64),
    ) -> Axes {
        let (r, c) = (index / cols, index % cols);
        let cw = 1.0 / cols as f64;
        let ch = 1.0 / rows as f64;
        let left = 70.0 / fig.width as f64;
        let right = 20.0 / fig.width as f64;
        let bottom = 50.0 / fig.height as f64;
        let top = 45.0 / fig.height as f64;
        Axes::new(
            fig,
            [
                c as f64 * cw + left,
                1.0 - (r + 1) as f64 * ch + bottom,
                cw - left - right,
                ch - bottom - top,
            ],
            xlim,
            ylim,
        )
    }

    pub fn tx(&self, x: f64) -> f64 {
        self.x0 + (x - self.xlim.0) / (self.xlim.1 - self.xlim.0) * self.w
    }

    pub fn ty(&self, y: f64) -> f64 {
        let (lo, hi, v) = if self.log_y {
            (
                self.ylim.0.log10(),
                self.ylim.1.log10(),
                y.max(1e-300).log10(),
            )
        } else {
            (self.ylim.0, self.ylim.1, y)
        };
        self.y0 + (v - lo) / (hi - lo) * self.h
    }

    fn clip(&self, x: f64, y: f64) -> [f64; 2] {
        [
            x.clamp(self.x0, self.x0 + self.w),
            y.clamp(self.y0, self.y0 + self.h),
        ]
    }

    /// Frame, grid, ticks with labels, axis labels and title.
    pub fn decorate(
        &self,
        fig: &mut Figure,
        title: &str,
        xlabel: &str,
        ylabel: &str,
    ) -> Result<()> {
        let fs = 11.0;
        // grid and ticks, x
        let step = nice_step(self.xlim.1 - self.xlim.0, 6);
        let mut v = (self.xlim.0 / step).ceil() * step;
        while v <= self.xlim.1 + step * 1e-9 {
            let x = self.tx(v);
            fig.polyline(
                &[[x, self.y0], [x, self.y0 + self.h]],
                &Line::solid(GRID, 1.0),
            )?;
            fig.polyline(
                &[[x, self.y0], [x, self.y0 - 4.0]],
                &Line::solid(BLACK, 1.0),
            )?;
            fig.text(x, self.y0 - 7.0, &fmt_tick(v, step), fs, BLACK, 0.5, 1.0)?;
            v += step;
        }
        // y
        if self.log_y {
            let mut e = self.ylim.0.log10().floor() as i32;
            while (e as f64) <= self.ylim.1.log10() {
                let val = 10f64.powi(e);
                if val >= self.ylim.0 {
                    let y = self.ty(val);
                    fig.polyline(
                        &[[self.x0, y], [self.x0 + self.w, y]],
                        &Line::solid(GRID, 1.0),
                    )?;
                    fig.polyline(
                        &[[self.x0 - 4.0, y], [self.x0, y]],
                        &Line::solid(BLACK, 1.0),
                    )?;
                    fig.text(self.x0 - 7.0, y, &format!("1e{e}"), fs, BLACK, 1.0, 0.5)?;
                }
                e += 1;
            }
        } else {
            let step = nice_step(self.ylim.1 - self.ylim.0, 5);
            let mut v = (self.ylim.0 / step).ceil() * step;
            while v <= self.ylim.1 + step * 1e-9 {
                let y = self.ty(v);
                fig.polyline(
                    &[[self.x0, y], [self.x0 + self.w, y]],
                    &Line::solid(GRID, 1.0),
                )?;
                fig.polyline(
                    &[[self.x0 - 4.0, y], [self.x0, y]],
                    &Line::solid(BLACK, 1.0),
                )?;
                fig.text(self.x0 - 7.0, y, &fmt_tick(v, step), fs, BLACK, 1.0, 0.5)?;
                v += step;
            }
        }
        self.frame(fig)?;
        fig.text(
            self.x0 + self.w / 2.0,
            self.y0 + self.h + 8.0,
            title,
            13.0,
            BLACK,
            0.5,
            0.0,
        )?;
        fig.text(
            self.x0 + self.w / 2.0,
            self.y0 - 26.0,
            xlabel,
            fs,
            BLACK,
            0.5,
            1.0,
        )?;
        fig.text_rot(
            self.x0 - 48.0,
            self.y0 + self.h / 2.0,
            ylabel,
            fs,
            BLACK,
            0.5,
            0.0,
            90.0,
        )?;
        Ok(())
    }

    pub fn frame(&self, fig: &mut Figure) -> Result<()> {
        let (x0, y0, x1, y1) = (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h);
        fig.polyline(
            &[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
            &Line::solid(BLACK, 1.0),
        )
    }

    /// A data series, clipped to the axes and split at points outside them.
    pub fn line(
        &mut self,
        fig: &mut Figure,
        xs: &[f64],
        ys: &[f64],
        style: &Line,
        label: Option<&str>,
    ) -> Result<()> {
        let mut run: Vec<[f64; 2]> = Vec::new();
        for (&x, &y) in xs.iter().zip(ys) {
            let inside = x >= self.xlim.0 && x <= self.xlim.1 && y.is_finite();
            if inside {
                let (px, py) = (self.tx(x), self.ty(y));
                run.push(self.clip(px, py));
            } else if !run.is_empty() {
                fig.polyline(&run, style)?;
                run.clear();
            }
        }
        fig.polyline(&run, style)?;
        if let Some(l) = label {
            self.legend.push((l.to_string(), style.color));
        }
        Ok(())
    }

    pub fn vline(&self, fig: &mut Figure, x: f64, style: &Line) -> Result<()> {
        if x < self.xlim.0 || x > self.xlim.1 {
            return Ok(());
        }
        let px = self.tx(x);
        fig.polyline(&[[px, self.y0], [px, self.y0 + self.h]], style)
    }

    pub fn hline(&self, fig: &mut Figure, y: f64, style: &Line) -> Result<()> {
        let py = self.ty(y);
        if py < self.y0 || py > self.y0 + self.h {
            return Ok(());
        }
        fig.polyline(&[[self.x0, py], [self.x0 + self.w, py]], style)
    }

    /// `axvspan`: a shaded vertical band.
    pub fn vspan(
        &mut self,
        fig: &mut Figure,
        x0: f64,
        x1: f64,
        color: Rgba,
        label: Option<&str>,
    ) -> Result<()> {
        let (a, b) = (self.tx(x0.max(self.xlim.0)), self.tx(x1.min(self.xlim.1)));
        fig.fill(
            &[
                [a, self.y0],
                [b, self.y0],
                [b, self.y0 + self.h],
                [a, self.y0 + self.h],
            ],
            color,
            None,
        )?;
        if let Some(l) = label {
            self.legend.push((l.to_string(), color));
        }
        Ok(())
    }

    /// A bar from 0 to `height` centred on `x`.
    pub fn bar(
        &self,
        fig: &mut Figure,
        x: f64,
        width: f64,
        height: f64,
        face: Rgba,
        edge: Option<Rgba>,
    ) -> Result<()> {
        let (a, b) = (self.tx(x - width / 2.0), self.tx(x + width / 2.0));
        let (y0, y1) = (
            self.ty(0.0f64.max(self.ylim.0).min(height)),
            self.ty(height.max(0.0f64.max(self.ylim.0))),
        );
        let (lo, hi) = if height >= 0.0 {
            (y0, y1)
        } else {
            (self.ty(height), self.ty(0.0))
        };
        fig.fill(
            &[[a, lo], [b, lo], [b, hi], [a, hi]],
            face,
            edge.map(|e| (e, 1.0)),
        )
    }

    /// A filled marker polygon (unit-size vertices scaled to `size` pixels) at data `(x, y)`.
    pub fn marker(
        &self,
        fig: &mut Figure,
        x: f64,
        y: f64,
        verts: &[[f64; 2]],
        size: f64,
        face: Rgba,
    ) -> Result<()> {
        let (cx, cy) = (self.tx(x), self.ty(y));
        let pts: Vec<[f64; 2]> = verts
            .iter()
            .map(|[vx, vy]| [cx + vx * size, cy + vy * size])
            .collect();
        fig.fill(&pts, face, None)
    }

    /// A heatmap filling the axes; `values` is `(rows, cols)` with row 0 at the
    /// bottom (matplotlib `origin='lower'`), mapped through viridis.
    pub fn heatmap(&self, fig: &mut Figure, values: &Array2<f32>) -> Result<()> {
        let (rows, cols) = values.dim();
        let (lo, hi) = values
            .iter()
            .filter(|v| v.is_finite())
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let (w, h) = (self.w.round() as usize, self.h.round() as usize);
        let img = Array3::from_shape_fn((h, w, 3), |(r, c, k)| {
            // image row 0 is the top of the axes
            let src_r = ((h - 1 - r) as f64 / h as f64 * rows as f64) as usize;
            let src_c = (c as f64 / w as f64 * cols as f64) as usize;
            let v = values[[src_r.min(rows - 1), src_c.min(cols - 1)]];
            let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.0 };
            viridis(t)[k]
        });
        // draw_image flips rows (matplotlib images are bottom-up)
        let flipped = Array3::from_shape_fn((h, w, 3), |(r, c, k)| img[[h - 1 - r, c, k]]);
        fig.image(self.x0, self.y0, &flipped)
    }

    /// Add a legend entry without drawing anything.
    pub fn add_legend(&mut self, label: &str, color: Rgba) {
        self.legend.push((label.to_string(), color));
    }

    /// Legend box in the upper right (or upper left with `left`).
    pub fn legend(&self, fig: &mut Figure, left: bool) -> Result<()> {
        if self.legend.is_empty() {
            return Ok(());
        }
        let row = 16.0;
        let width = 12.0
            + 26.0
            + self
                .legend
                .iter()
                .map(|(s, _)| s.chars().count() as f64 * 6.2)
                .fold(0.0, f64::max);
        let height = row * self.legend.len() as f64 + 8.0;
        let x = if left {
            self.x0 + 8.0
        } else {
            self.x0 + self.w - width - 8.0
        };
        let y = self.y0 + self.h - height - 8.0;
        fig.fill(
            &[
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height],
            ],
            [1.0, 1.0, 1.0, 0.85],
            Some((LIGHT, 1.0)),
        )?;
        for (i, (label, color)) in self.legend.iter().enumerate() {
            let yy = y + height - 4.0 - row * (i as f64 + 0.5);
            fig.polyline(
                &[[x + 6.0, yy], [x + 28.0, yy]],
                &Line::solid(with_alpha(*color, 1.0), 2.5),
            )?;
            fig.text(x + 34.0, yy, label, 10.0, BLACK, 0.0, 0.5)?;
        }
        Ok(())
    }
}

/// A compact viridis approximation (five-stop piecewise linear).
pub fn viridis(t: f32) -> [f32; 3] {
    const STOPS: [[f32; 3]; 5] = [
        [0.267, 0.005, 0.329],
        [0.231, 0.322, 0.545],
        [0.129, 0.569, 0.549],
        [0.369, 0.788, 0.384],
        [0.993, 0.906, 0.144],
    ];
    let t = t.clamp(0.0, 1.0) * 4.0;
    let i = (t.floor() as usize).min(3);
    let f = t - i as f32;
    let (a, b) = (STOPS[i], STOPS[i + 1]);
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

/// `np.linspace(a, b, n)`.
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}
