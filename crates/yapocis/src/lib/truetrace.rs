//! `yapocis.lib.truetrace`: bitmap contour tracing.
//!
//! GPU: jump-flood signed distance plus marching squares ([`TrueTrace::trace_segments`]).
//! CPU: thresholding, Felzenszwalb EDT, marching squares, segment chaining,
//! Douglas–Peucker, region nesting and SVG writers, ported function by function
//! from the Python module (numpy's float32 arithmetic and ordering included).

use crate::interfaces::truetrace;
use crate::rt::{CallArg, CallOptions};
use crate::{Device, Error, Result};
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;
use std::fmt::Write as _;

pub struct TrueTrace {
    pub kernels: truetrace::Truetrace,
}

/// Segments as an `(N, 4)` array of `[x0, y0, x1, y1]`.
pub type Segments = Array2<f32>;
/// A polyline / ring as an `(N, 2)` array.
pub type Polyline = Array2<f32>;

pub const INF32: f32 = 1e30;

impl TrueTrace {
    pub fn load(device: &Device) -> Result<TrueTrace> {
        Ok(TrueTrace {
            kernels: truetrace::Truetrace::load(device)?,
        })
    }

    /// `TrueTrace.trace_segments(mask_u8, iso=0.0)`: segments of the iso-contour
    /// of the signed distance field of an `H×W` mask (non-zero is foreground),
    /// plus `(W, H)`. The field is the jump-flood distance to the background
    /// minus the distance to the foreground (positive inside, the convention of
    /// `sdf_from_mask_cpu`), so iso 0 is the boundary between the two pixel
    /// classes. Every launch uses one thread per pixel, as the Python wrapper
    /// does since 2026-10-21.
    pub fn trace_segments(
        &self,
        mask: &ArrayView2<'_, u8>,
        iso: f32,
    ) -> Result<(Segments, (usize, usize))> {
        let (h, w) = mask.dim();
        let mut m = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                m[x * h + y] = (mask[[y, x]] != 0) as i32;
            }
        }
        let inverse: Vec<i32> = m.iter().map(|&v| 1 - v).collect();
        let device = self.kernels.device().clone();
        let opts = CallOptions {
            global: Some(vec![w * h]),
            async_exec: true,
            ..Default::default()
        };
        let d_paper = self.nearest(&inverse, w, h, &opts)?;
        let d_ink = self.nearest(&m, w, h, &opts)?;
        let sdf: Vec<f32> = d_paper.iter().zip(&d_ink).map(|(&p, &i)| p - i).collect();
        let iso = off_field_iso(&sdf, iso);
        let program = self.kernels.program();
        let dist = device.upload(&sdf, &[w, h])?;
        let max_segments = w.saturating_sub(1) * h.saturating_sub(1) * 2;
        let counter = device.upload(&[0u32], &[1])?;
        let segments = device.upload(
            &vec![f32::NAN; (max_segments * 4).max(1)],
            &[(max_segments * 4).max(1)],
        )?;
        let mut args = [
            CallArg::Buffer(dist.as_any()),
            CallArg::Scalar(iso as f64),
            CallArg::Buffer(counter.as_any()),
            CallArg::Buffer(segments.as_any()),
        ];
        program.call("marching_squares", &mut args, &opts)?;
        device.wait_all()?;
        let count = (counter.to_vec()?[0] as usize).min(max_segments);
        let data = segments.to_vec()?;
        let segs = Array2::from_shape_vec((count, 4), data[..count * 4].to_vec())?;
        Ok((sort_segments(&segs.view()), (w, h)))
    }

    /// Jump-flood distance (pixels) from every pixel to the nearest nonzero
    /// pixel of `mask_int` (`(W, H)` column-major, as the kernels index), zero
    /// on those pixels.
    fn nearest(
        &self,
        mask_int: &[i32],
        w: usize,
        h: usize,
        opts: &CallOptions,
    ) -> Result<Vec<f32>> {
        let device = self.kernels.device().clone();
        let program = self.kernels.program();
        let m_dev = device.upload(mask_int, &[w, h])?;
        let seed_a = device.upload(&vec![-1.0f32; w * h * 4], &[w, h, 4])?;
        let seed_b = device.upload(&vec![-1.0f32; w * h * 4], &[w, h, 4])?;
        let dist = device.upload(&vec![INF32; w * h], &[w, h])?;
        let mut args = [
            CallArg::Buffer(m_dev.as_any()),
            CallArg::Buffer(seed_a.as_any()),
            CallArg::Buffer(dist.as_any()),
        ];
        program.call("init_seed_from_mask", &mut args, opts)?;
        let max_dim = w.max(h);
        let mut jump = 1usize;
        while (jump << 1) < max_dim {
            jump <<= 1;
        }
        let (mut cur_in, mut cur_out) = (&seed_a, &seed_b);
        while jump >= 1 {
            let mut args = [
                CallArg::Scalar(jump as f64),
                CallArg::Buffer(cur_in.as_any()),
                CallArg::Buffer(dist.as_any()),
                CallArg::Buffer(cur_out.as_any()),
            ];
            program.call("jfa_step", &mut args, opts)?;
            std::mem::swap(&mut cur_in, &mut cur_out);
            jump >>= 1;
        }
        device.wait_all()?;
        Ok(dist.to_vec()?)
    }
}

/// `sort_segments`: segments `[x0, y0, x1, y1]` in a canonical order (by x0,
/// then y0, x1, y1), so chaining and simplification do not depend on the order
/// a kernel's atomic counter or a scan emitted them in.
pub fn sort_segments(segs: &ArrayView2<'_, f32>) -> Segments {
    let mut rows: Vec<[f32; 4]> = segs
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect();
    rows.sort_by(|a, b| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .find(|o| *o != std::cmp::Ordering::Equal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), 4), flat).expect("4 columns")
}

// ----------------------------- CPU helpers ----------------------------------

/// PIL's `convert("L")` luma: `(R*19595 + G*38470 + B*7471 + 0x8000) >> 16`.
pub fn luma_u8(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 0x8000) >> 16) as u8
}

/// `load_mask_from_image(path, invert=False, threshold=None)`.
///
/// Decodes with the `image` crate and converts to grayscale with PIL's luma
/// formula; JPEG decoders may differ by a level or two from Pillow's, so masks
/// of real photographs can differ on threshold-boundary pixels.
pub fn load_mask_from_image(
    path: impl AsRef<std::path::Path>,
    invert: bool,
    threshold: Option<f64>,
) -> Result<Array2<u8>> {
    let img = image::open(path.as_ref())
        .map_err(|e| Error::Invalid(format!("{}: {e}", path.as_ref().display())))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    let gray = Array2::from_shape_fn((h as usize, w as usize), |(y, x)| {
        let p = img.get_pixel(x as u32, y as u32);
        luma_u8(p[0], p[1], p[2])
    });
    Ok(mask_from_gray(&gray.view(), invert, threshold))
}

/// The thresholding half of [`load_mask_from_image`], for an already grayscale image.
pub fn mask_from_gray(
    gray: &ArrayView2<'_, u8>,
    invert: bool,
    threshold: Option<f64>,
) -> Array2<u8> {
    let thr = match threshold {
        None => otsu_threshold_u8(gray),
        Some(t) if (0.0..=1.0).contains(&t) => t * 255.0,
        Some(t) => t,
    };
    gray.mapv(|g| {
        let m = (g as f64 >= thr) as u8;
        if invert {
            1 - m
        } else {
            m
        }
    })
}

/// `otsu_threshold_u8(gray)`: the bin maximising between-class variance.
pub fn otsu_threshold_u8(gray: &ArrayView2<'_, u8>) -> f64 {
    let mut hist = [0f64; 256];
    for &g in gray.iter() {
        hist[g as usize] += 1.0;
    }
    let total = gray.len() as f64;
    let mut omega = 0.0;
    let mut mu = 0.0;
    let mut omegas = [0f64; 256];
    let mut mus = [0f64; 256];
    for i in 0..256 {
        let prob = hist[i] / total;
        omega += prob;
        mu += prob * i as f64;
        omegas[i] = omega;
        mus[i] = mu;
    }
    let mu_t = mus[255];
    let mut best = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    for i in 0..256 {
        let num = mu_t * omegas[i] - mus[i];
        let sigma = num * num / (omegas[i] * (1.0 - omegas[i]) + 1e-20);
        if sigma.is_nan() {
            continue;
        }
        if sigma > best {
            best = sigma;
            best_i = i;
        }
    }
    best_i as f64
}

/// `edt_1d(f)`: squared distance transform of a sampled function (Felzenszwalb).
#[allow(clippy::needless_range_loop)]
pub fn edt_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = ((f[q] + (q * q) as f64) - (f[v[k]] + (v[k] * v[k]) as f64))
            / (2.0 * (q as f64 - v[k] as f64));
        while k > 0 && s <= z[k] {
            k -= 1;
            s = ((f[q] + (q * q) as f64) - (f[v[k]] + (v[k] * v[k]) as f64))
                / (2.0 * (q as f64 - v[k] as f64));
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    let mut g = vec![0f64; n];
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let dq = q as f64 - v[k] as f64;
        g[q] = dq * dq + f[v[k]];
    }
    g
}

/// `edt_2d(mask_bool)`: Euclidean distance to the nearest `true` pixel.
pub fn edt_2d(mask: &ArrayView2<'_, bool>) -> Array2<f64> {
    let (h, w) = mask.dim();
    let inf = 1e12;
    let mut tmp = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        let row: Vec<f64> = (0..w)
            .map(|x| if mask[[y, x]] { 0.0 } else { inf })
            .collect();
        let g = edt_1d(&row);
        for x in 0..w {
            tmp[[y, x]] = g[x];
        }
    }
    let mut out = Array2::<f64>::zeros((h, w));
    for x in 0..w {
        let col: Vec<f64> = (0..h).map(|y| tmp[[y, x]]).collect();
        let g = edt_1d(&col);
        for y in 0..h {
            out[[y, x]] = g[y].sqrt();
        }
    }
    out
}

/// `sdf_from_mask_cpu(mask_u8)`: positive inside, negative outside.
pub fn sdf_from_mask_cpu(mask: &ArrayView2<'_, u8>) -> Array2<f32> {
    let b = mask.mapv(|v| v != 0);
    let nb = b.mapv(|v| !v);
    let d_true = edt_2d(&b.view());
    let d_false = edt_2d(&nb.view());
    Array2::from_shape_fn(mask.dim(), |ij| (d_false[ij] - d_true[ij]) as f32)
}

/// `marching_squares_cpu(dist, iso=0.0)` (float32 arithmetic, as numpy).
pub fn marching_squares_cpu(dist: &ArrayView2<'_, f32>, iso: f32) -> Segments {
    let (h, w) = dist.dim();
    let mut segs: Vec<[f32; 4]> = Vec::new();
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            let v00 = dist[[y, x]] - iso;
            let v10 = dist[[y, x + 1]] - iso;
            let v11 = dist[[y + 1, x + 1]] - iso;
            let v01 = dist[[y + 1, x]] - iso;
            let mut c = 0u8;
            if v00 > 0.0 {
                c |= 1;
            }
            if v10 > 0.0 {
                c |= 2;
            }
            if v11 > 0.0 {
                c |= 4;
            }
            if v01 > 0.0 {
                c |= 8;
            }
            if c == 0 || c == 15 {
                continue;
            }
            let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
            let eps = 1e-20f32;
            let (xf, yf) = (x as f32, y as f32);
            let t0 = v00 / (v00 - v10 + eps);
            let p0 = (lerp(t0, xf, xf + 1.0), yf);
            let t1 = v10 / (v10 - v11 + eps);
            let p1 = (xf + 1.0, lerp(t1, yf, yf + 1.0));
            let t2 = v01 / (v01 - v11 + eps);
            let p2 = (lerp(t2, xf, xf + 1.0), yf + 1.0);
            let t3 = v00 / (v00 - v01 + eps);
            let p3 = (xf, lerp(t3, yf, yf + 1.0));
            let pts = [p0, p1, p2, p3];
            let mut emit = |a: usize, b: usize| segs.push([pts[a].0, pts[a].1, pts[b].0, pts[b].1]);
            match c {
                1 | 14 => emit(3, 0),
                2 | 13 => emit(0, 1),
                3 | 12 => emit(3, 1),
                4 | 11 => emit(1, 2),
                6 | 9 => emit(0, 2),
                7 | 8 => emit(3, 2),
                5 => {
                    let d = v00 * v11 - v10 * v01;
                    if d > 0.0 {
                        emit(3, 0);
                        emit(1, 2);
                    } else {
                        emit(0, 1);
                        emit(2, 3);
                    }
                }
                10 => {
                    let d = v00 * v11 - v10 * v01;
                    if d > 0.0 {
                        emit(0, 1);
                        emit(2, 3);
                    } else {
                        emit(3, 0);
                        emit(1, 2);
                    }
                }
                _ => {}
            }
        }
    }
    segs_to_array(&segs)
}

fn segs_to_array(segs: &[[f32; 4]]) -> Segments {
    let flat: Vec<f32> = segs.iter().flat_map(|s| s.iter().copied()).collect();
    Array2::from_shape_vec((segs.len(), 4), flat).unwrap()
}

/// `write_svg_segments(path, segs, w, h)` as text.
pub fn svg_segments(segs: &ArrayView2<'_, f32>, w: usize, h: usize) -> String {
    let mut s = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\"><g fill=\"none\" stroke=\"black\" stroke-width=\"0.75\">"
    );
    let hf = (h as f32) - 1.0;
    for row in segs.rows() {
        let (x0, y0, x1, y1) = (row[0], row[1], row[2], row[3]);
        write!(
            s,
            "<line x1=\"{:?}\" y1=\"{:?}\" x2=\"{:?}\" y2=\"{:?}\" />",
            x1,
            hf - y1,
            x0,
            hf - y0
        )
        .unwrap();
    }
    s.push_str("</g></svg>");
    s
}

pub fn write_svg_segments(
    path: impl AsRef<std::path::Path>,
    segs: &ArrayView2<'_, f32>,
    w: usize,
    h: usize,
) -> Result<()> {
    std::fs::write(path.as_ref(), svg_segments(segs, w, h))
        .map_err(|e| Error::Invalid(format!("{}: {e}", path.as_ref().display())))?;
    Ok(())
}

/// `off_field_iso(dist, iso)`: `iso` raised in float32 steps of 1e-3 px until
/// no value of the field equals it. At an iso equal to a pixel-centre distance
/// (1, sqrt 2, 2, ...) marching squares puts contour points on those pixels'
/// centres, where the outlines on either side of a stroke meet; values at the
/// iso already count as outside, so raising it changes no classification.
pub fn off_field_iso(dist: &[f32], iso: f32) -> f32 {
    let mut iso = iso;
    while dist.contains(&iso) {
        iso += 1e-3f32;
    }
    iso
}

/// A point used as a dictionary key: exact (Python `(float(x), float(y))`) when
/// `tol <= 0`, else quantised (`round(x / tol) * tol`).
#[derive(Clone, Copy, Debug)]
struct Key(f64, f64);

impl Key {
    fn new(x: f32, y: f32, tol: f64) -> Key {
        let q = |v: f32| {
            let r = if tol <= 0.0 {
                v as f64
            } else {
                (v as f64 / tol).round_ties_even() * tol
            };
            // Python's dict treats -0.0 and 0.0 as one key
            if r == 0.0 {
                0.0
            } else {
                r
            }
        };
        Key(q(x), q(y))
    }
    fn bits(&self) -> (u64, u64) {
        (self.0.to_bits(), self.1.to_bits())
    }
    fn lt(&self, other: &Key) -> bool {
        (self.0, self.1) < (other.0, other.1)
    }
}

/// Insertion-ordered map of point keys to adjacency lists, as a Python dict.
struct Adjacency {
    keys: Vec<Key>,
    index: HashMap<(u64, u64), usize>,
    lists: Vec<Vec<usize>>,
}

impl Adjacency {
    fn id(&mut self, k: Key) -> usize {
        if let Some(&i) = self.index.get(&k.bits()) {
            return i;
        }
        self.keys.push(k);
        self.lists.push(Vec::new());
        self.index.insert(k.bits(), self.keys.len() - 1);
        self.keys.len() - 1
    }
}

/// `chain_segments_to_polylines(segs, tol=0.0)`: endpoints match exactly when
/// `tol <= 0` (a crossing shared by two cells is the same float in both). A
/// `tol` grid merges contours closer than `tol`: `trace_mask` chained with 0.25
/// until 2026-10-21, which above iso 0.5 fused the two outlines of a 1 px
/// stroke and filled long chords as black wedges.
pub fn chain_segments_to_polylines(segs: &ArrayView2<'_, f32>, tol: f64) -> Vec<Polyline> {
    if segs.is_empty() {
        return Vec::new();
    }
    let mut adj = Adjacency {
        keys: Vec::new(),
        index: HashMap::new(),
        lists: Vec::new(),
    };
    for row in segs.rows() {
        let a = adj.id(Key::new(row[0], row[1], tol));
        let b = adj.id(Key::new(row[2], row[3], tol));
        adj.lists[a].push(b);
        adj.lists[b].push(a);
    }
    let mut visited: HashMap<(usize, usize), bool> = HashMap::new();
    let mut polylines: Vec<Polyline> = Vec::new();
    let n = adj.keys.len();

    let to_poly = |path: &[usize], adj: &Adjacency| -> Polyline {
        let flat: Vec<f32> = path
            .iter()
            .flat_map(|&i| [adj.keys[i].0 as f32, adj.keys[i].1 as f32])
            .collect();
        Array2::from_shape_vec((path.len(), 2), flat).unwrap()
    };

    fn pop_edge(
        adj: &mut Adjacency,
        visited: &mut HashMap<(usize, usize), bool>,
        u: usize,
    ) -> Option<usize> {
        while let Some(v) = adj.lists[u].pop() {
            let e = if adj.keys[u].lt(&adj.keys[v]) {
                (u, v)
            } else {
                (v, u)
            };
            if !visited.get(&e).copied().unwrap_or(false) {
                visited.insert(e, true);
                return Some(v);
            }
        }
        None
    }

    let mut starts: Vec<usize> = (0..n).filter(|&i| adj.lists[i].len() != 2).collect();
    if starts.is_empty() {
        starts = vec![0];
    }
    let mut seen = vec![false; n];
    for s in starts {
        if seen[s] && adj.lists[s].is_empty() {
            continue;
        }
        let mut path = vec![s];
        let mut u = s;
        while let Some(v) = pop_edge(&mut adj, &mut visited, u) {
            path.push(v);
            seen[u] = true;
            u = v;
        }
        if path.len() >= 2 {
            polylines.push(to_poly(&path, &adj));
        }
    }
    for u in 0..n {
        while let Some(v) = pop_edge(&mut adj, &mut visited, u) {
            let mut path = vec![u, v];
            let mut w = v;
            while let Some(nxt) = pop_edge(&mut adj, &mut visited, w) {
                path.push(nxt);
                w = nxt;
            }
            if path.len() >= 2 {
                polylines.push(to_poly(&path, &adj));
            }
        }
    }
    polylines
}

fn norm2(dx: f32, dy: f32) -> f32 {
    (dx * dx + dy * dy).sqrt()
}

/// `douglas_peucker(poly, tol)`.
pub fn douglas_peucker(poly: &ArrayView2<'_, f32>, tol: f32) -> Polyline {
    let n = poly.nrows();
    if n <= 2 || tol <= 0.0 {
        return poly.to_owned();
    }
    let closed = norm2(
        poly[[0, 0]] - poly[[n - 1, 0]],
        poly[[0, 1]] - poly[[n - 1, 1]],
    ) < 1e-9;
    let open: Array2<f32> = if closed {
        poly.slice(ndarray::s![..n - 1, ..]).to_owned()
    } else {
        poly.to_owned()
    };
    let m = open.nrows();
    if closed && m <= 2 {
        let mut out = open.clone();
        out.push_row(open.row(0)).unwrap();
        return out;
    }
    let mut keep = vec![false; m];
    keep[0] = true;
    keep[m - 1] = true;
    let mut stack = vec![(0usize, m - 1)];
    while let Some((i0, i1)) = stack.pop() {
        if i1 <= i0 + 1 {
            continue;
        }
        let (ax, ay) = (open[[i0, 0]], open[[i0, 1]]);
        let (bx, by) = (open[[i1, 0]], open[[i1, 1]]);
        let (sx, sy) = (bx - ax, by - ay);
        let seg2 = (sx * sx + sy * sy) as f64;
        let mut dists = Vec::with_capacity(i1 - i0 - 1);
        for j in i0 + 1..i1 {
            let (px, py) = (open[[j, 0]] - ax, open[[j, 1]] - ay);
            let d = if seg2 <= 0.0 {
                norm2(px, py)
            } else {
                let t = ((px * sx + py * sy) as f64 / seg2).clamp(0.0, 1.0) as f32;
                let (qx, qy) = (ax + t * sx, ay + t * sy);
                norm2(open[[j, 0]] - qx, open[[j, 1]] - qy)
            };
            dists.push(d);
        }
        let mut j_rel = 0usize;
        for (k, &d) in dists.iter().enumerate() {
            if d > dists[j_rel] {
                j_rel = k;
            }
        }
        let dmax = dists[j_rel];
        let j = i0 + 1 + j_rel;
        if dmax > tol {
            keep[j] = true;
            stack.push((i0, j));
            stack.push((j, i1));
        }
    }
    let rows: Vec<usize> = (0..m).filter(|&i| keep[i]).collect();
    let mut out = open.select(ndarray::Axis(0), &rows);
    if closed {
        let first = out.row(0).to_owned();
        out.push_row(first.view()).unwrap();
    }
    out
}

/// `polygon_area(poly)`: the signed shoelace area over consecutive points, in
/// float64 with an exactly rounded sum (Python's `math.fsum`), as the Python
/// package computes it since 2026-10-21 so that no host's BLAS order matters.
pub fn polygon_area(poly: &ArrayView2<'_, f32>) -> f64 {
    let n = poly.nrows();
    if n < 2 {
        return 0.0;
    }
    let terms: Vec<f64> = (0..n - 1)
        .map(|i| {
            let (x0, y0) = (poly[[i, 0]] as f64, poly[[i, 1]] as f64);
            let (x1, y1) = (poly[[i + 1, 0]] as f64, poly[[i + 1, 1]] as f64);
            x0 * y1 - y0 * x1
        })
        .collect();
    0.5 * npexact::fsum::fsum(&terms)
}

/// `close_and_filter_polylines(polys, close_tol_px=1.0, min_area_px2=0.0)`.
pub fn close_and_filter_polylines(
    polys: &[Polyline],
    close_tol_px: f32,
    min_area_px2: f64,
) -> Vec<Polyline> {
    let mut out = Vec::new();
    for poly in polys {
        let n = poly.nrows();
        if n < 2 {
            continue;
        }
        let mut p = poly.clone();
        if norm2(
            poly[[0, 0]] - poly[[n - 1, 0]],
            poly[[0, 1]] - poly[[n - 1, 1]],
        ) > close_tol_px
        {
            let first = p.row(0).to_owned();
            p.push_row(first.view()).unwrap();
        }
        if p.nrows() < 4 {
            continue;
        }
        if polygon_area(&p.view()).abs() < min_area_px2 {
            continue;
        }
        out.push(p);
    }
    out
}

/// `point_in_polygon(pt, poly)` (crossing number).
pub fn point_in_polygon(pt: (f64, f64), poly: &ArrayView2<'_, f32>) -> bool {
    let (x, y) = pt;
    let mut cnt = 0;
    for i in 0..poly.nrows().saturating_sub(1) {
        let (x0, y0) = (poly[[i, 0]] as f64, poly[[i, 1]] as f64);
        let (x1, y1) = (poly[[i + 1, 0]] as f64, poly[[i + 1, 1]] as f64);
        if (y0 > y) != (y1 > y) {
            let t = (y - y0) / (y1 - y0 + 1e-20);
            let xr = x0 + t * (x1 - x0);
            if xr > x {
                cnt ^= 1;
            }
        }
    }
    cnt != 0
}

/// A nested region: an outer ring and its holes.
#[derive(Debug, Clone)]
pub struct Region {
    pub outer: Polyline,
    pub holes: Vec<Polyline>,
}

fn reversed(r: &Polyline) -> Polyline {
    let mut v = r.clone();
    v.invert_axis(ndarray::Axis(0));
    v
}

/// `build_regions_with_holes(rings)`: rings grouped into nested regions by
/// centroid-in-polygon tests, nesting depth deciding outers (even) and holes
/// (odd, reversed to CW); regions sorted by area.
#[allow(clippy::needless_range_loop, clippy::map_entry)]
pub fn build_regions_with_holes(rings: &[Polyline]) -> Vec<Region> {
    if rings.is_empty() {
        return Vec::new();
    }
    struct Rec {
        ring: Polyline,
        area: f64,
        centroid: (f64, f64),
    }
    let mut recs: Vec<Rec> = rings
        .iter()
        .map(|r| {
            let mut area = polygon_area(&r.view());
            let rr = if area < 0.0 {
                area = -area;
                reversed(r)
            } else {
                r.clone()
            };
            // float64, exactly rounded, as the Python package's math.fsum centroid
            let n = rr.nrows() - 1;
            let xs: Vec<f64> = rr
                .slice(ndarray::s![..n, 0])
                .iter()
                .map(|&v| v as f64)
                .collect();
            let ys: Vec<f64> = rr
                .slice(ndarray::s![..n, 1])
                .iter()
                .map(|&v| v as f64)
                .collect();
            let cx = npexact::fsum::fsum(&xs) / n as f64;
            let cy = npexact::fsum::fsum(&ys) / n as f64;
            Rec {
                ring: rr,
                area,
                centroid: (cx, cy),
            }
        })
        .collect();
    recs.sort_by(|a, b| {
        a.area
            .partial_cmp(&b.area)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = recs.len();
    let mut parents: Vec<isize> = vec![-1; n];
    for i in 0..n {
        let c = recs[i].centroid;
        let mut best: isize = -1;
        let mut best_area = 1e30;
        for j in i + 1..n {
            if recs[j].area < best_area && point_in_polygon(c, &recs[j].ring.view()) {
                best = j as isize;
                best_area = recs[j].area;
            }
        }
        parents[i] = best;
    }
    // Nesting depth decides: an unparented ring, or one inside a hole (even
    // depth), is an outer; a ring directly inside an outer (odd depth) is that
    // outer's hole, reversed to CW. Python's dict keeps insertion order.
    let mut order: Vec<usize> = Vec::new();
    let mut regions: HashMap<usize, Region> = HashMap::new();
    for (i, &p) in parents.iter().enumerate() {
        let mut depth = 0;
        let mut q = p;
        while q != -1 {
            depth += 1;
            q = parents[q as usize];
        }
        if depth % 2 == 0 {
            if !regions.contains_key(&i) {
                order.push(i);
                regions.insert(
                    i,
                    Region {
                        outer: recs[i].ring.clone(),
                        holes: Vec::new(),
                    },
                );
            }
        } else {
            let p = p as usize;
            if !regions.contains_key(&p) {
                order.push(p);
                regions.insert(
                    p,
                    Region {
                        outer: recs[p].ring.clone(),
                        holes: Vec::new(),
                    },
                );
            }
            regions
                .get_mut(&p)
                .unwrap()
                .holes
                .push(reversed(&recs[i].ring));
        }
    }
    let mut keyed: Vec<(usize, Region)> = order
        .into_iter()
        .map(|k| (k, regions.remove(&k).unwrap()))
        .collect();
    keyed.sort_by(|a, b| {
        polygon_area(&a.1.outer.view())
            .partial_cmp(&polygon_area(&b.1.outer.view()))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    keyed.into_iter().map(|(_, r)| r).collect()
}

/// `_ring_to_cubic_beziers(ring, tension)`: Catmull–Rom control points.
pub fn ring_to_cubic_beziers(ring: &ArrayView2<'_, f32>, tension: f32) -> Vec<[[f32; 2]; 4]> {
    let n = ring.nrows().saturating_sub(1);
    if n < 2 {
        return Vec::new();
    }
    let s = (1.0 - tension) / 6.0;
    let p = |i: usize| [ring[[i % n, 0]], ring[[i % n, 1]]];
    (0..n)
        .map(|i| {
            let p0 = p((i + n - 1) % n);
            let p1 = p(i);
            let p2 = p(i + 1);
            let p3 = p(i + 2);
            let c1 = [p1[0] + s * (p2[0] - p0[0]), p1[1] + s * (p2[1] - p0[1])];
            let c2 = [p2[0] - s * (p3[0] - p1[0]), p2[1] - s * (p3[1] - p1[1])];
            [p1, c1, c2, p2]
        })
        .collect()
}

/// Smoothing mode of [`svg_regions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Curve {
    None,
    Cubic,
}

/// `write_svg_regions(...)` as text.
///
/// With `Curve::Cubic` the Python code smooths holes but, through a leftover
/// assignment, writes the outer ring as a polyline; that is reproduced. With
/// `Curve::None` the Python code raises `UnboundLocalError`; this port writes
/// polylines for both, which is what it evidently intends.
#[allow(clippy::too_many_arguments)]
pub fn svg_regions(
    regions: &[Region],
    w: usize,
    h: usize,
    stroke_width: f32,
    fill: Option<&str>,
    curve: Curve,
    tension: f32,
) -> String {
    let hf = (h as f32) - 1.0;
    let fmt_xy = |p: [f32; 2]| (p[0], hf - p[1]);
    let poly_path = |r: &Polyline| -> String {
        let (x, y) = fmt_xy([r[[0, 0]], r[[0, 1]]]);
        let mut cmds = vec![format!("M {x:.3} {y:.3}")];
        for i in 1..r.nrows() {
            let (x, y) = fmt_xy([r[[i, 0]], r[[i, 1]]]);
            cmds.push(format!("L {x:.3} {y:.3}"));
        }
        cmds.push("Z".into());
        cmds.join(" ")
    };
    let cubic_path = |r: &Polyline| -> String {
        let bez = ring_to_cubic_beziers(&r.view(), tension);
        if bez.is_empty() {
            return poly_path(r);
        }
        let (x0, y0) = fmt_xy(bez[0][0]);
        let mut cmds = vec![format!("M {x0:.3} {y0:.3}")];
        for b in &bez {
            let (x1, y1) = fmt_xy(b[1]);
            let (x2, y2) = fmt_xy(b[2]);
            let (x3, y3) = fmt_xy(b[3]);
            cmds.push(format!("C {x1:.3} {y1:.3} {x2:.3} {y2:.3} {x3:.3} {y3:.3}"));
        }
        cmds.push("Z".into());
        cmds.join(" ")
    };
    let mut s = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\"><g fill=\"{}\" stroke=\"black\" stroke-width=\"{stroke_width:.3}\" fill-rule=\"evenodd\">",
        fill.unwrap_or("none")
    );
    for reg in regions {
        let mut d = poly_path(&reg.outer);
        for hole in &reg.holes {
            d.push(' ');
            d.push_str(&match curve {
                Curve::Cubic => cubic_path(hole),
                Curve::None => poly_path(hole),
            });
        }
        write!(s, "<path d=\"{d}\" />").unwrap();
    }
    s.push_str("</g></svg>");
    s
}

#[allow(clippy::too_many_arguments)]
pub fn write_svg_regions(
    path: impl AsRef<std::path::Path>,
    regions: &[Region],
    w: usize,
    h: usize,
    stroke_width: f32,
    fill: Option<&str>,
    curve: Curve,
    tension: f32,
) -> Result<()> {
    std::fs::write(
        path.as_ref(),
        svg_regions(regions, w, h, stroke_width, fill, curve, tension),
    )
    .map_err(|e| Error::Invalid(format!("{}: {e}", path.as_ref().display())))?;
    Ok(())
}

/// `trace_bitmap(mask_u8, simplify_tol_px=0.75, min_area_px2=4.0, chain_tol_px=0.0)`:
/// closed rings from a binary mask, entirely on the CPU.
pub fn trace_bitmap(
    mask: &ArrayView2<'_, u8>,
    simplify_tol_px: f32,
    min_area_px2: f64,
    chain_tol_px: f64,
) -> Vec<Polyline> {
    let field = mask.mapv(|v| v as f32);
    let segs = marching_squares_cpu(&field.view(), 0.5);
    let polys = chain_segments_to_polylines(&segs.view(), chain_tol_px);
    let simp: Vec<Polyline> = polys
        .iter()
        .filter(|p| p.nrows() >= 3)
        .map(|p| douglas_peucker(&p.view(), simplify_tol_px))
        .collect();
    close_and_filter_polylines(&simp, 1.0, min_area_px2)
}

/// `build_circle_mask(w, h, r)`.
pub fn build_circle_mask(w: usize, h: usize, r: usize) -> Array2<u8> {
    let (cx, cy) = ((w / 2) as i64, (h / 2) as i64);
    Array2::from_shape_fn((h, w), |(y, x)| {
        let (dx, dy) = (x as i64 - cx, y as i64 - cy);
        (dx * dx + dy * dy <= (r * r) as i64) as u8
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edt_of_single_seed() {
        let f = [1e12, 1e12, 0.0, 1e12, 1e12];
        assert_eq!(edt_1d(&f), vec![4.0, 1.0, 0.0, 1.0, 4.0]);
    }

    #[test]
    fn circle_traces_to_rings() {
        let m = build_circle_mask(48, 40, 15);
        let sdf = sdf_from_mask_cpu(&m.view());
        assert!(sdf[[20, 24]] > 10.0 && sdf[[0, 0]] < 0.0);
        let rings = trace_bitmap(&m.view(), 0.75, 4.0, 0.25);
        assert_eq!(rings.len(), 1);
        let area = polygon_area(&rings[0].view()).abs();
        assert!((area - std::f64::consts::PI * 225.0).abs() < 60.0, "{area}");
        let regions = build_regions_with_holes(&rings);
        let svg = svg_regions(&regions, 48, 40, 1.0, None, Curve::Cubic, 0.0);
        assert!(svg.starts_with("<svg xmlns"));
        assert!(svg.contains("<path d=\"M "));
    }
}

// ----------------------------- Filling traced regions ------------------------

/// `ring_to_polygon(ring, curve, tension, segments)`: a closed ring as a polygon,
/// its vertices as they are (`curve = "none"`) or its Catmull-Rom cubics
/// (`ring_to_cubic_beziers`) each flattened into `segments` steps by
/// `paths::flatten_cubic` (`curve = "cubic"`).
pub fn ring_to_polygon(
    ring: &ArrayView2<'_, f32>,
    curve: &str,
    tension: f32,
    segments: usize,
) -> Result<Vec<[f64; 2]>> {
    match curve {
        "none" => Ok(ring
            .rows()
            .into_iter()
            .take(ring.nrows().saturating_sub(1))
            .map(|r| [r[0] as f64, r[1] as f64])
            .collect()),
        "cubic" => {
            let mut poly = Vec::new();
            for [p1, c1, c2, p2] in ring_to_cubic_beziers(ring, tension) {
                let pts = super::paths::flatten_cubic(
                    [p1[0] as f64, p1[1] as f64],
                    [c1[0] as f64, c1[1] as f64],
                    [c2[0] as f64, c2[1] as f64],
                    [p2[0] as f64, p2[1] as f64],
                    segments,
                );
                poly.extend_from_slice(&pts[..pts.len() - 1]);
            }
            Ok(poly)
        }
        other => Err(Error::Invalid(format!(
            "curve must be 'none' or 'cubic', not {other:?}"
        ))),
    }
}

/// `fill_polygons(polys, shape, ink, paper)`: even-odd scanline fill of closed
/// polygons (float64 vertices, implicitly closed) as an `(h, w)` float32 raster
/// of `paper` with `ink` inside. Pixel (row, col) is sampled at (x = col,
/// y = row); an edge counts on scanline y when `min(y0, y1) <= y < max(y0, y1)`,
/// its crossing is `x0 + (y - y0) * (x1 - x0) / (y1 - y0)`, and each pair of
/// sorted crossings `[xa, xb)` covers the pixels with `xa <= col < xb`. The
/// same IEEE float64 operations as the Python package's, so the pixels match
/// on every host.
pub fn fill_polygons(
    polys: &[Vec<[f64; 2]>],
    shape: (usize, usize),
    ink: f32,
    paper: f32,
) -> Array2<f32> {
    let (h, w) = shape;
    let mut out = Array2::<f32>::from_elem((h, w), paper);
    let mut edges: Vec<[f64; 4]> = Vec::new();
    for poly in polys {
        let n = poly.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let [x0, y0] = poly[i];
            let [x1, y1] = poly[(i + 1) % n];
            if y0 != y1 {
                edges.push([x0, y0, x1, y1]);
            }
        }
    }
    if edges.is_empty() {
        return out;
    }
    let ylo: Vec<f64> = edges.iter().map(|e| e[1].min(e[3])).collect();
    let yhi: Vec<f64> = edges.iter().map(|e| e[1].max(e[3])).collect();
    let y_first = (ylo.iter().copied().fold(f64::INFINITY, f64::min).ceil() as i64).max(0) as usize;
    let y_last = (yhi.iter().copied().fold(f64::NEG_INFINITY, f64::max).ceil() as i64)
        .min(h as i64)
        .max(0) as usize;
    let mut xc: Vec<f64> = Vec::new();
    for y in y_first..y_last {
        let yf = y as f64;
        xc.clear();
        for (k, e) in edges.iter().enumerate() {
            if ylo[k] <= yf && yf < yhi[k] {
                let [x0, y0, x1, y1] = *e;
                xc.push(x0 + (yf - y0) * (x1 - x0) / (y1 - y0));
            }
        }
        if xc.is_empty() {
            continue;
        }
        xc.sort_by(|a, b| a.total_cmp(b));
        let mut k = 0;
        while k + 1 < xc.len() {
            let a = (xc[k].ceil() as i64).max(0);
            let b = (xc[k + 1].ceil() as i64).min(w as i64);
            if a < b {
                out.slice_mut(ndarray::s![y, a as usize..b as usize])
                    .fill(ink);
            }
            k += 2;
        }
    }
    out
}

/// `fill_regions(regions, shape, ...)`: nested regions rasterised as `ink` on
/// a `paper` canvas of `(h, w)`: every ring, outer or hole, flattened by
/// [`ring_to_polygon`] and filled together with the even-odd rule
/// ([`fill_polygons`]), so the nesting takes care of the holes. (Until
/// 2026-10-21 each region was triangulated with earcut and drawn through the
/// graphics4 `mesh_triangles` kernel; the C++ and Rust earcut libraries
/// triangulate differently and the kernel's coverage depended on that.)
pub fn fill_regions(
    regions: &[Region],
    shape: (usize, usize),
    curve: &str,
    tension: f32,
    segments: usize,
    ink: f32,
    paper: f32,
) -> Result<Array2<f32>> {
    let mut polys: Vec<Vec<[f64; 2]>> = Vec::new();
    for region in regions {
        for ring in std::iter::once(&region.outer).chain(region.holes.iter()) {
            let poly = ring_to_polygon(&ring.view(), curve, tension, segments)?;
            if poly.len() >= 3 {
                polys.push(poly);
            }
        }
    }
    Ok(fill_polygons(&polys, shape, ink, paper))
}

/// Parameters of [`trace_mask`], the potrace replacement.
#[derive(Debug, Clone)]
pub struct TraceParams {
    pub iso: f32,
    pub simplify_tol_px: f32,
    pub min_area_px2: f64,
    pub chain_tol_px: f64,
    pub curve: String,
    pub tension: f32,
    pub segments: usize,
    /// `"gpu"` (jump flood) or `"cpu"` (exact EDT)
    pub engine: String,
}

impl Default for TraceParams {
    fn default() -> Self {
        TraceParams {
            iso: 0.0,
            simplify_tol_px: 0.75,
            min_area_px2: 4.0,
            chain_tol_px: 0.0,
            curve: "cubic".to_string(),
            tension: 0.0,
            segments: 8,
            engine: "gpu".to_string(),
        }
    }
}

/// `trace_mask(mask, ...)`: a binary mask (nonzero = ink) traced into smooth
/// filled shapes, as a raster of 1.0 (paper) and 0.0 (ink): the signed distance
/// field (GPU jump flood or CPU EDT) traced at `iso`, chained, simplified,
/// closed and filtered by area, nested, and filled by [`fill_regions`].
pub fn trace_mask(
    device: &Device,
    mask: &ArrayView2<'_, u8>,
    params: &TraceParams,
) -> Result<Array2<f32>> {
    let mask: Array2<u8> = mask.mapv(|v| (v != 0) as u8);
    let (h, w) = mask.dim();
    let segs = match params.engine.as_str() {
        "gpu" => {
            TrueTrace::load(device)?
                .trace_segments(&mask.view(), params.iso)?
                .0
        }
        "cpu" => {
            let sdf = sdf_from_mask_cpu(&mask.view());
            let iso = off_field_iso(sdf.as_slice().expect("contiguous field"), params.iso);
            sort_segments(&marching_squares_cpu(&sdf.view(), iso).view())
        }
        other => {
            return Err(Error::Invalid(format!(
                "engine must be 'gpu' or 'cpu', not {other:?}"
            )))
        }
    };
    let polys = chain_segments_to_polylines(&segs.view(), params.chain_tol_px);
    let simp: Vec<Polyline> = polys
        .iter()
        .filter(|p| p.nrows() >= 3)
        .map(|p| douglas_peucker(&p.view(), params.simplify_tol_px))
        .collect();
    let rings = close_and_filter_polylines(&simp, 1.0, params.min_area_px2);
    let regions = build_regions_with_holes(&rings);
    fill_regions(
        &regions,
        (h, w),
        &params.curve,
        params.tension,
        params.segments,
        0.0,
        1.0,
    )
}

#[cfg(test)]
mod region_tests {
    use super::*;

    #[test]
    fn block_with_hole_and_speckle_nest_as_one_region_plus_one() {
        let mut mask = Array2::<u8>::zeros((64, 80));
        mask.slice_mut(ndarray::s![10..50, 12..68]).fill(1);
        mask.slice_mut(ndarray::s![22..38, 30..50]).fill(0);
        mask.slice_mut(ndarray::s![55..58, 5..8]).fill(1);
        let rings = trace_bitmap(&mask.view(), 0.0, 0.0, 0.25);
        assert_eq!(rings.len(), 3, "rings");
        let regions = build_regions_with_holes(&rings);
        let holes: Vec<usize> = regions.iter().map(|r| r.holes.len()).collect();
        assert_eq!(regions.len(), 2, "regions: {holes:?}");
        assert_eq!(holes.iter().sum::<usize>(), 1);
    }
}
