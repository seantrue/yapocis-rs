//! `graphics.cl`: rasterise lines, rectangles, circles, triangles, ellipses
//! and sprites into a one-channel float image.
//!
//! Every kernel is one work item per pixel: it reads its pixel, blends each
//! primitive over it in order, and writes it back, so pixels run in parallel
//! and primitives in sequence. The image is addressed `i = x * height + y`.
//!
//! What a primitive covers is the same here and in `graphics4.cl`; only the
//! blending differs (value/alpha here, RGBA "over" there). So the coverage
//! tests live in this module as `Cover`-returning functions, shared with
//! [`super::graphics4`].

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("lines", lines),
    ("rects", rects),
    ("circles", circles),
    ("triangles", triangles),
    ("mesh_triangles", mesh_triangles),
    ("ellipses", ellipses),
    ("composite", composite),
];

// ---------------------------------------------------------------------------
// OpenCL built-ins
// ---------------------------------------------------------------------------

/// OpenCL `clamp`, which is `fmin(fmax(x, lo), hi)`: a NaN becomes `lo`.
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

/// OpenCL `smoothstep`; the kernels also call it with `edge0 > edge1`.
pub(super) fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// OpenCL `mix`.
fn mix(x: f32, y: f32, a: f32) -> f32 {
    x + (y - x) * a
}

/// `dxdy2(a, b)`: squared distance.
fn dxdy2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let (dx, dy) = (ax - bx, ay - by);
    dx * dx + dy * dy
}

/// `dist_to_segment(p, a, b)`.
fn dist_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let pa = [p[0] - a[0], p[1] - a[1]];
    let ba = [b[0] - a[0], b[1] - a[1]];
    let h = clamp(
        (pa[0] * ba[0] + pa[1] * ba[1]) / (ba[0] * ba[0] + ba[1] * ba[1]),
        0.0,
        1.0,
    );
    let (dx, dy) = (pa[0] - h * ba[0], pa[1] - h * ba[1]);
    (dx * dx + dy * dy).sqrt()
}

// ---------------------------------------------------------------------------
// Coverage, shared with graphics4
// ---------------------------------------------------------------------------

/// How one primitive covers one pixel.
#[derive(Clone, Copy)]
pub(super) enum Cover {
    /// Not drawn.
    Skip,
    /// The hard-edge path: drawn at the primitive's alpha.
    Hard,
    /// The anti-aliased path: drawn at the primitive's alpha times this weight.
    Soft(f32),
}

/// Line `c = [x0, y0, x1, y1]` with width `rdist` (the threshold is
/// `rdist / 2 + 1/2`, computed in double as the `.5` literals make it).
pub(super) fn line(c: &[f32], rdist: f32, aa: f32, x: f32, y: f32) -> Cover {
    let threshold = (rdist as f64 * 0.5 + 0.5) as f32;
    let threshold2 = threshold * threshold;
    let (sx, sy, ex, ey) = (c[0], c[1], c[2], c[3]);
    let linemag2 = dxdy2(ex, ey, sx, sy);
    let (dx, dy) = (ex - sx, ey - sy);
    let (ux, uy) = ((x - sx) * dx, (y - sy) * dy);
    // A zero-length line is a dot at its start.
    let u = if linemag2 > 0.0 {
        (ux + uy) / linemag2
    } else {
        0.0
    };
    let dist2 = dxdy2(x, y, sx + u * dx, sy + u * dy);
    if !(0.0..=1.0).contains(&u) {
        Cover::Skip
    } else if aa > 0.0 {
        Cover::Soft(smoothstep(threshold + aa, threshold - aa, dist2.sqrt()))
    } else if dist2 < threshold2 {
        Cover::Hard
    } else {
        Cover::Skip
    }
}

/// Rectangle `c = [x0, y0, x1, y1]`, corners in either order, edges inclusive.
pub(super) fn rect(c: &[f32], aa: f32, x: f32, y: f32) -> Cover {
    let (l, r) = (c[0].min(c[2]), c[0].max(c[2]));
    let (b, t) = (c[1].min(c[3]), c[1].max(c[3]));
    let inside = l <= x && x <= r && b <= y && y <= t;
    if !inside {
        Cover::Skip
    } else if aa > 0.0 {
        let edge_dist = (x - l).min(r - x).min((y - b).min(t - y));
        Cover::Soft(smoothstep(0.0, aa, edge_dist))
    } else {
        Cover::Hard
    }
}

/// Circle `c = [cx, cy, radius]`.
pub(super) fn circle(c: &[f32], aa: f32, x: f32, y: f32) -> Cover {
    let (cx, cy, radius) = (c[0], c[1], c[2]);
    let dist = dxdy2(cx, cy, x, y).sqrt();
    if aa > 0.0 {
        Cover::Soft(smoothstep(radius + aa, radius - aa, dist))
    } else if dist <= radius {
        Cover::Hard
    } else {
        Cover::Skip
    }
}

/// `signex(p1, p2, p3)`.
fn signex(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> f32 {
    (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
}

/// The three vertices of `c = [x1, y1, x2, y2, x3, y3]`.
fn vertices(c: &[f32]) -> [[f32; 2]; 3] {
    [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]]]
}

/// The anti-aliased weight inside a triangle: by distance to its nearest edge.
fn triangle_soft(p: [f32; 2], [v1, v2, v3]: [[f32; 2]; 3], aa: f32) -> Cover {
    let edge_dist = dist_to_segment(p, v1, v2)
        .min(dist_to_segment(p, v2, v3))
        .min(dist_to_segment(p, v3, v1));
    Cover::Soft(smoothstep(0.0, aa, edge_dist))
}

/// Triangle `c`; pixels on any edge are inside.
pub(super) fn triangle(c: &[f32], aa: f32, x: f32, y: f32) -> Cover {
    let p = [x, y];
    let [v1, v2, v3] = vertices(c);
    let d = [signex(p, v1, v2), signex(p, v2, v3), signex(p, v3, v1)];
    let has_neg = d.iter().any(|&d| d < 0.0);
    let has_pos = d.iter().any(|&d| d > 0.0);
    if has_neg && has_pos {
        Cover::Skip
    } else if aa > 0.0 {
        triangle_soft(p, [v1, v2, v3], aa)
    } else {
        Cover::Hard
    }
}

/// `tri_edge`: exactly antisymmetric in `a` and `b` (computed with the
/// endpoints in a canonical order), so two triangles sharing an edge get
/// exactly opposite values on it.
fn tri_edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    let swap = b[0] < a[0] || (b[0] == a[0] && b[1] < a[1]);
    let (s, e) = if swap { (b, a) } else { (a, b) };
    let v = (e[0] - s[0]) * (p[1] - s[1]) - (e[1] - s[1]) * (p[0] - s[0]);
    if swap {
        -v
    } else {
        v
    }
}

/// `tri_owns`: which direction of a shared edge owns the pixels on it.
fn tri_owns(dx: f32, dy: f32) -> bool {
    dy > 0.0 || (dy == 0.0 && dx > 0.0)
}

/// `tri_inside`: the fill rule for meshes; a pixel on an edge shared by two
/// triangles belongs to exactly one of them, and degenerate triangles are empty.
fn tri_inside(p: [f32; 2], [v1, v2, v3]: [[f32; 2]; 3]) -> bool {
    let area = tri_edge(v1, v2, v3);
    if area == 0.0 {
        return false;
    }
    let s = if area > 0.0 { 1.0 } else { -1.0 };
    let edge = |a: [f32; 2], b: [f32; 2]| {
        let e = s * tri_edge(a, b, p);
        e > 0.0 || (e == 0.0 && tri_owns(s * (b[0] - a[0]), s * (b[1] - a[1])))
    };
    edge(v1, v2) && edge(v2, v3) && edge(v3, v1)
}

/// Mesh triangle `c`, with the [`tri_inside`] fill rule.
pub(super) fn mesh_triangle(c: &[f32], aa: f32, x: f32, y: f32) -> Cover {
    let p = [x, y];
    let v = vertices(c);
    if !tri_inside(p, v) {
        Cover::Skip
    } else if aa > 0.0 {
        triangle_soft(p, v, aa)
    } else {
        Cover::Hard
    }
}

/// Ellipse `c = [cx, cy, ax, ay]` (centre and semi-axes).
pub(super) fn ellipse(c: &[f32], aa: f32, x: f32, y: f32) -> Cover {
    let (dx, dy) = ((x - c[0]) / c[2], (y - c[1]) / c[3]);
    let d = (dx * dx + dy * dy).sqrt();
    if aa > 0.0 {
        let avg_radius = (c[2] + c[3]) * 0.5;
        Cover::Soft(smoothstep(aa, -aa, (d - 1.0) * avg_radius))
    } else if d <= 1.0 {
        Cover::Hard
    } else {
        Cover::Skip
    }
}

/// The floats each primitive takes in `coords`, by kernel.
pub(super) fn coords_per(symbol: &str) -> usize {
    match symbol {
        "circles" => 3,
        "triangles" | "mesh_triangles" => 6,
        _ => 4, // lines, rects, ellipses
    }
}

/// Pixel `i`'s position: `x = i / height`, `y = i % height`.
pub(super) fn position(i: usize, height: usize) -> (f32, f32) {
    ((i / height) as f32, (i % height) as f32)
}

// ---------------------------------------------------------------------------
// The kernels
// ---------------------------------------------------------------------------

/// `value * alpha + color * (1.0 - alpha)`; the `1.0` is a double literal.
fn blend(value: f32, alpha: f32, color: f32) -> f32 {
    ((value * alpha) as f64 + color as f64 * (1.0 - alpha as f64)) as f32
}

/// The same with `1.0f`, as `ellipses` writes it.
fn blend_f32(value: f32, alpha: f32, color: f32) -> f32 {
    value * alpha + color * (1.0 - alpha)
}

/// Draw `n` primitives over the image: `cover(k, x, y)` says how primitive
/// `k` covers pixel `(x, y)`, and `colors` holds `(value, alpha)` per primitive.
fn draw(
    image: &mut [f32],
    height: usize,
    n: usize,
    colors: &[f32],
    blend: fn(f32, f32, f32) -> f32,
    cover: impl Fn(usize, f32, f32) -> Cover + Sync,
) {
    par::map_chunks(image, 1, |i, pixel| {
        let (x, y) = position(i, height);
        let mut color = pixel[0];
        for k in 0..n {
            let (value, alpha) = (colors[2 * k], colors[2 * k + 1]);
            match cover(k, x, y) {
                Cover::Skip => {}
                Cover::Hard => color = blend(value, alpha, color),
                Cover::Soft(t) => color = mix(color, blend(value, alpha, color), t * alpha),
            }
        }
        pixel[0] = color;
    });
}

/// `lines(int width, int height, float* a, int nelements, float2* coords,
/// float* rdist, float* aa_widths, float2* colors)`.
fn lines(a: &Args) {
    let height = a.int(1) as usize;
    let n = a.int(3) as usize;
    let coords = a.input::<f32>(4);
    let rdist = a.input::<f32>(5);
    let aa = a.input::<f32>(6);
    let colors = a.input::<f32>(7);
    let mut image = a.output::<f32>(2);
    draw(
        &mut image[..a.global_size()],
        height,
        n,
        &colors,
        blend,
        |k, x, y| line(&coords[4 * k..], rdist[k], aa[k], x, y),
    );
}

/// The kernels that take `(width, height, a, n, coords, aa_widths, colors)`.
fn shapes(a: &Args, symbol: &str, cover: fn(&[f32], f32, f32, f32) -> Cover) {
    let height = a.int(1) as usize;
    let n = a.int(3) as usize;
    let coords = a.input::<f32>(4);
    let aa = a.input::<f32>(5);
    let colors = a.input::<f32>(6);
    let mut image = a.output::<f32>(2);
    let per = coords_per(symbol);
    let blend = if symbol == "ellipses" {
        blend_f32
    } else {
        blend
    };
    draw(
        &mut image[..a.global_size()],
        height,
        n,
        &colors,
        blend,
        |k, x, y| cover(&coords[per * k..], aa[k], x, y),
    );
}

/// `rects(int width, int height, float* a, int nlines, float4* coords, float* aa_widths, float2* colors)`.
fn rects(a: &Args) {
    shapes(a, "rects", rect)
}

/// `circles(int width, int height, float* a, int nlines, float* coords, float* aa_widths, float2* colors)`.
fn circles(a: &Args) {
    shapes(a, "circles", circle)
}

/// `triangles(int width, int height, float* a, int nlines, float2* coords, float* aa_widths, float2* colors)`.
fn triangles(a: &Args) {
    shapes(a, "triangles", triangle)
}

/// `mesh_triangles(int width, int height, float* a, int nlines, float2* coords, float* aa_widths, float2* colors)`.
fn mesh_triangles(a: &Args) {
    shapes(a, "mesh_triangles", mesh_triangle)
}

/// `ellipses(int width, int height, float* a, int nlines, float2* coords, float* aa_widths, float2* colors)`.
fn ellipses(a: &Args) {
    shapes(a, "ellipses", ellipse)
}

/// Where sprite `coord = [x, y, w, h]` (starting at `ipixel` in the pixel
/// buffer) puts pixel `(x, y)`: its index in the pixel buffer, or `None`
/// when the pixel is outside the sprite or the index outside the buffer.
pub(super) fn sprite_source(
    coord: &[i32],
    ipixel: i32,
    npixels: i32,
    x: i32,
    y: i32,
) -> Option<usize> {
    // lb = coord.s10 and yt = lb + coord.s32, so x pairs with coord[1], y with coord[0].
    let (left, bottom) = (coord[1], coord[0]);
    let (right, top) = (left + coord[3], bottom + coord[2]);
    let outside = x < left || y < bottom || x >= right || y >= top;
    let source = ipixel + (x - left) * coord[2] + (y - bottom);
    (!outside && 0 <= source && source < npixels).then_some(source as usize)
}

/// `composite(int width, int height, float* a, int n, int4* coords,
/// float2* pixels, int* ipixels, int npixels)`: blend sprites of
/// `(value, alpha)` pixels over the image.
fn composite(a: &Args) {
    let height = a.int(1) as usize;
    let n = a.int(3) as usize;
    let coords = a.input::<i32>(4);
    let pixels = a.input::<[f32; 2]>(5);
    let ipixels = a.input::<i32>(6);
    let npixels = a.int(7);
    let mut image = a.output::<f32>(2);
    par::map_chunks(&mut image[..a.global_size()], 1, |i, pixel| {
        let (x, y) = ((i / height) as i32, (i % height) as i32);
        let mut color = pixel[0];
        for j in 0..n {
            if let Some(s) = sprite_source(&coords[4 * j..], ipixels[j], npixels, x, y) {
                let [value, alpha] = pixels[s];
                color = blend(value, alpha, color);
            }
        }
        pixel[0] = color;
    });
}
