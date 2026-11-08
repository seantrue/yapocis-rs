//! `graphics4.cl`: the [`super::graphics`] primitives on an RGBA image
//! (`float4` per pixel), with RGBA colours composited Porter-Duff "over".
//!
//! Coverage is shared with `graphics.cl` (see `graphics::Cover`); the difference is
//! the blend. One work item per pixel, primitives in order.

use super::graphics::{
    circle, coords_per, ellipse, line, mesh_triangle, position, rect, sprite_source, triangle,
    Cover,
};
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

/// Porter-Duff "over": `rgba`'s colour at alpha `alpha` over `pixel`.
fn over(pixel: &mut [f32], rgba: &[f32], alpha: f32) {
    for c in 0..3 {
        pixel[c] = rgba[c] * alpha + pixel[c] * (1.0 - alpha);
    }
    pixel[3] = alpha + pixel[3] * (1.0 - alpha);
}

/// Draw `n` primitives over the image: `cover(k, x, y)` says how primitive
/// `k` covers pixel `(x, y)`, and `colors` holds an RGBA per primitive.
fn draw(
    image: &mut [f32],
    height: usize,
    n: usize,
    colors: &[f32],
    cover: impl Fn(usize, f32, f32) -> Cover + Sync,
) {
    par::map_chunks(image, 4, |i, pixel| {
        let (x, y) = position(i, height);
        for k in 0..n {
            let rgba = &colors[4 * k..4 * k + 4];
            match cover(k, x, y) {
                Cover::Skip => {}
                Cover::Hard => over(pixel, rgba, rgba[3]),
                Cover::Soft(t) => over(pixel, rgba, t * rgba[3]),
            }
        }
    });
}

/// `lines(int width, int height, float4* a, int nelements, float2* coords,
/// float* rdist, float* aa_widths, float4* colors)`.
fn lines(a: &Args) {
    let height = a.int(1) as usize;
    let n = a.int(3) as usize;
    let coords = a.input::<f32>(4);
    let rdist = a.input::<f32>(5);
    let aa = a.input::<f32>(6);
    let colors = a.input::<f32>(7);
    let mut image = a.output::<f32>(2);
    draw(
        &mut image[..4 * a.global_size()],
        height,
        n,
        &colors,
        |k, x, y| line(&coords[4 * k..], rdist[k], aa[k], x, y),
    );
}

/// The kernels that take `(width, height, a, n, coords, aa_widths, colors)`.
fn shapes(a: &Args, symbol: &str, cover: impl Fn(&[f32], f32, f32, f32) -> Cover + Sync) {
    let height = a.int(1) as usize;
    let n = a.int(3) as usize;
    let coords = a.input::<f32>(4);
    let aa = a.input::<f32>(5);
    let colors = a.input::<f32>(6);
    let mut image = a.output::<f32>(2);
    let per = coords_per(symbol);
    draw(
        &mut image[..4 * a.global_size()],
        height,
        n,
        &colors,
        |k, x, y| cover(&coords[per * k..], aa[k], x, y),
    );
}

/// `rects(int width, int height, float4* a, int nelements, float4* coords, float* aa_widths, float4* colors)`.
fn rects(a: &Args) {
    shapes(a, "rects", rect)
}

/// `circles(int width, int height, float4* a, int nelements, float* coords, float* aa_widths, float4* colors)`.
fn circles(a: &Args) {
    shapes(a, "circles", circle)
}

/// `triangles(int width, int height, float4* a, int nelements, float2* coords, float* aa_widths, float4* colors)`.
fn triangles(a: &Args) {
    shapes(a, "triangles", triangle)
}

/// `mesh_triangles(int width, int height, float4* a, int nelements, float2* coords, float* aa_widths, float4* colors)`.
fn mesh_triangles(a: &Args) {
    shapes(a, "mesh_triangles", mesh_triangle)
}

/// `ellipses(int width, int height, float4* a, int nelements, float4* coords, float* aa_widths, float4* colors)`.
fn ellipses(a: &Args) {
    shapes(a, "ellipses", ellipse)
}

/// `composite(int width, int height, float4* a, int nelements, int4* coords,
/// float4* pixels, int* ipixels, int npixels)`: composite RGBA sprites.
fn composite(a: &Args) {
    let height = a.int(1) as usize;
    let n = a.int(3) as usize;
    let coords = a.input::<i32>(4);
    let pixels = a.input::<[f32; 4]>(5);
    let ipixels = a.input::<i32>(6);
    let npixels = a.int(7);
    let mut image = a.output::<f32>(2);
    par::map_chunks(&mut image[..4 * a.global_size()], 4, |i, pixel| {
        let (x, y) = ((i / height) as i32, (i % height) as i32);
        for j in 0..n {
            if let Some(s) = sprite_source(&coords[4 * j..], ipixels[j], npixels, x, y) {
                let rgba = pixels[s];
                over(pixel, &rgba, rgba[3]);
            }
        }
    });
}
