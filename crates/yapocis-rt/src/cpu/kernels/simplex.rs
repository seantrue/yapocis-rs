//! `simplex.cl`: 2-D and 3-D simplex noise (Perlin 2001) summed over octaves
//! (fBm), one pixel per work item. Images are addressed `gid = x * height + y`.
//!
//! The 3-D noise uses the radius `0.6`, which makes it slightly discontinuous
//! across simplex boundaries: where `floor(x + s)` lands within an ulp of an
//! integer, a fast-math GPU build (`-cl-fast-relaxed-math`) and this exact
//! port can pick neighbouring cells and differ by up to ~1e-3.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[("simplex2d", simplex2d), ("simplex3d", simplex3d)];

/// `hash_int`: integer hash of a lattice point (wrapping, as the GPU's int
/// arithmetic). The result is never negative: `h >> 16` is an arithmetic
/// shift, so XOR with it clears the sign bit, and `hash % 12` is a valid index.
fn hash_int(x: i32, y: i32, z: i32, seed: i32) -> i32 {
    let mut h = seed;
    h ^= x.wrapping_mul(374761393);
    h ^= y.wrapping_mul(668265263);
    h ^= z.wrapping_mul(1264258069);
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

const GRAD2: [[f32; 2]; 12] = [
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [0.0, -1.0],
    [-1.0, 1.0],
    [0.0, -1.0],
];

const GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

fn floor_int(v: f32) -> i32 {
    v.floor() as i32
}

/// `simplex2d_single(x, y, seed, tileable, tile_w, tile_h)`.
#[allow(clippy::excessive_precision)] // the OpenCL literals, verbatim
fn simplex2d_single(x: f32, y: f32, seed: i32, tileable: bool, tile_w: i32, tile_h: i32) -> f32 {
    const F2: f32 = 0.3660254037844386; // (sqrt(3) - 1) / 2
    const G2: f32 = 0.21132486540518713; // (3 - sqrt(3)) / 6

    let s = (x + y) * F2;
    let i = floor_int(x + s);
    let j = floor_int(y + s);
    let t = (i + j) as f32 * G2;
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);

    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };
    let x1 = x0 - i1 as f32 + G2;
    let y1 = y0 - j1 as f32 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    // Lattice coordinates, wrapped for tiling; C's `%` keeps the sign.
    let wrap = |v: i32, n: i32| {
        if !tileable {
            return v;
        }
        let r = v % n;
        if r < 0 {
            r + n
        } else {
            r
        }
    };
    let (ii, jj) = (wrap(i, tile_w), wrap(j, tile_h));

    let corner = |t: f32, gx: i32, gy: i32, dx: f32, dy: f32| {
        if t < 0.0 {
            return 0.0;
        }
        let t = t * t;
        let g = GRAD2[(hash_int(gx, gy, 0, seed) & 11) as usize];
        t * t * (g[0] * dx + g[1] * dy)
    };
    let n0 = corner(0.5 - x0 * x0 - y0 * y0, ii, jj, x0, y0);
    let n1 = corner(
        0.5 - x1 * x1 - y1 * y1,
        wrap(ii + i1, tile_w),
        wrap(jj + j1, tile_h),
        x1,
        y1,
    );
    let n2 = corner(
        0.5 - x2 * x2 - y2 * y2,
        wrap(ii + 1, tile_w),
        wrap(jj + 1, tile_h),
        x2,
        y2,
    );
    70.0 * (n0 + n1 + n2)
}

/// `simplex3d_single(x, y, z, seed)`.
#[allow(clippy::excessive_precision)] // the OpenCL literals, verbatim
fn simplex3d_single(x: f32, y: f32, z: f32, seed: i32) -> f32 {
    const F3: f32 = 0.3333333333333333;
    const G3: f32 = 0.16666666666666666;

    let s = (x + y + z) * F3;
    let i = floor_int(x + s);
    let j = floor_int(y + s);
    let k = floor_int(z + s);
    let t = (i + j + k) as f32 * G3;
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);
    let z0 = z - (k as f32 - t);

    // Which of the six tetrahedra: the second and third corners' offsets.
    let ((i1, j1, k1), (i2, j2, k2)) = if x0 >= y0 {
        if y0 >= z0 {
            ((1, 0, 0), (1, 1, 0))
        } else if x0 >= z0 {
            ((1, 0, 0), (1, 0, 1))
        } else {
            ((0, 0, 1), (1, 0, 1))
        }
    } else if y0 < z0 {
        ((0, 0, 1), (0, 1, 1))
    } else if x0 < z0 {
        ((0, 1, 0), (0, 1, 1))
    } else {
        ((0, 1, 0), (1, 1, 0))
    };

    let corner = |dx: f32, dy: f32, dz: f32, gx: i32, gy: i32, gz: i32| {
        let t = 0.6 - dx * dx - dy * dy - dz * dz;
        if t < 0.0 {
            return 0.0;
        }
        let t = t * t;
        let g = GRAD3[(hash_int(gx, gy, gz, seed) % 12) as usize];
        t * t * (g[0] * dx + g[1] * dy + g[2] * dz)
    };
    let n0 = corner(x0, y0, z0, i, j, k);
    let n1 = corner(
        x0 - i1 as f32 + G3,
        y0 - j1 as f32 + G3,
        z0 - k1 as f32 + G3,
        i + i1,
        j + j1,
        k + k1,
    );
    let n2 = corner(
        x0 - i2 as f32 + 2.0 * G3,
        y0 - j2 as f32 + 2.0 * G3,
        z0 - k2 as f32 + 2.0 * G3,
        i + i2,
        j + j2,
        k + k2,
    );
    let n3 = corner(
        x0 - 1.0 + 3.0 * G3,
        y0 - 1.0 + 3.0 * G3,
        z0 - 1.0 + 3.0 * G3,
        i + 1,
        j + 1,
        k + 1,
    );
    32.0 * (n0 + n1 + n2 + n3)
}

/// fBm normalisation: `clamp((total / max_value + 1) * 0.5, 0, 1)`.
fn to_unit(total: f32, max_value: f32) -> f32 {
    ((total / max_value + 1.0) * 0.5).clamp(0.0, 1.0)
}

/// `simplex2d(int width, int height, float* output, int seed, float scale,
/// float offset_x, float offset_y, int octaves, float persistence,
/// float lacunarity, int tileable)`.
fn simplex2d(a: &Args) {
    let width = a.int(0);
    let height = a.int(1);
    let mut output = a.output::<f32>(2);
    let seed = a.int(3);
    let scale = a.float(4);
    let (offset_x, offset_y) = (a.float(5), a.float(6));
    let octaves = a.int(7);
    let persistence = a.float(8);
    let lacunarity = a.float(9);
    let tileable = a.int(10) != 0;
    // Work items past width * height return without writing.
    let items = a.global_size().min((width.max(0) * height.max(0)) as usize);
    par::map(&mut output[..items], |gid| {
        let x = (gid / height as usize) as i32;
        let y = (gid % height as usize) as i32;
        let (mut total, mut amplitude, mut frequency, mut max_value) = (0f32, 1f32, scale, 0f32);
        for octave in 0..octaves {
            let nx = (x as f32 / width as f32) * frequency + offset_x;
            let ny = (y as f32 / height as f32) * frequency + offset_y;
            let tile_w = ((width as f32 / frequency) as i32).max(1);
            let tile_h = ((height as f32 / frequency) as i32).max(1);
            let noise = simplex2d_single(nx, ny, seed + octave, tileable, tile_w, tile_h);
            total += noise * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        to_unit(total, max_value)
    });
}

/// `simplex3d(int width, int height, int depth, float* output, int seed,
/// float scale, float offset_x, float offset_y, float offset_z, int octaves,
/// float persistence, float lacunarity)`: `gid = z * width * height + x * height + y`.
fn simplex3d(a: &Args) {
    let (width, height, depth) = (a.int(0), a.int(1), a.int(2));
    let mut output = a.output::<f32>(3);
    let seed = a.int(4);
    let scale = a.float(5);
    let (offset_x, offset_y, offset_z) = (a.float(6), a.float(7), a.float(8));
    let octaves = a.int(9);
    let persistence = a.float(10);
    let lacunarity = a.float(11);
    let plane = (width.max(0) * height.max(0)) as usize;
    let items = a.global_size().min(plane * depth.max(0) as usize);
    par::map(&mut output[..items], |gid| {
        let z = (gid / plane) as i32;
        let rem = gid % plane;
        let x = (rem / height as usize) as i32;
        let y = (rem % height as usize) as i32;
        let (mut total, mut amplitude, mut frequency, mut max_value) = (0f32, 1f32, scale, 0f32);
        for octave in 0..octaves {
            let nx = (x as f32 / width as f32) * frequency + offset_x;
            let ny = (y as f32 / height as f32) * frequency + offset_y;
            let nz = (z as f32 / depth as f32) * frequency + offset_z;
            total += simplex3d_single(nx, ny, nz, seed + octave) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        to_unit(total, max_value)
    });
}
