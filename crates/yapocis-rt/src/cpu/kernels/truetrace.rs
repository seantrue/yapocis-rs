//! `truetrace.cl`: a jump-flood distance transform from a mask, and marching
//! squares over the distance field. Images are addressed `i = x * height + y`.

use crate::cpu::{par, Args};

pub const KERNELS: super::Table = &[
    ("init_seed_from_mask", init_seed_from_mask),
    ("jfa_step", jfa_step),
    ("marching_squares", marching_squares),
];

/// A seed as the kernels store it: `(x, y, 0, 0)`; `x < 0` means none.
type Seed = [f32; 4];

/// `init_seed_from_mask(int width, int height, int* mask, float4* seed, float* dist)`:
/// mask pixels seed themselves at distance 0; the rest have no seed and
/// distance `1e30`.
fn init_seed_from_mask(a: &Args) {
    let width = a.int(0);
    let height = a.int(1);
    let mask = a.input::<i32>(2);
    let mut seed = a.output::<Seed>(3);
    let mut dist = a.output::<f32>(4);
    let items = a.global_size().min((width.max(0) * height.max(0)) as usize);
    par::map2(&mut seed[..items], &mut dist[..items], |idx| {
        let (x, y) = (idx as i32 / height, idx as i32 % height);
        if mask[idx] != 0 {
            ([x as f32, y as f32, 0.0, 0.0], 0.0)
        } else {
            ([-1.0, -1.0, 0.0, 0.0], 1e30)
        }
    });
}

/// `jfa_step(int width, int height, int jump, float4* seed_in, float* dist_inout,
/// float4* seed_out)`: each pixel takes the nearest seed among itself and its
/// eight neighbours at distance `jump`. A work item reads and writes only its
/// own `dist`, so the old values are copied and the pixels run in parallel.
fn jfa_step(a: &Args) {
    let width = a.int(0);
    let height = a.int(1);
    let jump = a.int(2);
    let seed_in = a.input::<Seed>(3);
    let mut dist = a.output::<f32>(4);
    let mut seed_out = a.output::<Seed>(5);
    let items = a.global_size().min((width.max(0) * height.max(0)) as usize);
    let dist_in = dist[..items].to_vec();
    const DIRS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    par::map2(&mut seed_out[..items], &mut dist[..items], |idx| {
        let (x, y) = (idx as i32 / height, idx as i32 % height);
        let mut best_seed = seed_in[idx];
        let mut best_dist = dist_in[idx];
        for (dx, dy) in DIRS {
            let (nx, ny) = (x + dx * jump, y + dy * jump);
            if nx < 0 || ny < 0 || nx >= width || ny >= height {
                continue;
            }
            let nseed = seed_in[(nx * height + ny) as usize];
            if nseed[0] < 0.0 {
                continue; // no seed
            }
            let ddx = nseed[0] - x as f32;
            let ddy = nseed[1] - y as f32;
            let d2 = ddx * ddx + ddy * ddy;
            if d2 < best_dist * best_dist {
                best_dist = d2.sqrt();
                best_seed = nseed;
            }
        }
        (best_seed, best_dist)
    });
}

/// `marching_squares(int width, int height, float* dist, float iso,
/// uint* seg_counter, float* segments_out)`: up to two segments per cell
/// `(x, y)..(x+1, y+1)`, appended as `(ax, ay, bx, by)` at slots taken from
/// `atomic_inc(seg_counter)`. The GPU leaves the slot order undefined (the
/// callers sort or treat the segments as a set); here cells take slots in
/// work-item order.
fn marching_squares(a: &Args) {
    let width = a.int(0);
    let height = a.int(1);
    let dist = a.input::<f32>(2);
    let iso = a.float(3);
    let mut seg_counter = a.output::<u32>(4);
    let mut segments_out = a.output::<f32>(5);
    for idx in 0..a.global_size() {
        let (x, y) = (idx as i32 / height, idx as i32 % height);
        if x >= width - 1 || y >= height - 1 {
            continue;
        }
        let at = |xx: i32, yy: i32| dist[(xx * height + yy) as usize] - iso;
        let v00 = at(x, y);
        let v10 = at(x + 1, y);
        let v11 = at(x + 1, y + 1);
        let v01 = at(x, y + 1);

        let mut c = 0;
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

        // Crossing points on the four cell edges: bottom, right, top, left.
        let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
        let (fx, fy) = (x as f32, y as f32);
        let t0 = v00 / (v00 - v10 + 1e-20);
        let t1 = v10 / (v10 - v11 + 1e-20);
        let t2 = v01 / (v01 - v11 + 1e-20);
        let t3 = v00 / (v00 - v01 + 1e-20);
        let p = [
            [lerp(t0, fx, fx + 1.0), fy],
            [fx + 1.0, lerp(t1, fy, fy + 1.0)],
            [lerp(t2, fx, fx + 1.0), fy + 1.0],
            [fx, lerp(t3, fy, fy + 1.0)],
        ];

        let pairs: &[(usize, usize)] = match c {
            1 | 14 => &[(3, 0)],
            2 | 13 => &[(0, 1)],
            3 | 12 => &[(3, 1)],
            4 | 11 => &[(1, 2)],
            6 | 9 => &[(0, 2)],
            7 | 8 => &[(3, 2)],
            5 | 10 => {
                // Saddle: the asymptotic decider picks the pairing.
                let d = v00 * v11 - v10 * v01;
                if (c == 5) == (d > 0.0) {
                    &[(3, 0), (1, 2)]
                } else {
                    &[(0, 1), (2, 3)]
                }
            }
            _ => continue,
        };

        for &(i, j) in pairs {
            let slot = seg_counter[0];
            seg_counter[0] = slot.wrapping_add(1);
            let base = slot as usize * 4;
            let (pa, pb) = (p[i], p[j]);
            // The kernel trusts the caller to size segments_out; so does this.
            segments_out[base..base + 4].copy_from_slice(&[pa[0], pa[1], pb[0], pb[1]]);
        }
    }
}
