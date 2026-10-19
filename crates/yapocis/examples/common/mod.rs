//! Shared helpers for the examples: device selection, the test image, and
//! writing results as PNGs (the Rust counterpart of `show_array`).
//!
//! Images are `(rows, cols[, channels])` float32 in [0, 1], like `imread` in
//! the Python package. Output goes to `$YAPOCIS_EXAMPLES_OUT`, else
//! `target/examples-out/<example>/`.
#![allow(dead_code)]

use ndarray::{Array2, Array3, ArrayD, Axis};
use std::path::PathBuf;
use yapocis::Device;

pub mod kernels;
pub mod plot;

/// The device the Python examples would use: `YAPOCIS_BACKEND` / `YAPOCIS_DEVICE`.
pub fn device() -> Device {
    let d = Device::default().expect("no GPU device (build with --features metal, ocl or cuda)");
    println!("device: {:?}", d);
    d
}

pub fn out_dir(example: &str) -> PathBuf {
    let base = std::env::var("YAPOCIS_EXAMPLES_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/examples-out")
        });
    let dir = base.join(example);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/data")
        .join(name)
}

/// `get_test_image()`: the standard RGB test image as (rows, cols, 3) floats.
pub fn test_image() -> Array3<f32> {
    load_rgb(&data("test.jpg"))
}

pub fn load_rgb(path: &std::path::Path) -> Array3<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = img.dimensions();
    Array3::from_shape_fn((h as usize, w as usize, 3), |(y, x, c)| {
        img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    })
}

/// Linear stretch of an array to [0, 1] (what `show_array` does before display).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn normalize(a: &ArrayD<f32>) -> ArrayD<f32> {
    let finite = a.iter().cloned().filter(|v| v.is_finite());
    let (lo, hi) = finite.fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
    if !(hi > lo) {
        return a.mapv(|_| 0.0);
    }
    a.mapv(|v| {
        if v.is_finite() {
            (v - lo) / (hi - lo)
        } else {
            0.0
        }
    })
}

/// `show_array(a, title)`: save a 2-D or (rows, cols, 3|4) array as a PNG,
/// stretched to [0, 1]. Returns the path.
pub fn show(example: &str, title: &str, a: &ArrayD<f32>) -> PathBuf {
    let n = normalize(a);
    let path = out_dir(example).join(format!("{}.png", title.replace(' ', "_")));
    match n.ndim() {
        2 => {
            let (h, w) = (n.shape()[0], n.shape()[1]);
            let img = image::GrayImage::from_fn(w as u32, h as u32, |x, y| {
                image::Luma([(n[[y as usize, x as usize]] * 255.0) as u8])
            });
            img.save(&path).unwrap();
        }
        3 => {
            let (h, w) = (n.shape()[0], n.shape()[1]);
            let img = image::RgbImage::from_fn(w as u32, h as u32, |x, y| {
                let p =
                    |c: usize| (n[[y as usize, x as usize, c.min(n.shape()[2] - 1)]] * 255.0) as u8;
                image::Rgb([p(0), p(1), p(2)])
            });
            img.save(&path).unwrap();
        }
        d => panic!("cannot show a {d}-D array"),
    }
    println!("{title}: {}", path.display());
    path
}

/// Save without stretching (values already in [0, 1]).
pub fn save_unit(example: &str, title: &str, a: &ArrayD<f32>) -> PathBuf {
    let clipped = a.mapv(|v| v.clamp(0.0, 1.0));
    let path = out_dir(example).join(format!("{}.png", title.replace(' ', "_")));
    let n = clipped;
    let (h, w) = (n.shape()[0], n.shape()[1]);
    if n.ndim() == 2 {
        image::GrayImage::from_fn(w as u32, h as u32, |x, y| {
            image::Luma([(n[[y as usize, x as usize]] * 255.0) as u8])
        })
        .save(&path)
        .unwrap();
    } else {
        image::RgbImage::from_fn(w as u32, h as u32, |x, y| {
            let p = |c: usize| (n[[y as usize, x as usize, c]] * 255.0) as u8;
            image::Rgb([p(0), p(1), p(2)])
        })
        .save(&path)
        .unwrap();
    }
    println!("{title}: {}", path.display());
    path
}

/// A seeded uniform generator so examples are reproducible without a rand dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    pub fn uniform(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Standard normal (Box-Muller), like `np.random.randn`.
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.uniform() as f64).max(1e-12);
        let u2 = self.uniform() as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// `np.linspace(0, end, n, dtype=float32)`.
pub fn linspace_f32(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * i as f32 / (n - 1) as f32)
        .collect()
}

/// Today's UTC date as `YYYY-MM-DD` (`datetime.date.today()` in the titles).
pub fn today() -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        / 86_400;
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// A row of `=` like the Python examples print.
pub fn rule(n: usize) -> String {
    "=".repeat(n)
}

pub fn plane(a: &Array3<f32>, c: usize) -> Array2<f32> {
    a.index_axis(Axis(2), c).to_owned()
}

/// Wall-clock helper.
pub fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let v = f();
    println!("{label}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    v
}
