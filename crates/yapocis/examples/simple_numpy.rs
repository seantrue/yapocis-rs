//! `examples/simple_numpy.py`: the CPU baseline for `simple_yapocis`: add two
//! 50,000-element arrays on the host and report the time and the norms.
//!
//! cargo run -p yapocis --example simple_numpy --release

mod common;

use ndarray::Array1;

fn main() {
    let mut rng = common::Rng::new(1);
    let a = Array1::from_iter((0..50_000).map(|_| rng.uniform() as f64));
    let b = Array1::from_iter((0..50_000).map(|_| rng.uniform() as f64));
    let t = std::time::Instant::now();
    let a_plus_b = &a + &b;
    println!("Elapsed: {:.6} s", t.elapsed().as_secs_f64());
    let err = (&a_plus_b - &(&a + &b)).mapv(|v| v * v).sum().sqrt();
    let norm = a_plus_b.mapv(|v| v * v).sum().sqrt();
    println!("({err}, {norm})");
}
