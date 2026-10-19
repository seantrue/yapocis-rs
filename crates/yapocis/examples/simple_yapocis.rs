//! `examples/simple_yapocis.py`: add two arrays with the `demo` interface and
//! check the result against the CPU.
//!
//! cargo run -p yapocis --example simple_yapocis --features metal

mod common;

use ndarray::Array1;
use yapocis::interfaces::demo::Demo;

fn main() -> yapocis::Result<()> {
    let device = common::device();
    let mut rng = common::Rng::new(1);
    let a = Array1::from_iter((0..50_000).map(|_| rng.uniform())).into_dyn();
    let b = Array1::from_iter((0..50_000).map(|_| rng.uniform())).into_dyn();

    let t = std::time::Instant::now();
    let demo = Demo::load(&device)?;
    let a_plus_b = demo.sum(&a.view(), &b.view())?;
    let elapsed = t.elapsed();

    let expected = &a + &b;
    let err = (&a_plus_b - &expected).mapv(|v| v * v).sum().sqrt();
    let norm = a_plus_b.mapv(|v| v * v).sum().sqrt();
    println!("(|gpu - cpu|, |gpu|) = ({err}, {norm})");
    println!(
        "Elapsed (compile + run): {:.3} ms",
        elapsed.as_secs_f64() * 1e3
    );
    let t = std::time::Instant::now();
    demo.sum(&a.view(), &b.view())?;
    println!(
        "Elapsed (cached program): {:.3} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
