//! `examples/mandelbrot.py`: render the Mandelbrot set on the GPU and write it
//! with the red palette the Tk viewer used (no window; the PNG is the output).
//!
//! cargo run -p yapocis --example mandelbrot --features metal --release

mod common;

use ndarray::Array1;
use num_complex::Complex;
use yapocis::lib::Mandelbrot;

const W: usize = 1024;
const H: usize = 1024;

fn main() -> yapocis::Result<()> {
    let device = common::device();
    let mb = Mandelbrot::load(&device)?;
    // draw(-2.13, 0.77, -1.3, 1.3): xx along columns, yy from top (y2) down
    let (x1, x2, y1, y2) = (-2.13f32, 0.77f32, -1.3f32, 1.3f32);
    let q: Vec<Complex<f32>> = (0..H)
        .flat_map(|r| {
            let y = y2 + (y1 - y2) * r as f32 / H as f32;
            (0..W).map(move |c| Complex::new(x1 + (x2 - x1) * c as f32 / W as f32, y))
        })
        .collect();
    let q = Array1::from_vec(q).into_dyn();
    mb.calc(&q.view(), 256)?; // warm-up: compiles the program
    let output = common::timed("Main took", || mb.calc(&q.view(), 256))?;

    let max = *output.iter().max().unwrap() as f32;
    let img = image::RgbImage::from_fn(W as u32, H as u32, |x, y| {
        let v = (output[[y as usize * W + x as usize]] as f32 / max * 255.0) as u8;
        image::Rgb([v, 0, 0])
    });
    let path = common::out_dir("mandelbrot").join("mandelbrot.png");
    img.save(&path).unwrap();
    println!("Mandelbrot: {}", path.display());
    Ok(())
}
