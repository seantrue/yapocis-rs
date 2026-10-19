//! `examples/gaussians.py`: median filtering, HSI conversion, Gaussian blur,
//! difference of Gaussians and gradient on the standard test image.
//!
//! cargo run -p yapocis --example gaussians --features metal --release

mod common;

use ndarray::{s, Array3};
use yapocis::lib::image::join_channels;
use yapocis::lib::{Gaussian, Gradient, Hsi, Median};

const EX: &str = "gaussians";

fn main() -> yapocis::Result<()> {
    let device = common::device();
    let median = Median::load(&device)?;
    let gradient = Gradient::load(&device)?;
    let hsi = Hsi::load(&device)?;
    let gaussian = Gaussian::load(&device)?;

    let image = common::test_image();
    let image: Array3<f32> = image.slice(s![..;2, ..;2, ..]).to_owned();
    common::show(EX, "Original", &image.clone().into_dyn());

    let mut rng = common::Rng::new(7);
    let noisy = image.mapv(|v| v * 0.8 + rng.uniform() * 0.2);
    common::show(EX, "Noisy", &noisy.clone().into_dyn());

    let (r, g, b) = (
        common::plane(&noisy, 0),
        common::plane(&noisy, 1),
        common::plane(&noisy, 2),
    );
    let filt = |p: &ndarray::Array2<f32>| median.median3x3(&p.view().into_dyn(), 100);
    let (r, g, b) = (filt(&r)?, filt(&g)?, filt(&b)?);
    let to2 = |a: ndarray::ArrayD<f32>| a.into_dimensionality::<ndarray::Ix2>().unwrap();
    let (r, g, b) = (to2(r), to2(g), to2(b));
    common::show(EX, "Filtered", &join_channels(r.view(), g.view(), b.view()));

    let (h, s, i) = hsi.rgb2hsi(
        &r.view().into_dyn(),
        &g.view().into_dyn(),
        &b.view().into_dyn(),
    )?;
    common::show(EX, "I", &i);
    common::show(EX, "S", &s);
    common::show(EX, "H", &h);

    let blur = gaussian.gauss_image(&i.view(), 3)?;
    common::show(EX, "Blur", &blur);
    let more_blurred = gaussian.gauss_image(&i.view(), 4)?;
    let dog = &blur - &more_blurred;
    common::show(EX, "DOG", &dog);

    let (grad, angle) = gradient.gradient(&i.view(), 5)?;
    common::show(EX, "Gradient", &grad);
    common::show(EX, "Angle", &angle);
    Ok(())
}
