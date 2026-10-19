//! `examples/matplotlib_histogram.py`: a 50-bin histogram of 10,000 normal
//! samples, titled with today's date.
//!
//! The Python example renders through the yapocis matplotlib backend
//! (`yapocis.lib.mplbackend`). Rust has no matplotlib, so this draws the same
//! visible content (bars, axes, ticks, title) directly with
//! `yapocis::lib::paths` and `typeset` via `examples/common/plot.rs`; it does
//! not reproduce matplotlib's layout engine.
//!
//! cargo run -p yapocis --example paths_histogram --features metal --release

mod common;

use common::plot::{Axes, Figure, BLUE};

fn main() -> yapocis::Result<()> {
    let device = common::device();
    let mut rng = common::Rng::new(12);
    let data: Vec<f64> = (0..10_000).map(|_| rng.normal() as f64).collect();
    let (lo, hi) = data
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let bins = 50;
    let width = (hi - lo) / bins as f64;
    let mut counts = vec![0usize; bins];
    for v in &data {
        counts[(((v - lo) / width) as usize).min(bins - 1)] += 1;
    }
    let top = *counts.iter().max().unwrap() as f64;

    // 6 x 4 inches at 100 dpi, like fig.set_size_inches(6, 4)
    let mut fig = Figure::new(&device, 600, 400)?;
    let pad = (hi - lo) * 0.05;
    let ax = Axes::new(
        &fig,
        [0.125, 0.11, 0.775, 0.77],
        (lo - pad, hi + pad),
        (0.0, top * 1.05),
    );
    for (i, &c) in counts.iter().enumerate() {
        let x = lo + (i as f64 + 0.5) * width;
        ax.bar(&mut fig, x, width, c as f64, BLUE, None)?;
    }
    ax.decorate(
        &mut fig,
        &format!("{}: Normal distribution", common::today()),
        "",
        "",
    )?;
    fig.save(&common::out_dir("paths_histogram").join("matplotlib_histogram.png"))
}
