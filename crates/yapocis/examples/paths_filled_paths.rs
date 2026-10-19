//! `examples/matplotlib_filled_paths.py`: filled paths on one axes: an orange
//! unit square, a translucent pink diamond, a translucent green circle, and a
//! translucent blue region under a sine curve, with axis lines and two labels.
//!
//! The Python example renders matplotlib `PathPatch`es through the yapocis
//! matplotlib backend. Rust has no matplotlib, so this draws the same visible
//! content directly with `yapocis::lib::paths` (earcut fills, strokes) and
//! `typeset` via `examples/common/plot.rs`; it does not reproduce matplotlib's
//! layout engine. The circle is flattened to 64 segments, as matplotlib's
//! `Path.circle` Bézier arcs are when the backend draws them.
//!
//! cargo run -p yapocis --example paths_filled_paths --features metal --release

mod common;

use common::plot::{with_alpha, Axes, Figure, Line, BLACK, BLUE};

fn main() -> yapocis::Result<()> {
    let device = common::device();
    // 5 x 5 inches at 100 dpi
    let mut fig = Figure::new(&device, 500, 500)?;
    let ax = Axes::new(&fig, [0.125, 0.11, 0.775, 0.77], (-2.0, 2.0), (-2.0, 2.0));
    let p = |x: f64, y: f64| [ax.tx(x), ax.ty(y)];

    ax.vline(&mut fig, 0.0, &Line::solid([1.0, 1.0, 0.0, 1.0], 1.5))?;
    ax.hline(&mut fig, 0.0, &Line::solid([0.5, 0.5, 0.5, 1.0], 1.5))?;
    fig.text(
        p(-1.1, -1.1)[0],
        p(-1.1, -1.1)[1],
        "lb",
        12.0,
        BLACK,
        0.0,
        0.0,
    )?;
    fig.text(p(1.1, 1.1)[0], p(1.1, 1.1)[1], "rt", 12.0, BLACK, 0.0, 0.0)?;

    // the square: MOVETO (0,0), LINETO (0,1), (1,1), (1,0), CLOSEPOLY
    let square = [p(0.0, 0.0), p(0.0, 1.0), p(1.0, 1.0), p(1.0, 0.0)];
    fig.fill(&square, [1.0, 0.647, 0.0, 1.0], Some((BLACK, 1.0)))?;
    let diamond = [p(0.0, -1.0), p(1.0, 0.0), p(0.0, 1.0), p(-1.0, 0.0)];
    fig.fill(
        &diamond,
        [1.0, 0.753, 0.796, 0.5],
        Some((with_alpha(BLACK, 0.5), 1.0)),
    )?;
    let circle: Vec<[f64; 2]> = (0..64)
        .map(|i| {
            let a = i as f64 / 64.0 * std::f64::consts::TAU;
            p(-1.0 + 0.5 * a.cos(), 1.0 + 0.5 * a.sin())
        })
        .collect();
    fig.fill(
        &circle,
        [0.0, 0.502, 0.0, 0.5],
        Some((with_alpha(BLACK, 0.5), 1.0)),
    )?;

    // the sine region: x in [-1, 1), y = 0.5 sin(5 pi x) - 0.5, closed at y = -1.5
    let mut xs: Vec<f64> = (0..200).map(|i| -1.0 + i as f64 * 0.01).collect();
    let mut ys: Vec<f64> = xs
        .iter()
        .map(|x| (x * std::f64::consts::PI * 5.0).sin() * 0.5 - 0.5)
        .collect();
    xs[0] = xs[1];
    ys[0] = -1.5;
    let n = ys.len();
    ys[n - 1] = -1.5;
    xs[n - 1] = xs[n - 2];
    let line: Vec<[f64; 2]> = xs.iter().zip(&ys).map(|(&x, &y)| p(x, y)).collect();
    fig.fill(&line, with_alpha(BLUE, 0.5), None)?;
    fig.polyline(&line, &Line::solid(BLUE, 1.5))?;

    ax.decorate(&mut fig, &common::today(), "", "")?;
    fig.save(&common::out_dir("paths_filled_paths").join("matplot_filled_paths.png"))
}
