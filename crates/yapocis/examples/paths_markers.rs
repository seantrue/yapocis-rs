//! `examples/matplotlib_markers.py`: six scatter plots of the same ten points,
//! coloured by distance from the origin, with different marker shapes:
//! circle, the text glyph α, a custom vertex path, a pentagon `(5, 0)`, a star
//! `(5, 1)` and an asterisk `(5, 2)`.
//!
//! The Python example renders through the yapocis matplotlib backend. Rust has
//! no matplotlib, so this draws the same visible content directly with
//! `yapocis::lib::paths` (earcut-filled marker polygons, strokes) and
//! `typeset` (the α glyph and titles) via `examples/common/plot.rs`; it does not
//! reproduce matplotlib's layout or its shared axes. NumPy's seeded points
//! (`np.random.seed(19680801)`) are replaced by the examples' own generator.
//!
//! cargo run -p yapocis --example paths_markers --features metal --release

mod common;

use common::plot::{viridis, Axes, Figure, Line};
use yapocis::lib::typeset::render_text;

fn regular(n: usize, radius_inner: Option<f64>) -> Vec<[f64; 2]> {
    // matplotlib markers point up: first vertex at 90 degrees
    let step = std::f64::consts::TAU / n as f64;
    let mut v = Vec::new();
    for i in 0..n {
        let a = std::f64::consts::FRAC_PI_2 + step * i as f64;
        v.push([a.cos(), a.sin()]);
        if let Some(r) = radius_inner {
            let b = a + step / 2.0;
            v.push([r * b.cos(), r * b.sin()]);
        }
    }
    v
}

fn main() -> yapocis::Result<()> {
    let device = common::device();
    let mut rng = common::Rng::new(19_680_801);
    let x: Vec<f64> = (0..10).map(|_| rng.uniform() as f64).collect();
    let y: Vec<f64> = (0..10).map(|_| rng.uniform() as f64).collect();
    let z: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a * a + b * b).sqrt())
        .collect();
    let (zlo, zhi) = z
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let color = |v: f64| {
        let c = viridis(((v - zlo) / (zhi - zlo)) as f32);
        [c[0], c[1], c[2], 1.0]
    };
    // s=200 points² -> marker radius sqrt(200)/2 points, at 100 dpi
    let size = 200f64.sqrt() / 2.0 / 72.0 * 100.0;

    let mut fig = Figure::new(&device, 1000, 1000)?;
    // plt.title(...) after plt.show() retitles the current (last) axes
    let last_title = format!("{}: markers", common::today());
    let titles = [
        "marker='o'",
        "marker=r'$\\alpha$'",
        "marker=verts",
        "marker=(5, 0)",
        "marker=(5, 1)",
        last_title.as_str(),
    ];
    let verts = vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]];
    for (idx, title) in titles.iter().enumerate() {
        let ax = Axes::subplot(&fig, 2, 3, idx, (-0.05, 1.05), (-0.05, 1.05));
        for i in 0..10 {
            let c = color(z[i]);
            match idx {
                0 => ax.marker(&mut fig, x[i], y[i], &regular(32, None), size, c)?,
                1 => {
                    // a glyph as the marker, centred on the point
                    let t = render_text("α", "default", Some(size * 2.2), None)?;
                    let (rows, cols) = t.block.dim();
                    let (cx, cy) = (ax.tx(x[i]), ax.ty(y[i]));
                    let canvas_y = fig.height as f64 - (cy - rows as f64 / 2.0);
                    fig.r.draw_text_image(
                        t.block.view(),
                        (cx - cols as f64 / 2.0) as i64,
                        canvas_y as i64,
                        0.0,
                        c,
                    )?;
                }
                2 => ax.marker(&mut fig, x[i], y[i], &verts, size / 2f64.sqrt(), c)?,
                3 => ax.marker(&mut fig, x[i], y[i], &regular(5, None), size, c)?,
                4 => ax.marker(&mut fig, x[i], y[i], &regular(5, Some(0.38)), size, c)?,
                _ => {
                    let (cx, cy) = (ax.tx(x[i]), ax.ty(y[i]));
                    for p in regular(5, None) {
                        fig.polyline(
                            &[[cx, cy], [cx + p[0] * size, cy + p[1] * size]],
                            &Line::solid(c, 2.0),
                        )?;
                    }
                }
            }
        }
        ax.decorate(&mut fig, title, "", "")?;
    }
    fig.save(&common::out_dir("paths_markers").join("matplotlib_markers.png"))
}
