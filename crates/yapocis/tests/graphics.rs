//! Replay the graphics-layer parity fixtures (`yapocis-tools bundle
//! --parity-lib ... --generators graphics,graphics4,kerneldraw,typeset,paths`)
//! through the Rust ports and compare with the Python output canvases.

mod common;

use common::{allclose_f32, devices, lib_index, mismatch_fraction, open_npz, read_any, Npy};
use ndarray::{Array2, Array3, ArrayD, Axis, IxDyn};
use ndarray_npy::NpzReader;
use serde_json::Value;
use std::fs::File;
use yapocis::lib::graphics::{BlockAlpha, Graphics1, Graphics4, Style, TextAlpha, TextStyle};
use yapocis::lib::kerneldraw::{Draw, Item, KernelDraw1, KernelDraw3, KernelDraw4, TextItem};
use yapocis::lib::paths::PathRenderer;
use yapocis::lib::typeset;
use yapocis::{Backend, Device};

type Npz = NpzReader<File>;

fn f32_arr(npz: &mut Npz, key: &str) -> ArrayD<f32> {
    let n = read_any(npz, key);
    match n.dtype {
        yapocis::DType::Float32 => n.to_array::<f32>(),
        yapocis::DType::Float64 => n.to_array::<f64>().mapv(|v| v as f32),
        yapocis::DType::Int32 => n.to_array::<i32>().mapv(|v| v as f32),
        yapocis::DType::Int64 => n.to_array::<i64>().mapv(|v| v as f32),
        yapocis::DType::Uint8 => n.to_array::<u8>().mapv(|v| v as f32),
        other => panic!("{key}: unexpected dtype {other}"),
    }
}

fn rows(a: &ArrayD<f32>) -> Vec<Vec<f32>> {
    a.outer_iter()
        .map(|r| r.iter().cloned().collect())
        .collect()
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn rgba(v: &Value) -> [f32; 4] {
    let f = f32s(v);
    [f[0], f[1], f[2], f[3]]
}

fn opt_f32(v: Option<&Value>) -> Option<f32> {
    v.and_then(|x| x.as_f64()).map(|x| x as f32)
}

fn vec_or_key_f32(npz: &mut Npz, v: Option<&Value>) -> Option<Vec<f32>> {
    match v {
        Some(Value::String(k)) => Some(f32_arr(npz, k).iter().cloned().collect()),
        Some(Value::Array(a)) => Some(a.iter().map(|x| x.as_f64().unwrap() as f32).collect()),
        _ => None,
    }
}

fn style1(npz: &mut Npz, s: &Value) -> Style<f32> {
    Style {
        color: opt_f32(s.get("color")),
        colors: vec_or_key_f32(npz, s.get("colors")),
        alpha: opt_f32(s.get("alpha")),
        alphas: vec_or_key_f32(npz, s.get("alphas")),
        width: opt_f32(s.get("width")),
        widths: vec_or_key_f32(npz, s.get("widths")),
        distance: opt_f32(s.get("distance")),
        aa_width: opt_f32(s.get("aa_width")),
        aa_widths: vec_or_key_f32(npz, s.get("aa_widths")),
    }
}

fn colors4(npz: &mut Npz, v: Option<&Value>) -> Option<Vec<[f32; 4]>> {
    match v {
        Some(Value::String(k)) => {
            let a = f32_arr(npz, k);
            Some(a.outer_iter().map(|r| [r[0], r[1], r[2], r[3]]).collect())
        }
        Some(Value::Array(a)) => Some(a.iter().map(rgba).collect()),
        _ => None,
    }
}

fn style4(npz: &mut Npz, s: &Value) -> Style<[f32; 4]> {
    Style {
        color: s.get("color").filter(|v| v.is_array()).map(rgba),
        colors: colors4(npz, s.get("colors")),
        alpha: opt_f32(s.get("alpha")),
        alphas: vec_or_key_f32(npz, s.get("alphas")),
        width: opt_f32(s.get("width")),
        widths: vec_or_key_f32(npz, s.get("widths")),
        distance: opt_f32(s.get("distance")),
        aa_width: opt_f32(s.get("aa_width")),
        aa_widths: vec_or_key_f32(npz, s.get("aa_widths")),
    }
}

fn locations2(npz: &mut Npz, key: &str) -> Vec<(f64, f64)> {
    f32_arr(npz, key)
        .outer_iter()
        .map(|r| (r[0] as f64, r[1] as f64))
        .collect()
}

fn locations3(npz: &mut Npz, key: &str) -> Vec<(usize, f64, f64)> {
    f32_arr(npz, key)
        .outer_iter()
        .map(|r| (r[0] as usize, r[1] as f64, r[2] as f64))
        .collect()
}

fn blocks2(npz: &mut Npz, keys: &Value) -> Vec<Array2<f32>> {
    keys.as_array()
        .unwrap()
        .iter()
        .map(|k| {
            f32_arr(npz, k.as_str().unwrap())
                .into_dimensionality()
                .unwrap()
        })
        .collect()
}

fn blocks3(npz: &mut Npz, keys: &Value) -> Vec<Array3<f32>> {
    keys.as_array()
        .unwrap()
        .iter()
        .map(|k| {
            f32_arr(npz, k.as_str().unwrap())
                .into_dimensionality()
                .unwrap()
        })
        .collect()
}

fn block_alpha(npz: &mut Npz, v: &Value) -> Option<BlockAlpha> {
    match v {
        Value::Null => None,
        Value::String(k) => Some(BlockAlpha::Mask(
            f32_arr(npz, k).into_dimensionality().unwrap(),
        )),
        other => Some(BlockAlpha::Value(other.as_f64().unwrap() as f32)),
    }
}

fn text_alpha(v: &Value) -> Option<TextAlpha> {
    match v {
        Value::Null => None,
        Value::String(s) if s == "mask" => Some(TextAlpha::Mask),
        Value::String(s) if s == "bug" => Some(TextAlpha::Bug),
        other => Some(TextAlpha::Value(other.as_f64().unwrap() as f32)),
    }
}

fn opt_list<T>(v: Option<&Value>, f: impl Fn(&Value) -> Option<T>) -> Option<Vec<Option<T>>> {
    v.and_then(|x| x.as_array())
        .map(|a| a.iter().map(f).collect())
}

fn text_style<C: Clone>(rec: &Value, color: impl Fn(&Value) -> Option<C>) -> TextStyle<C> {
    TextStyle {
        font: rec.get("font").and_then(|v| v.as_str()).map(String::from),
        fonts: opt_list(rec.get("fonts"), |v| v.as_str().map(String::from)),
        size: rec.get("size").and_then(|v| v.as_f64()),
        sizes: opt_list(rec.get("sizes"), |v| v.as_f64()),
        alpha: rec.get("alpha").and_then(text_alpha),
        alphas: opt_list(rec.get("alphas"), text_alpha),
        color: rec.get("color").and_then(&color),
        colors: opt_list(rec.get("colors"), |v| color(v)),
        rotation: rec.get("rotation").and_then(|v| v.as_f64()),
        rotations: opt_list(rec.get("rotations"), |v| v.as_f64()),
    }
}

fn texts(rec: &Value) -> Vec<String> {
    rec["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_string())
        .collect()
}

const SHAPES: [&str; 6] = [
    "lines",
    "rects",
    "circles",
    "triangles",
    "mesh_triangles",
    "ellipses",
];

/// Compare a canvas with the Python output: nearly all pixels must agree
/// (allclose) and the rest, at most `tolerance_fraction`, may differ by edge
/// effects of float accumulation order in rasterisation.
fn check_canvas(
    label: &str,
    got: &ArrayD<f32>,
    want: &ArrayD<f32>,
    rtol: f64,
    atol: f64,
    tolerance_fraction: f64,
) {
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    // YAPOCIS_PARITY_DUMP=<dir>: save the Rust canvas for visual comparison
    // (claude-debug/graphics_parity_montage.py in the Python repo).
    if let Ok(dir) = std::env::var("YAPOCIS_PARITY_DUMP") {
        let name: String = label
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let _ = std::fs::create_dir_all(&dir);
        ndarray_npy::write_npy(format!("{dir}/{name}.npy"), got).expect("dump canvas");
    }
    let g = got.as_slice().unwrap();
    let w = want.as_slice().unwrap();
    if let Err(e) = allclose_f32(g, w, rtol, atol) {
        let frac = mismatch_fraction(g, w, (atol * 10.0) as f32);
        assert!(
            frac <= tolerance_fraction,
            "{label}: {e}; {:.4}% of pixels differ",
            frac * 100.0
        );
        eprintln!(
            "{label}: {:.4}% of pixels differ ({e}), within tolerance",
            frac * 100.0
        );
    }
}

fn run_graphics1(device: &Device, rec: &Value, npz: &mut Npz) -> ArrayD<f32> {
    let g = Graphics1::load(device).unwrap();
    let mut canvas = f32_arr(npz, rec["canvas"].as_str().unwrap());
    for op in rec["ops"].as_array().unwrap() {
        let name = op["op"].as_str().unwrap();
        let mut view = canvas.view_mut();
        if SHAPES.contains(&name) {
            let coords = rows(&f32_arr(npz, op["coords"].as_str().unwrap()));
            let style = style1(npz, &op["style"]);
            match name {
                "lines" => g.lines(&mut view, &coords, &style),
                "rects" => g.rects(&mut view, &coords, &style),
                "circles" => g.circles(&mut view, &coords, &style),
                "triangles" => g.triangles(&mut view, &coords, &style),
                "mesh_triangles" => g.mesh_triangles(&mut view, &coords, &style),
                _ => g.ellipses(&mut view, &coords, &style),
            }
            .unwrap();
        } else if name == "composite" {
            let locations = locations2(npz, op["locations"].as_str().unwrap());
            let blocks = blocks2(npz, &op["blocks"]);
            let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
            let alpha = op.get("alpha").and_then(|v| block_alpha(npz, v));
            let alphas = op.get("alphas").filter(|v| !v.is_null()).map(|v| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| block_alpha(npz, x))
                    .collect()
            });
            g.composite(&mut view, &locations, &views, alpha, alphas)
                .unwrap();
        } else if name == "sprites" {
            let locations = locations3(npz, op["locations"].as_str().unwrap());
            let sprites = blocks2(npz, &op["sprites"]);
            let views: Vec<_> = sprites.iter().map(|b| b.view()).collect();
            let alpha = op.get("alpha").and_then(|v| block_alpha(npz, v));
            let alphas = op.get("alphas").filter(|v| !v.is_null()).map(|v| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| block_alpha(npz, x))
                    .collect()
            });
            g.sprites(&mut view, &locations, &views, alpha, alphas)
                .unwrap();
        } else if name == "draw_strings" {
            let locations = locations2(npz, op["locations"].as_str().unwrap());
            let t = texts(op);
            let strs: Vec<&str> = t.iter().map(String::as_str).collect();
            let style = text_style::<f32>(op, |v| v.as_f64().map(|x| x as f32));
            g.draw_strings(&mut view, &locations, &strs, &style)
                .unwrap();
        } else {
            panic!("unknown op {name}");
        }
    }
    canvas
}

fn run_graphics4(device: &Device, rec: &Value, npz: &mut Npz) -> ArrayD<f32> {
    let g = Graphics4::load(device).unwrap();
    let mut canvas = f32_arr(npz, rec["canvas"].as_str().unwrap());
    for op in rec["ops"].as_array().unwrap() {
        let name = op["op"].as_str().unwrap();
        let mut view = canvas.view_mut();
        if SHAPES.contains(&name) {
            let coords = rows(&f32_arr(npz, op["coords"].as_str().unwrap()));
            let style = style4(npz, &op["style"]);
            match name {
                "lines" => g.lines(&mut view, &coords, &style),
                "rects" => g.rects(&mut view, &coords, &style),
                "circles" => g.circles(&mut view, &coords, &style),
                "triangles" => g.triangles(&mut view, &coords, &style),
                "mesh_triangles" => g.mesh_triangles(&mut view, &coords, &style),
                _ => g.ellipses(&mut view, &coords, &style),
            }
            .unwrap();
        } else if name == "composite" {
            let locations = locations2(npz, op["locations"].as_str().unwrap());
            let blocks = blocks3(npz, &op["blocks"]);
            let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
            g.composite(&mut view, &locations, &views).unwrap();
        } else if name == "sprites" {
            let locations = locations3(npz, op["locations"].as_str().unwrap());
            let sprites = blocks3(npz, &op["sprites"]);
            let views: Vec<_> = sprites.iter().map(|b| b.view()).collect();
            g.sprites(&mut view, &locations, &views).unwrap();
        } else if name == "draw_strings" {
            let locations = locations2(npz, op["locations"].as_str().unwrap());
            let t = texts(op);
            let strs: Vec<&str> = t.iter().map(String::as_str).collect();
            let style = text_style::<[f32; 4]>(op, |v| v.as_array().map(|_| rgba(v)));
            g.draw_strings(&mut view, &locations, &strs, &style)
                .unwrap();
        } else {
            panic!("unknown op {name}");
        }
    }
    canvas
}

fn item_kw<C>(kw: &Value, color: impl Fn(&Value) -> Option<C>) -> Item<C> {
    Item {
        color: kw.get("color").and_then(color),
        alpha: opt_f32(kw.get("alpha")),
        width: opt_f32(kw.get("width")),
        distance: opt_f32(kw.get("distance")),
        aa_width: opt_f32(kw.get("aa_width")),
    }
}

fn text_item<C>(kw: &Value, color: impl Fn(&Value) -> Option<C>) -> TextItem<C> {
    TextItem {
        font: kw.get("font").and_then(|v| v.as_str()).map(String::from),
        size: kw.get("size").and_then(|v| v.as_f64()),
        alpha: kw.get("alpha").and_then(text_alpha),
        color: kw.get("color").and_then(color),
        rotation: kw.get("rotation").and_then(|v| v.as_f64()),
    }
}

fn draw_kw<C: Clone>(
    npz: &mut Npz,
    s: &Value,
    color: impl Fn(&Value) -> Option<C>,
    colors: impl Fn(&mut Npz, Option<&Value>) -> Option<Vec<C>>,
) -> Draw<C> {
    Draw {
        color: s.get("color").and_then(color),
        colors: colors(npz, s.get("colors")),
        alpha: opt_f32(s.get("alpha")),
        width: opt_f32(s.get("width")),
        widths: vec_or_key_f32(npz, s.get("widths")),
        distance: opt_f32(s.get("distance")),
        aa_width: opt_f32(s.get("aa_width")),
        aa_widths: vec_or_key_f32(npz, s.get("aa_widths")),
    }
}

fn coords_of(op: &Value) -> Vec<f32> {
    f32s(&op["coords"])
}

macro_rules! kd_items {
    ($kd:expr, $kind:expr, $c:expr, $kw:expr) => {
        match $kind {
            "line" => $kd.line([$c[0], $c[1], $c[2], $c[3]], $kw),
            "rect" => $kd.rect([$c[0], $c[1], $c[2], $c[3]], $kw),
            "circle" => $kd.circle([$c[0], $c[1], $c[2]], $kw),
            "triangle" => $kd.triangle([$c[0], $c[1], $c[2], $c[3], $c[4], $c[5]], $kw),
            "mesh_triangle" => $kd.mesh_triangle([$c[0], $c[1], $c[2], $c[3], $c[4], $c[5]], $kw),
            "ellipse" => $kd.ellipse([$c[0], $c[1], $c[2], $c[3]], $kw),
            other => panic!("unknown item {other}"),
        }
    };
}

macro_rules! kd_shapes {
    ($kd:expr, $name:expr, $coords:expr, $kw:expr) => {
        match $name {
            "lines" => $kd.lines(&$coords, &$kw),
            "rects" => $kd.rects(&$coords, &$kw),
            "circles" => $kd.circles(&$coords, &$kw),
            "triangles" => $kd.triangles(&$coords, &$kw),
            "mesh_triangles" => $kd.mesh_triangles(&$coords, &$kw),
            _ => $kd.ellipses(&$coords, &$kw),
        }
        .unwrap()
    };
}

fn run_kd4(device: &Device, rec: &Value, npz: &mut Npz) -> ArrayD<f32> {
    let canvas = f32_arr(npz, rec["canvas"].as_str().unwrap());
    let chunk = rec["chunk_size"].as_u64().map(|c| c as usize);
    let mut kd = KernelDraw4::new(device, &canvas.view(), chunk).unwrap();
    let color4 = |v: &Value| v.as_array().map(|_| rgba(v));
    for op in rec["ops"].as_array().unwrap() {
        let name = op["op"].as_str().unwrap();
        if SHAPES.contains(&name) {
            let coords = rows(&f32_arr(npz, op["coords"].as_str().unwrap()));
            let kw = draw_kw(npz, &op["style"], color4, colors4);
            kd_shapes!(kd, name, coords, kw);
        } else if name == "item" {
            let kind = op["kind"].as_str().unwrap();
            if kind == "draw_string" {
                let xy = f32s(&op["xy"]);
                kd.draw_string(
                    (xy[0] as f64, xy[1] as f64),
                    op["text"].as_str().unwrap(),
                    text_item(&op["kw"], color4),
                );
            } else {
                let c = coords_of(op);
                kd_items!(kd, kind, c, item_kw(&op["kw"], color4));
            }
        } else if name == "composite" {
            let locations = locations2(npz, op["locations"].as_str().unwrap());
            let blocks = blocks3(npz, &op["blocks"]);
            let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
            kd.composite(&locations, &views).unwrap();
        } else if name == "sprites" {
            let locations = locations3(npz, op["locations"].as_str().unwrap());
            let sprites = blocks3(npz, &op["sprites"]);
            let views: Vec<_> = sprites.iter().map(|b| b.view()).collect();
            kd.sprites(&locations, &views).unwrap();
        } else if name == "draw_strings" {
            let locations = locations2(npz, op["locations"].as_str().unwrap());
            let t = texts(op);
            let strs: Vec<&str> = t.iter().map(String::as_str).collect();
            kd.draw_strings(&locations, &strs, &text_style::<[f32; 4]>(op, color4))
                .unwrap();
        } else if name == "render" {
            kd.render().unwrap();
        } else {
            panic!("unknown op {name}");
        }
    }
    kd.read().unwrap()
}

fn run_kd1(device: &Device, rec: &Value, npz: &mut Npz) -> ArrayD<f32> {
    let canvas = f32_arr(npz, rec["canvas"].as_str().unwrap());
    let chunk = rec["chunk_size"].as_u64().map(|c| c as usize);
    let mut kd = KernelDraw1::new(device, &canvas.view(), chunk).unwrap();
    let color1 = |v: &Value| v.as_f64().map(|x| x as f32);
    for op in rec["ops"].as_array().unwrap() {
        let name = op["op"].as_str().unwrap();
        if SHAPES.contains(&name) {
            let coords = rows(&f32_arr(npz, op["coords"].as_str().unwrap()));
            let kw = draw_kw(npz, &op["style"], color1, vec_or_key_f32);
            kd_shapes!(kd, name, coords, kw);
        } else if name == "item" {
            let kind = op["kind"].as_str().unwrap();
            if kind == "draw_string" {
                let xy = f32s(&op["xy"]);
                kd.draw_string(
                    (xy[0] as f64, xy[1] as f64),
                    op["text"].as_str().unwrap(),
                    text_item(&op["kw"], color1),
                );
            } else {
                let c = coords_of(op);
                kd_items!(kd, kind, c, item_kw(&op["kw"], color1));
            }
        } else if name == "render" {
            kd.render().unwrap();
        } else {
            panic!("unknown op {name}");
        }
    }
    kd.read().unwrap()
}

fn run_kd3(device: &Device, rec: &Value, npz: &mut Npz) -> ArrayD<f32> {
    let canvas = f32_arr(npz, rec["canvas"].as_str().unwrap());
    let chunk = rec["chunk_size"].as_u64().map(|c| c as usize);
    let mut kd = KernelDraw3::new(device, &canvas.view(), chunk).unwrap();
    let color3 = |v: &Value| {
        v.as_array().map(|_| {
            let f = f32s(v);
            [f[0], f[1], f[2]]
        })
    };
    let colors3 = |npz: &mut Npz, v: Option<&Value>| -> Option<Vec<[f32; 3]>> {
        match v {
            Some(Value::String(k)) => Some(
                f32_arr(npz, k)
                    .outer_iter()
                    .map(|r| [r[0], r[1], r[2]])
                    .collect(),
            ),
            _ => None,
        }
    };
    for op in rec["ops"].as_array().unwrap() {
        let name = op["op"].as_str().unwrap();
        if SHAPES.contains(&name) {
            let coords = rows(&f32_arr(npz, op["coords"].as_str().unwrap()));
            let kw = draw_kw(npz, &op["style"], color3, colors3);
            kd_shapes!(kd, name, coords, kw);
        } else if name == "item" {
            let kind = op["kind"].as_str().unwrap();
            if kind == "draw_string" {
                let xy = f32s(&op["xy"]);
                kd.draw_string(
                    (xy[0] as f64, xy[1] as f64),
                    op["text"].as_str().unwrap(),
                    text_item(&op["kw"], |v| v.as_f64().map(|x| x as f32)),
                );
            } else {
                let c = coords_of(op);
                kd_items!(kd, kind, c, item_kw(&op["kw"], color3));
            }
        } else if name == "render" {
            kd.render().unwrap();
        } else {
            panic!("unknown op {name}");
        }
    }
    kd.read().unwrap().into_dyn()
}

fn fixtures(backend: Backend, generator: &str) -> Option<(Vec<Value>, f64, f64)> {
    let index = lib_index(backend)?;
    let cases = index["generators"].get(generator)?.as_array()?.clone();
    Some((
        cases,
        index["rtol"].as_f64().unwrap_or(1e-4),
        index["atol"].as_f64().unwrap_or(1e-4),
    ))
}

fn replay(
    generator: &str,
    run: impl Fn(&Device, &Value, &mut Npz) -> ArrayD<f32>,
    tolerance_fraction: f64,
) {
    let mut any = false;
    for device in devices() {
        let Some((cases, rtol, atol)) = fixtures(device.backend(), generator) else {
            eprintln!("no {generator} fixtures for {:?}", device.backend());
            continue;
        };
        any = true;
        for rec in &cases {
            let name = rec["name"].as_str().unwrap();
            let path = common::parity_lib_dir(device.backend()).join(rec["file"].as_str().unwrap());
            let mut npz = open_npz(&path);
            let got = run(&device, rec, &mut npz);
            let want = f32_arr(&mut npz, rec["output"].as_str().unwrap());
            check_canvas(
                &format!("{:?} {generator}.{name}", device.backend()),
                &got,
                &want,
                rtol,
                atol,
                tolerance_fraction,
            );
        }
        println!("{:?}: {generator}: {} cases", device.backend(), cases.len());
    }
    assert!(any, "no backend with {generator} fixtures");
}

// Rasterisation differences: the kernels are the same text on both sides and
// run on the same device, so canvases should agree to float tolerance. A small
// fraction is allowed for edge pixels where Python's float32 coordinate
// coercion and ours could round differently.
const EDGE_TOLERANCE: f64 = 0.002;

#[test]
fn graphics1_parity() {
    replay("graphics", run_graphics1, EDGE_TOLERANCE);
}

#[test]
fn graphics4_parity() {
    replay("graphics4", run_graphics4, EDGE_TOLERANCE);
}

#[test]
fn kerneldraw_parity() {
    replay(
        "kerneldraw",
        |device, rec, npz| match rec["kind"].as_str().unwrap() {
            "kd4" => run_kd4(device, rec, npz),
            "kd1" => run_kd1(device, rec, npz),
            _ => run_kd3(device, rec, npz),
        },
        EDGE_TOLERANCE,
    );
}

#[test]
fn typeset_parity() {
    let mut any = false;
    for backend in [Backend::Metal, Backend::Ocl, Backend::Cuda] {
        let Some((cases, _, _)) = fixtures(backend, "typeset") else {
            continue;
        };
        any = true;
        let rec = &cases[0];
        let path = common::parity_lib_dir(backend).join(rec["file"].as_str().unwrap());
        let mut npz = open_npz(&path);
        for case in rec["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let want = f32_arr(&mut npz, case["block"].as_str().unwrap());
            let got = typeset::render_text(
                case["text"].as_str().unwrap(),
                case["font"].as_str().unwrap(),
                case["size"].as_f64(),
                case["rotation"].as_f64(),
            )
            .unwrap();
            let bbox: Vec<usize> = case["bbox"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(got.bbox.to_vec(), bbox, "{name}: bbox");
            assert_eq!(got.block.shape(), want.shape(), "{name}: block shape");
            let g: Vec<f32> = got.block.iter().cloned().collect();
            allclose_f32(&g, want.as_slice().unwrap(), 0.0, 1e-6)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        for o in rec["offsets"].as_array().unwrap() {
            let (dx, dy) = typeset::compute_offset(
                o["angle"].as_f64().unwrap(),
                o["h"].as_u64().unwrap() as usize,
                o["w"].as_u64().unwrap() as usize,
            );
            let want = o["dxdy"].as_array().unwrap();
            assert_eq!(
                (dx, dy),
                (want[0].as_f64().unwrap(), want[1].as_f64().unwrap()),
                "offset {o}"
            );
        }
        println!(
            "typeset: {} text cases match FreeType output",
            rec["cases"].as_array().unwrap().len()
        );
        break; // text rendering has no device dependence
    }
    assert!(any, "no typeset fixtures");
}

fn run_paths(device: &Device, rec: &Value, npz: &mut Npz) -> ArrayD<f32> {
    let (w, h) = (
        rec["width"].as_u64().unwrap() as usize,
        rec["height"].as_u64().unwrap() as usize,
    );
    let mut r = PathRenderer::new(device, w, h).unwrap();
    let name = rec["name"].as_str().unwrap();
    match name {
        "dashed" => {
            for s in rec["segments"].as_array().unwrap() {
                let xyxy: Vec<f64> = s["xyxy"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let pattern: Vec<f64> = s["pattern"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                r.draw_dashed_line(
                    xyxy[0],
                    xyxy[1],
                    xyxy[2],
                    xyxy[3],
                    rgba(&s["color"]),
                    s["linewidth"].as_f64().unwrap() as f32,
                    &pattern,
                    s["offset"].as_f64().unwrap(),
                );
            }
            r.kd.render().unwrap();
        }
        "images" => {
            for im in rec["images"].as_array().unwrap() {
                let key = im["key"].as_str().unwrap();
                let a = f32_arr(npz, key);
                let (x, y) = (im["x"].as_f64().unwrap(), im["y"].as_f64().unwrap());
                if a.ndim() == 2 {
                    r.draw_gray_image(x, y, a.view().into_dimensionality().unwrap())
                        .unwrap();
                } else {
                    let a3: Array3<f32> = a.into_dimensionality().unwrap();
                    // uint8 images were saved as uint8; Python scales them by 255
                    let scaled = if key == "rgb8" {
                        a3.mapv(|v| v / 255.0)
                    } else {
                        a3
                    };
                    r.draw_image(x, y, scaled.view()).unwrap();
                }
            }
            r.kd.render().unwrap();
        }
        "text" => {
            for t in rec["texts"].as_array().unwrap() {
                let cov = f32_arr(npz, t["key"].as_str().unwrap()).mapv(|v| v / 255.0);
                let cov2: Array2<f32> = cov.into_dimensionality().unwrap();
                r.draw_text_image(
                    cov2.view(),
                    t["x"].as_i64().unwrap(),
                    t["y"].as_i64().unwrap() + 1,
                    t["angle"].as_f64().unwrap(),
                    rgba(&t["rgba"]),
                )
                .unwrap();
            }
            r.kd.render().unwrap();
        }
        _ => {
            let polygons: Vec<Vec<[f64; 2]>> = rec["polygons"]
                .as_array()
                .unwrap()
                .iter()
                .map(|k| {
                    let a = read_any(npz, k.as_str().unwrap());
                    let pts: Vec<f64> = match a.dtype {
                        yapocis::DType::Float64 => a.as_slice::<f64>().to_vec(),
                        _ => a.as_slice::<f32>().iter().map(|v| *v as f64).collect(),
                    };
                    pts.chunks(2).map(|p| [p[0], p[1]]).collect()
                })
                .collect();
            let face = rec["face"].as_array().map(|_| rgba(&rec["face"]));
            let edge = rgba(&rec["edge"]);
            let lw = rec["linewidth_px"].as_f64().unwrap();
            let dashes = rec.get("dashes").and_then(|d| d.as_array()).map(|d| {
                (
                    d[0].as_f64().unwrap(),
                    d[1].as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect(),
                )
            });
            r.draw_path(&polygons, edge, face, lw, dashes).unwrap();
        }
    }
    r.read().unwrap().into_dyn()
}

#[test]
fn paths_parity() {
    // The rotated text bitmap is resampled by PIL in Python and by our own
    // bilinear rotation here; they are close but not bit-identical, so the
    // text case allows a larger fraction of differing pixels.
    let mut any = false;
    for device in devices() {
        let Some((cases, rtol, atol)) = fixtures(device.backend(), "paths") else {
            continue;
        };
        any = true;
        for rec in &cases {
            let name = rec["name"].as_str().unwrap();
            let path = common::parity_lib_dir(device.backend()).join(rec["file"].as_str().unwrap());
            let mut npz = open_npz(&path);
            let got = run_paths(&device, rec, &mut npz);
            let want = f32_arr(&mut npz, rec["output"].as_str().unwrap());
            let tol = if name == "text" { 0.05 } else { EDGE_TOLERANCE };
            check_canvas(
                &format!("{:?} paths.{name}", device.backend()),
                &got,
                &want,
                rtol,
                atol,
                tol,
            );
        }
        println!("{:?}: paths: {} cases", device.backend(), cases.len());
    }
    assert!(any, "no paths fixtures");
}

#[test]
fn canvases_are_not_empty() {
    // sanity: the fixtures actually drew something
    for backend in [Backend::Metal, Backend::Ocl] {
        let Some((cases, _, _)) = fixtures(backend, "graphics") else {
            continue;
        };
        let rec = &cases[0];
        let mut npz =
            open_npz(&common::parity_lib_dir(backend).join(rec["file"].as_str().unwrap()));
        let out = f32_arr(&mut npz, "output");
        assert!(out.iter().any(|&v| v > 0.0));
        let _ = out.index_axis(Axis(0), 0);
        let _ = IxDyn(&[1]);
        let _: Option<Npy> = None;
        break;
    }
}
