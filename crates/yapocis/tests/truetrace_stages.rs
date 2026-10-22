//! `trace_mask`'s stages against the Python package's on a real stroke image
//! (`YAPOCIS_TRACE_STAGES` names an `.npz` written by impressionist's
//! `claude-debug/rustport_trace_stages_probe.py`): segments, polylines,
//! simplified polylines, rings, regions and the filled raster, each compared
//! exactly, so a drift is pinned to the first stage that differs.

use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use yapocis::lib::truetrace::{
    build_regions_with_holes, chain_segments_to_polylines, close_and_filter_polylines,
    douglas_peucker, fill_regions, Polyline, TrueTrace,
};
use yapocis::Device;

fn ragged(npz: &mut NpzReader<std::fs::File>, name: &str) -> Vec<Polyline> {
    let flat: Array2<f32> = npz.by_name(&format!("{name}.npy")).unwrap();
    let offsets: Array1<i64> = npz.by_name(&format!("{name}_offsets.npy")).unwrap();
    offsets
        .windows(2)
        .into_iter()
        .map(|w| {
            flat.slice(ndarray::s![w[0] as usize..w[1] as usize, ..])
                .to_owned()
        })
        .collect()
}

fn compare_polys(label: &str, got: &[Polyline], want: &[Polyline]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {} polylines vs {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.nrows(),
            w.nrows(),
            "{label}[{i}]: {} points vs {}",
            g.nrows(),
            w.nrows()
        );
        for (k, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}[{i}] value {k}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn trace_stages_match_python() {
    let Some(path) = std::env::var_os("YAPOCIS_TRACE_STAGES") else {
        eprintln!("YAPOCIS_TRACE_STAGES not set: skipping");
        return;
    };
    let mut npz = NpzReader::new(std::fs::File::open(&path).unwrap()).unwrap();
    let mask: Array2<u8> = npz.by_name("mask.npy").unwrap();
    let segs: Array2<f32> = npz.by_name("segs.npy").unwrap();
    let fill: Array2<f32> = npz.by_name("fill.npy").unwrap();
    let polys = ragged(&mut npz, "polys");
    let simp = ragged(&mut npz, "simp");
    let rings = ragged(&mut npz, "rings");
    let outers = ragged(&mut npz, "outers");
    let holes = ragged(&mut npz, "holes");
    let holes_per_region: Array1<i64> = npz.by_name("holes_per_region.npy").unwrap();

    let device = Device::default().unwrap();
    let tracer = TrueTrace::load(&device).unwrap();
    let (my_segs, _) = tracer.trace_segments(&mask.view(), 0.75).unwrap();
    assert_eq!(my_segs.dim(), segs.dim(), "segments: shape");
    let n = my_segs
        .iter()
        .zip(segs.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(n, 0, "segments: {n} values differ");
    println!("segments match ({} rows)", segs.nrows());

    let my_polys = chain_segments_to_polylines(&segs.view(), 0.0);
    compare_polys("polys", &my_polys, &polys);
    println!("polylines match ({})", polys.len());

    let my_simp: Vec<Polyline> = polys
        .iter()
        .filter(|p| p.nrows() >= 3)
        .map(|p| douglas_peucker(&p.view(), 0.75))
        .collect();
    compare_polys("simp", &my_simp, &simp);
    println!("simplified match ({})", simp.len());

    let my_rings = close_and_filter_polylines(&simp, 1.0, 4.0);
    compare_polys("rings", &my_rings, &rings);
    println!("rings match ({})", rings.len());

    let my_regions = build_regions_with_holes(&rings);
    let my_outers: Vec<Polyline> = my_regions.iter().map(|r| r.outer.clone()).collect();
    compare_polys("outers", &my_outers, &outers);
    let my_counts: Vec<i64> = my_regions.iter().map(|r| r.holes.len() as i64).collect();
    assert_eq!(my_counts, holes_per_region.to_vec(), "holes per region");
    let my_holes: Vec<Polyline> = my_regions
        .iter()
        .flat_map(|r| r.holes.iter().cloned())
        .collect();
    compare_polys("holes", &my_holes, &holes);
    println!("regions match ({})", outers.len());

    let my_fill = fill_regions(&my_regions, mask.dim(), "none", 0.0, 8, 0.0, 1.0).unwrap();
    let diffs: Vec<usize> = my_fill
        .iter()
        .zip(fill.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .collect();
    assert!(
        diffs.is_empty(),
        "fill: {} pixels differ; first {:?}",
        diffs.len(),
        &diffs[..diffs.len().min(8)]
    );
    println!("fill matches");
}
