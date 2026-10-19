//! `examples/vgg16.py`: ImageNet classification with VGG16, or FC6 embeddings
//! (`--embeddings`), for one image or a test set.
//!
//! Weights come from `--weights PATH` or `$YAPOCIS_MODELS/vgg16/weights.npz`
//! (convert them with `yapocis-tools models vgg16` in the Python package); the
//! example exits with a message when they are absent. Class names are vendored,
//! so `--labels` is not needed; the Rust model is always device-resident, so
//! `--resident` is accepted and ignored; test mode uses the three images
//! vendored with the crate.
//!
//! cargo run -p yapocis --example vgg16 --features metal --release -- IMAGE [--embeddings]
//! cargo run -p yapocis --example vgg16 --features metal --release -- --test

mod common;

use std::path::PathBuf;
use yapocis::lib::vgg16::Vgg16Layer;
use yapocis::lib::Vgg16;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn classify(
    vgg: &Vgg16,
    labels: &[&str],
    path: &std::path::Path,
    embeddings: bool,
) -> yapocis::Result<f64> {
    println!("\nProcessing: {}", path.display());
    println!("{}", "-".repeat(60));
    let t = std::time::Instant::now();
    let img = image::open(path).expect("image").to_rgb8();
    println!("Image size: ({}, {}, 3)", img.height(), img.width());
    let pre = vgg.preprocess(&img);
    let pre_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("Preprocessing time: {pre_ms:.2} ms");
    let t = std::time::Instant::now();
    if embeddings {
        let f = vgg.forward(&pre.view(), Vgg16Layer::Fc6)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("Inference time: {ms:.2} ms");
        let v = f.as_slice().unwrap();
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        println!("\nFC6 Embeddings: shape=({},)", v.len());
        println!("  Mean: {mean:.4}");
        println!("  Std:  {std:.4}");
        println!("  Min:  {:.4}", v.iter().cloned().fold(f32::MAX, f32::min));
        println!("  Max:  {:.4}", v.iter().cloned().fold(f32::MIN, f32::max));
        println!(
            "  L2 norm: {:.4}",
            v.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].abs().partial_cmp(&v[a].abs()).unwrap());
        println!("\nTop 10 embedding dimensions:");
        for (i, d) in idx.iter().take(10).enumerate() {
            println!("  {}. Dim {d:4}: {:+.4}", i + 1, v[*d]);
        }
        Ok(pre_ms + ms)
    } else {
        let probs = vgg.forward(&pre.view(), Vgg16Layer::Probabilities)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("Inference time: {ms:.2} ms");
        println!("Total time: {:.2} ms", pre_ms + ms);
        println!("\nTop 5 predictions:");
        for (i, (c, p)) in yapocis::lib::cnn::top_k(probs.as_slice().unwrap(), 5)
            .iter()
            .enumerate()
        {
            println!("  {}. {:30}  {:6.2}%", i + 1, labels[*c], p * 100.0);
        }
        Ok(pre_ms + ms)
    }
}

fn main() -> yapocis::Result<()> {
    let test = flag("--test");
    let image_arg = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--") && !a.ends_with(".npz"));
    if !test && image_arg.is_none() {
        eprintln!("error: either provide an image path or use --test");
        std::process::exit(2);
    }
    let device = common::device();
    let weights = match arg("--weights")
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(|| yapocis::lib::cnn::weights_path("vgg16"))
    {
        Ok(p) if p.exists() => p,
        Ok(p) => {
            println!("ERROR: Weights file not found: {}", p.display());
            std::process::exit(1);
        }
        Err(e) => {
            println!("ERROR: {e}\n\nVGG16 weights are required to run this example.");
            println!("Convert them with `yapocis-tools models vgg16` and set YAPOCIS_MODELS,");
            println!("or pass --weights path/to/weights.npz.");
            std::process::exit(1);
        }
    };
    println!("Using weights from: {}", weights.display());
    let vgg = Vgg16::load(&device, &weights)?;
    let labels = Vgg16::labels();
    let embeddings = flag("--embeddings");
    if test {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let images = [
            root.join("examples/data/test.jpg"),
            root.join("tests/data/golden_retriever.jpg"),
            root.join("tests/data/lemon.jpg"),
        ];
        println!(
            "\nFound {} image assets\n{}",
            images.len(),
            common::rule(60)
        );
        let mut total = 0.0;
        for (i, p) in images.iter().enumerate() {
            println!(
                "\n[{}/{}] {}",
                i + 1,
                images.len(),
                p.file_name().unwrap().to_string_lossy()
            );
            total += classify(&vgg, &labels, p, embeddings)?;
        }
        let avg = total / images.len() as f64;
        println!(
            "\n{}\nProcessed {}/{} images successfully",
            common::rule(60),
            images.len(),
            images.len()
        );
        println!("Average time per image: {avg:.2} ms");
        println!("Throughput: {:.2} images/sec", 1000.0 / avg);
    } else {
        classify(
            &vgg,
            &labels,
            std::path::Path::new(&image_arg.unwrap()),
            embeddings,
        )?;
    }
    Ok(())
}
