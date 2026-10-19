//! `examples/resnet50.py`: ImageNet classification (top-k) and 2048-d
//! embeddings with ResNet50, for one image or a test set.
//!
//! Weights come from `--weights PATH` or `$YAPOCIS_MODELS/resnet50/weights.npz`
//! (convert them with `yapocis-tools models resnet50` in the Python package);
//! the example exits with a message when they are absent. `--labels` is not
//! needed: the class names are vendored. The Rust model always keeps weights and
//! activations on the device, so `--resident` is accepted and ignored. Test mode
//! uses the three images vendored with the crate instead of the Python package's
//! 14 sample images.
//!
//! cargo run -p yapocis --example resnet50 --features metal --release -- IMAGE [--top-k 10] [--embeddings]
//! cargo run -p yapocis --example resnet50 --features metal --release -- --test [--max-images 2]

mod common;

use std::path::PathBuf;
use yapocis::lib::ResNet50;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

pub fn test_images() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        root.join("examples/data/test.jpg"),
        root.join("tests/data/golden_retriever.jpg"),
        root.join("tests/data/lemon.jpg"),
    ]
}

fn stats(e: &[f32]) -> (f32, f32, f32, f32, f32) {
    let n = e.len() as f32;
    let mean = e.iter().sum::<f32>() / n;
    let std = (e.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    let l2 = e.iter().map(|v| v * v).sum::<f32>().sqrt();
    let min = e.iter().cloned().fold(f32::MAX, f32::min);
    let max = e.iter().cloned().fold(f32::MIN, f32::max);
    (mean, std, l2, min, max)
}

fn main() -> yapocis::Result<()> {
    let positional: Vec<String> = {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut out = Vec::new();
        let mut skip = false;
        for a in &raw {
            if skip {
                skip = false;
                continue;
            }
            if [
                "--weights",
                "--labels",
                "--top-k",
                "--max-images",
                "--model",
            ]
            .contains(&a.as_str())
            {
                skip = true;
            } else if !a.starts_with("--") {
                out.push(a.clone());
            }
        }
        out
    };
    let test = flag("--test");
    if !test && positional.is_empty() {
        eprintln!("error: either provide an image path or use --test");
        std::process::exit(2);
    }
    let top_k: usize = arg("--top-k").map(|s| s.parse().unwrap()).unwrap_or(5);
    println!(
        "Using backend: {}",
        std::env::var("YAPOCIS_BACKEND").unwrap_or_else(|_| "auto".into())
    );
    let device = common::device();
    let weights = match arg("--weights") {
        Some(p) => Ok(PathBuf::from(p)),
        None => yapocis::lib::cnn::weights_path("resnet50"),
    };
    let weights = match weights {
        Ok(p) if p.exists() => p,
        Ok(p) => {
            println!("ERROR: Weights file not found: {}", p.display());
            std::process::exit(1);
        }
        Err(e) => {
            println!("ERROR: {e}\n\nResNet50 weights are required to run this example.");
            println!("Convert them with `yapocis-tools models resnet50` and set YAPOCIS_MODELS,");
            println!("or pass --weights path/to/weights.npz.");
            std::process::exit(1);
        }
    };
    println!("Using weights: {}", weights.display());
    println!("Loading ResNet50...");
    let resnet = ResNet50::load(&device, &weights)?;
    let labels = ResNet50::labels();
    println!("Loaded {} ImageNet class labels\n", labels.len());

    if test {
        let mut images = test_images();
        if let Some(m) = arg("--max-images") {
            images.truncate(m.parse().unwrap());
        }
        let (mut total, mut ok) = (0.0, 0);
        for (i, path) in images.iter().enumerate() {
            println!(
                "[{}/{}] {}",
                i + 1,
                images.len(),
                path.file_name().unwrap().to_string_lossy()
            );
            let img = image::open(path).expect("image").to_rgb8();
            let t = std::time::Instant::now();
            let top = resnet.predict(&img, 3)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            total += ms;
            ok += 1;
            println!(
                "  -> {:30} {:6.2}% ({ms:.1} ms)\n",
                labels[top[0].0],
                top[0].1 * 100.0
            );
        }
        let avg = total / ok as f64;
        println!("{}", common::rule(60));
        println!("Processed {ok}/{} images successfully", images.len());
        println!("Average time per image: {avg:.2} ms");
        println!("Throughput: {:.2} images/sec", 1000.0 / avg);
        println!("{}", common::rule(60));
    } else {
        let path = &positional[0];
        println!("\nClassifying: {path}");
        let img = image::open(path).expect("image").to_rgb8();
        let t = std::time::Instant::now();
        let top = resnet.predict(&img, top_k)?;
        println!(
            "Inference time: {:.2} ms\n",
            t.elapsed().as_secs_f64() * 1e3
        );
        println!("Top {top_k} predictions:");
        for (i, (idx, p)) in top.iter().enumerate() {
            println!("  {}. {:25} {:6.2}%", i + 1, labels[*idx], p * 100.0);
        }
        if flag("--embeddings") {
            let e = resnet.extract_embeddings(&img)?;
            let (mean, std, l2, min, max) = stats(e.as_slice().unwrap());
            println!("\nEmbeddings (2048-dimensional):");
            println!("  Mean: {mean:.4}");
            println!("  Std:  {std:.4}");
            println!("  L2 norm: {l2:.4}");
            println!("  Min: {min:.4}, Max: {max:.4}");
        }
    }
    Ok(())
}
