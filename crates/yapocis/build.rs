//! Generate the typed interface structs for the vendored Yapocis3 bundle with
//! `yapocis_bundle::codegen`. The output goes to `$OUT_DIR/generated.rs` and is
//! included by `src/interfaces.rs`.

use std::path::PathBuf;
use yapocis_bundle::codegen::{generate, Options};

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let bundle = root.join("bundle");
    let text = std::fs::read_to_string(bundle.join("manifest.json")).expect("bundle/manifest.json");
    let generated = generate(
        &text,
        &bundle,
        &Options {
            yapocis_path: "crate".to_string(),
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    for path in &generated.rerun_if_changed {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!(
        "cargo:rustc-env=YAPOCIS_BUNDLE_VERSION={}",
        generated.yapocis_version
    );
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("generated.rs");
    std::fs::write(&out, generated.code).unwrap();
}
