//! `yapocis.lib.convolve` and the runtime-weight side of `yapocis.lib.gaussian`.
//!
//! In Python, `Convolve1D(name=..., conv=[...])` and `Gaussian(gaussian_weights=...)`
//! render the `convolve` and `convolvesep` Mako templates with the caller's
//! weights baked into the kernel source. The bundle only carries named
//! variants, so this module ports those two templates to Rust: it generates
//! the same source text (kernel bodies with unrolled weights) for each backend,
//! reusing the shim prologue of the bundled variant, and builds the matching
//! interface description the runtime needs.

use crate::interfaces::{convolve_box3, convolvesep};
use crate::rt::{CallArg, CallOptions};
use crate::{host, Backend, Device, Error, Program, Result};
use ndarray::{ArrayD, ArrayViewD};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use yapocis_bundle::manifest::CudaInfo;
use yapocis_bundle::{
    DType, Dim, InterfaceEntry, KernelEntry, ParamEntry, ParamKind, Role, SourceEntry,
};

/// A weight as the template writes it: Python `repr(float)`, which like
/// Rust's `{:?}` is the shortest round-trip decimal (`0.1`, `1.0`, `1e-05`/`1e-5`).
pub fn weight_literal(w: f64) -> String {
    format!("{w:?}")
}

/// The text of a bundled source before its first kernel definition: the shim
/// and helper macros each backend needs.
fn prologue<'a>(source: &'a str, markers: &[&str]) -> &'a str {
    let cut = markers
        .iter()
        .filter_map(|m| source.find(m))
        .min()
        .unwrap_or(source.len());
    // back up to the start of that line
    let start = source[..cut].rfind('\n').map(|i| i + 1).unwrap_or(0);
    &source[..start]
}

fn backend_source(
    backend: Backend,
    bundled: fn(&str) -> Option<&'static str>,
) -> Result<&'static str> {
    bundled(backend.name()).ok_or(Error::BackendUnavailable(backend))
}

// ---------------------------------------------------------------------------
// convolve.mako
// ---------------------------------------------------------------------------

/// Render `convolve.mako` for `backend` with `name` and `weights`.
pub fn convolve_source(backend: Backend, name: &str, weights: &[f64]) -> Result<String> {
    let bundled = backend_source(backend, convolve_box3::source)?;
    let left = weights.len() / 2;
    let n = weights.len();
    let lit: Vec<String> = weights.iter().map(|w| weight_literal(*w)).collect();
    let mut s = String::new();
    match backend {
        Backend::Ocl => {
            s.push_str(prologue(bundled, &["__kernel\nvoid box3("]));
            s.push_str(&format!(
                "__kernel\nvoid {name}(int width,  __global float* a, __global float* ret )\n{{\n    size_t i = get_global_id(0);\n    float sum;\n    size_t right = width-{left};\n    sum = 0.0;\n    if ((i > (size_t){left}) &&  (i < right)) {{\n"
            ));
            for (j, l) in lit.iter().enumerate() {
                s.push_str(&format!(
                    "       sum = sum + a[i+({})] * {l};\n",
                    j as i64 - left as i64
                ));
            }
            s.push_str("    }\n    ret[i] = sum;\n    return;\n}\n\n");
            s.push_str(&format!(
                "__kernel void {name}_fast(int width,  __global float* a, __global float* ret, int chunk)\n{{\n    size_t i = get_global_id(0) * chunk;\n    float sum;\n    size_t right = width-{left};\n    volatile __local float buffer[FAST];\n    for (int j = 0; j < {n}; j++) {{\n        buffer[j] = i+j >= width ? 0.0 : a[i+j];\n    }}\n    for (int j = 0; j < chunk-{n}; j++) {{\n        sum = 0.0;\n        int ipixel = i+j >= right ? right-1 : i+j;\n        if ((i > (size_t){left}) &&  (i < right)) {{\n"
            ));
            for l in &lit {
                s.push_str(&format!("           sum = sum + a[j] * {l};\n"));
            }
            s.push_str("        }\n        ret[ipixel] = sum;\n    }\n    return;\n}\n");
        }
        Backend::Metal => {
            s.push_str(prologue(bundled, &["kernel void box3("]));
            s.push_str(&format!(
                "kernel void {name}(constant int &width [[ buffer(0) ]],\n                    device float* a [[ buffer(1) ]],\n                    device float* ret [[ buffer(2) ]],\n                    uint i [[ thread_position_in_grid ]])\n{{\n    float sum = 0.0f;\n    size_t right = width-{left};\n    if ((i > (size_t){left}) &&  (i < right)) {{\n"
            ));
            for (j, l) in lit.iter().enumerate() {
                s.push_str(&format!(
                    "        sum = sum + a[i+({})] * {l};\n",
                    j as i64 - left as i64
                ));
            }
            s.push_str("    }\n    ret[i] = sum;\n}\n\n");
            s.push_str(&format!(
                "kernel void {name}_fast(constant int &width [[ buffer(0) ]],\n                         device float* a [[ buffer(1) ]],\n                         device float* ret [[ buffer(2) ]],\n                         constant int &chunk [[ buffer(3) ]],\n                         uint i [[ thread_position_in_grid ]])\n{{\n    size_t idx = i * (size_t)chunk;\n    float sum = 0.0f;\n    size_t right = width-{left};\n    threadgroup float buffer[FAST];\n    for (int j = 0; j < {n}; j++) {{\n        buffer[j] = idx+j >= (size_t)width ? 0.0f : a[idx+j];\n    }}\n    for (int j = 0; j < chunk-{n}; j++) {{\n        sum = 0.0f;\n        size_t ipixel = idx+j >= right ? (right-1) : (idx+j);\n        if ((idx > (size_t){left}) &&  (idx < right)) {{\n"
            ));
            for (k, l) in lit.iter().enumerate() {
                s.push_str(&format!("            sum = sum + buffer[{k}] * {l};\n"));
            }
            s.push_str("        }\n        ret[ipixel] = sum;\n    }\n}\n");
        }
        Backend::Cuda => {
            s.push_str(prologue(bundled, &["extern \"C\" __kernel\nvoid box3("]));
            s.push_str(&format!(
                "extern \"C\" __kernel\nvoid {name}(int width,  float* a, float* ret )\n{{\n    size_t i = get_global_id(0);\n    if (i >= (size_t)width) return;\n    float sum;\n    size_t right = width-{left};\n    sum = 0.0;\n    if ((i > (size_t){left}) &&  (i < right)) {{\n"
            ));
            for (j, l) in lit.iter().enumerate() {
                s.push_str(&format!(
                    "       sum = sum + a[i+({})] * {l};\n",
                    j as i64 - left as i64
                ));
            }
            s.push_str("    }\n    ret[i] = sum;\n    return;\n}\n\n");
            s.push_str(&format!(
                "extern \"C\" __kernel void {name}_fast(int width,  float* a, float* ret, int chunk)\n{{\n    size_t i = get_global_id(0) * chunk;\n    float sum;\n    size_t right = width-{left};\n    volatile __local float buffer[FAST];\n    for (int j = 0; j < {n}; j++) {{\n        buffer[j] = i+j >= width ? 0.0 : a[i+j];\n    }}\n    for (int j = 0; j < chunk-{n}; j++) {{\n        sum = 0.0;\n        int ipixel = i+j >= right ? right-1 : i+j;\n        if ((i > (size_t){left}) &&  (i < right)) {{\n"
            ));
            for l in &lit {
                s.push_str(&format!("           sum = sum + a[j] * {l};\n"));
            }
            s.push_str("        }\n        ret[ipixel] = sum;\n    }\n    return;\n}\n");
        }
    }
    Ok(s)
}

fn dim(name: &str, d: Dim) -> ParamEntry {
    ParamEntry {
        name: name.into(),
        kind: ParamKind::Dim,
        dtype: DType::Int32,
        ctype: "int".into(),
        role: None,
        dim: Some(d),
        of: Some(name.into()),
        derived: true,
    }
}

fn buffer(name: &str, role: Role) -> ParamEntry {
    ParamEntry {
        name: name.into(),
        kind: ParamKind::Buffer,
        dtype: DType::Float32,
        ctype: "float".into(),
        role: Some(role),
        dim: None,
        of: None,
        derived: false,
    }
}

fn out_like(of: &str) -> ParamEntry {
    ParamEntry {
        name: format!("out_like_{of}"),
        kind: ParamKind::OutLike,
        dtype: DType::Float32,
        ctype: "float".into(),
        role: None,
        dim: None,
        of: Some(of.into()),
        derived: true,
    }
}

fn int_scalar(name: &str) -> ParamEntry {
    ParamEntry {
        name: name.into(),
        kind: ParamKind::Scalar,
        dtype: DType::Int32,
        ctype: "int".into(),
        role: None,
        dim: None,
        of: None,
        derived: false,
    }
}

fn kernel(name: &str, symbol: &str, params: Vec<ParamEntry>) -> KernelEntry {
    let user_args = params
        .iter()
        .filter(|p| !p.derived)
        .map(|p| p.name.clone())
        .collect();
    let returns = params
        .iter()
        .filter(|p| p.kind == ParamKind::OutLike || matches!(p.role, Some(Role::Inout | Role::Out)))
        .map(|p| p.name.clone())
        .collect();
    KernelEntry {
        name: name.into(),
        symbol: symbol.into(),
        params,
        user_args,
        returns,
        dispatch_size: None,
        cuda: CudaInfo {
            takes_global_size: Some(false),
        },
        traits: serde_json::Value::Object(Default::default()),
    }
}

fn sources_for(
    id: &str,
    bundled_entry: &InterfaceEntry,
    render: impl Fn(Backend) -> Result<String>,
) -> Result<(BTreeMap<String, SourceEntry>, BTreeMap<String, String>)> {
    let mut sources = BTreeMap::new();
    let mut texts = BTreeMap::new();
    for backend in [Backend::Ocl, Backend::Metal, Backend::Cuda] {
        let Some(bundled) = bundled_entry.sources.get(backend.name()) else {
            continue;
        };
        let text = render(backend)?;
        sources.insert(
            backend.name().to_string(),
            SourceEntry {
                path: format!("generated/{}/{id}", backend.name()),
                sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
                build_options: bundled.build_options.clone(),
            },
        );
        texts.insert(backend.name().to_string(), text);
    }
    Ok((sources, texts))
}

/// The interface description `convolve.interface` yields for `name`, plus the
/// generated sources per backend.
pub fn convolve_entry(
    name: &str,
    weights: &[f64],
) -> Result<(InterfaceEntry, BTreeMap<String, String>)> {
    let bundled = crate::MANIFEST
        .interface(convolve_box3::ID)
        .expect("bundled convolve variant");
    let id = format!("convolve.{name}");
    let (sources, texts) = sources_for(&id, bundled, |b| convolve_source(b, name, weights))?;
    let entry = InterfaceEntry {
        name: "convolve".into(),
        variant: name.into(),
        id,
        interface_text: format!(
            "interface convolve {{\n    kernel {name}(size_of a, in float a, out_like a);\n    kernel {name}_fast(size_of a, in float a, out_like a, int chunk);\n}};\n"
        ),
        kernels: vec![
            kernel(name, name, vec![dim("a", Dim::SizeOf), buffer("a", Role::In), out_like("a")]),
            kernel(
                &format!("{name}_fast"),
                &format!("{name}_fast"),
                vec![dim("a", Dim::SizeOf), buffer("a", Role::In), out_like("a"), int_scalar("chunk")],
            ),
        ],
        sources,
        context: serde_json::json!({ "name": name, "conv": weights }),
    };
    Ok((entry, texts))
}

/// `Convolve1D(name=..., conv=[...])`: a 1-D convolution with weights chosen at runtime.
pub struct Convolve1D {
    program: Program,
    name: String,
}

impl Convolve1D {
    pub fn new(device: &Device, name: &str, weights: &[f64]) -> Result<Convolve1D> {
        if weights.is_empty() {
            return Err(Error::Invalid(
                "convolution weights must not be empty".into(),
            ));
        }
        let (entry, texts) = convolve_entry(name, weights)?;
        let text = texts
            .get(device.backend().name())
            .ok_or(Error::BackendUnavailable(device.backend()))?;
        Ok(Convolve1D {
            program: device.load(&entry, text)?,
            name: name.into(),
        })
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    /// `cnv.<name>(a)`: convolve a 1-D float array; the `len/2` samples at each
    /// end are zero, as in the kernel.
    pub fn convolve(&self, a: &ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
        let std = host::standard(a);
        let mut args = [CallArg::In(crate::rt::HostView::of(
            std.as_slice().unwrap(),
            a.shape(),
        ))];
        host::one_output(
            self.program
                .call(&self.name, &mut args, &CallOptions::default())?,
        )
    }

    /// `cnv.<name>_fast(a, chunk)`: the chunked variant. The caller passes the
    /// global size Python callers pass (`len / chunk` threads).
    pub fn convolve_fast(
        &self,
        a: &ArrayViewD<'_, f32>,
        chunk: i32,
        global: usize,
    ) -> Result<ArrayD<f32>> {
        let std = host::standard(a);
        let mut args = [
            CallArg::In(crate::rt::HostView::of(std.as_slice().unwrap(), a.shape())),
            CallArg::Scalar(chunk as f64),
        ];
        let opts = CallOptions {
            global: Some(vec![global]),
            ..Default::default()
        };
        host::one_output(
            self.program
                .call(&format!("{}_fast", self.name), &mut args, &opts)?,
        )
    }
}

// ---------------------------------------------------------------------------
// convolvesep.mako
// ---------------------------------------------------------------------------

/// Render `convolvesep.mako` for `backend` with named weight sets.
pub fn convolvesep_source(backend: Backend, convs: &[(String, Vec<f64>)]) -> Result<String> {
    let bundled = backend_source(backend, convolvesep::source)?;
    let mut s = String::new();
    match backend {
        Backend::Ocl => s.push_str(prologue(bundled, &["__kernel\nvoid gauss"])),
        Backend::Metal => s.push_str(prologue(bundled, &["kernel void gauss"])),
        Backend::Cuda => s.push_str(prologue(bundled, &["extern \"C\" __kernel\nvoid gauss"])),
    }
    for (name, conv) in convs {
        let margin = conv.len() / 2;
        let lit: Vec<String> = conv.iter().map(|w| weight_literal(*w)).collect();
        let off = |j: usize| j as i64 - margin as i64;
        match backend {
            Backend::Ocl => {
                s.push_str(&format!(
                    "__kernel\nvoid {name}(int vertical, int width, int height, __global float* a, __global float* ret )\n{{\n    size_t i = get_global_id(0);\n    size_t n = width*height;\n    float sum = 0.0;\n    if (!vertical && ((size_t)({margin}*height) < i) && (i < (n-{margin}*height))){{\n       // Horizontal convolution\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "\t\tsum = sum + a[i+({}*height)] * {l};\n",
                        off(j)
                    ));
                }
                s.push_str(&format!(
                    "    }}\n    if (vertical && ({margin} < i) && (i < (n-{margin}))) {{\n       // Vertical convolution\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!("\t\tsum = sum + a[i+({})] * {l};\n", off(j)));
                }
                s.push_str("    }\n    ret[i] = sum;\n    return;\n}\n");
                s.push_str(&format!(
                    "__kernel\nvoid {name}_h(int width, int height, __global float* a, __global float* ret )\n{{\n    size_t i = get_global_id(0);\n    size_t n = width*height;\n    float sum = 0.0;\n    int x = X(i);\n    int y = Y(i);\n    // Horizontal convolution\n    int inside = (x >= {margin}) && (x<width-{margin}) && (y >= {margin}) && (y < height-{margin});\n    //int inside = (((size_t)({margin}*height) < i) && (i < (n-{margin}*height)));\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "    sum += inside ? a[i+({}*height)] * {l} : 0.0f;\n",
                        off(j)
                    ));
                }
                s.push_str("    ret[i] = sum;\n    return;\n}\n");
                s.push_str(&format!(
                    "__kernel\nvoid {name}_v(int width, int height, __global float* a, __global float* ret )\n{{\n    size_t i = get_global_id(0);\n    size_t n = width*height;\n    int x = X(i);\n    int y = Y(i);\n    // Vertical convolution\n    float sum = 0.0;\n    int inside = (x >= {margin}) && (x<width-{margin}) && (y >= {margin}) && (y < height-{margin});\n    //int inside = ({margin} < i) && (i < (n-{margin}));\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "    sum +=  inside ?  a[i+({})] * {l} : 0.0;\n",
                        off(j)
                    ));
                }
                s.push_str("\n    ret[i] = sum;\n    return;\n}\n");
            }
            Backend::Metal => {
                s.push_str(&format!(
                    "kernel void {name}(constant int &vertical [[ buffer(0) ]],\n                    constant int &width [[ buffer(1) ]],\n                    constant int &height [[ buffer(2) ]],\n                    const device float* a [[ buffer(3) ]],\n                    device float* ret [[ buffer(4) ]],\n                    uint i [[ thread_position_in_grid ]])\n{{\n    size_t n = width*height;\n    float sum = 0.0f;\n    if (!vertical && ((size_t)({margin}*height) < i) && (i < (n-{margin}*height))){{\n       // Horizontal convolution\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "        sum = sum + a[i+({}*height)] * {l};\n",
                        off(j)
                    ));
                }
                s.push_str(&format!(
                    "    }}\n    if (vertical && ({margin} < i) && (i < (n-{margin}))) {{\n       // Vertical convolution\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!("        sum = sum + a[i+({})] * {l};\n", off(j)));
                }
                s.push_str("    }\n    ret[i] = sum;\n}\n\n");
                s.push_str(&format!(
                    "kernel void {name}_h(constant int &width [[ buffer(0) ]],\n                      constant int &height [[ buffer(1) ]],\n                      const device float* a [[ buffer(2) ]],\n                      device float* ret [[ buffer(3) ]],\n                      uint i [[ thread_position_in_grid ]])\n{{\n    size_t n = width*height;\n    float sum = 0.0f;\n    int x = X(i);\n    int y = Y(i);\n    // Horizontal convolution\n    int inside = (x >= {margin}) && (x<width-{margin}) && (y >= {margin}) && (y < height-{margin});\n    //int inside = (((size_t)({margin}*height) < i) && (i < (n-{margin}*height)));\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "    sum += inside ? a[i+({}*height)] * {l} : 0.0f;\n",
                        off(j)
                    ));
                }
                s.push_str("    ret[i] = sum;\n}\n\n");
                s.push_str(&format!(
                    "kernel void {name}_v(constant int &width [[ buffer(0) ]],\n                      constant int &height [[ buffer(1) ]],\n                      const device float* a [[ buffer(2) ]],\n                      device float* ret [[ buffer(3) ]],\n                      uint i [[ thread_position_in_grid ]])\n{{\n    size_t n = width*height;\n    int x = X(i);\n    int y = Y(i);\n    // Vertical convolution\n    float sum = 0.0f;\n    int inside = (x >= {margin}) && (x<width-{margin}) && (y >= {margin}) && (y < height-{margin});\n    //int inside = ({margin} < i) && (i < (n-{margin}));\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "    sum +=  inside ?  a[i+({})] * {l} : 0.0f;\n",
                        off(j)
                    ));
                }
                s.push_str("\n    ret[i] = sum;\n}\n");
            }
            Backend::Cuda => {
                s.push_str(&format!(
                    "extern \"C\" __kernel\nvoid {name}(int vertical, int width, int height, float* a, float* ret )\n{{\n    size_t i = get_global_id(0);\n    size_t n = width*height;\n    if (i >= n) return;\n    float sum = 0.0;\n    if (!vertical && ((size_t)({margin}*height) < i) && (i < (n-{margin}*height))){{\n       // Horizontal convolution\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "\t\tsum = sum + a[i+({}*height)] * {l};\n",
                        off(j)
                    ));
                }
                s.push_str(&format!(
                    "    }}\n    if (vertical && ({margin} < i) && (i < (n-{margin}))) {{\n       // Vertical convolution\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!("\t\tsum = sum + a[i+({})] * {l};\n", off(j)));
                }
                s.push_str("    }\n    ret[i] = sum;\n    return;\n}\n");
                s.push_str(&format!(
                    "extern \"C\" __kernel\nvoid {name}_h(int width, int height, float* a, float* ret )\n{{\n    size_t i = get_global_id(0);\n    size_t n = width*height;\n    if (i >= n) return;\n    float sum = 0.0;\n    int x = X(i);\n    int y = Y(i);\n    // Horizontal convolution\n    // CUDA X(i) runs along height and Y(i) along width (helpers.mako)\n    int inside = (x >= {margin}) && (x<height-{margin}) && (y >= {margin}) && (y < width-{margin});\n    //int inside = (((size_t)({margin}*height) < i) && (i < (n-{margin}*height)));\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "    sum += inside ? a[i+({}*height)] * {l} : 0.0f;\n",
                        off(j)
                    ));
                }
                s.push_str("    ret[i] = sum;\n    return;\n}\n");
                s.push_str(&format!(
                    "extern \"C\" __kernel\nvoid {name}_v(int width, int height, float* a, float* ret )\n{{\n    size_t i = get_global_id(0);\n    size_t n = width*height;\n    if (i >= n) return;\n    int x = X(i);\n    int y = Y(i);\n    // Vertical convolution\n    float sum = 0.0;\n    // CUDA X(i) runs along height and Y(i) along width (helpers.mako)\n    int inside = (x >= {margin}) && (x<height-{margin}) && (y >= {margin}) && (y < width-{margin});\n    //int inside = ({margin} < i) && (i < (n-{margin}));\n"
                ));
                for (j, l) in lit.iter().enumerate() {
                    s.push_str(&format!(
                        "    sum +=  inside ?  a[i+({})] * {l} : 0.0;\n",
                        off(j)
                    ));
                }
                s.push_str("\n    ret[i] = sum;\n    return;\n}\n");
            }
        }
    }
    Ok(s)
}

/// The interface description `convolvesep.interface` yields for `convs`, plus
/// the generated sources per backend.
pub fn convolvesep_entry(
    convs: &[(String, Vec<f64>)],
) -> Result<(InterfaceEntry, BTreeMap<String, String>)> {
    let bundled = crate::MANIFEST
        .interface(convolvesep::ID)
        .expect("bundled convolvesep");
    let names: Vec<&str> = convs.iter().map(|(n, _)| n.as_str()).collect();
    let id = format!("convolvesep.{}", names.join("-"));
    let (sources, texts) = sources_for(&id, bundled, |b| convolvesep_source(b, convs))?;
    let mut kernels = Vec::new();
    let mut text = String::from("interface convolvesep {\n");
    for name in &names {
        text.push_str(&format!(
            "        kernel {name}(int vertical, width_of input, height_of input, in float input, out_like input);\n        extension res of {name}(int vertical, width_of input, height_of input, resident float input, resident float output);\n        kernel {name}_h(width_of input, height_of input, resident float input, resident float output);\n        kernel {name}_v(width_of input, height_of input, resident float input, resident float output);\n"
        ));
        let dims = || vec![dim("input", Dim::WidthOf), dim("input", Dim::HeightOf)];
        let mut main = vec![int_scalar("vertical")];
        main.extend(dims());
        main.push(buffer("input", Role::In));
        main.push(out_like("input"));
        kernels.push(kernel(name, name, main));
        let mut hv = dims();
        hv.push(buffer("input", Role::Resident));
        hv.push(buffer("output", Role::Resident));
        kernels.push(kernel(
            &format!("{name}_h"),
            &format!("{name}_h"),
            hv.clone(),
        ));
        kernels.push(kernel(&format!("{name}_v"), &format!("{name}_v"), hv));
        let mut res = vec![int_scalar("vertical")];
        res.extend(dims());
        res.push(buffer("input", Role::Resident));
        res.push(buffer("output", Role::Resident));
        kernels.push(kernel(&format!("{name}_res"), name, res));
    }
    text.push_str("    };\n");
    let entry = InterfaceEntry {
        name: "convolvesep".into(),
        variant: names.join("-"),
        id,
        interface_text: text,
        kernels,
        sources,
        context: serde_json::json!({ "convs": convs.iter().map(|(n, w)| serde_json::json!([n, w])).collect::<Vec<_>>() }),
    };
    Ok((entry, texts))
}

/// Compile a `convolvesep` program for arbitrary weight sets on `device`.
/// Names follow Python's `Gaussian`: `gauss<width>`.
pub fn load_convolvesep(device: &Device, weights: &[Vec<f64>]) -> Result<Program> {
    let convs: Vec<(String, Vec<f64>)> = weights
        .iter()
        .map(|w| (format!("gauss{}", w.len()), w.clone()))
        .collect();
    let (entry, texts) = convolvesep_entry(&convs)?;
    let text = texts
        .get(device.backend().name())
        .ok_or(Error::BackendUnavailable(device.backend()))?;
    Ok(device.load(&entry, text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_match_python_repr() {
        assert_eq!(weight_literal(0.1), "0.1");
        assert_eq!(weight_literal(1.0), "1.0");
        assert_eq!(weight_literal(0.3333333333), "0.3333333333");
        assert_eq!(weight_literal(0.3989422804014327), "0.3989422804014327");
    }

    #[test]
    fn generated_box3_matches_bundled_text() {
        // The bundled convolve.box3 variant was rendered by Mako from the same
        // weights, so the Rust renderer must reproduce its tokens exactly.
        let weights = [0.3333333333, 0.3333333333, 0.3333333333];
        for backend in [Backend::Ocl, Backend::Metal, Backend::Cuda] {
            let bundled = convolve_box3::source(backend.name()).unwrap();
            let generated = convolve_source(backend, "box3", &weights).unwrap();
            assert_same_tokens(&generated, bundled, backend);
        }
    }

    /// Mako's `%for` lines leave different blank lines and indentation than
    /// the Rust renderer; compare the token stream, which is what the compiler sees.
    fn tokens(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    /// Assert equal token streams, reporting the first difference with context.
    fn assert_same_tokens(generated: &str, bundled: &str, backend: Backend) {
        let g = tokens(generated);
        let b = tokens(bundled);
        // Weight literals may differ in the last ulp: numpy's pairwise sum and
        // exp are not bit-identical to Rust's, which float32 kernels never see.
        let same = |x: &&str, y: &&str| -> bool {
            if x == y {
                return true;
            }
            let num = |t: &str| t.trim_end_matches(';').parse::<f64>().ok();
            match (num(x), num(y)) {
                (Some(a), Some(b)) => (a - b).abs() <= 1e-14 * a.abs().max(b.abs()),
                _ => false,
            }
        };
        if let Some(i) = (0..g.len().max(b.len())).find(|&i| match (g.get(i), b.get(i)) {
            (Some(x), Some(y)) => !same(x, y),
            _ => true,
        }) {
            let lo = i.saturating_sub(6);
            panic!(
                "{backend:?}: token {i} differs\n generated: {:?}\n bundled:   {:?}",
                &g[lo..(i + 6).min(g.len())],
                &b[lo..(i + 6).min(b.len())]
            );
        }
    }

    #[test]
    fn generated_convolvesep_matches_bundled_text() {
        let ladder = crate::lib::gaussian::default_weights();
        let convs: Vec<(String, Vec<f64>)> = ladder
            .iter()
            .map(|w| (format!("gauss{}", w.len()), w.clone()))
            .collect();
        for backend in [Backend::Ocl, Backend::Metal, Backend::Cuda] {
            let bundled = convolvesep::source(backend.name()).unwrap();
            let generated = convolvesep_source(backend, &convs).unwrap();
            assert_same_tokens(&generated, bundled, backend);
        }
    }
}
