//! `examples/simple_openclose.py`: the same `sum` kernel written inline and run
//! through the untyped runtime API, with no bundle involved. Shows what the
//! generated interfaces do for you.
//!
//! cargo run -p yapocis --example simple_openclose --features ocl

mod common;

use yapocis::rt::{CallArg, CallOptions, HostView, InterfaceEntry, OutputValue};
use yapocis::Backend;

const OCL: &str = r#"
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}
"#;

const METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void sum(device const float *a [[ buffer(0) ]], device const float *b [[ buffer(1) ]],
                device float *c [[ buffer(2) ]], uint gid [[ thread_position_in_grid ]])
{ c[gid] = a[gid] + b[gid]; }
"#;

const CUDA: &str = r#"
extern "C" __global__ void sum(const float *a, const float *b, float *c, unsigned long long n)
{
  unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) c[gid] = a[gid] + b[gid];
}
"#;

fn main() -> yapocis::Result<()> {
    let device = common::device();
    let (source, takes_global_size) = match device.backend() {
        Backend::Ocl => (OCL, false),
        Backend::Metal => (METAL, false),
        Backend::Cuda => (CUDA, true),
    };
    // Describe the kernel the way the interface `kernel sum(in float a, in float b, out_like a);` does.
    let manifest_entry = serde_json::json!({
        "name": "inline", "variant": "default", "id": "inline", "interface_text": "",
        "kernels": [{
            "name": "sum", "symbol": "sum",
            "params": [
                {"name": "a", "kind": "buffer", "dtype": "float32", "ctype": "float", "role": "in"},
                {"name": "b", "kind": "buffer", "dtype": "float32", "ctype": "float", "role": "in"},
                {"name": "out_like_a", "kind": "out_like", "dtype": "float32", "ctype": "float", "of": "a", "derived": true}
            ],
            "user_args": ["a", "b"], "returns": ["out_like_a"],
            "cuda": {"takes_global_size": takes_global_size}
        }],
        "sources": {device.backend().name(): {"path": "inline", "sha256": "", "build_options": []}}
    });
    let entry: InterfaceEntry = serde_json::from_value(manifest_entry).unwrap();

    let mut rng = common::Rng::new(2);
    let a: Vec<f32> = (0..50_000).map(|_| rng.uniform()).collect();
    let b: Vec<f32> = (0..50_000).map(|_| rng.uniform()).collect();
    let shape = [a.len()];

    let t = std::time::Instant::now();
    let program = device.load(&entry, source)?;
    let mut args = [
        CallArg::In(HostView::of(&a, &shape)),
        CallArg::In(HostView::of(&b, &shape)),
    ];
    let out = program.call("sum", &mut args, &CallOptions::default())?;
    let elapsed = t.elapsed();
    let OutputValue::Host { bytes, .. } = &out.outputs[0].value else {
        unreachable!()
    };
    let c: &[f32] = bytemuck::cast_slice(bytes);
    let err: f32 = c
        .iter()
        .zip(&a)
        .zip(&b)
        .map(|((c, a), b)| (c - (a + b)).powi(2))
        .sum::<f32>()
        .sqrt();
    let norm: f32 = c.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("(|gpu - cpu|, |gpu|) = ({err}, {norm})");
    println!("Elapsed: {:.3} ms", elapsed.as_secs_f64() * 1e3);
    Ok(())
}
