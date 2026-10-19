//! Per-backend launch geometry: default global sizes and the CUDA grid/block
//! and Metal grid/threadgroup conversions (`_normalize_sizes` and
//! `_calculate_grid_block` in the Python backends, plus pymetallic's default
//! threadgroup size).

use crate::derive::ArgShape;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Ocl,
    Metal,
    Cuda,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Ocl => "ocl",
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
        }
    }

    pub fn parse(s: &str) -> Option<Backend> {
        Some(match s.to_ascii_lowercase().as_str() {
            "ocl" | "opencl" => Backend::Ocl,
            "metal" | "metallic" | "mtl" => Backend::Metal,
            "cuda" => Backend::Cuda,
            _ => return None,
        })
    }
}

/// Resolved launch geometry: a 3-D grid plus a 3-D local size (CUDA block or
/// Metal threadgroup). OpenCL uses `global`/`local` directly and leaves `local`
/// to the driver when the caller gave none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    pub global: Vec<usize>,
    pub local: Option<Vec<usize>>,
}

/// The global size used when the caller gave none and the kernel has no `dispatch_size`.
///
/// * OpenCL and CUDA: the first argument with a `.size` (the first array), flattened.
/// * Metal: the first array; `(shape[0]*shape[1], 1)` when it is 2-D or more.
pub fn default_global_size(backend: Backend, args: &[ArgShape]) -> Option<Vec<usize>> {
    let first = args.iter().find_map(|a| a.shape())?;
    Some(match backend {
        Backend::Ocl | Backend::Cuda => vec![first.iter().product()],
        Backend::Metal => {
            if first.len() > 1 {
                vec![first[0] * first[1], 1]
            } else {
                vec![first[0]]
            }
        }
    })
}

fn pad3(v: &[usize]) -> [usize; 3] {
    let mut out = [1usize; 3];
    for (i, x) in v.iter().take(3).enumerate() {
        out[i] = *x;
    }
    out
}

/// `cu/kernel.py _calculate_grid_block`: block defaults to (256,1,1); grid = ceil(global/block).
pub fn cuda_grid_block(global: &[usize], local: Option<&[usize]>) -> ([usize; 3], [usize; 3]) {
    let block = local.map(pad3).unwrap_or([256, 1, 1]);
    let gs = pad3(global);
    let mut grid = [1usize; 3];
    for i in 0..3 {
        grid[i] = gs[i].div_ceil(block[i]);
    }
    (grid, block)
}

/// `metallic/kernel.py _normalize_sizes` plus pymetallic's default threadgroup:
/// the grid is the global size padded to 3-D; the threadgroup is the caller's
/// local size padded, else `(min(max(1, grid.x), 256), 1, 1)`.
pub fn metal_grid_tgs(global: &[usize], local: Option<&[usize]>) -> ([usize; 3], [usize; 3]) {
    let grid = pad3(global);
    let tgs = local.map(pad3).unwrap_or([grid[0].clamp(1, 256), 1, 1]);
    (grid, tgs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::DType;

    #[test]
    fn defaults_per_backend() {
        let args = [
            ArgShape::Scalar(3.0),
            ArgShape::array(&[7, 5, 3], DType::Float32),
        ];
        assert_eq!(default_global_size(Backend::Ocl, &args), Some(vec![105]));
        assert_eq!(default_global_size(Backend::Cuda, &args), Some(vec![105]));
        assert_eq!(
            default_global_size(Backend::Metal, &args),
            Some(vec![35, 1])
        );
        assert_eq!(
            default_global_size(Backend::Metal, &[ArgShape::array(&[11], DType::Int32)]),
            Some(vec![11])
        );
        assert_eq!(
            default_global_size(Backend::Ocl, &[ArgShape::Scalar(1.0)]),
            None
        );
    }

    #[test]
    fn geometry() {
        assert_eq!(cuda_grid_block(&[1000], None), ([4, 1, 1], [256, 1, 1]));
        assert_eq!(
            cuda_grid_block(&[64, 64], Some(&[16, 16])),
            ([4, 4, 1], [16, 16, 1])
        );
        assert_eq!(metal_grid_tgs(&[35, 1], None), ([35, 1, 1], [35, 1, 1]));
        assert_eq!(metal_grid_tgs(&[5000], None), ([5000, 1, 1], [256, 1, 1]));
        assert_eq!(metal_grid_tgs(&[8, 8], Some(&[8])), ([8, 8, 1], [8, 1, 1]));
    }
}
