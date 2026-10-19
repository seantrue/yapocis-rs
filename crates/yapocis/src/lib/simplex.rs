//! `yapocis.lib.simplex`: 2-D and 3-D simplex noise with fBm, turbulence
//! and ridged variants.

use crate::interfaces::simplex;
use crate::{Device, Result};
use ndarray::{Array2, Array3};

pub struct Simplex {
    pub kernels: simplex::Simplex,
}

/// Parameters of `Simplex.simplex2d` with the Python defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct Params2d {
    pub seed: i32,
    pub scale: f32,
    pub offset: (f32, f32),
    pub octaves: i32,
    pub persistence: f32,
    pub lacunarity: f32,
    pub tileable: bool,
}

impl Default for Params2d {
    fn default() -> Self {
        Params2d {
            seed: 0,
            scale: 1.0,
            offset: (0.0, 0.0),
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            tileable: false,
        }
    }
}

/// Parameters of `Simplex.simplex3d` with the Python defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct Params3d {
    pub seed: i32,
    pub scale: f32,
    pub offset: (f32, f32, f32),
    pub octaves: i32,
    pub persistence: f32,
    pub lacunarity: f32,
}

impl Default for Params3d {
    fn default() -> Self {
        Params3d {
            seed: 0,
            scale: 1.0,
            offset: (0.0, 0.0, 0.0),
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
        }
    }
}

impl Simplex {
    pub fn load(device: &Device) -> Result<Simplex> {
        Ok(Simplex {
            kernels: simplex::Simplex::load(device)?,
        })
    }

    /// `Simplex.simplex2d(width, height, ...)`: a `(height, width)` array in [0, 1].
    pub fn simplex2d(&self, width: usize, height: usize, p: &Params2d) -> Result<Array2<f32>> {
        let mut output = Array2::<f32>::zeros((height, width)).into_dyn();
        self.kernels.simplex2d(
            &mut output.view_mut(),
            p.seed,
            p.scale,
            p.offset.0,
            p.offset.1,
            p.octaves,
            p.persistence,
            p.lacunarity,
            p.tileable as i32,
        )?;
        Ok(output.into_dimensionality()?)
    }

    /// `Simplex.simplex3d(width, height, depth, ...)`: a `(depth, height, width)` array.
    pub fn simplex3d(
        &self,
        width: usize,
        height: usize,
        depth: usize,
        p: &Params3d,
    ) -> Result<Array3<f32>> {
        let mut output = Array3::<f32>::zeros((depth, height, width)).into_dyn();
        self.kernels.simplex3d(
            width as i32,
            height as i32,
            depth as i32,
            &mut output.view_mut(),
            p.seed,
            p.scale,
            p.offset.0,
            p.offset.1,
            p.offset.2,
            p.octaves,
            p.persistence,
            p.lacunarity,
        )?;
        Ok(output.into_dimensionality()?)
    }

    /// `Simplex.turbulence(width, height, seed, scale, octaves=6)`: `|2*noise - 1|`.
    pub fn turbulence(
        &self,
        width: usize,
        height: usize,
        seed: i32,
        scale: f32,
        octaves: i32,
    ) -> Result<Array2<f32>> {
        let noise = self.simplex2d(
            width,
            height,
            &Params2d {
                seed,
                scale,
                octaves,
                ..Default::default()
            },
        )?;
        Ok(noise.mapv(|v| (2.0 * v - 1.0).abs()))
    }

    /// `Simplex.ridged(...)`: `(1 - turbulence)^2`.
    pub fn ridged(
        &self,
        width: usize,
        height: usize,
        seed: i32,
        scale: f32,
        octaves: i32,
    ) -> Result<Array2<f32>> {
        let turb = self.turbulence(width, height, seed, scale, octaves)?;
        Ok(turb.mapv(|t| {
            let v = 1.0 - t;
            v * v
        }))
    }
}

/// `make_simplex2d`: one-off noise without keeping a `Simplex` around.
pub fn make_simplex2d(
    device: &Device,
    width: usize,
    height: usize,
    p: &Params2d,
) -> Result<Array2<f32>> {
    Simplex::load(device)?.simplex2d(width, height, p)
}

/// `make_simplex3d`.
pub fn make_simplex3d(
    device: &Device,
    width: usize,
    height: usize,
    depth: usize,
    p: &Params3d,
) -> Result<Array3<f32>> {
    Simplex::load(device)?.simplex3d(width, height, depth, p)
}
