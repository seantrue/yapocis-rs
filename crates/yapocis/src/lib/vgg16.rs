//! `yapocis.lib.vgg16`: VGG16 classification and FC6/FC7 embeddings.
//!
//! Weights are uploaded once at load (as `VGG16Resident.allocate_buffers`
//! does) and every forward pass keeps its activations on the device with a
//! single wait at the end, which is the `VGG16Resident` + async behaviour.
//! Numerically this is the same kernel sequence as the plain `VGG16` class.

use super::cnn::{
    check_input, imagenet_labels, preprocess_vgg16, top_k, weights_path, DeviceWeights, Layers,
    Weights,
};
use crate::{Buffer, Device, Error, Result};
use image::RgbImage;
use ndarray::{Array1, Array3, ArrayView3};

/// Where to stop a forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vgg16Layer {
    /// 4096-d FC6 features (after ReLU)
    Fc6,
    /// 4096-d FC7 features (after ReLU)
    Fc7,
    /// 1000 class probabilities
    Probabilities,
}

pub struct Vgg16 {
    pub layers: Layers,
    pub weights: DeviceWeights,
}

/// (block, layers, out_channels) for the five convolutional blocks.
const BLOCKS: [(usize, usize, usize); 5] = [
    (1, 2, 64),
    (2, 2, 128),
    (3, 3, 256),
    (4, 3, 512),
    (5, 3, 512),
];

impl Vgg16 {
    /// Load the kernels and the weights at `path` (`weights.npz`).
    pub fn load(device: &Device, path: impl AsRef<std::path::Path>) -> Result<Vgg16> {
        let weights = Weights::load(path)?;
        Ok(Vgg16 {
            layers: Layers::load(device)?,
            weights: DeviceWeights::upload(device, &weights)?,
        })
    }

    /// Load with the weights found through `YAPOCIS_MODELS` (`vgg16/weights.npz`).
    pub fn load_default(device: &Device) -> Result<Vgg16> {
        Vgg16::load(device, weights_path("vgg16")?)
    }

    pub fn labels() -> Vec<&'static str> {
        imagenet_labels("vgg16")
    }

    pub fn preprocess(&self, image: &RgbImage) -> Array3<f32> {
        preprocess_vgg16(image)
    }

    /// `VGG16._get_conv_layer_index`: global conv index from block and layer (1-based).
    fn conv_index(block: usize, layer_in_block: usize) -> usize {
        const OFFSETS: [usize; 5] = [0, 2, 4, 7, 10];
        OFFSETS[block - 1] + layer_in_block - 1
    }

    fn conv_block(
        &self,
        input: &Buffer<f32>,
        block: usize,
        layers: usize,
        out_channels: usize,
    ) -> Result<Buffer<f32>> {
        let (h, w) = (input.shape()[1], input.shape()[2]);
        let mut current = input.clone();
        for i in 1..=layers {
            let idx = Self::conv_index(block, i);
            let weights = self.weights.get(&format!("conv{idx}_weights"))?;
            let bias = self.weights.get(&format!("conv{idx}_bias"))?;
            let conv = self.layers.alloc(&[out_channels, h, w])?;
            self.layers
                .conv2d_3x3(&current, weights, bias, &conv, out_channels)?;
            self.layers.relu(&conv, &conv)?;
            current = conv;
        }
        let pool = self.layers.alloc(&[out_channels, h / 2, w / 2])?;
        self.layers.max_pool_2x2(&current, &pool)?;
        Ok(pool)
    }

    /// Forward pass from a preprocessed (3, 224, 224) tensor.
    pub fn forward(&self, image: &ArrayView3<'_, f32>, stop_at: Vgg16Layer) -> Result<Array1<f32>> {
        check_input(image)?;
        let mut x = self.layers.upload(&image.to_owned().into_dyn().view())?;
        for (block, layers, channels) in BLOCKS {
            x = self.conv_block(&x, block, layers, channels)?;
        }
        // flatten: the (512, 7, 7) buffer is already contiguous row-major
        let flat = self.layers.alloc(&[512 * 7 * 7])?;
        self.layers.wait()?;
        flat.write(&x.to_vec()?)?;

        let fc6 = self.layers.alloc(&[4096])?;
        self.layers.fully_connected(
            &flat,
            self.weights.get("fc0_weights")?,
            self.weights.get("fc0_bias")?,
            &fc6,
        )?;
        self.layers.relu(&fc6, &fc6)?;
        if stop_at == Vgg16Layer::Fc6 {
            self.layers.wait()?;
            return self.layers.download_1d(&fc6);
        }
        let fc7 = self.layers.alloc(&[4096])?;
        self.layers.fully_connected(
            &fc6,
            self.weights.get("fc1_weights")?,
            self.weights.get("fc1_bias")?,
            &fc7,
        )?;
        self.layers.relu(&fc7, &fc7)?;
        if stop_at == Vgg16Layer::Fc7 {
            self.layers.wait()?;
            return self.layers.download_1d(&fc7);
        }
        let fc8 = self.layers.alloc(&[1000])?;
        self.layers.fully_connected(
            &fc7,
            self.weights.get("fc2_weights")?,
            self.weights.get("fc2_bias")?,
            &fc8,
        )?;
        self.layers.softmax(&fc8, &fc8)?;
        self.layers.wait()?;
        self.layers.download_1d(&fc8)
    }

    /// `VGG16.predict`: top-k `(class index, probability)` for an image.
    pub fn predict(&self, image: &RgbImage, k: usize) -> Result<Vec<(usize, f32)>> {
        let probs = self.forward(&self.preprocess(image).view(), Vgg16Layer::Probabilities)?;
        Ok(top_k(probs.as_slice().unwrap(), k))
    }

    /// `VGG16.extract_features`: 4096-d FC6 or FC7 embedding.
    pub fn extract_features(&self, image: &RgbImage, layer: Vgg16Layer) -> Result<Array1<f32>> {
        if layer == Vgg16Layer::Probabilities {
            return Err(Error::Invalid("layer must be Fc6 or Fc7".into()));
        }
        self.forward(&self.preprocess(image).view(), layer)
    }
}
