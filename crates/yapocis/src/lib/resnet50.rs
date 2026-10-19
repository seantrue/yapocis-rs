//! `yapocis.lib.resnet50`: ResNet50 classification and 2048-d embeddings.
//!
//! Weights (batch norm already fused by `yapocis-tools models`) are uploaded
//! once at load; a forward pass keeps activations on the device and waits
//! once, as `ResNet50Resident` with async execution does.

use super::cnn::{
    check_input, imagenet_labels, preprocess_resnet50, top_k, weights_path, DeviceWeights, Layers,
    Weights,
};
use crate::{Buffer, Device, Result};
use image::RgbImage;
use ndarray::{Array1, Array3, ArrayView3};

/// Where to stop a forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResNet50Layer {
    /// 2048-d features after global average pooling
    Pool,
    /// 1000 class probabilities
    Probabilities,
}

pub struct ResNet50 {
    pub layers: Layers,
    pub weights: DeviceWeights,
}

/// (block index, in, mid, out, stride, downsample) for the 16 bottleneck blocks.
const BLOCKS: [(usize, usize, usize, usize, usize, bool); 16] = [
    (0, 64, 64, 256, 1, true),
    (1, 256, 64, 256, 1, false),
    (2, 256, 64, 256, 1, false),
    (3, 256, 128, 512, 2, true),
    (4, 512, 128, 512, 1, false),
    (5, 512, 128, 512, 1, false),
    (6, 512, 128, 512, 1, false),
    (7, 512, 256, 1024, 2, true),
    (8, 1024, 256, 1024, 1, false),
    (9, 1024, 256, 1024, 1, false),
    (10, 1024, 256, 1024, 1, false),
    (11, 1024, 256, 1024, 1, false),
    (12, 1024, 256, 1024, 1, false),
    (13, 1024, 512, 2048, 2, true),
    (14, 2048, 512, 2048, 1, false),
    (15, 2048, 512, 2048, 1, false),
];

impl ResNet50 {
    pub fn load(device: &Device, path: impl AsRef<std::path::Path>) -> Result<ResNet50> {
        let weights = Weights::load(path)?;
        Ok(ResNet50 {
            layers: Layers::load(device)?,
            weights: DeviceWeights::upload(device, &weights)?,
        })
    }

    /// Load with the weights found through `YAPOCIS_MODELS` (`resnet50/weights.npz`).
    pub fn load_default(device: &Device) -> Result<ResNet50> {
        ResNet50::load(device, weights_path("resnet50")?)
    }

    pub fn labels() -> Vec<&'static str> {
        imagenet_labels("resnet50")
    }

    pub fn preprocess(&self, image: &RgbImage) -> Array3<f32> {
        preprocess_resnet50(image)
    }

    /// `ResNet50._bottleneck_block`: 1×1 → 3×3 (stride 1 or 2) → 1×1 with a
    /// skip connection (identity or 1×1 projection), ReLU after the add.
    #[allow(clippy::too_many_arguments)]
    fn bottleneck(
        &self,
        x: &Buffer<f32>,
        block: usize,
        mid: usize,
        out: usize,
        stride: usize,
        downsample: bool,
    ) -> Result<Buffer<f32>> {
        let (h, w) = (x.shape()[1], x.shape()[2]);
        let (h_out, w_out) = if stride == 2 { (h / 2, w / 2) } else { (h, w) };
        let wt = |name: &str| self.weights.get(&format!("block{block}_{name}"));

        let out1 = self.layers.alloc(&[mid, h, w])?;
        self.layers
            .conv2d_1x1(x, wt("conv1_weights")?, wt("conv1_bias")?, &out1, mid)?;
        self.layers.relu(&out1, &out1)?;

        let out2 = self.layers.alloc(&[mid, h_out, w_out])?;
        if stride == 2 {
            self.layers.conv2d_3x3_stride2(
                &out1,
                wt("conv2_weights")?,
                wt("conv2_bias")?,
                &out2,
                mid,
            )?;
        } else {
            self.layers
                .conv2d_3x3(&out1, wt("conv2_weights")?, wt("conv2_bias")?, &out2, mid)?;
        }
        self.layers.relu(&out2, &out2)?;

        let out3 = self.layers.alloc(&[out, h_out, w_out])?;
        self.layers
            .conv2d_1x1(&out2, wt("conv3_weights")?, wt("conv3_bias")?, &out3, out)?;

        let shortcut = if downsample {
            let proj = self.layers.alloc(&[out, h_out, w_out])?;
            if stride == 2 {
                self.layers.conv2d_1x1_stride2(
                    x,
                    wt("downsample_weights")?,
                    wt("downsample_bias")?,
                    &proj,
                    out,
                )?;
            } else {
                self.layers.conv2d_1x1(
                    x,
                    wt("downsample_weights")?,
                    wt("downsample_bias")?,
                    &proj,
                    out,
                )?;
            }
            proj
        } else {
            x.clone()
        };

        let result = self.layers.alloc(&[out, h_out, w_out])?;
        self.layers.add_tensors(&out3, &shortcut, &result)?;
        self.layers.relu(&result, &result)?;
        Ok(result)
    }

    /// Forward pass from a preprocessed (3, 224, 224) tensor.
    pub fn forward(
        &self,
        image: &ArrayView3<'_, f32>,
        stop_at: ResNet50Layer,
    ) -> Result<Array1<f32>> {
        check_input(image)?;
        let input = self.layers.upload(&image.to_owned().into_dyn().view())?;

        // stem: 7×7 stride 2 → (64, 112, 112), ReLU, 3×3 stride 2 max pool → (64, 56, 56)
        let stem = self.layers.alloc(&[64, 112, 112])?;
        self.layers.conv2d_7x7_stride2(
            &input,
            self.weights.get("stem_weights")?,
            self.weights.get("stem_bias")?,
            &stem,
            64,
        )?;
        self.layers.relu(&stem, &stem)?;
        let pooled = self.layers.alloc(&[64, 56, 56])?;
        self.layers.max_pool_3x3_stride2(&stem, &pooled)?;

        let mut x = pooled;
        for (block, _in, mid, out, stride, downsample) in BLOCKS {
            x = self.bottleneck(&x, block, mid, out, stride, downsample)?;
        }

        let pool = self.layers.alloc(&[2048])?;
        self.layers.global_avg_pool(&x, &pool)?;
        if stop_at == ResNet50Layer::Pool {
            self.layers.wait()?;
            return self.layers.download_1d(&pool);
        }
        let logits = self.layers.alloc(&[1000])?;
        self.layers.fully_connected(
            &pool,
            self.weights.get("fc_weights")?,
            self.weights.get("fc_bias")?,
            &logits,
        )?;
        self.layers.softmax(&logits, &logits)?;
        self.layers.wait()?;
        self.layers.download_1d(&logits)
    }

    /// `ResNet50.predict`: top-k `(class index, probability)` for an image.
    pub fn predict(&self, image: &RgbImage, k: usize) -> Result<Vec<(usize, f32)>> {
        let probs = self.forward(&self.preprocess(image).view(), ResNet50Layer::Probabilities)?;
        Ok(top_k(probs.as_slice().unwrap(), k))
    }

    /// `ResNet50.extract_embeddings`: the 2048-d pooled features.
    pub fn extract_embeddings(&self, image: &RgbImage) -> Result<Array1<f32>> {
        self.forward(&self.preprocess(image).view(), ResNet50Layer::Pool)
    }
}
