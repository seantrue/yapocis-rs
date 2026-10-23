//! Shared CNN machinery for the VGG16 and ResNet50 ports (`yapocis.lib.vgg16`,
//! `yapocis.lib.resnet50`, `yapocis.models`): weight files, class labels,
//! ImageNet preprocessing, and the layer operations over device buffers.
//!
//! Layers run on the `cnn` interface the way the Python `*Resident` classes
//! do: activations stay on the device, kernels are enqueued asynchronously
//! and the caller waits once. The host-array convenience methods upload,
//! run one layer and read back, which is what the plain Python classes do.

use crate::interfaces::cnn::Cnn;
use crate::rt::{CallArg, CallOptions};
use crate::{Buffer, Device, Error, Result};
use image::{DynamicImage, RgbImage};
use ndarray::{Array1, Array3, ArrayD, ArrayView3, IxDyn};
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// ImageNet normalisation (PyTorch style), as in both Python classes.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Path of `weights.npz` for a model: `$YAPOCIS_MODELS/<model>/weights.npz`,
/// else the crate's `models/<model>/weights.npz` (not vendored; a checkout may
/// place it there). Mirrors `yapocis.models.get_weights_path`.
pub fn weights_path(model: &str) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(dir) = std::env::var("YAPOCIS_MODELS") {
        if !dir.is_empty() {
            candidates.push(PathBuf::from(dir).join(model).join("weights.npz"));
        }
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join(model)
            .join("weights.npz"),
    );
    candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .ok_or_else(|| {
            Error::Invalid(format!(
                "{model} weights not found; set YAPOCIS_MODELS to a directory holding {model}/weights.npz \
                 (looked at {})",
                candidates
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// ImageNet class labels shipped with the models (`classes.txt`).
pub fn imagenet_labels(model: &str) -> Vec<&'static str> {
    let text = match model {
        "vgg16" => include_str!("../../models/vgg16/classes.txt"),
        "resnet50" => include_str!("../../models/resnet50/classes.txt"),
        _ => "",
    };
    text.lines().map(str::trim).collect()
}

/// Model weights loaded from `weights.npz` (the layout `yapocis-tools models` writes).
pub struct Weights {
    pub tensors: HashMap<String, ArrayD<f32>>,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Weights> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?;
        let mut npz =
            NpzReader::new(file).map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?;
        let names = npz
            .names()
            .map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))?;
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let key = name.trim_end_matches(".npy").to_string();
            let array: ArrayD<f32> = npz
                .by_name(&name)
                .map_err(|e| Error::Invalid(format!("{}: {key}: {e}", path.display())))?;
            tensors.insert(key, array.as_standard_layout().to_owned());
        }
        Ok(Weights { tensors })
    }

    pub fn get(&self, name: &str) -> Result<&ArrayD<f32>> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("weights have no tensor {name:?}")))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Weights uploaded once to a device (`*Resident.allocate_buffers` pre-writes them).
pub struct DeviceWeights {
    pub buffers: HashMap<String, Buffer<f32>>,
}

impl DeviceWeights {
    pub fn upload(device: &Device, weights: &Weights) -> Result<DeviceWeights> {
        let mut buffers = HashMap::with_capacity(weights.len());
        for (name, tensor) in &weights.tensors {
            let std = tensor.as_standard_layout();
            buffers.insert(
                name.clone(),
                device.upload(std.as_slice().unwrap(), tensor.shape())?,
            );
        }
        Ok(DeviceWeights { buffers })
    }

    pub fn get(&self, name: &str) -> Result<&Buffer<f32>> {
        self.buffers
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("weights have no tensor {name:?}")))
    }
}

/// Indices and probabilities of the `k` largest entries, largest first
/// (`np.argsort(probs)[::-1][:k]`).
pub fn top_k(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.into_iter().map(|i| (i, probs[i])).collect()
}

fn hwc_to_normalised_chw(img: &RgbImage) -> Array3<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = Array3::<f32>::zeros((3, h, w));
    for (x, y, px) in img.enumerate_pixels() {
        for c in 0..3 {
            let v = px.0[c] as f32 / 255.0;
            out[[c, y as usize, x as usize]] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    out
}

/// `VGG16.preprocess_image`: resize to 224×224 (PIL BILINEAR, exactly: [`super::pil`]),
/// scale to [0, 1], ImageNet-normalise, (3, 224, 224).
pub fn preprocess_vgg16(img: &RgbImage) -> Array3<f32> {
    let resized = if img.width() == 224 && img.height() == 224 {
        img.clone()
    } else {
        super::pil::resize_bilinear(img, 224, 224)
    };
    hwc_to_normalised_chw(&resized)
}

/// `Image.fromarray((image * 255).astype(np.uint8))` of an (H, W, 3) float image in [0, 1].
pub fn quantize(image: &ArrayView3<'_, f32>) -> RgbImage {
    let (h, w, _) = image.dim();
    RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let v = |c: usize| (image[[y as usize, x as usize, c]] * 255.0) as u8;
        image::Rgb([v(0), v(1), v(2)])
    })
}

/// `VGG16.preprocess_image` of an (H, W, 3) float image in [0, 1]: one already 224×224 is
/// normalised as it is; any other is quantised to 8 bits and resized.
pub fn preprocess_vgg16_array(image: &ArrayView3<'_, f32>) -> Array3<f32> {
    let (h, w, _) = image.dim();
    if (h, w) != (224, 224) {
        return preprocess_vgg16(&quantize(image));
    }
    let mut out = Array3::<f32>::zeros((3, h, w));
    for ((y, x, c), &v) in image.indexed_iter() {
        if c < 3 {
            out[[c, y, x]] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    out
}

/// `ResNet50.preprocess_image`: shortest edge to 256 (PIL BILINEAR, exactly), centre
/// crop 224×224, scale to [0, 1], ImageNet-normalise, (3, 224, 224).
pub fn preprocess_resnet50(img: &RgbImage) -> Array3<f32> {
    let (w, h) = (img.width(), img.height());
    // Python: int(256 * w / h) truncates
    let (new_w, new_h) = if h < w {
        ((256.0 * w as f64 / h as f64) as u32, 256)
    } else {
        (256, (256.0 * h as f64 / w as f64) as u32)
    };
    let resized = super::pil::resize_bilinear(img, new_w, new_h);
    let left = (new_w - 224) / 2;
    let top = (new_h - 224) / 2;
    let cropped = DynamicImage::ImageRgb8(resized)
        .crop_imm(left, top, 224, 224)
        .to_rgb8();
    hwc_to_normalised_chw(&cropped)
}

/// Build an RGB image from an (H, W, 3) `u8` array (what `np.array(PIL image)` holds).
pub fn rgb_from_hwc(hwc: &ndarray::ArrayView3<'_, u8>) -> RgbImage {
    let (h, w, _) = hwc.dim();
    let mut img = RgbImage::new(w as u32, h as u32);
    for (x, y, px) in img.enumerate_pixels_mut() {
        for c in 0..3 {
            px.0[c] = hwc[[y as usize, x as usize, c]];
        }
    }
    img
}

/// Layer operations on device buffers (the `_res` kernels), enqueued
/// asynchronously; call [`Layers::wait`] before reading results.
pub struct Layers {
    pub cnn: Cnn,
    opts: CallOptions,
}

impl Layers {
    pub fn load(device: &Device) -> Result<Layers> {
        Ok(Layers {
            cnn: Cnn::load(device)?,
            opts: CallOptions {
                async_exec: true,
                ..Default::default()
            },
        })
    }

    pub fn device(&self) -> &Device {
        self.cnn.device()
    }

    pub fn wait(&self) -> Result<()> {
        Ok(self.device().wait_all()?)
    }

    pub fn alloc(&self, shape: &[usize]) -> Result<Buffer<f32>> {
        Ok(self.device().alloc::<f32>(shape)?)
    }

    pub fn upload(&self, a: &ndarray::ArrayViewD<'_, f32>) -> Result<Buffer<f32>> {
        let std = a.as_standard_layout();
        Ok(self.device().upload(std.as_slice().unwrap(), a.shape())?)
    }

    fn call(&self, kernel: &str, args: &mut [CallArg<'_>]) -> Result<()> {
        self.cnn.program().call(kernel, args, &self.opts)?;
        Ok(())
    }

    fn conv(
        &self,
        kernel: &str,
        input: &Buffer<f32>,
        weights: &Buffer<f32>,
        bias: &Buffer<f32>,
        output: &Buffer<f32>,
        out_channels: usize,
    ) -> Result<()> {
        self.call(
            kernel,
            &mut [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(weights.as_any()),
                CallArg::Buffer(bias.as_any()),
                CallArg::Buffer(output.as_any()),
                CallArg::Scalar(out_channels as f64),
            ],
        )
    }

    /// 3×3 convolution, padding 1, stride 1. `output` is (out_channels, H, W).
    pub fn conv2d_3x3(
        &self,
        input: &Buffer<f32>,
        weights: &Buffer<f32>,
        bias: &Buffer<f32>,
        output: &Buffer<f32>,
        out_channels: usize,
    ) -> Result<()> {
        self.conv("conv2d_3x3_res", input, weights, bias, output, out_channels)
    }

    pub fn conv2d_1x1(
        &self,
        input: &Buffer<f32>,
        weights: &Buffer<f32>,
        bias: &Buffer<f32>,
        output: &Buffer<f32>,
        out_channels: usize,
    ) -> Result<()> {
        self.conv("conv2d_1x1_res", input, weights, bias, output, out_channels)
    }

    /// 7×7, padding 3, stride 2 (ResNet stem). `output` is (out_channels, H/2, W/2).
    pub fn conv2d_7x7_stride2(
        &self,
        input: &Buffer<f32>,
        weights: &Buffer<f32>,
        bias: &Buffer<f32>,
        output: &Buffer<f32>,
        out_channels: usize,
    ) -> Result<()> {
        self.conv(
            "conv2d_7x7_stride2_res",
            input,
            weights,
            bias,
            output,
            out_channels,
        )
    }

    pub fn conv2d_3x3_stride2(
        &self,
        input: &Buffer<f32>,
        weights: &Buffer<f32>,
        bias: &Buffer<f32>,
        output: &Buffer<f32>,
        out_channels: usize,
    ) -> Result<()> {
        self.conv(
            "conv2d_3x3_stride2_res",
            input,
            weights,
            bias,
            output,
            out_channels,
        )
    }

    pub fn conv2d_1x1_stride2(
        &self,
        input: &Buffer<f32>,
        weights: &Buffer<f32>,
        bias: &Buffer<f32>,
        output: &Buffer<f32>,
        out_channels: usize,
    ) -> Result<()> {
        self.conv(
            "conv2d_1x1_stride2_res",
            input,
            weights,
            bias,
            output,
            out_channels,
        )
    }

    /// ReLU; `output` may be `input` for an in-place pass.
    pub fn relu(&self, input: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.call(
            "relu_activation_res",
            &mut [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(output.as_any()),
            ],
        )
    }

    pub fn softmax(&self, input: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.call(
            "softmax_res",
            &mut [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(output.as_any()),
            ],
        )
    }

    /// 2×2 max pooling, stride 2. `output` is (C, H/2, W/2).
    pub fn max_pool_2x2(&self, input: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.call(
            "max_pool_2x2_res",
            &mut [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(output.as_any()),
            ],
        )
    }

    /// 3×3 max pooling, stride 2, padding 1. `output` is (C, H/2, W/2).
    pub fn max_pool_3x3_stride2(&self, input: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.call(
            "max_pool_3x3_stride2_res",
            &mut [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(output.as_any()),
            ],
        )
    }

    /// Global average pooling. `output` is (C,).
    pub fn global_avg_pool(&self, input: &Buffer<f32>, output: &Buffer<f32>) -> Result<()> {
        self.call(
            "global_avg_pool_res",
            &mut [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(output.as_any()),
            ],
        )
    }

    /// y = Wx + b. `weights` is (out, in), `output` is (out,).
    pub fn fully_connected(
        &self,
        input: &Buffer<f32>,
        weights: &Buffer<f32>,
        bias: &Buffer<f32>,
        output: &Buffer<f32>,
    ) -> Result<()> {
        self.call(
            "fully_connected_res",
            &mut [
                CallArg::Buffer(input.as_any()),
                CallArg::Buffer(weights.as_any()),
                CallArg::Buffer(bias.as_any()),
                CallArg::Buffer(output.as_any()),
            ],
        )
    }

    /// Element-wise `output = a + b` (residual connections).
    pub fn add_tensors(
        &self,
        a: &Buffer<f32>,
        b: &Buffer<f32>,
        output: &Buffer<f32>,
    ) -> Result<()> {
        self.call(
            "add_tensors_res",
            &mut [
                CallArg::Buffer(a.as_any()),
                CallArg::Buffer(b.as_any()),
                CallArg::Buffer(output.as_any()),
            ],
        )
    }

    /// Read a buffer back as a dynamic array after [`Layers::wait`].
    pub fn download(&self, buf: &Buffer<f32>) -> Result<ArrayD<f32>> {
        Ok(ArrayD::from_shape_vec(IxDyn(buf.shape()), buf.to_vec()?)?)
    }

    /// Read a (C,) buffer back as a vector.
    pub fn download_1d(&self, buf: &Buffer<f32>) -> Result<Array1<f32>> {
        Ok(Array1::from_vec(buf.to_vec()?))
    }
}

/// Validate a (3, 224, 224) preprocessed tensor.
pub(crate) fn check_input(image: &ArrayView3<'_, f32>) -> Result<()> {
    if image.dim() != (3, 224, 224) {
        return Err(Error::Invalid(format!(
            "expected a preprocessed (3, 224, 224) tensor, got {:?}",
            image.shape()
        )));
    }
    Ok(())
}
