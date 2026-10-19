//! Typed device buffers.

use crate::device::Device;
use crate::error::{Error, Result};
use crate::raw::RawBuffer;
use std::marker::PhantomData;
use std::sync::Arc;
use yapocis_bundle::DType;

/// Host element types that map onto interface dtypes.
pub trait Element: bytemuck::Pod + Send + Sync + 'static {
    const DTYPE: DType;
}

macro_rules! element {
    ($($t:ty => $d:expr),* $(,)?) => { $(impl Element for $t { const DTYPE: DType = $d; })* };
}
element! {
    i8 => DType::Int8, i16 => DType::Int16, i32 => DType::Int32, i64 => DType::Int64,
    u8 => DType::Uint8, u16 => DType::Uint16, u32 => DType::Uint32, u64 => DType::Uint64,
    f32 => DType::Float32, f64 => DType::Float64,
    half::f16 => DType::Float16,
    num_complex::Complex<f32> => DType::Complex64,
    num_complex::Complex<f64> => DType::Complex128,
}

/// A type-erased device buffer: the handle plus its logical shape and dtype.
/// This is what `resident` parameters and dimension derivation work with.
#[derive(Clone)]
pub struct BufferAny {
    pub(crate) device: Device,
    pub(crate) raw: Arc<dyn RawBuffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

impl BufferAny {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len_bytes(&self) -> usize {
        self.len() * self.dtype.size_of()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn alloc(device: Device, shape: &[usize], dtype: DType) -> Result<BufferAny> {
        let bytes = shape.iter().product::<usize>() * dtype.size_of();
        let raw = device.raw.alloc(bytes.max(1))?;
        Ok(BufferAny {
            device,
            raw: Arc::from(raw),
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Wrap an existing raw buffer (from the buffer cache) with a new shape.
    pub(crate) fn from_raw(
        device: Device,
        raw: Arc<dyn RawBuffer>,
        shape: &[usize],
        dtype: DType,
    ) -> BufferAny {
        BufferAny {
            device,
            raw,
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub(crate) fn upload_bytes(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.raw.len_bytes() {
            return Err(Error::BufferSize {
                have: self.raw.len_bytes(),
                need: data.len(),
            });
        }
        self.device.raw.upload(self.raw.as_ref(), data)
    }

    pub(crate) fn download_bytes(&self, out: &mut [u8]) -> Result<()> {
        if out.len() > self.raw.len_bytes() {
            return Err(Error::BufferSize {
                have: self.raw.len_bytes(),
                need: out.len(),
            });
        }
        self.device.raw.download(self.raw.as_ref(), out)
    }

    /// Read the whole buffer back as bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.len_bytes()];
        self.download_bytes(&mut out)?;
        Ok(out)
    }

    /// Reinterpret as a typed buffer; errors if the dtype differs.
    pub fn typed<T: Element>(self) -> std::result::Result<Buffer<T>, BufferAny> {
        if self.dtype == T::DTYPE {
            Ok(Buffer {
                any: self,
                _t: PhantomData,
            })
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Debug for BufferAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer<{}>{:?}", self.dtype, self.shape)
    }
}

/// A typed device buffer with a logical shape (row-major, like the numpy
/// arrays the Python runtime uploads).
#[derive(Clone, Debug)]
pub struct Buffer<T: Element> {
    pub(crate) any: BufferAny,
    _t: PhantomData<T>,
}

impl<T: Element> Buffer<T> {
    pub(crate) fn alloc(device: Device, shape: &[usize]) -> Result<Buffer<T>> {
        Ok(Buffer {
            any: BufferAny::alloc(device, shape, T::DTYPE)?,
            _t: PhantomData,
        })
    }

    pub(crate) fn upload(device: Device, data: &[T], shape: &[usize]) -> Result<Buffer<T>> {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "shape {shape:?} does not match {} elements",
            data.len()
        );
        let buf = Buffer::<T>::alloc(device, shape)?;
        buf.any.upload_bytes(bytemuck::cast_slice(data))?;
        Ok(buf)
    }

    pub fn shape(&self) -> &[usize] {
        &self.any.shape
    }

    pub fn len(&self) -> usize {
        self.any.len()
    }

    pub fn is_empty(&self) -> bool {
        self.any.is_empty()
    }

    pub fn device(&self) -> &Device {
        &self.any.device
    }

    /// Overwrite the buffer contents.
    pub fn write(&self, data: &[T]) -> Result<()> {
        self.any.upload_bytes(bytemuck::cast_slice(data))
    }

    /// Read the buffer back into `out` (same element count as the buffer).
    pub fn read_into(&self, out: &mut [T]) -> Result<()> {
        self.any.download_bytes(bytemuck::cast_slice_mut(out))
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut v = vec![T::zeroed(); self.len()];
        self.read_into(&mut v)?;
        Ok(v)
    }

    pub fn as_any(&self) -> &BufferAny {
        &self.any
    }

    pub fn into_any(self) -> BufferAny {
        self.any
    }
}

impl<T: Element> From<Buffer<T>> for BufferAny {
    fn from(b: Buffer<T>) -> BufferAny {
        b.any
    }
}
