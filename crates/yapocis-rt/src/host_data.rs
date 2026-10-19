//! Host memory for call outputs.
//!
//! An `out_like` result is downloaded straight into a `Vec` of its element
//! type, so the typed layer can take it over without a copy ([`HostData::into_vec`]).
//! Large outputs are allocated zeroed with `alloc_zeroed`, so the allocator
//! does not write the pages before the download does.
//!
//! On Linux, `YAPOCIS_HUGE_PAGES=1` also marks large outputs for transparent
//! huge pages (`MADV_HUGEPAGE`). It is off by default: with the common
//! `defrag=madvise` setting every huge-page fault may compact memory
//! synchronously, so on a host with fragmented memory a 64 MB output took
//! 220 ms instead of 43 ms (yellowfin, 2026-10-19), while without the hint it
//! took a steady 65 ms.

use crate::buffer::Element;
use std::any::Any;
use yapocis_bundle::DType;

/// Outputs at least this large are allocated with `alloc_zeroed` and may get
/// the huge-page hint (numpy uses 4 MiB too).
const HUGE_PAGE_HINT_BYTES: usize = 4 << 20;

trait HostVec: Send + Sync {
    fn bytes(&self) -> &[u8];
    fn bytes_mut(&mut self) -> &mut [u8];
    fn as_any(&self) -> &dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<T: bytemuck::Pod + Send + Sync> HostVec for Vec<T> {
    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// A host array's storage, typed by its dtype but readable as bytes.
pub struct HostData {
    inner: Box<dyn HostVec>,
}

impl HostData {
    /// Zeroed storage for `len` elements of `dtype`.
    pub fn zeroed(dtype: DType, len: usize) -> HostData {
        let inner: Box<dyn HostVec> = match dtype {
            DType::Bool | DType::Uint8 => Box::new(zeroed_vec::<u8>(len)),
            DType::Int8 => Box::new(zeroed_vec::<i8>(len)),
            DType::Int16 => Box::new(zeroed_vec::<i16>(len)),
            DType::Int32 => Box::new(zeroed_vec::<i32>(len)),
            DType::Int64 => Box::new(zeroed_vec::<i64>(len)),
            DType::Uint16 => Box::new(zeroed_vec::<u16>(len)),
            DType::Uint32 => Box::new(zeroed_vec::<u32>(len)),
            DType::Uint64 => Box::new(zeroed_vec::<u64>(len)),
            DType::Float16 => Box::new(zeroed_vec::<half::f16>(len)),
            DType::Float32 => Box::new(zeroed_vec::<f32>(len)),
            DType::Float64 => Box::new(zeroed_vec::<f64>(len)),
            DType::Complex64 => Box::new(zeroed_vec::<num_complex::Complex<f32>>(len)),
            DType::Complex128 => Box::new(zeroed_vec::<num_complex::Complex<f64>>(len)),
        };
        HostData { inner }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.inner.bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.bytes_mut()
    }

    /// Take the storage as a `Vec<T>` without copying. Fails (returning the
    /// data unchanged) if `T` is not the element type it was allocated with.
    pub fn into_vec<T: Element>(self) -> Result<Vec<T>, HostData> {
        if !self.inner.as_any().is::<Vec<T>>() {
            return Err(self);
        }
        Ok(*self
            .inner
            .into_any()
            .downcast::<Vec<T>>()
            .expect("type checked above"))
    }

    /// The bytes as a `Vec<u8>` (copies unless the dtype is one byte wide).
    pub fn into_bytes(self) -> Vec<u8> {
        match self.into_vec::<u8>() {
            Ok(v) => v,
            Err(d) => d.as_bytes().to_vec(),
        }
    }
}

impl std::ops::Deref for HostData {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::fmt::Debug for HostData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostData({} bytes)", self.as_bytes().len())
    }
}

/// A zeroed `Vec<T>` from `alloc_zeroed`, so large allocations get fresh,
/// untouched pages (calloc does not write them); see the module docs.
pub fn zeroed_vec<T: bytemuck::Pod>(len: usize) -> Vec<T> {
    let bytes = len * std::mem::size_of::<T>();
    if bytes < HUGE_PAGE_HINT_BYTES {
        return vec![T::zeroed(); len];
    }
    let layout = std::alloc::Layout::array::<T>(len).expect("allocation size overflows");
    // SAFETY: the layout is non-zero sized (bytes >= 4 MiB) and all-zero bytes
    // are a valid T (Pod). The Vec takes ownership with the same layout.
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        advise_huge_pages(ptr, bytes);
        Vec::from_raw_parts(ptr as *mut T, len, len)
    }
}

/// `YAPOCIS_HUGE_PAGES=1` turns the hint on (read once); see the module docs.
#[cfg(target_os = "linux")]
fn huge_pages_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("YAPOCIS_HUGE_PAGES").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(target_os = "linux")]
fn advise_huge_pages(ptr: *mut u8, bytes: usize) {
    if !huge_pages_enabled() {
        return;
    }
    const HUGE: usize = 2 << 20;
    let start = (ptr as usize).div_ceil(HUGE) * HUGE;
    let end = (ptr as usize + bytes) / HUGE * HUGE;
    if end > start {
        // SAFETY: [start, end) lies inside the allocation just made. madvise
        // is only a hint; failure (THP disabled) leaves normal pages.
        unsafe {
            libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge_pages(_ptr: *mut u8, _bytes: usize) {}
