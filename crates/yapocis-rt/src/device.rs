//! Device selection and the backend-agnostic [`Device`] handle.

use crate::buffer::{Buffer, Element};
use crate::buffer_cache::{BufferCache, BufferCacheStats};
use crate::error::{Error, Result};
use crate::program::Program;
use crate::raw::RawDevice;
use std::sync::Arc;
use yapocis_bundle::{Backend, InterfaceEntry};

/// A GPU device on one backend. Cheap to clone; all clones share the context
/// and in-order queue (one queue per device, as in the Python device managers).
#[derive(Clone)]
pub struct Device {
    pub(crate) raw: Arc<dyn RawDevice>,
    pub(crate) buffers: Arc<BufferCache>,
}

/// Backends compiled into this build and usable on this platform.
#[allow(clippy::vec_init_then_push)]
pub fn available_backends() -> Vec<Backend> {
    let mut v = Vec::new();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v.push(Backend::Metal);
    #[cfg(feature = "cuda")]
    v.push(Backend::Cuda);
    #[cfg(feature = "ocl")]
    v.push(Backend::Ocl);
    v
}

impl Device {
    /// Open the default device.
    ///
    /// `YAPOCIS_BACKEND` selects the backend (`ocl`, `metal`, `cuda` or their
    /// aliases) and `YAPOCIS_DEVICE` the device by name or index, as in Python.
    /// Without `YAPOCIS_BACKEND` the first backend in [`available_backends`]
    /// that opens a device is used (Metal on macOS, then CUDA, then OpenCL).
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Result<Device> {
        let selector = std::env::var("YAPOCIS_DEVICE")
            .ok()
            .filter(|s| !s.is_empty());
        if let Ok(name) = std::env::var("YAPOCIS_BACKEND") {
            if !name.is_empty() {
                let backend =
                    Backend::parse(&name).ok_or_else(|| Error::UnknownBackend(name.clone()))?;
                return Device::open(backend, selector.as_deref());
            }
        }
        let mut last = None;
        for backend in available_backends() {
            match Device::open(backend, selector.as_deref()) {
                Ok(d) => return Ok(d),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or(Error::BackendUnavailable(Backend::Ocl)))
    }

    /// Open a device on `backend`. `selector` is an index or a device name
    /// (exact, else case-insensitive substring); `None` picks the backend's
    /// preferred device the way the Python device managers do.
    pub fn open(backend: Backend, selector: Option<&str>) -> Result<Device> {
        let raw: Arc<dyn RawDevice> = match backend {
            #[cfg(feature = "ocl")]
            Backend::Ocl => Arc::new(crate::ocl::OclDevice::open(selector)?),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => Arc::new(crate::metal::MetalDevice::open(selector)?),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Arc::new(crate::cuda::CudaDevice::open(selector)?),
            #[allow(unreachable_patterns)]
            other => return Err(Error::BackendUnavailable(other)),
        };
        tracing::info!(backend = backend.name(), device = %raw.name(), "opened device");
        Ok(Device {
            raw,
            buffers: Arc::new(BufferCache::from_env()),
        })
    }

    pub fn backend(&self) -> Backend {
        self.raw.backend()
    }

    pub fn name(&self) -> String {
        self.raw.name()
    }

    /// Compile one interface variant for this device. `source` is the rendered
    /// text for this backend (from the bundle's `sources[backend].path`).
    pub fn load(&self, entry: &InterfaceEntry, source: &str) -> Result<Program> {
        Program::load(self.clone(), entry, source)
    }

    /// Load an interface variant from a bundle directory on disk.
    pub fn load_from_bundle(
        &self,
        bundle_dir: impl AsRef<std::path::Path>,
        entry: &InterfaceEntry,
    ) -> Result<Program> {
        let src = entry
            .sources
            .get(self.backend().name())
            .ok_or_else(|| Error::BackendUnavailable(self.backend()))?;
        let text = std::fs::read_to_string(bundle_dir.as_ref().join(&src.path))?;
        self.load(entry, &text)
    }

    /// Upload a contiguous host slice with the given logical shape.
    pub fn upload<T: Element>(&self, data: &[T], shape: &[usize]) -> Result<Buffer<T>> {
        Buffer::upload(self.clone(), data, shape)
    }

    /// Allocate an uninitialised device buffer with the given shape.
    pub fn alloc<T: Element>(&self, shape: &[usize]) -> Result<Buffer<T>> {
        Buffer::alloc(self.clone(), shape)
    }

    /// Drop the device buffers cached for host-array arguments (Python's
    /// `purge`); see [`crate::buffer_cache`].
    pub fn purge_buffers(&self) {
        self.buffers.clear()
    }

    pub fn buffer_cache_stats(&self) -> BufferCacheStats {
        self.buffers.stats()
    }

    /// Block until every launch on this device has completed.
    pub fn wait_all(&self) -> Result<()> {
        self.raw.wait_all()
    }

    pub(crate) fn same_as(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.raw, &other.raw)
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({:?}, {})", self.backend(), self.name())
    }
}

/// Pick a device from `names` by `selector`: an index, an exact name, or a
/// case-insensitive substring. Shared by the backends.
pub(crate) fn select(
    backend: Backend,
    names: &[String],
    selector: Option<&str>,
    preferred: impl Fn(&[String]) -> Option<usize>,
) -> Result<usize> {
    if names.is_empty() {
        return Err(Error::NoDevices(backend));
    }
    let Some(sel) = selector else {
        return Ok(preferred(names).unwrap_or(names.len() - 1));
    };
    if let Ok(i) = sel.parse::<usize>() {
        if i < names.len() {
            return Ok(i);
        }
    }
    if let Some(i) = names.iter().position(|n| n == sel) {
        return Ok(i);
    }
    let lower = sel.to_ascii_lowercase();
    if let Some(i) = names
        .iter()
        .position(|n| n.to_ascii_lowercase().contains(&lower))
    {
        return Ok(i);
    }
    Err(Error::NoSuchDevice {
        backend,
        selector: sel.to_string(),
        available: names.to_vec(),
    })
}

/// First name containing any of `keys` (case-insensitive), in key order.
#[allow(dead_code)]
pub(crate) fn prefer(names: &[String], keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(i) = names
            .iter()
            .position(|n| n.to_ascii_lowercase().contains(key))
        {
            return Some(i);
        }
    }
    None
}
