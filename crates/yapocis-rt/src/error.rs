use yapocis_bundle::Backend;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("backend {0:?} is not compiled in (enable the Cargo feature) or not available on this platform")]
    BackendUnavailable(Backend),
    #[error("unknown backend {0:?}")]
    UnknownBackend(String),
    #[error("no {backend:?} device matches {selector:?}; available: {available:?}")]
    NoSuchDevice {
        backend: Backend,
        selector: String,
        available: Vec<String>,
    },
    #[error("no {0:?} devices found")]
    NoDevices(Backend),
    #[error("{backend:?} compile failed for {interface}:\n{log}")]
    Compile {
        backend: Backend,
        interface: String,
        log: String,
    },
    #[error("kernel {0:?} is not in this program")]
    NoSuchKernel(String),
    #[error("{0}")]
    Derive(#[from] yapocis_bundle::DeriveError),
    #[error("{0}")]
    Dispatch(#[from] yapocis_bundle::DispatchError),
    #[error("parameter {param:?} expects dtype {expected} but the argument is {got}")]
    DType {
        param: String,
        expected: yapocis_bundle::DType,
        got: yapocis_bundle::DType,
    },
    #[error("parameter {param:?} ({role}) needs {needs}")]
    ArgKind {
        param: String,
        role: String,
        needs: &'static str,
    },
    #[error("buffer belongs to another device")]
    ForeignBuffer,
    #[error("buffer holds {have} bytes but {need} were requested")]
    BufferSize { have: usize, need: usize },
    #[error("no global size: pass CallOptions::global or give the kernel array arguments")]
    NoGlobalSize,
    #[error("{backend:?} error in {op}: {detail}")]
    Backend {
        backend: Backend,
        op: &'static str,
        detail: String,
    },
    #[error("manifest: {0}")]
    Manifest(#[from] yapocis_bundle::ManifestError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn backend(backend: Backend, op: &'static str, detail: impl ToString) -> Error {
        Error::Backend {
            backend,
            op,
            detail: detail.to_string(),
        }
    }
}
