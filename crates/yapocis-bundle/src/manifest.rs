//! `manifest.json` schema (version 1), mirroring `yapocis.codegen.bundle`.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// The manifest schema version this crate understands.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("cannot read manifest: {0}")]
    Io(#[from] std::io::Error),
    #[error("cannot parse manifest: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported bundle schema version {found} (this crate understands {SCHEMA_VERSION})")]
    Schema { found: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub yapocis_version: String,
    pub backends: Vec<String>,
    pub interfaces: Vec<InterfaceEntry>,
}

impl Manifest {
    /// Parse a manifest and check its schema version.
    pub fn from_json(text: &str) -> Result<Manifest, ManifestError> {
        #[derive(Deserialize)]
        struct Probe {
            schema_version: u32,
        }
        let probe: Probe = serde_json::from_str(text)?;
        if probe.schema_version != SCHEMA_VERSION {
            return Err(ManifestError::Schema {
                found: probe.schema_version,
            });
        }
        Ok(serde_json::from_str(text)?)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Manifest, ManifestError> {
        Manifest::from_json(&std::fs::read_to_string(path)?)
    }

    /// Look up an interface variant by its id (`name` or `name.variant`).
    pub fn interface(&self, id: &str) -> Option<&InterfaceEntry> {
        self.interfaces.iter().find(|e| e.id == id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceEntry {
    pub name: String,
    pub variant: String,
    pub id: String,
    pub interface_text: String,
    pub kernels: Vec<KernelEntry>,
    /// backend name -> rendered source
    pub sources: BTreeMap<String, SourceEntry>,
    /// Template context overrides that produced this variant (JSON-safe).
    #[serde(default)]
    pub context: serde_json::Value,
}

impl InterfaceEntry {
    pub fn kernel(&self, name: &str) -> Option<&KernelEntry> {
        self.kernels.iter().find(|k| k.name == name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceEntry {
    /// Path relative to the bundle directory, e.g. `ocl/demo.cl`.
    pub path: String,
    pub sha256: String,
    pub build_options: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelEntry {
    /// Name exposed to callers (extensions are `<base>_<ext>`).
    pub name: String,
    /// Function name in the compiled program.
    pub symbol: String,
    /// Kernel arguments in order; on Metal the index is the `[[buffer(i)]]` slot.
    pub params: Vec<ParamEntry>,
    /// Non-derived parameter names, in the order the caller passes them.
    pub user_args: Vec<String>,
    /// Parameters read back and returned, in declaration order.
    pub returns: Vec<String>,
    /// Pre-parsed `dispatch_size` trait, one expression per global-size axis.
    #[serde(default)]
    pub dispatch_size: Option<Vec<DispatchNode>>,
    #[serde(default)]
    pub cuda: CudaInfo,
    #[serde(default)]
    pub traits: serde_json::Value,
}

impl KernelEntry {
    pub fn param(&self, name: &str) -> Option<&ParamEntry> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Position of a user argument by parameter name.
    pub fn user_arg_index(&self, name: &str) -> Option<usize> {
        self.user_args.iter().position(|n| n == name)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CudaInfo {
    /// The CUDA signature ends with `GLOBAL_SIZE_PARAM`: pass the real global
    /// size as a trailing `u64` because the grid is rounded up to whole blocks.
    #[serde(default)]
    pub takes_global_size: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamEntry {
    pub name: String,
    pub kind: ParamKind,
    /// Element type (numpy name).
    pub dtype: DType,
    /// IDL type name (`float`, `int`, `complex64`, ...).
    pub ctype: String,
    #[serde(default)]
    pub role: Option<Role>,
    #[serde(default)]
    pub dim: Option<Dim>,
    /// Source parameter for `dim` and `out_like` entries.
    #[serde(default)]
    pub of: Option<String>,
    #[serde(default)]
    pub derived: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    Buffer,
    Scalar,
    Dim,
    OutLike,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    In,
    Out,
    Inout,
    Resident,
}

/// Element types that appear in interfaces (numpy names in the manifest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

impl DType {
    pub fn size_of(self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::Uint8 => 1,
            DType::Int16 | DType::Uint16 | DType::Float16 => 2,
            DType::Int32 | DType::Uint32 | DType::Float32 => 4,
            DType::Int64 | DType::Uint64 | DType::Float64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    pub fn numpy_name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Uint8 => "uint8",
            DType::Uint16 => "uint16",
            DType::Uint32 => "uint32",
            DType::Uint64 => "uint64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.numpy_name())
    }
}

/// A derived dimension parameter (`width_of a`, `dim_2_of input`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    /// `size_of` and `sizeof`: total element count
    SizeOf,
    /// `length_of`: `len(arg)`, i.e. `shape[0]`
    LengthOf,
    /// `shape[0]`
    WidthOf,
    /// `shape[1]`, or 1 for a 1-D array
    HeightOf,
    /// `shape[2]`
    DepthOf,
    /// `dim_N_of`: `shape[N]`
    DimN(usize),
}

impl Dim {
    pub fn parse(s: &str) -> Option<Dim> {
        Some(match s {
            "size_of" | "sizeof" => Dim::SizeOf,
            "length_of" | "lengthof" => Dim::LengthOf,
            "width_of" | "widthof" => Dim::WidthOf,
            "height_of" | "heightof" => Dim::HeightOf,
            "depth_of" | "depthof" => Dim::DepthOf,
            _ => {
                let n = s.strip_prefix("dim_")?.strip_suffix("_of")?;
                Dim::DimN(n.parse().ok()?)
            }
        })
    }

    pub fn name(&self) -> String {
        match self {
            Dim::SizeOf => "size_of".into(),
            Dim::LengthOf => "length_of".into(),
            Dim::WidthOf => "width_of".into(),
            Dim::HeightOf => "height_of".into(),
            Dim::DepthOf => "depth_of".into(),
            Dim::DimN(n) => format!("dim_{n}_of"),
        }
    }
}

impl Serialize for Dim {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name())
    }
}

impl<'de> Deserialize<'de> for Dim {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Dim, D::Error> {
        let s = String::deserialize(d)?;
        Dim::parse(&s).ok_or_else(|| de::Error::custom(format!("unknown dimension role {s:?}")))
    }
}

/// One node of a pre-parsed `dispatch_size` expression.
///
/// Serialized as `{"lit": 64}`, `{"dim": "dim_1_of", "of": "input", "param_index": 0}`,
/// `{"op": "+", "lhs": ..., "rhs": ...}` or `{"op": "neg", "arg": ...}`.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchNode {
    Lit(f64),
    Dim {
        dim: String,
        of: String,
        /// Index of the parameter in the full parameter list; `dispatch_expr`
        /// applies it to the user argument tuple, and so must an evaluator.
        param_index: usize,
    },
    Binary {
        op: String,
        lhs: Box<DispatchNode>,
        rhs: Box<DispatchNode>,
    },
    Neg(Box<DispatchNode>),
}

impl Serialize for DispatchNode {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut m = s.serialize_map(None)?;
        match self {
            DispatchNode::Lit(v) => {
                if v.fract() == 0.0 {
                    m.serialize_entry("lit", &(*v as i64))?
                } else {
                    m.serialize_entry("lit", v)?
                }
            }
            DispatchNode::Dim {
                dim,
                of,
                param_index,
            } => {
                m.serialize_entry("dim", dim)?;
                m.serialize_entry("of", of)?;
                m.serialize_entry("param_index", param_index)?;
            }
            DispatchNode::Binary { op, lhs, rhs } => {
                m.serialize_entry("op", op)?;
                m.serialize_entry("lhs", lhs)?;
                m.serialize_entry("rhs", rhs)?;
            }
            DispatchNode::Neg(arg) => {
                m.serialize_entry("op", "neg")?;
                m.serialize_entry("arg", arg)?;
            }
        }
        m.end()
    }
}

impl<'de> Deserialize<'de> for DispatchNode {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<DispatchNode, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            lit: Option<f64>,
            dim: Option<String>,
            of: Option<String>,
            param_index: Option<usize>,
            op: Option<String>,
            lhs: Option<DispatchNode>,
            rhs: Option<DispatchNode>,
            arg: Option<DispatchNode>,
        }
        let raw = Raw::deserialize(d)?;
        if let Some(v) = raw.lit {
            return Ok(DispatchNode::Lit(v));
        }
        if let Some(dim) = raw.dim {
            return Ok(DispatchNode::Dim {
                dim,
                of: raw.of.ok_or_else(|| de::Error::missing_field("of"))?,
                param_index: raw
                    .param_index
                    .ok_or_else(|| de::Error::missing_field("param_index"))?,
            });
        }
        match raw.op.as_deref() {
            Some("neg") => Ok(DispatchNode::Neg(Box::new(
                raw.arg.ok_or_else(|| de::Error::missing_field("arg"))?,
            ))),
            Some(op) => Ok(DispatchNode::Binary {
                op: op.to_string(),
                lhs: Box::new(raw.lhs.ok_or_else(|| de::Error::missing_field("lhs"))?),
                rhs: Box::new(raw.rhs.ok_or_else(|| de::Error::missing_field("rhs"))?),
            }),
            None => Err(de::Error::custom("dispatch node needs lit, dim or op")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_roundtrip() {
        for s in [
            "size_of",
            "length_of",
            "width_of",
            "height_of",
            "depth_of",
            "dim_7_of",
        ] {
            assert_eq!(Dim::parse(s).unwrap().name(), s);
        }
        assert_eq!(Dim::parse("sizeof"), Some(Dim::SizeOf));
        assert_eq!(Dim::parse("dim_x_of"), None);
        assert_eq!(Dim::parse("bogus"), None);
    }

    #[test]
    fn dispatch_node_json() {
        let text =
            r#"{"op":"/","lhs":{"dim":"height_of","of":"input","param_index":0},"rhs":{"lit":2}}"#;
        let node: DispatchNode = serde_json::from_str(text).unwrap();
        match &node {
            DispatchNode::Binary { op, lhs, rhs } => {
                assert_eq!(op, "/");
                assert!(matches!(**lhs, DispatchNode::Dim { param_index: 0, .. }));
                assert_eq!(**rhs, DispatchNode::Lit(2.0));
            }
            other => panic!("{other:?}"),
        }
        let back = serde_json::to_string(&node).unwrap();
        let again: DispatchNode = serde_json::from_str(&back).unwrap();
        assert_eq!(again, node);
    }

    #[test]
    fn schema_version_is_checked() {
        let err = Manifest::from_json(r#"{"schema_version": 99}"#).unwrap_err();
        assert!(matches!(err, ManifestError::Schema { found: 99 }));
    }
}
