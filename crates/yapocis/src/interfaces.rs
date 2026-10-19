//! Generated interface structs: one module per bundled interface variant,
//! each with a struct exposing one typed method per kernel (plus a `_with`
//! variant taking [`CallOptions`]).
//!
//! Argument conventions follow the Python runtime: `in` parameters take
//! `&ArrayViewD<T>`, `inout`/`out` take `&mut ArrayViewMutD<T>` and are written
//! in place, `resident` take `&Buffer<T>`, scalars take their element type, and
//! `out_like` outputs are returned as new arrays. Dimension parameters
//! (`width_of a`, ...) are derived from the arrays, never passed.

#![allow(non_snake_case, clippy::too_many_arguments, clippy::type_complexity)]

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
