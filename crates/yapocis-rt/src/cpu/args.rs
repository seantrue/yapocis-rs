//! What a CPU kernel sees: its arguments, by position, and the launch size.
//!
//! Arguments are in the kernel's parameter order, exactly as the OpenCL kernel
//! receives them (derived dimensions included). A port reads them like this:
//!
//! ```ignore
//! fn add(a: &Args) {
//!     let input1 = a.input::<f32>(0);
//!     let input2 = a.input::<f32>(1);
//!     let mut output = a.output::<f32>(2);
//!     par::map(&mut output[..a.global_size()], |i| input1[i] + input2[i]);
//! }
//! ```
//!
//! An [`Input`] or [`Output`] is a plain slice once taken (`input1[i]` is an
//! ordinary index, so loops over it optimise as loops over a slice) and holds
//! the buffer's lock until it is dropped.
//!
//! Take inputs before outputs. An input that is also bound to another
//! argument of the same launch (an in-place call) is copied, so it reads the
//! values from before the launch, as a GPU kernel with no write-after-read
//! hazard would. An output on a buffer the kernel may not write (an `in`
//! argument used in place) is a private copy whose writes are dropped, as a
//! GPU's write to an `in` parameter's temporary buffer is never read back.

use super::CpuBuffer;
use crate::raw::RawArg;
use bytemuck::Pod;
use std::ops::{Deref, DerefMut};
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

pub struct Args<'a> {
    args: &'a [RawArg<'a>],
    global: &'a [usize],
    local: Option<&'a [usize]>,
}

impl<'a> Args<'a> {
    pub(crate) fn new(
        args: &'a [RawArg<'a>],
        global: &'a [usize],
        local: Option<&'a [usize]>,
    ) -> Self {
        Args {
            args,
            global,
            local,
        }
    }

    /// The global size, as OpenCL's `get_global_size` (one to three axes).
    pub fn global(&self) -> &[usize] {
        self.global
    }

    /// `get_global_size(axis)`: 1 for an axis the launch does not have.
    pub fn global_dim(&self, axis: usize) -> usize {
        self.global.get(axis).copied().unwrap_or(1)
    }

    /// Total work items: the product of the global size.
    pub fn global_size(&self) -> usize {
        self.global.iter().product()
    }

    /// The local size the caller asked for, if any (only kernels whose
    /// results depend on their work-group shape look at it).
    pub fn local(&self) -> Option<&[usize]> {
        self.local
    }

    /// A scalar argument of type `T` (`int` is `i32`, `float` is `f32`).
    pub fn scalar<T: Pod>(&self, i: usize) -> T {
        match &self.args[i] {
            RawArg::Bytes(b) => bytemuck::pod_read_unaligned(b),
            RawArg::Buffer(_) => panic!("argument {i} is a buffer, not a scalar"),
        }
    }

    /// `int` argument.
    pub fn int(&self, i: usize) -> i32 {
        self.scalar::<i32>(i)
    }

    /// `float` argument.
    pub fn float(&self, i: usize) -> f32 {
        self.scalar::<f32>(i)
    }

    fn buffer(&self, i: usize) -> &'a CpuBuffer {
        match self.args[i] {
            RawArg::Buffer(b) => b
                .as_any()
                .downcast_ref::<CpuBuffer>()
                .expect("a CPU buffer"),
            RawArg::Bytes(_) => panic!("argument {i} is a scalar, not a buffer"),
        }
    }

    /// Whether buffer argument `i` is also bound to another argument.
    fn aliased(&self, i: usize) -> bool {
        let me = self.buffer(i) as *const CpuBuffer;
        self.args.iter().enumerate().any(|(j, a)| {
            j != i
                && matches!(a, RawArg::Buffer(b)
                    if b.as_any().downcast_ref::<CpuBuffer>()
                        .is_some_and(|b| std::ptr::eq(b, me)))
        })
    }

    /// A read-only view of buffer argument `i` as `[T]`.
    pub fn input<T: Pod>(&self, i: usize) -> Input<'a, T> {
        let b = self.buffer(i);
        let (guard, bytes) = b
            .try_read()
            .unwrap_or_else(|| panic!("argument {i}: take inputs before outputs"));
        let data: &'a [T] = typed(bytes);
        if self.aliased(i) {
            Input {
                data: Vec::from(data).into(),
                _guard: None,
            }
        } else {
            Input {
                data: View::Borrowed(data),
                _guard: Some(guard),
            }
        }
    }

    /// A writable view of buffer argument `i` as `[T]`, holding its current
    /// contents (so `inout` parameters read and write through it).
    pub fn output<T: Pod>(&self, i: usize) -> Output<'a, T> {
        let b = self.buffer(i);
        if !b.writable {
            let (_guard, bytes) = b
                .try_read()
                .unwrap_or_else(|| panic!("argument {i} is bound to a writable argument"));
            let copy: Vec<T> = typed(bytes).to_vec();
            return Output {
                data: copy.into(),
                _guard: None,
            };
        }
        let (guard, bytes) = b
            .try_write()
            .unwrap_or_else(|| panic!("argument {i} is bound to two writable arguments"));
        Output {
            data: ViewMut::Borrowed(bytemuck::cast_slice_mut(trim_mut::<T>(bytes))),
            _guard: Some(guard),
        }
    }
}

/// The whole elements of `bytes` as `[T]` (the buffer is 8-byte aligned).
fn typed<T: Pod>(bytes: &[u8]) -> &[T] {
    let n = bytes.len() / std::mem::size_of::<T>();
    bytemuck::cast_slice(&bytes[..n * std::mem::size_of::<T>()])
}

fn trim_mut<T>(bytes: &mut [u8]) -> &mut [u8] {
    let n = bytes.len() / std::mem::size_of::<T>();
    &mut bytes[..n * std::mem::size_of::<T>()]
}

/// The slice an [`Input`] serves: the buffer's own, or a copy.
enum View<'a, T> {
    Borrowed(&'a [T]),
    Owned(Box<[T]>),
}

impl<T> From<Vec<T>> for View<'_, T> {
    fn from(v: Vec<T>) -> Self {
        View::Owned(v.into_boxed_slice())
    }
}

/// See [`Args::input`].
pub struct Input<'a, T> {
    data: View<'a, T>,
    // Field order: `data` is dropped before the guard it depends on.
    _guard: Option<RwLockReadGuard<'a, ()>>,
}

impl<T> Deref for Input<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.data {
            View::Borrowed(s) => s,
            View::Owned(b) => b,
        }
    }
}

/// The slice an [`Output`] serves: the buffer's own, or a private copy.
enum ViewMut<'a, T> {
    Borrowed(&'a mut [T]),
    Owned(Box<[T]>),
}

impl<T> From<Vec<T>> for ViewMut<'_, T> {
    fn from(v: Vec<T>) -> Self {
        ViewMut::Owned(v.into_boxed_slice())
    }
}

/// See [`Args::output`].
pub struct Output<'a, T> {
    data: ViewMut<'a, T>,
    _guard: Option<RwLockWriteGuard<'a, ()>>,
}

impl<T> Deref for Output<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.data {
            ViewMut::Borrowed(s) => s,
            ViewMut::Owned(b) => b,
        }
    }
}

impl<T> DerefMut for Output<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.data {
            ViewMut::Borrowed(s) => s,
            ViewMut::Owned(b) => b,
        }
    }
}
