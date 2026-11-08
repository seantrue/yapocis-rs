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
//! Take inputs before outputs. An input that is also bound to another
//! argument of the same launch (an in-place call) is copied, so it reads the
//! values from before the launch, as a GPU kernel with no write-after-read
//! hazard would.

use super::CpuBuffer;
use crate::raw::RawArg;
use bytemuck::Pod;
use std::marker::PhantomData;
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
        let n = b.len / std::mem::size_of::<T>();
        if self.aliased(i) {
            let words = b
                .words
                .try_read()
                .unwrap_or_else(|_| panic!("argument {i}: take inputs before outputs"));
            let data =
                bytemuck::cast_slice::<u8, T>(&super::bytes(&words, b.len)[..n * size::<T>()])
                    .to_vec();
            Input::Copied(data)
        } else {
            let guard = b.words.read().unwrap_or_else(|e| e.into_inner());
            Input::Shared(guard, n, PhantomData)
        }
    }

    /// A writable view of buffer argument `i` as `[T]`, holding its current
    /// contents (so `inout` parameters read and write through it).
    pub fn output<T: Pod>(&self, i: usize) -> Output<'a, T> {
        let b = self.buffer(i);
        let n = b.len / std::mem::size_of::<T>();
        let guard = b
            .words
            .try_write()
            .unwrap_or_else(|_| panic!("argument {i} is bound to two writable arguments"));
        Output(guard, n, PhantomData)
    }
}

fn size<T>() -> usize {
    std::mem::size_of::<T>()
}

/// See [`Args::input`].
pub enum Input<'a, T> {
    Shared(RwLockReadGuard<'a, Vec<u64>>, usize, PhantomData<T>),
    Copied(Vec<T>),
}

impl<T: Pod> Deref for Input<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Input::Shared(words, n, _) => {
                bytemuck::cast_slice(&bytemuck::cast_slice::<u64, u8>(words)[..n * size::<T>()])
            }
            Input::Copied(v) => v,
        }
    }
}

/// See [`Args::output`].
pub struct Output<'a, T>(RwLockWriteGuard<'a, Vec<u64>>, usize, PhantomData<T>);

impl<T: Pod> Deref for Output<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        bytemuck::cast_slice(&bytemuck::cast_slice::<u64, u8>(&self.0)[..self.1 * size::<T>()])
    }
}

impl<T: Pod> DerefMut for Output<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let n = self.1 * size::<T>();
        bytemuck::cast_slice_mut(&mut bytemuck::cast_slice_mut::<u64, u8>(&mut self.0)[..n])
    }
}
