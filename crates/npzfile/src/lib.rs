//! numpy `.npz` archives for ndarray: [`NpzReader`] and [`NpzWriter`] with ndarray-npy's API
//! (its `npz` module, 0.10), on `zip` 9.
//!
//! ndarray-npy 0.10 pins `zip` 6, which has open advisories (a parser differential leading to
//! file overwrite, unbounded symlink targets, legacy-decoder and AES-entry bugs, all fixed in
//! `zip` 9). Its `.npy` reading and writing are kept; only the archive layer is replaced, so
//! ndarray-npy is used here without its `npz` feature.
//!
//! Reading accepts stored and deflated members (`np.savez`, `np.savez_compressed`); writing
//! stores, as `np.savez` does. Member names carry `.npy`; `names()` strips it, and `by_name`
//! accepts a name with or without it.

use std::io::{BufWriter, Read, Seek, Write};

use ndarray::{ArrayBase, DataOwned, Dimension};
use ndarray_npy::{ReadNpyExt, ReadableElement, WriteNpyExt};
use zip::result::ZipError;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// What went wrong reading or writing an archive.
#[derive(Debug)]
pub enum Error {
    Zip(ZipError),
    ReadNpy(ndarray_npy::ReadNpyError),
    WriteNpy(ndarray_npy::WriteNpyError),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Zip(e) => write!(f, "zip error: {e}"),
            Error::ReadNpy(e) => write!(f, "error reading npy from npz: {e}"),
            Error::WriteNpy(e) => write!(f, "error writing npy to npz: {e}"),
            Error::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Zip(e) => Some(e),
            Error::ReadNpy(e) => Some(e),
            Error::WriteNpy(e) => Some(e),
            Error::Io(e) => Some(e),
        }
    }
}

impl From<ZipError> for Error {
    fn from(e: ZipError) -> Self {
        Error::Zip(e)
    }
}

impl From<ndarray_npy::ReadNpyError> for Error {
    fn from(e: ndarray_npy::ReadNpyError) -> Self {
        Error::ReadNpy(e)
    }
}

impl From<ndarray_npy::WriteNpyError> for Error {
    fn from(e: ndarray_npy::WriteNpyError) -> Self {
        Error::WriteNpy(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Reads the arrays of an `.npz` archive.
pub struct NpzReader<R: Read + Seek> {
    zip: ZipArchive<R>,
}

impl<R: Read + Seek> NpzReader<R> {
    pub fn new(reader: R) -> Result<NpzReader<R>> {
        Ok(NpzReader {
            zip: ZipArchive::new(reader)?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.zip.len() == 0
    }

    pub fn len(&self) -> usize {
        self.zip.len()
    }

    /// The arrays' names, without `.npy`, in archive order.
    pub fn names(&mut self) -> Result<Vec<String>> {
        (0..self.zip.len())
            .map(|i| {
                let file = self.zip.by_index(i)?;
                let name = file.name()?;
                Ok(name.strip_suffix(".npy").unwrap_or(&name).to_owned())
            })
            .collect()
    }

    /// The array named `name` (with or without `.npy`).
    pub fn by_name<S, D>(&mut self, name: &str) -> Result<ArrayBase<S, D>>
    where
        S::Elem: ReadableElement,
        S: DataOwned,
        D: Dimension,
    {
        match self.zip.by_name(name) {
            Ok(file) => return Ok(ArrayBase::<S, D>::read_npy(file)?),
            Err(ZipError::FileNotFound) => {}
            Err(e) => return Err(e.into()),
        }
        Ok(ArrayBase::<S, D>::read_npy(
            self.zip.by_name(&format!("{name}.npy"))?,
        )?)
    }

    /// The array at `index`, in archive order.
    pub fn by_index<S, D>(&mut self, index: usize) -> Result<ArrayBase<S, D>>
    where
        S::Elem: ReadableElement,
        S: DataOwned,
        D: Dimension,
    {
        Ok(ArrayBase::<S, D>::read_npy(self.zip.by_index(index)?)?)
    }
}

/// Writes arrays to an `.npz` archive, stored (uncompressed), as `np.savez` does.
pub struct NpzWriter<W: Write + Seek> {
    zip: ZipWriter<W>,
    options: SimpleFileOptions,
}

impl<W: Write + Seek> NpzWriter<W> {
    pub fn new(writer: W) -> NpzWriter<W> {
        NpzWriter {
            zip: ZipWriter::new(writer),
            options: SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
        }
    }

    /// Add `array` as `name.npy`.
    pub fn add_array<N, T>(&mut self, name: N, array: &T) -> Result<()>
    where
        N: Into<String>,
        T: WriteNpyExt + ?Sized,
    {
        self.zip.start_file(name.into() + ".npy", self.options)?;
        array.write_npy(BufWriter::new(&mut self.zip))?;
        Ok(())
    }

    pub fn finish(self) -> Result<W> {
        let mut writer = self.zip.finish()?;
        writer.flush()?;
        Ok(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ArrayD, IxDyn};

    #[test]
    fn writes_and_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.npz");
        let a = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32);
        let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i32, -2]).unwrap();
        let mut w = NpzWriter::new(std::fs::File::create(&path).unwrap());
        w.add_array("a", &a).unwrap();
        w.add_array("b", &b).unwrap();
        w.finish().unwrap();
        let mut r = NpzReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.names().unwrap(), ["a", "b"]);
        let got: Array2<f32> = r.by_name("a").unwrap();
        assert_eq!(got, a);
        let got: ArrayD<i32> = r.by_name("b.npy").unwrap();
        assert_eq!(got, b);
        assert!(matches!(
            r.by_name::<ndarray::OwnedRepr<f32>, IxDyn>("c"),
            Err(Error::Zip(ZipError::FileNotFound))
        ));
    }
}
