//! `np.fft.fftshift` and `np.fft.ifftshift` of a row-major 2-D plane. (The
//! transforms themselves are not here: numpy's pocketfft has not been ported.)

/// `np.fft.fftshift` of an `h` × `w` row-major plane: the zero frequency moved to
/// the centre (each axis rolled by `n / 2`). `None` when `v.len() != h * w`.
pub fn fftshift_2d<T: Copy>(v: &[T], h: usize, w: usize) -> Option<Vec<T>> {
    roll_2d(v, h, w, h / 2, w / 2)
}

/// `np.fft.ifftshift`, the inverse of [`fftshift_2d`] (each axis rolled by
/// `-(n / 2)`, which differs from `fftshift` for odd sizes).
pub fn ifftshift_2d<T: Copy>(v: &[T], h: usize, w: usize) -> Option<Vec<T>> {
    roll_2d(v, h, w, h - h / 2, w - w / 2)
}

/// `np.roll` of both axes by `dy` and `dx` (non-negative, below `h` and `w`).
fn roll_2d<T: Copy>(v: &[T], h: usize, w: usize, dy: usize, dx: usize) -> Option<Vec<T>> {
    if h.checked_mul(w)? != v.len() {
        return None;
    }
    let mut out = Vec::with_capacity(v.len());
    for y in 0..h {
        let sy = (y + h - dy) % h;
        for x in 0..w {
            let sx = (x + w - dx) % w;
            out.push(*v.get(sy * w + sx)?);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_and_even() {
        // np.fft.fftshift(np.arange(5)) == [3, 4, 0, 1, 2]
        assert_eq!(
            fftshift_2d(&[0, 1, 2, 3, 4], 1, 5),
            Some(vec![3, 4, 0, 1, 2])
        );
        assert_eq!(
            ifftshift_2d(&[3, 4, 0, 1, 2], 1, 5),
            Some(vec![0, 1, 2, 3, 4])
        );
        assert_eq!(fftshift_2d(&[0, 1, 2, 3], 1, 4), Some(vec![2, 3, 0, 1]));
        assert_eq!(fftshift_2d(&[0, 1, 2], 2, 2), None);
    }
}
