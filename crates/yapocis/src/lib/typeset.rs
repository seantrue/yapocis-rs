//! `yapocis.lib.typeset`: text rasterisation with FreeType, matching the
//! Python module glyph for glyph (same FreeType calls in the same order, same
//! fixed-point arithmetic), so text blocks composited by the graphics layers
//! compare pixel-wise with the Python package.
//!
//! Needs the `text` Cargo feature (on by default) and the system FreeType library.

use crate::fonts;
use crate::{Error, Result};
use freetype::face::{KerningMode, LoadFlag};
use freetype::ffi::{FT_Fixed, FT_Pos};
use freetype::{Library, Matrix, Vector};
use ndarray::Array2;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// `compute_360_to_180`: map a rotation to (-180, 180].
pub fn compute_360_to_180(rotation: f64) -> f64 {
    let mut r = rotation.rem_euclid(360.0);
    if r > 180.0 {
        r = -180.0 + (r - 180.0);
    }
    r
}

/// Python `round(x, 4)`.
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

thread_local! {
    static LIBRARY: RefCell<Option<Library>> = const { RefCell::new(None) };
    static FACES: RefCell<HashMap<&'static str, Rc<freetype::Face>>> = RefCell::new(HashMap::new());
}

fn face_for(font_name: &str) -> Result<Rc<freetype::Face>> {
    let (name, bytes) = fonts::font(font_name);
    FACES.with(|faces| {
        if let Some(f) = faces.borrow().get(name) {
            return Ok(f.clone());
        }
        let face = LIBRARY.with(|lib| -> Result<freetype::Face> {
            let mut lib = lib.borrow_mut();
            if lib.is_none() {
                *lib = Some(
                    Library::init().map_err(|e| Error::Invalid(format!("FreeType init: {e}")))?,
                );
            }
            lib.as_ref()
                .unwrap()
                .new_memory_face(bytes.to_vec(), 0)
                .map_err(|e| Error::Invalid(format!("FreeType face {name}: {e}")))
        })?;
        let face = Rc::new(face);
        faces.borrow_mut().insert(name, face.clone());
        Ok(face)
    })
}

struct GlyphInfo {
    kerning_x: i64,
    left: i64,
    rows: i64,
    top: i64,
    width: i64,
    advance_x: i64,
    advance_y: i64,
    /// `rows * width` coverage bytes, top row first (as FreeType renders it)
    data: Vec<u8>,
}

/// Widens a FreeType position (a C `long`) to `i64`: 32-bit on Windows, already `i64` on
/// Linux/macOS, where clippy would otherwise call the conversion useless.
#[allow(clippy::useless_conversion)]
fn wide(v: FT_Pos) -> i64 {
    i64::from(v)
}

/// One pass over `text`: FreeType metrics per glyph, with the Python module's
/// transform (rotation matrix in 16.16, zero delta) and kerning calls.
fn layout(face: &freetype::Face, text: &str, size: f64, angle_deg: f64) -> Result<Vec<GlyphInfo>> {
    face.set_char_size((size * 64.0) as isize, 0, 72, 72)
        .map_err(|e| Error::Invalid(format!("set_char_size: {e}")))?;
    let angle = (angle_deg / 180.0) * std::f64::consts::PI;
    let mut matrix = Matrix {
        // FT_Fixed is a C `long`: 64-bit on Linux/macOS, 32-bit on Windows.
        xx: (angle.cos() * 65536.0) as FT_Fixed,
        xy: (-angle.sin() * 65536.0) as FT_Fixed,
        yx: (angle.sin() * 65536.0) as FT_Fixed,
        yy: (angle.cos() * 65536.0) as FT_Fixed,
    };
    let mut delta = Vector { x: 0, y: 0 };
    face.set_transform(&mut matrix, &mut delta);
    let mut previous: u32 = 0;
    let mut out = Vec::with_capacity(text.chars().count());
    for c in text.chars() {
        face.load_char(c as usize, LoadFlag::RENDER)
            .map_err(|e| Error::Invalid(format!("load_char {c:?}: {e}")))?;
        let right = face.get_char_index(c as usize).unwrap_or(0);
        let kerning = face
            .get_kerning(previous, right, KerningMode::KerningDefault)
            .map_err(|e| Error::Invalid(format!("get_kerning: {e}")))?;
        previous = right;
        let glyph = face.glyph();
        let bitmap = glyph.bitmap();
        let (width, rows, pitch) = (
            bitmap.width() as i64,
            bitmap.rows() as i64,
            bitmap.pitch() as i64,
        );
        let buffer = bitmap.buffer();
        let mut data = Vec::with_capacity((rows * width) as usize);
        for j in 0..rows {
            let start = (j * pitch) as usize;
            data.extend_from_slice(&buffer[start..start + width as usize]);
        }
        out.push(GlyphInfo {
            kerning_x: wide(kerning.x),
            left: glyph.bitmap_left() as i64,
            rows,
            top: glyph.bitmap_top() as i64,
            width,
            advance_x: wide(glyph.advance().x),
            advance_y: wide(glyph.advance().y),
            data,
        });
    }
    Ok(out)
}

/// `extent_freetype(text, font_name, size, angle)`: the box `(xmin, ymin, xmax, ymax)` text
/// covers, y up, with the pen starting at the origin. `size` `None` is 32. Python's kerning
/// cache isn't ported: it stores entries under the wrong key, which only matters for a font with
/// kerning, and none of the bundled fonts has any.
pub fn extent_freetype(
    text: &str,
    font_name: &str,
    size: Option<f64>,
    angle: f64,
) -> Result<(i64, i64, i64, i64)> {
    let size = size.unwrap_or(32.0);
    let angle = round4(compute_360_to_180(angle));
    let face = face_for(font_name)?;
    let glyphs = layout(&face, text, size, angle)?;
    Ok(extent(&glyphs))
}

/// The box `glyphs` cover, as `(xmin, ymin, xmax, ymax)`.
fn extent(glyphs: &[GlyphInfo]) -> (i64, i64, i64, i64) {
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (0i64, 0i64, 0i64, 0i64);
    let (mut pen_x, mut pen_y) = (0i64, 0i64);
    for g in glyphs {
        pen_x += g.kerning_x;
        let x0 = (pen_x >> 6) + g.left;
        let x1 = x0 + g.width;
        let y0 = (pen_y >> 6) - (g.rows - g.top);
        let y1 = y0 + g.rows;
        xmin = xmin.min(x0);
        xmax = xmax.max(x1);
        ymin = ymin.min(y0);
        ymax = ymax.max(y1);
        pen_x += g.advance_x;
        pen_y += g.advance_y;
    }
    (xmin, ymin, xmax, ymax)
}

/// `render_freetype`: the uncropped coverage image, bottom row first (the
/// Python function builds it with y up, callers flip it).
fn render_freetype(
    face: &freetype::Face,
    text: &str,
    size: f64,
    angle_deg: f64,
) -> Result<Array2<u8>> {
    let angle_deg = round4(angle_deg);
    let glyphs = layout(face, text, size, angle_deg)?;
    let (xmin, ymin, xmax, ymax) = extent(&glyphs);
    let (height, width) = ((ymax - ymin) as usize, (xmax - xmin) as usize);
    let mut l = Array2::<u8>::zeros((height, width));
    let (mut pen_x, mut pen_y) = (0i64, 0i64);
    for g in &glyphs {
        pen_x += g.kerning_x;
        let x = (pen_x >> 6) - xmin + g.left;
        let y = (pen_y >> 6) - ymin - (g.rows - g.top);
        if !g.data.is_empty() {
            // L[y:y+rows, x:x+width] |= Z[::-1, :]
            for j in 0..g.rows {
                let src_row = g.rows - 1 - j;
                let dst_row = y + j;
                if dst_row < 0 || dst_row as usize >= height {
                    continue;
                }
                for i in 0..g.width {
                    let dst_col = x + i;
                    if dst_col < 0 || dst_col as usize >= width {
                        continue;
                    }
                    let v = g.data[(src_row * g.width + i) as usize];
                    let cell = &mut l[[dst_row as usize, dst_col as usize]];
                    *cell |= v;
                }
            }
        }
        pen_x += g.advance_x;
        pen_y += g.advance_y;
    }
    Ok(l)
}

/// A rendered text block: the bounding box `[0, 0, rows, cols]` and the
/// coverage in `[0, 1]`, top row first.
pub struct TextBlock {
    pub bbox: [usize; 4],
    pub block: Array2<f32>,
}

/// `render_text(text, font_name, size, rotation, crop=True)`: rasterise, flip
/// to top-row-first, and crop to the inked area with a one-pixel margin.
/// Empty text gives a 1×1 zero block.
pub fn render_text(
    text: &str,
    font_name: &str,
    size: Option<f64>,
    rotation: Option<f64>,
) -> Result<TextBlock> {
    let size = size.unwrap_or(32.0);
    let rotation = compute_360_to_180(rotation.unwrap_or(0.0));
    let face = face_for(font_name)?;
    let rendered = render_freetype(&face, text, size, rotation)?;
    let (h, w) = rendered.dim();
    // rendered[::-1, :]
    let flipped = Array2::from_shape_fn((h, w), |(r, c)| rendered[[h - 1 - r, c]]);
    let col_ink: Vec<usize> = (0..w)
        .filter(|&c| (0..h).any(|r| flipped[[r, c]] != 0))
        .collect();
    let row_ink: Vec<usize> = (0..h)
        .filter(|&r| (0..w).any(|c| flipped[[r, c]] != 0))
        .collect();
    if row_ink.is_empty() || col_ink.is_empty() {
        return Ok(TextBlock {
            bbox: [0, 0, 1, 1],
            block: Array2::zeros((1, 1)),
        });
    }
    let (y1, y2) = (row_ink[0], *row_ink.last().unwrap());
    let (x1, x2) = (col_ink[0], *col_ink.last().unwrap());
    let r0 = y1.saturating_sub(1);
    let r1 = (y2 + 2).min(h);
    let c0 = x1.saturating_sub(1);
    let c1 = (x2 + 2).min(w);
    let cropped = flipped
        .slice(ndarray::s![r0..r1, c0..c1])
        .mapv(|v| v as f32 / 255.0);
    let (rows, cols) = cropped.dim();
    Ok(TextBlock {
        bbox: [0, 0, rows, cols],
        block: cropped,
    })
}

/// `compute_offset(angle, h, w)`: where to shift a text block's location so
/// the text is anchored by its rotation. Floats because `-w / 2` is a true
/// division in Python at -90°; callers truncate when building coordinates.
pub fn compute_offset(angle: f64, h: usize, w: usize) -> (f64, f64) {
    let (h, w) = (h as f64, w as f64);
    // Python: -h // 2 (floor division)
    let neg_half_floor = |v: f64| (-v / 2.0).floor();
    let (mut dx, mut dy) = (0.0, 0.0);
    if angle == 0.0 {
        dy = neg_half_floor(h);
    } else if angle > 0.0 && angle < 90.0 {
        dy = -h;
    } else if angle == 90.0 {
        dx = neg_half_floor(w);
        dy = -h;
    } else if (90.0..180.0).contains(&angle) {
        dx = -w;
        dy = -h;
    } else if angle == 180.0 || angle == -180.0 {
        dx = -w;
        dy = neg_half_floor(h);
    } else if angle > -90.0 && angle < 0.0 {
        // nothing
    } else if angle == -90.0 {
        dx = -w / 2.0;
    } else if angle > -180.0 && angle < -90.0 {
        dx = -w;
    }
    (dx, dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_normalisation() {
        assert_eq!(compute_360_to_180(0.0), 0.0);
        assert_eq!(compute_360_to_180(190.0), -170.0);
        assert_eq!(compute_360_to_180(-30.0), -30.0);
        assert_eq!(compute_360_to_180(360.0), 0.0);
        assert_eq!(compute_360_to_180(180.0), 180.0);
    }

    #[test]
    fn offsets() {
        assert_eq!(compute_offset(0.0, 5, 10), (0.0, -3.0));
        assert_eq!(compute_offset(90.0, 5, 11), (-6.0, -5.0));
        assert_eq!(compute_offset(-90.0, 5, 11), (-5.5, 0.0));
        assert_eq!(compute_offset(180.0, 5, 11), (-11.0, -3.0));
    }

    #[test]
    fn renders_text() {
        let t = render_text("Hi", "default", Some(32.0), None).unwrap();
        assert_eq!(t.bbox[2], t.block.nrows());
        assert!(t.block.nrows() > 10 && t.block.ncols() > 10);
        assert!(t.block.iter().any(|&v| v == 1.0));
        let e = render_text("", "default", None, None).unwrap();
        assert_eq!(e.block.dim(), (1, 1));
        let r = render_text("Hi", "bold", Some(20.0), Some(90.0)).unwrap();
        assert!(r.block.nrows() > r.block.ncols());
    }
}
