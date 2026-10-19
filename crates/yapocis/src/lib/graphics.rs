//! `yapocis.lib.abstract_graphics`, `graphics` (one channel) and `graphics4`
//! (RGBA): rasterising lines, rectangles, circles, triangles, mesh triangles,
//! ellipses, composited blocks, sprites and text onto a canvas.
//!
//! The host logic is the Python wrappers' logic: coordinate swizzling into the
//! kernels' (row, col) order on every backend but CUDA, scalar-or-vector
//! coercion of colours/widths/alphas (with the same "a falsy scalar falls back
//! to the vector or the default" rule as `to_vector`), composite/sprite
//! packing with `ipixels` offsets (including the sprite offset table layout),
//! and text placement through `lib::typeset`.

use crate::rt::{CallArg, CallOptions, HostView, HostViewMut};
use crate::{Buffer, Device, Error, Result};
use ndarray::{ArrayView2, ArrayView3, ArrayViewMutD};
use yapocis_bundle::Backend;

// ---------------------------------------------------------------------------
// coercion helpers (abstract_graphics.py)
// ---------------------------------------------------------------------------

/// `to_vector(scalar, vector, n, default)`: a truthy (non-zero) scalar fills
/// the vector; otherwise the vector is used if given, else the default.
pub fn to_vector(
    scalar: Option<f32>,
    vector: Option<&[f32]>,
    n: usize,
    default: f32,
) -> Result<Vec<f32>> {
    match scalar {
        Some(s) if s != 0.0 => Ok(vec![s; n]),
        _ => match vector {
            Some(v) => {
                if v.len() != n {
                    return Err(Error::Invalid(format!(
                        "value length {} does not match specified {n}",
                        v.len()
                    )));
                }
                Ok(v.to_vec())
            }
            None => Ok(vec![default; n]),
        },
    }
}

/// `to_list(scalar, vector, n, default)`: the vector wins when given; `None`
/// entries become the default.
pub fn to_list<T: Clone>(
    scalar: Option<T>,
    vector: Option<&[Option<T>]>,
    n: usize,
    default: Option<T>,
) -> Result<Vec<Option<T>>> {
    let out: Vec<Option<T>> = match vector {
        Some(v) => v.to_vec(),
        None => vec![scalar.or(default.clone()); n],
    };
    if out.len() != n {
        return Err(Error::Invalid(format!(
            "expected {n} values, got {}",
            out.len()
        )));
    }
    Ok(out.into_iter().map(|v| v.or(default.clone())).collect())
}

/// `KERNEL_ORDER`: how one shape's (x, y) coordinate row is reordered for the
/// (row, col) kernels. CUDA kernels take (x, y) as given.
pub fn kernel_order(shape: &str) -> &'static [usize] {
    match shape {
        "lines" | "rects" | "ellipses" => &[1, 0, 3, 2],
        "circles" => &[1, 0, 2],
        "triangles" | "mesh_triangles" => &[1, 0, 3, 2, 5, 4],
        _ => panic!("no kernel order for {shape}"),
    }
}

/// `to_kernel_order`: reorder coordinate rows unless the backend takes (x, y).
pub fn to_kernel_order(coords: &[Vec<f32>], shape: &str, backend: Backend) -> Vec<Vec<f32>> {
    if backend == Backend::Cuda {
        return coords.to_vec();
    }
    let order = kernel_order(shape);
    coords
        .iter()
        .map(|row| order.iter().map(|&i| row[i]).collect())
        .collect()
}

/// Per-primitive styling: scalar (`color`) or per-shape (`colors`) values,
/// with the Python keyword names. `C` is `f32` for one-channel graphics and
/// `[f32; 4]` for RGBA.
#[derive(Debug, Clone)]
pub struct Style<C> {
    pub color: Option<C>,
    pub colors: Option<Vec<C>>,
    /// One-channel only; RGBA carries alpha in the colour.
    pub alpha: Option<f32>,
    pub alphas: Option<Vec<f32>>,
    pub width: Option<f32>,
    pub widths: Option<Vec<f32>>,
    /// Alias for `width` (lines); giving both is an error.
    pub distance: Option<f32>,
    pub aa_width: Option<f32>,
    pub aa_widths: Option<Vec<f32>>,
}

impl<C> Default for Style<C> {
    fn default() -> Self {
        Style {
            color: None,
            colors: None,
            alpha: None,
            alphas: None,
            width: None,
            widths: None,
            distance: None,
            aa_width: None,
            aa_widths: None,
        }
    }
}

impl<C: Clone> Style<C> {
    pub fn color(mut self, c: C) -> Self {
        self.color = Some(c);
        self
    }
    pub fn colors(mut self, c: Vec<C>) -> Self {
        self.colors = Some(c);
        self
    }
    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = Some(a);
        self
    }
    pub fn alphas(mut self, a: Vec<f32>) -> Self {
        self.alphas = Some(a);
        self
    }
    pub fn width(mut self, w: f32) -> Self {
        self.width = Some(w);
        self
    }
    pub fn widths(mut self, w: Vec<f32>) -> Self {
        self.widths = Some(w);
        self
    }
    pub fn distance(mut self, d: f32) -> Self {
        self.distance = Some(d);
        self
    }
    pub fn aa_width(mut self, w: f32) -> Self {
        self.aa_width = Some(w);
        self
    }
    pub fn aa_widths(mut self, w: Vec<f32>) -> Self {
        self.aa_widths = Some(w);
        self
    }

    /// `coerce_widths`: `width` (or `distance`) fills, else `widths`, else 1.0.
    pub fn widths_for(&self, n: usize) -> Result<Vec<f32>> {
        let truthy = |v: Option<f32>| matches!(v, Some(x) if x != 0.0);
        if truthy(self.width) && self.widths.is_some() {
            return Err(Error::Invalid("Only one of width and widths".into()));
        }
        if truthy(self.width) && truthy(self.distance) {
            return Err(Error::Invalid("Only one of width/distance".into()));
        }
        let width = if self.width.is_none() {
            self.distance
        } else {
            self.width
        };
        to_vector(width, self.widths.as_deref(), n, 1.0)
    }

    /// `coerce_aa_widths`: `aa_width` fills, else `aa_widths`, else 0.0.
    pub fn aa_widths_for(&self, n: usize) -> Result<Vec<f32>> {
        if self.aa_width.is_some() && self.aa_widths.is_some() {
            return Err(Error::Invalid("Only one of aa_width and aa_widths".into()));
        }
        to_vector(self.aa_width, self.aa_widths.as_deref(), n, 0.0)
    }
}

impl Style<f32> {
    /// `Graphics1Setup.coerce_colors`: `(n, 2)` rows of `[colour, alpha]`.
    pub fn colors_for(&self, n: usize) -> Result<Vec<f32>> {
        let truthy = |v: Option<f32>| matches!(v, Some(x) if x != 0.0);
        if truthy(self.color) && self.colors.is_some() {
            return Err(Error::Invalid("Only one of color and colors".into()));
        }
        if truthy(self.alpha) && self.alphas.is_some() {
            return Err(Error::Invalid("Only one of alpha and alphas".into()));
        }
        let colors = to_vector(self.color, self.colors.as_deref(), n, 1.0)?;
        let alphas = to_vector(self.alpha, self.alphas.as_deref(), n, 1.0)?;
        Ok(colors
            .iter()
            .zip(&alphas)
            .flat_map(|(c, a)| [*c, *a])
            .collect())
    }
}

impl Style<[f32; 4]> {
    /// `GraphicsSetup4.coerce_colors`: `(n, 4)` RGBA rows; one of `color` or
    /// `colors` is required.
    pub fn colors_for(&self, n: usize) -> Result<Vec<f32>> {
        if self.color.is_some() && self.colors.is_some() {
            return Err(Error::Invalid("Only one of color and colors".into()));
        }
        match (&self.color, &self.colors) {
            (Some(c), _) => Ok((0..n).flat_map(|_| *c).collect()),
            (None, Some(cs)) => {
                if cs.len() != n {
                    return Err(Error::Invalid(format!(
                        "expected {n} colours, got {}",
                        cs.len()
                    )));
                }
                Ok(cs.iter().flat_map(|c| *c).collect())
            }
            (None, None) => Err(Error::Invalid("Missing one of color or colors".into())),
        }
    }
}

/// Per-block opacity for `composite` (one-channel).
#[derive(Debug, Clone)]
pub enum BlockAlpha {
    /// A constant over the block.
    Value(f32),
    /// One value per pixel, the block's shape.
    Mask(ndarray::Array2<f32>),
}

/// Text opacity, as the Python `alpha` keyword of `draw_strings`.
#[derive(Debug, Clone)]
pub enum TextAlpha {
    /// `"mask"`: the glyph coverage.
    Mask,
    /// `"bug"`: coverage with a floor of 0.33.
    Bug,
    /// A constant.
    Value(f32),
}

/// `draw_strings` styling. `C` as in [`Style`]; one-channel ignores colours.
#[derive(Debug, Clone)]
pub struct TextStyle<C> {
    pub font: Option<String>,
    pub fonts: Option<Vec<Option<String>>>,
    pub size: Option<f64>,
    pub sizes: Option<Vec<Option<f64>>>,
    pub alpha: Option<TextAlpha>,
    pub alphas: Option<Vec<Option<TextAlpha>>>,
    pub color: Option<C>,
    pub colors: Option<Vec<Option<C>>>,
    pub rotation: Option<f64>,
    pub rotations: Option<Vec<Option<f64>>>,
}

impl<C> Default for TextStyle<C> {
    fn default() -> Self {
        TextStyle {
            font: None,
            fonts: None,
            size: None,
            sizes: None,
            alpha: None,
            alphas: None,
            color: None,
            colors: None,
            rotation: None,
            rotations: None,
        }
    }
}

/// Where a drawing call writes: a host image written back, or a resident buffer.
pub enum Target<'a, 'b> {
    Host(&'a mut ArrayViewMutD<'b, f32>),
    Resident(&'a Buffer<f32>),
}

impl Target<'_, '_> {
    fn shape(&self) -> Vec<usize> {
        match self {
            Target::Host(a) => a.shape().to_vec(),
            Target::Resident(b) => b.shape().to_vec(),
        }
    }
}

/// What `setup_texts` returns for one-channel text: shifted locations, the
/// coverage blocks and the per-block alphas.
pub type Texts1 = (
    Vec<(f64, f64)>,
    Vec<ndarray::Array2<f32>>,
    Vec<Option<BlockAlpha>>,
);
/// What `setup_texts` returns for RGBA text: shifted locations and RGBA blocks.
pub type Texts4 = (Vec<(f64, f64)>, Vec<ndarray::Array3<f32>>);

/// A packed composite: `coords` rows `[x, y, w, h]`, the pixel buffer and the
/// per-element pixel offsets.
pub struct Packed {
    pub coords: Vec<i32>,
    pub n: usize,
    /// Pixel buffer and its logical shape as the kernel sees it.
    pub pixels: Vec<f32>,
    pub pixels_shape: Vec<usize>,
    pub ipixels: Vec<i32>,
}

// ---------------------------------------------------------------------------
// the shared drawing engine
// ---------------------------------------------------------------------------

/// What differs between the one-channel and RGBA interfaces.
pub trait Channels: Sized {
    /// Colour type of a shape.
    type Color: Clone;
    const CHANNELS: usize;
    /// Python's `coerce_coords` dtype: `int32` for one channel (coordinates are
    /// truncated), `float32` for RGBA.
    fn coerce_coord(v: f32) -> f32;
    fn colors_for(style: &Style<Self::Color>, n: usize) -> Result<Vec<f32>>;
    fn color_width() -> usize;
}

/// One-channel graphics (`Graphics1`): `(h, w)` canvases, `[colour, alpha]` shape colours.
pub struct OneChannel;
/// RGBA graphics (`Graphics4`): `(h, w, 4)` canvases, RGBA shape colours.
pub struct FourChannel;

impl Channels for OneChannel {
    type Color = f32;
    const CHANNELS: usize = 1;
    fn coerce_coord(v: f32) -> f32 {
        v as i32 as f32
    }
    fn colors_for(style: &Style<f32>, n: usize) -> Result<Vec<f32>> {
        style.colors_for(n)
    }
    fn color_width() -> usize {
        2
    }
}

impl Channels for FourChannel {
    type Color = [f32; 4];
    const CHANNELS: usize = 4;
    fn coerce_coord(v: f32) -> f32 {
        v
    }
    fn colors_for(style: &Style<[f32; 4]>, n: usize) -> Result<Vec<f32>> {
        style.colors_for(n)
    }
    fn color_width() -> usize {
        4
    }
}

/// The drawing engine behind [`Graphics1`] and [`Graphics4`]: holds the compiled
/// interface and runs kernels on host images or resident buffers.
pub struct Engine<K: Channels> {
    program: crate::rt::Program,
    device: Device,
    _k: std::marker::PhantomData<K>,
}

impl<K: Channels> Engine<K> {
    fn new(device: &Device, program: crate::rt::Program) -> Engine<K> {
        Engine {
            program,
            device: device.clone(),
            _k: std::marker::PhantomData,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn program(&self) -> &crate::rt::Program {
        &self.program
    }

    fn backend(&self) -> Backend {
        self.device.backend()
    }

    /// `coerce_coords` + `to_kernel_order`.
    pub(crate) fn coords(
        &self,
        coords: &[Vec<f32>],
        width: usize,
        shape: &str,
    ) -> Result<Vec<f32>> {
        for c in coords {
            if c.len() != width {
                return Err(Error::Invalid(format!(
                    "{shape}: expected {width} values per shape, got {}",
                    c.len()
                )));
            }
        }
        let coerced: Vec<Vec<f32>> = coords
            .iter()
            .map(|c| c.iter().map(|&v| K::coerce_coord(v)).collect())
            .collect();
        Ok(to_kernel_order(&coerced, shape, self.backend()).concat())
    }

    /// Run one kernel: `target` first, then the arrays, with the per-pixel global size.
    pub(crate) fn run(
        &self,
        kernel: &str,
        target: Target<'_, '_>,
        args: &[(&[f32], Vec<usize>)],
        ints: &[(usize, &[i32], Vec<usize>)],
        global: Option<usize>,
        async_exec: bool,
    ) -> Result<()> {
        let shape = target.shape();
        let size: usize = shape.iter().product();
        let global = global.unwrap_or(size / K::CHANNELS);
        let mut call_args: Vec<CallArg<'_>> = Vec::with_capacity(args.len() + ints.len() + 1);
        match target {
            Target::Host(a) => {
                let slice = a.as_slice_mut().ok_or(Error::NotContiguous)?;
                call_args.push(CallArg::InOut(HostViewMut::of(slice, &shape)));
            }
            Target::Resident(b) => call_args.push(CallArg::Buffer(b.as_any())),
        }
        // interleave int arrays at their positions (composite: coords, pixels, ipixels)
        let mut fi = 0;
        let total = args.len() + ints.len();
        for pos in 0..total {
            if let Some((_, data, sh)) = ints.iter().find(|(p, _, _)| *p == pos) {
                call_args.push(CallArg::In(HostView::of(data, sh)));
            } else {
                let (data, sh) = &args[fi];
                fi += 1;
                call_args.push(CallArg::In(HostView::of(data, sh)));
            }
        }
        let opts = CallOptions {
            global: Some(vec![global]),
            async_exec,
            ..Default::default()
        };
        self.program.call(kernel, &mut call_args, &opts)?;
        Ok(())
    }

    /// Lines, rects, circles, triangles, mesh triangles or ellipses.
    pub fn shapes(
        &self,
        shape: &str,
        target: Target<'_, '_>,
        coords: &[Vec<f32>],
        style: &Style<K::Color>,
        async_exec: bool,
    ) -> Result<()> {
        if coords.is_empty() {
            return Ok(());
        }
        let width = match shape {
            "lines" | "rects" | "ellipses" => 4,
            "circles" => 3,
            "triangles" | "mesh_triangles" => 6,
            other => return Err(Error::Invalid(format!("unknown shape {other}"))),
        };
        let n = coords.len();
        let flat = self.coords(coords, width, shape)?;
        let colors = K::colors_for(style, n)?;
        let aa = style.aa_widths_for(n)?;
        let kernel = if matches!(target, Target::Resident(_)) {
            format!("{shape}_res")
        } else {
            shape.to_string()
        };
        if shape == "lines" {
            let widths = style.widths_for(n)?;
            self.run(
                &kernel,
                target,
                &[
                    (&flat, vec![n, width]),
                    (&widths, vec![n]),
                    (&aa, vec![n]),
                    (&colors, vec![n, K::color_width()]),
                ],
                &[],
                None,
                async_exec,
            )
        } else {
            self.run(
                &kernel,
                target,
                &[
                    (&flat, vec![n, width]),
                    (&aa, vec![n]),
                    (&colors, vec![n, K::color_width()]),
                ],
                &[],
                None,
                async_exec,
            )
        }
    }

    /// Run `composite` with a packed block set.
    pub fn composite_packed(
        &self,
        target: Target<'_, '_>,
        packed: &Packed,
        async_exec: bool,
    ) -> Result<()> {
        let kernel = if matches!(target, Target::Resident(_)) {
            "composite_res"
        } else {
            "composite"
        };
        self.run(
            kernel,
            target,
            &[(&packed.pixels, packed.pixels_shape.clone())],
            &[
                (0, &packed.coords, vec![packed.n, 4]),
                (2, &packed.ipixels, vec![packed.ipixels.len()]),
            ],
            None,
            async_exec,
        )
    }
}

// ---------------------------------------------------------------------------
// Graphics1
// ---------------------------------------------------------------------------

/// `yapocis.lib.graphics.Graphics1`.
pub struct Graphics1 {
    pub engine: Engine<OneChannel>,
}

impl Graphics1 {
    pub fn load(device: &Device) -> Result<Graphics1> {
        let entry = crate::MANIFEST
            .interface("graphics")
            .expect("graphics in manifest");
        let source = crate::interfaces::graphics::source(device.backend().name())
            .ok_or(Error::BackendUnavailable(device.backend()))?;
        Ok(Graphics1 {
            engine: Engine::new(device, device.load(entry, source)?),
        })
    }

    pub fn device(&self) -> &Device {
        self.engine.device()
    }

    pub fn lines(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<f32>,
    ) -> Result<()> {
        self.engine
            .shapes("lines", Target::Host(image), coords, style, false)
    }
    pub fn rects(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<f32>,
    ) -> Result<()> {
        self.engine
            .shapes("rects", Target::Host(image), coords, style, false)
    }
    pub fn circles(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<f32>,
    ) -> Result<()> {
        self.engine
            .shapes("circles", Target::Host(image), coords, style, false)
    }
    pub fn triangles(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<f32>,
    ) -> Result<()> {
        self.engine
            .shapes("triangles", Target::Host(image), coords, style, false)
    }
    /// Triangles of a mesh: a pixel on an edge shared by two triangles is drawn once.
    pub fn mesh_triangles(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<f32>,
    ) -> Result<()> {
        self.engine
            .shapes("mesh_triangles", Target::Host(image), coords, style, false)
    }
    pub fn ellipses(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<f32>,
    ) -> Result<()> {
        self.engine
            .shapes("ellipses", Target::Host(image), coords, style, false)
    }

    /// `Graphics1Setup.setup_composite`: `[intensity, alpha]` pixel pairs.
    pub fn setup_composite(
        locations: &[(f64, f64)],
        blocks: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<Packed> {
        if locations.len() != blocks.len() {
            return Err(Error::Invalid("one location per block".into()));
        }
        if alpha.is_some() && alphas.is_some() {
            return Err(Error::Invalid("Only one of alpha and alphas".into()));
        }
        let alphas = to_list(alpha, alphas.as_deref(), locations.len(), None)?;
        let mut coords = Vec::with_capacity(locations.len() * 4);
        let mut ipixels = Vec::with_capacity(locations.len());
        let mut pixels: Vec<f32> = Vec::new();
        let mut ipixel = 0usize;
        for ((block, (x, y)), alpha) in blocks.iter().zip(locations).zip(&alphas) {
            let (height, width) = block.dim();
            coords.extend_from_slice(&[*x as i32, *y as i32, width as i32, height as i32]);
            ipixels.push(ipixel as i32);
            ipixel += block.len();
            let std = block.as_standard_layout();
            let flat = std.as_slice().unwrap();
            let alpha_flat: Vec<f32> = match alpha {
                None => vec![1.0; flat.len()],
                Some(BlockAlpha::Value(a)) => vec![*a; flat.len()],
                Some(BlockAlpha::Mask(m)) => {
                    if m.len() != flat.len() {
                        return Err(Error::Invalid("alpha mask must match the block".into()));
                    }
                    m.as_standard_layout().as_slice().unwrap().to_vec()
                }
            };
            for (p, a) in flat.iter().zip(alpha_flat) {
                pixels.push(*p);
                pixels.push(a);
            }
        }
        let n = locations.len();
        Ok(Packed {
            coords,
            n,
            pixels_shape: vec![pixels.len() / 2, 2],
            pixels,
            ipixels,
        })
    }

    /// `Graphics1Setup.setup_sprites`: locations are `(sprite index, x, y)`.
    /// The offset table lists the sprite offsets first and then one entry per
    /// location, as the Python code builds it (the kernel reads the first
    /// `len(locations)` entries).
    pub fn setup_sprites(
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<Packed> {
        if alpha.is_some() && alphas.is_some() {
            return Err(Error::Invalid("Only one of alpha and alphas".into()));
        }
        let alphas = to_list(alpha, alphas.as_deref(), sprites.len(), None)?;
        let mut pixels: Vec<f32> = Vec::new();
        let mut ipixels: Vec<i32> = Vec::new();
        let mut ipixel = 0usize;
        for (sprite, alpha) in sprites.iter().zip(&alphas) {
            let std = sprite.as_standard_layout();
            let flat = std.as_slice().unwrap();
            ipixels.push(ipixel as i32);
            ipixel += flat.len();
            let alpha_flat: Vec<f32> = match alpha {
                None => vec![1.0; flat.len()],
                Some(BlockAlpha::Value(a)) => vec![*a; flat.len()],
                Some(BlockAlpha::Mask(m)) => m.as_standard_layout().as_slice().unwrap().to_vec(),
            };
            for (p, a) in flat.iter().zip(alpha_flat) {
                pixels.push(*p);
                pixels.push(a);
            }
        }
        let mut coords = Vec::with_capacity(locations.len() * 4);
        for (isprite, x, y) in locations {
            let sprite = sprites
                .get(*isprite)
                .ok_or_else(|| Error::Invalid(format!("sprite {isprite} out of range")))?;
            let (height, width) = sprite.dim();
            coords.extend_from_slice(&[*x as i32, *y as i32, width as i32, height as i32]);
            ipixels.push(ipixels[*isprite]);
        }
        Ok(Packed {
            coords,
            n: locations.len(),
            pixels_shape: vec![pixels.len() / 2, 2],
            pixels,
            ipixels,
        })
    }

    /// `Graphics1Setup.setup_texts`: rasterise, anchor by rotation, and turn the
    /// text alphas into per-block alphas.
    #[cfg(feature = "text")]
    pub fn setup_texts(
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<f32>,
    ) -> Result<Texts1> {
        let n = locations.len();
        if texts.len() != n {
            return Err(Error::Invalid("one text per location".into()));
        }
        let fonts = to_list(
            style.font.clone(),
            style.fonts.as_deref(),
            n,
            Some("default".to_string()),
        )?;
        let sizes = to_list(style.size, style.sizes.as_deref(), n, Some(32.0))?;
        let alphas = to_list(style.alpha.clone(), style.alphas.as_deref(), n, None)?;
        let rotations = to_list(style.rotation, style.rotations.as_deref(), n, Some(0.0))?;
        let rotations: Vec<f64> = rotations
            .into_iter()
            .map(|r| super::typeset::compute_360_to_180(r.unwrap()))
            .collect();
        let mut blocks = Vec::with_capacity(n);
        let mut out_locations = Vec::with_capacity(n);
        let mut out_alphas = Vec::with_capacity(n);
        for i in 0..n {
            let t = super::typeset::render_text(
                texts[i],
                fonts[i].as_deref().unwrap(),
                sizes[i],
                Some(rotations[i]),
            )?;
            let (dx, dy) = super::typeset::compute_offset(rotations[i], t.bbox[2], t.bbox[3]);
            out_locations.push((locations[i].0 + dx, locations[i].1 + dy));
            out_alphas.push(match &alphas[i] {
                None => None,
                Some(TextAlpha::Value(v)) => Some(BlockAlpha::Value(*v)),
                Some(TextAlpha::Mask) => Some(BlockAlpha::Mask(t.block.clone())),
                Some(TextAlpha::Bug) => Some(BlockAlpha::Mask(t.block.mapv(|v| v.max(0.33)))),
            });
            blocks.push(t.block);
        }
        Ok((out_locations, blocks, out_alphas))
    }

    pub fn composite(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        locations: &[(f64, f64)],
        blocks: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<()> {
        let packed = Self::setup_composite(locations, blocks, alpha, alphas)?;
        self.engine
            .composite_packed(Target::Host(image), &packed, false)
    }

    pub fn sprites(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<()> {
        let packed = Self::setup_sprites(locations, sprites, alpha, alphas)?;
        self.engine
            .composite_packed(Target::Host(image), &packed, false)
    }

    #[cfg(feature = "text")]
    pub fn draw_strings(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<f32>,
    ) -> Result<()> {
        let (locations, blocks, alphas) = Self::setup_texts(locations, texts, style)?;
        let views: Vec<ArrayView2<'_, f32>> = blocks.iter().map(|b| b.view()).collect();
        let packed = Self::setup_composite(&locations, &views, None, Some(alphas))?;
        self.engine
            .composite_packed(Target::Host(image), &packed, false)
    }
}

// ---------------------------------------------------------------------------
// Graphics4
// ---------------------------------------------------------------------------

/// `yapocis.lib.graphics4.Graphics4`: RGBA canvases `(h, w, 4)`.
pub struct Graphics4 {
    pub engine: Engine<FourChannel>,
}

impl Graphics4 {
    pub fn load(device: &Device) -> Result<Graphics4> {
        let entry = crate::MANIFEST
            .interface("graphics4")
            .expect("graphics4 in manifest");
        let source = crate::interfaces::graphics4::source(device.backend().name())
            .ok_or(Error::BackendUnavailable(device.backend()))?;
        Ok(Graphics4 {
            engine: Engine::new(device, device.load(entry, source)?),
        })
    }

    pub fn device(&self) -> &Device {
        self.engine.device()
    }

    pub fn lines(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<[f32; 4]>,
    ) -> Result<()> {
        self.engine
            .shapes("lines", Target::Host(image), coords, style, false)
    }
    pub fn rects(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<[f32; 4]>,
    ) -> Result<()> {
        self.engine
            .shapes("rects", Target::Host(image), coords, style, false)
    }
    pub fn circles(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<[f32; 4]>,
    ) -> Result<()> {
        self.engine
            .shapes("circles", Target::Host(image), coords, style, false)
    }
    pub fn triangles(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<[f32; 4]>,
    ) -> Result<()> {
        self.engine
            .shapes("triangles", Target::Host(image), coords, style, false)
    }
    pub fn mesh_triangles(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<[f32; 4]>,
    ) -> Result<()> {
        self.engine
            .shapes("mesh_triangles", Target::Host(image), coords, style, false)
    }
    pub fn ellipses(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        coords: &[Vec<f32>],
        style: &Style<[f32; 4]>,
    ) -> Result<()> {
        self.engine
            .shapes("ellipses", Target::Host(image), coords, style, false)
    }

    /// `GraphicsSetup4.setup_composite`: RGBA blocks `(h, w, 4)` packed flat.
    pub fn setup_composite(
        locations: &[(f64, f64)],
        blocks: &[ArrayView3<'_, f32>],
    ) -> Result<Packed> {
        if locations.len() != blocks.len() {
            return Err(Error::Invalid("one location per block".into()));
        }
        let mut coords = Vec::with_capacity(locations.len() * 4);
        let mut ipixels = Vec::with_capacity(locations.len());
        let mut pixels: Vec<f32> = Vec::new();
        let mut ipixel = 0usize;
        for (block, (x, y)) in blocks.iter().zip(locations) {
            let (height, width, depth) = block.dim();
            if depth != 4 {
                return Err(Error::Invalid("blocks must be (h, w, 4)".into()));
            }
            let std = block.as_standard_layout();
            let flat = std.as_slice().unwrap();
            pixels.extend_from_slice(flat);
            coords.extend_from_slice(&[*x as i32, *y as i32, width as i32, height as i32]);
            ipixels.push(ipixel as i32);
            ipixel += flat.len() / 4;
        }
        Ok(Packed {
            coords,
            n: locations.len(),
            pixels_shape: vec![pixels.len()],
            pixels,
            ipixels,
        })
    }

    /// `GraphicsSetup4.setup_sprites`.
    pub fn setup_sprites(
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView3<'_, f32>],
    ) -> Result<Packed> {
        let mut pixels: Vec<f32> = Vec::new();
        let mut ipixels: Vec<i32> = Vec::new();
        let mut ipixel = 0usize;
        for sprite in sprites {
            let std = sprite.as_standard_layout();
            let flat = std.as_slice().unwrap();
            ipixels.push(ipixel as i32);
            ipixel += flat.len() / 4;
            pixels.extend_from_slice(flat);
        }
        let mut coords = Vec::with_capacity(locations.len() * 4);
        for (isprite, x, y) in locations {
            let sprite = sprites
                .get(*isprite)
                .ok_or_else(|| Error::Invalid(format!("sprite {isprite} out of range")))?;
            let (height, width, depth) = sprite.dim();
            if depth != 4 {
                return Err(Error::Invalid("sprites must be (h, w, 4)".into()));
            }
            coords.extend_from_slice(&[*x as i32, *y as i32, width as i32, height as i32]);
            ipixels.push(ipixels[*isprite]);
        }
        Ok(Packed {
            coords,
            n: locations.len(),
            pixels_shape: vec![pixels.len()],
            pixels,
            ipixels,
        })
    }

    /// `GraphicsSetup4.setup_texts`: coloured RGBA text blocks.
    #[cfg(feature = "text")]
    pub fn setup_texts(
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<[f32; 4]>,
    ) -> Result<Texts4> {
        let n = locations.len();
        if texts.len() != n {
            return Err(Error::Invalid("one text per location".into()));
        }
        let colors = to_list(
            style.color,
            style.colors.as_deref(),
            n,
            Some([1.0, 1.0, 1.0, 1.0]),
        )?;
        let fonts = to_list(
            style.font.clone(),
            style.fonts.as_deref(),
            n,
            Some("default".to_string()),
        )?;
        let sizes = to_list(style.size, style.sizes.as_deref(), n, Some(32.0))?;
        let alphas = to_list(
            style.alpha.clone(),
            style.alphas.as_deref(),
            n,
            Some(TextAlpha::Mask),
        )?;
        let rotations = to_list(style.rotation, style.rotations.as_deref(), n, Some(0.0))?;
        let rotations: Vec<f64> = rotations
            .into_iter()
            .map(|r| super::typeset::compute_360_to_180(r.unwrap()))
            .collect();
        let mut blocks = Vec::with_capacity(n);
        let mut out_locations = Vec::with_capacity(n);
        for i in 0..n {
            let t = super::typeset::render_text(
                texts[i],
                fonts[i].as_deref().unwrap(),
                sizes[i],
                Some(rotations[i]),
            )?;
            let (dx, dy) = super::typeset::compute_offset(rotations[i], t.bbox[2], t.bbox[3]);
            out_locations.push((locations[i].0 + dx, locations[i].1 + dy));
            let color = colors[i].unwrap();
            let (rows, cols) = t.block.dim();
            let mut rgba = ndarray::Array3::<f32>::zeros((rows, cols, 4));
            for r in 0..rows {
                for c in 0..cols {
                    let v = t.block[[r, c]];
                    rgba[[r, c, 0]] = v * color[0];
                    rgba[[r, c, 1]] = v * color[1];
                    rgba[[r, c, 2]] = v * color[2];
                    rgba[[r, c, 3]] = match &alphas[i] {
                        Some(TextAlpha::Mask) => v,
                        Some(TextAlpha::Bug) => v.max(0.33),
                        Some(TextAlpha::Value(a)) => *a,
                        None => color[3],
                    };
                }
            }
            blocks.push(rgba);
        }
        Ok((out_locations, blocks))
    }

    pub fn composite(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        locations: &[(f64, f64)],
        blocks: &[ArrayView3<'_, f32>],
    ) -> Result<()> {
        let packed = Self::setup_composite(locations, blocks)?;
        self.engine
            .composite_packed(Target::Host(image), &packed, false)
    }

    pub fn sprites(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView3<'_, f32>],
    ) -> Result<()> {
        let packed = Self::setup_sprites(locations, sprites)?;
        self.engine
            .composite_packed(Target::Host(image), &packed, false)
    }

    #[cfg(feature = "text")]
    pub fn draw_strings(
        &self,
        image: &mut ArrayViewMutD<'_, f32>,
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<[f32; 4]>,
    ) -> Result<()> {
        let (locations, blocks) = Self::setup_texts(locations, texts, style)?;
        let views: Vec<ArrayView3<'_, f32>> = blocks.iter().map(|b| b.view()).collect();
        let packed = Self::setup_composite(&locations, &views)?;
        self.engine
            .composite_packed(Target::Host(image), &packed, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_vector_rules() {
        assert_eq!(to_vector(Some(2.0), None, 3, 1.0).unwrap(), vec![2.0; 3]);
        // a falsy scalar falls back to the vector / default (Python `if scalar:`)
        assert_eq!(to_vector(Some(0.0), None, 2, 1.0).unwrap(), vec![1.0; 2]);
        assert_eq!(
            to_vector(Some(0.0), Some(&[0.5, 0.25]), 2, 1.0).unwrap(),
            vec![0.5, 0.25]
        );
        assert!(to_vector(None, Some(&[1.0]), 2, 1.0).is_err());
    }

    #[test]
    fn style_colors() {
        let s = Style::<f32>::default().color(0.5).alphas(vec![0.1, 0.2]);
        assert_eq!(s.colors_for(2).unwrap(), vec![0.5, 0.1, 0.5, 0.2]);
        let s = Style::<[f32; 4]>::default().color([1.0, 0.0, 0.0, 0.5]);
        assert_eq!(
            s.colors_for(2).unwrap(),
            vec![1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5]
        );
        assert!(Style::<[f32; 4]>::default().colors_for(1).is_err());
        let s = Style::<f32>::default().width(2.0).distance(3.0);
        assert!(s.widths_for(1).is_err());
        let s = Style::<f32>::default().distance(3.0);
        assert_eq!(s.widths_for(2).unwrap(), vec![3.0, 3.0]);
    }

    #[test]
    fn swizzle() {
        let c = vec![vec![1.0, 2.0, 3.0, 4.0]];
        assert_eq!(
            to_kernel_order(&c, "lines", Backend::Ocl),
            vec![vec![2.0, 1.0, 4.0, 3.0]]
        );
        assert_eq!(to_kernel_order(&c, "lines", Backend::Cuda), c);
        let t = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        assert_eq!(
            to_kernel_order(&t, "triangles", Backend::Metal),
            vec![vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0]]
        );
    }

    #[test]
    fn sprite_offsets_follow_python() {
        let a = ndarray::Array2::<f32>::ones((2, 3));
        let b = ndarray::Array2::<f32>::zeros((1, 2));
        let packed = Graphics1::setup_sprites(
            &[(1, 5.0, 6.0), (0, 1.0, 1.0)],
            &[a.view(), b.view()],
            None,
            None,
        )
        .unwrap();
        assert_eq!(packed.ipixels, vec![0, 6, 6, 0]);
        assert_eq!(packed.coords, vec![5, 6, 2, 1, 1, 1, 3, 2]);
        assert_eq!(packed.pixels_shape, vec![8, 2]);
    }
}
