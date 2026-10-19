//! `yapocis.lib.kerneldraw1/3/4`: a canvas kept resident on the device with
//! immediate drawing calls and display lists rendered in batches.
//!
//! Faithful to the Python `AbstractKernelDraw`, including its quirks: display
//! list items are rendered through the plural keyword columns, so a per-item
//! `alpha` or `distance` is ignored at render time (`coerce_colors` and
//! `coerce_widths` do not take `alphas`/`distances`); only the immediate calls
//! honour `alpha`, and only `width`/`widths` set a display-list line width.
//! `KernelDraw3` draws each channel through its own one-channel canvas.

#[cfg_attr(not(feature = "text"), allow(unused_imports))]
use super::graphics::{
    BlockAlpha, Channels, Engine, FourChannel, Graphics1, Graphics4, OneChannel, Packed, Style,
    Target, TextAlpha, TextStyle,
};
use crate::{Buffer, Device, Error, Result};
use ndarray::{Array2, Array3, ArrayD, ArrayView2, ArrayView3, ArrayViewD, Axis, IxDyn};

/// Python's default `chunk_size`.
pub const DEFAULT_CHUNK: usize = 500;

/// Keyword arguments of a display-list item (`line(xyxy, color=..., ...)`).
#[derive(Debug, Clone, Default)]
pub struct Item<C> {
    pub color: Option<C>,
    /// Recorded but ignored when the list renders (Python drops `alphas`).
    pub alpha: Option<f32>,
    pub width: Option<f32>,
    /// Recorded but ignored when the list renders (Python drops `distances`).
    pub distance: Option<f32>,
    pub aa_width: Option<f32>,
}

/// Keyword arguments of an immediate call (`ad.lines(coords, color=..., alpha=...)`).
#[derive(Debug, Clone)]
pub struct Draw<C> {
    pub color: Option<C>,
    pub colors: Option<Vec<C>>,
    /// Honoured for one-channel canvases (the alpha column of the colour pairs).
    pub alpha: Option<f32>,
    pub width: Option<f32>,
    pub widths: Option<Vec<f32>>,
    pub distance: Option<f32>,
    pub aa_width: Option<f32>,
    pub aa_widths: Option<Vec<f32>>,
}

impl<C> Default for Draw<C> {
    fn default() -> Self {
        Draw {
            color: None,
            colors: None,
            alpha: None,
            width: None,
            widths: None,
            distance: None,
            aa_width: None,
            aa_widths: None,
        }
    }
}

/// A queued text item (`draw_string`).
#[derive(Debug, Clone)]
pub struct TextItem<C> {
    pub font: Option<String>,
    pub size: Option<f64>,
    pub alpha: Option<TextAlpha>,
    pub color: Option<C>,
    pub rotation: Option<f64>,
}

impl<C> Default for TextItem<C> {
    fn default() -> Self {
        TextItem {
            font: None,
            size: None,
            alpha: None,
            color: None,
            rotation: None,
        }
    }
}

/// Colour coercion of `AbstractKernelDraw.coerce_colors` per colour width.
pub trait KdColor: Clone {
    const WIDTH: usize;
    /// Rows of the kernel colour array for `n` shapes.
    fn rows(
        color: Option<&Self>,
        colors: Option<&[Self]>,
        default: &Self,
        alpha: f32,
        n: usize,
    ) -> Result<Vec<f32>>;
}

impl KdColor for f32 {
    const WIDTH: usize = 1;
    fn rows(
        color: Option<&f32>,
        colors: Option<&[f32]>,
        default: &f32,
        alpha: f32,
        n: usize,
    ) -> Result<Vec<f32>> {
        let color = if color.is_none() && colors.is_none() {
            Some(default)
        } else {
            color
        };
        match color {
            Some(c) if *c != 0.0 => Ok((0..n).flat_map(|_| [*c, alpha]).collect()),
            _ => {
                let cs = colors
                    .ok_or_else(|| Error::Invalid("a zero colour needs explicit colors".into()))?;
                if cs.len() != n {
                    return Err(Error::Invalid(format!("Expected ({n}, 2) colours")));
                }
                Ok(cs.iter().flat_map(|c| [*c, alpha]).collect())
            }
        }
    }
}

impl KdColor for [f32; 4] {
    const WIDTH: usize = 4;
    fn rows(
        color: Option<&[f32; 4]>,
        colors: Option<&[[f32; 4]]>,
        default: &[f32; 4],
        _alpha: f32,
        n: usize,
    ) -> Result<Vec<f32>> {
        let color = if color.is_none() && colors.is_none() {
            Some(default)
        } else {
            color
        };
        match color {
            Some(c) => Ok((0..n).flat_map(|_| *c).collect()),
            None => {
                let cs = colors.unwrap();
                if cs.len() != n {
                    return Err(Error::Invalid(format!("Expected ({n}, 4) colours")));
                }
                Ok(cs.iter().flat_map(|c| *c).collect())
            }
        }
    }
}

enum Queued<C> {
    Shape(Vec<f32>, Item<C>),
    Sprite((usize, f64, f64), Array2<f32>, Option<BlockAlpha>),
    #[cfg_attr(not(feature = "text"), allow(dead_code))]
    Text((f64, f64), String, TextItem<C>),
}

/// `KernelDraw1` / `KernelDraw4`: the canvas lives on the device from
/// construction until [`read`](KernelDraw::read).
pub struct KernelDraw<K: Channels + Packing>
where
    K::Color: KdColor,
{
    engine: Engine<K>,
    canvas: Buffer<f32>,
    shape: Vec<usize>,
    pub chunk_size: Option<usize>,
    /// Default alpha for immediate calls (1.0).
    pub alpha: f32,
    /// Default colour when neither `color` nor `colors` is given.
    pub color: K::Color,
    global_size: usize,
    lists: std::collections::HashMap<&'static str, Vec<Queued<K::Color>>>,
}

pub type KernelDraw1 = KernelDraw<OneChannel>;
pub type KernelDraw4 = KernelDraw<FourChannel>;

const ORDER: [&str; 8] = [
    "rects",
    "circles",
    "ellipses",
    "triangles",
    "mesh_triangles",
    "lines",
    "draw_strings",
    "sprites",
];

impl KernelDraw<OneChannel> {
    /// `KernelDraw1(a)`: a 2-D canvas.
    pub fn new(
        device: &Device,
        canvas: &ArrayViewD<'_, f32>,
        chunk_size: Option<usize>,
    ) -> Result<Self> {
        let g = Graphics1::load(device)?;
        Self::with_engine(g.engine, canvas, chunk_size, 1.0)
    }
}

impl KernelDraw<FourChannel> {
    /// `KernelDraw4(a)`: an `(h, w, 4)` canvas.
    pub fn new(
        device: &Device,
        canvas: &ArrayViewD<'_, f32>,
        chunk_size: Option<usize>,
    ) -> Result<Self> {
        let g = Graphics4::load(device)?;
        Self::with_engine(g.engine, canvas, chunk_size, [1.0, 1.0, 1.0, 1.0])
    }
}

impl<K: Channels + Packing> KernelDraw<K>
where
    K::Color: KdColor,
{
    fn with_engine(
        engine: Engine<K>,
        canvas: &ArrayViewD<'_, f32>,
        chunk_size: Option<usize>,
        color: K::Color,
    ) -> Result<Self> {
        let std = canvas.as_standard_layout();
        let buffer = engine
            .device()
            .upload(std.as_slice().unwrap(), canvas.shape())?;
        Ok(KernelDraw {
            engine,
            canvas: buffer,
            shape: canvas.shape().to_vec(),
            chunk_size,
            alpha: 1.0,
            color,
            global_size: canvas.len() / K::Color::WIDTH,
            lists: std::collections::HashMap::new(),
        })
    }

    pub fn device(&self) -> &Device {
        self.engine.device()
    }

    pub fn canvas(&self) -> &Buffer<f32> {
        &self.canvas
    }

    /// `prepare_args`: coords (coerced), colours, widths and aa widths.
    fn prepare(
        &self,
        coords: &[Vec<f32>],
        kw: &Draw<K::Color>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let n = coords.len();
        let alpha = kw.alpha.unwrap_or(self.alpha);
        let colors = K::Color::rows(
            kw.color.as_ref(),
            kw.colors.as_deref(),
            &self.color,
            alpha,
            n,
        )?;
        let style: Style<K::Color> = Style {
            width: kw.width,
            widths: kw.widths.clone(),
            distance: kw.distance,
            aa_width: kw.aa_width,
            aa_widths: kw.aa_widths.clone(),
            ..Default::default()
        };
        Ok((colors, style.widths_for(n)?, style.aa_widths_for(n)?))
    }

    fn shapes(&self, shape: &str, coords: &[Vec<f32>], kw: &Draw<K::Color>) -> Result<()> {
        if coords.is_empty() {
            return Ok(());
        }
        let (colors, widths, aa) = self.prepare(coords, kw)?;
        // Engine::shapes recomputes colours from a Style; feed it the prepared vectors.
        let n = coords.len();
        let style = PreparedStyle {
            colors,
            widths,
            aa,
            n,
        };
        self.engine.shapes_prepared(
            shape,
            Target::Resident(&self.canvas),
            coords,
            &style,
            Some(self.global_size),
            true,
        )
    }

    pub fn lines(&self, coords: &[Vec<f32>], kw: &Draw<K::Color>) -> Result<()> {
        self.shapes("lines", coords, kw)
    }
    pub fn rects(&self, coords: &[Vec<f32>], kw: &Draw<K::Color>) -> Result<()> {
        self.shapes("rects", coords, kw)
    }
    pub fn circles(&self, coords: &[Vec<f32>], kw: &Draw<K::Color>) -> Result<()> {
        self.shapes("circles", coords, kw)
    }
    pub fn triangles(&self, coords: &[Vec<f32>], kw: &Draw<K::Color>) -> Result<()> {
        self.shapes("triangles", coords, kw)
    }
    pub fn mesh_triangles(&self, coords: &[Vec<f32>], kw: &Draw<K::Color>) -> Result<()> {
        self.shapes("mesh_triangles", coords, kw)
    }
    pub fn ellipses(&self, coords: &[Vec<f32>], kw: &Draw<K::Color>) -> Result<()> {
        self.shapes("ellipses", coords, kw)
    }

    fn composite_packed(&self, packed: &Packed) -> Result<()> {
        self.engine.composite_packed_global(
            Target::Resident(&self.canvas),
            packed,
            Some(self.global_size),
            true,
        )
    }

    // -- display lists --------------------------------------------------------

    fn push(&mut self, list: &'static str, coords: Vec<f32>, kw: Item<K::Color>) {
        self.lists
            .entry(list)
            .or_default()
            .push(Queued::Shape(coords, kw));
    }

    pub fn line(&mut self, xyxy: [f32; 4], kw: Item<K::Color>) {
        self.push("lines", xyxy.to_vec(), kw)
    }
    pub fn rect(&mut self, rect: [f32; 4], kw: Item<K::Color>) {
        self.push("rects", rect.to_vec(), kw)
    }
    pub fn circle(&mut self, xyr: [f32; 3], kw: Item<K::Color>) {
        self.push("circles", xyr.to_vec(), kw)
    }
    pub fn triangle(&mut self, t: [f32; 6], kw: Item<K::Color>) {
        self.push("triangles", t.to_vec(), kw)
    }
    pub fn mesh_triangle(&mut self, t: [f32; 6], kw: Item<K::Color>) {
        self.push("mesh_triangles", t.to_vec(), kw)
    }
    pub fn ellipse(&mut self, e: [f32; 4], kw: Item<K::Color>) {
        self.push("ellipses", e.to_vec(), kw)
    }
    /// Queue a one-channel sprite at `(sprite index, x, y)` (the index is into
    /// the sprites queued in the same batch, as the Python API expects).
    pub fn sprite(
        &mut self,
        location: (usize, f64, f64),
        sprite: Array2<f32>,
        alpha: Option<BlockAlpha>,
    ) {
        self.lists
            .entry("sprites")
            .or_default()
            .push(Queued::Sprite(location, sprite, alpha));
    }
    pub fn draw_string(&mut self, xy: (f64, f64), text: &str, kw: TextItem<K::Color>) {
        self.lists
            .entry("draw_strings")
            .or_default()
            .push(Queued::Text(xy, text.to_string(), kw));
    }

    /// Render one chunk of a shape list through the immediate call with the
    /// plural columns (`colors`, `widths`, `aa_widths`; `alphas`/`distances` dropped).
    fn render_shape_chunk(&self, shape: &str, chunk: &[Queued<K::Color>]) -> Result<()> {
        let mut coords = Vec::with_capacity(chunk.len());
        let mut colors = Vec::new();
        let mut widths = Vec::new();
        let mut aa_widths = Vec::new();
        for q in chunk {
            if let Queued::Shape(c, kw) = q {
                coords.push(c.clone());
                if let Some(col) = &kw.color {
                    colors.push(col.clone());
                }
                if let Some(w) = kw.width {
                    widths.push(w);
                }
                if let Some(a) = kw.aa_width {
                    aa_widths.push(a);
                }
            }
        }
        let n = coords.len();
        let column = |v: Vec<f32>, name: &str| -> Result<Option<Vec<f32>>> {
            if v.is_empty() {
                Ok(None)
            } else if v.len() == n {
                Ok(Some(v))
            } else {
                Err(Error::Invalid(format!(
                    "{name}: every item in a batch must give it, or none"
                )))
            }
        };
        let kw = Draw {
            color: None,
            colors: if colors.is_empty() {
                None
            } else if colors.len() == n {
                Some(colors)
            } else {
                return Err(Error::Invalid(
                    "color: every item in a batch must give it, or none".into(),
                ));
            },
            alpha: None,
            width: None,
            widths: column(widths, "width")?,
            distance: None,
            aa_width: None,
            aa_widths: column(aa_widths, "aa_width")?,
        };
        self.shapes(shape, &coords, &kw)
    }

    fn render_list(&mut self, name: &'static str) -> Result<()> {
        let Some(list) = self.lists.remove(name) else {
            return Ok(());
        };
        if list.is_empty() {
            return Ok(());
        }
        let chunks: Vec<&[Queued<K::Color>]> = match self.chunk_size {
            Some(c) if c > 0 => list.chunks(c).collect(),
            _ => vec![&list[..]],
        };
        for chunk in chunks {
            match name {
                "sprites" => {
                    let mut locations = Vec::new();
                    let mut sprites = Vec::new();
                    let mut alphas = Vec::new();
                    for q in chunk {
                        if let Queued::Sprite(loc, sprite, alpha) = q {
                            locations.push(*loc);
                            sprites.push(sprite.view());
                            alphas.push(alpha.clone());
                        }
                    }
                    self.sprites_views(&locations, &sprites, None, Some(alphas))?;
                }
                #[cfg(not(feature = "text"))]
                "draw_strings" => {
                    return Err(Error::Invalid("text needs the `text` feature".into()));
                }
                #[cfg(feature = "text")]
                "draw_strings" => {
                    let mut locations = Vec::new();
                    let mut texts: Vec<&str> = Vec::new();
                    let mut style = TextStyle::<K::Color> {
                        fonts: Some(Vec::new()),
                        sizes: Some(Vec::new()),
                        alphas: Some(Vec::new()),
                        colors: Some(Vec::new()),
                        rotations: Some(Vec::new()),
                        ..Default::default()
                    };
                    for q in chunk {
                        if let Queued::Text(xy, text, kw) = q {
                            locations.push(*xy);
                            texts.push(text);
                            style.fonts.as_mut().unwrap().push(kw.font.clone());
                            style.sizes.as_mut().unwrap().push(kw.size);
                            style.alphas.as_mut().unwrap().push(kw.alpha.clone());
                            style.colors.as_mut().unwrap().push(kw.color.clone());
                            style.rotations.as_mut().unwrap().push(kw.rotation);
                        }
                    }
                    self.draw_strings_style(&locations, &texts, &style)?;
                }
                shape => self.render_shape_chunk(shape, chunk)?,
            }
        }
        Ok(())
    }

    /// `render`: flush every display list in the Python order.
    pub fn render(&mut self) -> Result<()> {
        for name in ORDER {
            self.render_list(name)?;
        }
        Ok(())
    }

    /// `read`: render, wait, and return the canvas.
    pub fn read(&mut self) -> Result<ArrayD<f32>> {
        self.render()?;
        self.engine.device().wait_all()?;
        Ok(ArrayD::from_shape_vec(
            IxDyn(&self.shape),
            self.canvas.to_vec()?,
        )?)
    }

    // -- composite / sprites / text (channel-specific packing) ----------------

    fn sprites_views(
        &self,
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<()> {
        let packed = K::pack_sprites(locations, sprites, alpha, alphas)?;
        self.composite_packed(&packed)
    }

    #[cfg(feature = "text")]
    fn draw_strings_style(
        &self,
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<K::Color>,
    ) -> Result<()> {
        let packed = K::pack_texts(locations, texts, style)?;
        self.composite_packed(&packed)
    }

    /// Immediate `draw_strings`.
    #[cfg(feature = "text")]
    pub fn draw_strings(
        &self,
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<K::Color>,
    ) -> Result<()> {
        if locations.is_empty() {
            return Ok(());
        }
        self.draw_strings_style(locations, texts, style)
    }
}

impl KernelDraw<OneChannel> {
    /// Immediate `composite` of one-channel blocks.
    pub fn composite(
        &self,
        locations: &[(f64, f64)],
        blocks: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<()> {
        let packed = Graphics1::setup_composite(locations, blocks, alpha, alphas)?;
        self.composite_packed(&packed)
    }

    /// Immediate `sprites`.
    pub fn sprites(
        &self,
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<()> {
        if locations.is_empty() {
            return Ok(());
        }
        self.sprites_views(locations, sprites, alpha, alphas)
    }
}

impl KernelDraw<FourChannel> {
    /// Immediate `composite` of RGBA blocks.
    pub fn composite(
        &self,
        locations: &[(f64, f64)],
        blocks: &[ArrayView3<'_, f32>],
    ) -> Result<()> {
        let packed = Graphics4::setup_composite(locations, blocks)?;
        self.composite_packed(&packed)
    }

    /// Immediate `sprites` of RGBA sprites.
    pub fn sprites(
        &self,
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView3<'_, f32>],
    ) -> Result<()> {
        if locations.is_empty() {
            return Ok(());
        }
        let packed = Graphics4::setup_sprites(locations, sprites)?;
        self.composite_packed(&packed)
    }
}

/// Channel-specific packing used by the display lists.
pub trait Packing: Channels {
    fn pack_sprites(
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<Packed>;
    #[cfg(feature = "text")]
    fn pack_texts(
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<Self::Color>,
    ) -> Result<Packed>;
}

impl Packing for OneChannel {
    fn pack_sprites(
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
        alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<Packed> {
        Graphics1::setup_sprites(locations, sprites, alpha, alphas)
    }
    #[cfg(feature = "text")]
    fn pack_texts(
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<f32>,
    ) -> Result<Packed> {
        let (locations, blocks, alphas) = Graphics1::setup_texts(locations, texts, style)?;
        let views: Vec<ArrayView2<'_, f32>> = blocks.iter().map(|b| b.view()).collect();
        Graphics1::setup_composite(&locations, &views, None, Some(alphas))
    }
}

impl Packing for FourChannel {
    fn pack_sprites(
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        _alpha: Option<BlockAlpha>,
        _alphas: Option<Vec<Option<BlockAlpha>>>,
    ) -> Result<Packed> {
        // KernelDraw4.sprite queues one-channel blocks; the RGBA kernel needs (h, w, 4).
        let rgba: Vec<Array3<f32>> = sprites
            .iter()
            .map(|s| {
                let (h, w) = s.dim();
                Array3::from_shape_fn((h, w, 4), |(r, c, k)| if k == 3 { 1.0 } else { s[[r, c]] })
            })
            .collect();
        let views: Vec<ArrayView3<'_, f32>> = rgba.iter().map(|b| b.view()).collect();
        Graphics4::setup_sprites(locations, &views)
    }
    #[cfg(feature = "text")]
    fn pack_texts(
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<[f32; 4]>,
    ) -> Result<Packed> {
        let (locations, blocks) = Graphics4::setup_texts(locations, texts, style)?;
        let views: Vec<ArrayView3<'_, f32>> = blocks.iter().map(|b| b.view()).collect();
        Graphics4::setup_composite(&locations, &views)
    }
}

/// Colours/widths already coerced by `KernelDraw::prepare`.
pub struct PreparedStyle {
    pub colors: Vec<f32>,
    pub widths: Vec<f32>,
    pub aa: Vec<f32>,
    pub n: usize,
}

impl<K: Channels> Engine<K> {
    /// `shapes` with pre-coerced colour/width vectors (the KernelDraw path).
    pub fn shapes_prepared(
        &self,
        shape: &str,
        target: Target<'_, '_>,
        coords: &[Vec<f32>],
        style: &PreparedStyle,
        global: Option<usize>,
        async_exec: bool,
    ) -> Result<()> {
        let width = match shape {
            "lines" | "rects" | "ellipses" => 4,
            "circles" => 3,
            "triangles" | "mesh_triangles" => 6,
            other => return Err(Error::Invalid(format!("unknown shape {other}"))),
        };
        let n = coords.len();
        let flat = self.coords(coords, width, shape)?;
        let kernel = if matches!(target, Target::Resident(_)) {
            format!("{shape}_res")
        } else {
            shape.to_string()
        };
        let cw = style.colors.len() / n.max(1);
        if shape == "lines" {
            self.run(
                &kernel,
                target,
                &[
                    (&flat, vec![n, width]),
                    (&style.widths, vec![n]),
                    (&style.aa, vec![n]),
                    (&style.colors, vec![n, cw]),
                ],
                &[],
                global,
                async_exec,
            )
        } else {
            self.run(
                &kernel,
                target,
                &[
                    (&flat, vec![n, width]),
                    (&style.aa, vec![n]),
                    (&style.colors, vec![n, cw]),
                ],
                &[],
                global,
                async_exec,
            )
        }
    }

    /// `composite_packed` with an explicit global size.
    pub fn composite_packed_global(
        &self,
        target: Target<'_, '_>,
        packed: &Packed,
        global: Option<usize>,
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
            global,
            async_exec,
        )
    }
}

/// Queued shapes of a `KernelDraw3` list.
type Kd3Items = Vec<(Vec<f32>, Item<[f32; 3]>)>;
/// Queued text of a `KernelDraw3`.
type Kd3Text = ((f64, f64), String, TextItem<f32>);

/// `KernelDraw3`: an `(h, w, 3)` canvas drawn as three one-channel canvases.
pub struct KernelDraw3 {
    pub planes: [KernelDraw1; 3],
    shape: (usize, usize),
    pub chunk_size: Option<usize>,
    pub alpha: f32,
    pub color: Option<[f32; 3]>,
    lists: std::collections::HashMap<&'static str, Kd3Items>,
    texts: Vec<Kd3Text>,
}

impl KernelDraw3 {
    pub fn new(
        device: &Device,
        canvas: &ArrayViewD<'_, f32>,
        chunk_size: Option<usize>,
    ) -> Result<KernelDraw3> {
        if canvas.ndim() != 3 || canvas.shape()[2] != 3 {
            return Err(Error::Invalid(format!(
                "KernelDraw3 needs an (h, w, 3) canvas, got {:?}",
                canvas.shape()
            )));
        }
        let (h, w) = (canvas.shape()[0], canvas.shape()[1]);
        let plane = |c: usize| -> Result<KernelDraw1> {
            let p = canvas.index_axis(Axis(2), c).to_owned().into_dyn();
            KernelDraw1::new(device, &p.view(), chunk_size)
        };
        Ok(KernelDraw3 {
            planes: [plane(0)?, plane(1)?, plane(2)?],
            shape: (h, w),
            chunk_size,
            alpha: 1.0,
            color: None,
            lists: std::collections::HashMap::new(),
            texts: Vec::new(),
        })
    }

    /// `set_color`: the default colour (Python's `set_color` never changes alpha).
    pub fn set_color(&mut self, color: [f32; 3]) {
        self.color = Some(color);
    }

    fn colors(
        &self,
        color: Option<&[f32; 3]>,
        colors: Option<&[[f32; 3]]>,
        n: usize,
    ) -> Result<Vec<[f32; 3]>> {
        match (color, colors) {
            (Some(c), _) => Ok(vec![*c; n]),
            (None, Some(cs)) => {
                if cs.len() != n {
                    return Err(Error::Invalid(format!("expected {n} colours")));
                }
                Ok(cs.to_vec())
            }
            (None, None) => match self.color {
                Some(c) => Ok(vec![c; n]),
                None => Err(Error::Invalid(
                    "KernelDraw3 needs a colour (Python would draw NaN)".into(),
                )),
            },
        }
    }

    /// `render_immediate`: draw on every plane with that plane's colour component.
    fn immediate(&self, shape: &str, coords: &[Vec<f32>], kw: &Draw<[f32; 3]>) -> Result<()> {
        if coords.is_empty() {
            return Ok(());
        }
        let n = coords.len();
        let colors = self.colors(kw.color.as_ref(), kw.colors.as_deref(), n)?;
        let style: Style<f32> = Style {
            width: kw.width,
            widths: kw.widths.clone(),
            distance: kw.distance,
            ..Default::default()
        };
        let widths = style.widths_for(n)?;
        for (d, plane) in self.planes.iter().enumerate() {
            let draw = Draw::<f32> {
                colors: Some(colors.iter().map(|c| c[d]).collect()),
                widths: Some(widths.clone()),
                ..Default::default()
            };
            match shape {
                "lines" => plane.lines(coords, &draw)?,
                "rects" => plane.rects(coords, &draw)?,
                "circles" => plane.circles(coords, &draw)?,
                "triangles" => plane.triangles(coords, &draw)?,
                "mesh_triangles" => plane.mesh_triangles(coords, &draw)?,
                "ellipses" => plane.ellipses(coords, &draw)?,
                other => return Err(Error::Invalid(format!("unknown shape {other}"))),
            }
        }
        Ok(())
    }

    /// `lines(coords, distance=1.0, ...)`: KernelDraw3 lines take `distance`, not `width`.
    pub fn lines(&self, coords: &[Vec<f32>], kw: &Draw<[f32; 3]>) -> Result<()> {
        let mut kw = kw.clone();
        if kw.width.is_some() {
            return Err(Error::Invalid("Only one of width/distance".into()));
        }
        if kw.distance.is_none() {
            kw.distance = Some(1.0);
        }
        self.immediate("lines", coords, &kw)
    }
    pub fn rects(&self, coords: &[Vec<f32>], kw: &Draw<[f32; 3]>) -> Result<()> {
        self.immediate("rects", coords, kw)
    }
    pub fn circles(&self, coords: &[Vec<f32>], kw: &Draw<[f32; 3]>) -> Result<()> {
        self.immediate("circles", coords, kw)
    }
    pub fn triangles(&self, coords: &[Vec<f32>], kw: &Draw<[f32; 3]>) -> Result<()> {
        self.immediate("triangles", coords, kw)
    }
    pub fn mesh_triangles(&self, coords: &[Vec<f32>], kw: &Draw<[f32; 3]>) -> Result<()> {
        self.immediate("mesh_triangles", coords, kw)
    }
    pub fn ellipses(&self, coords: &[Vec<f32>], kw: &Draw<[f32; 3]>) -> Result<()> {
        self.immediate("ellipses", coords, kw)
    }

    /// `sprites`: the same one-channel sprites on every plane.
    pub fn sprites(
        &self,
        locations: &[(usize, f64, f64)],
        sprites: &[ArrayView2<'_, f32>],
        alpha: Option<BlockAlpha>,
    ) -> Result<()> {
        if locations.is_empty() {
            return Ok(());
        }
        for plane in &self.planes {
            plane.sprites(locations, sprites, alpha.clone(), None)?;
        }
        Ok(())
    }

    /// `draw_strings`: the same text on every plane (colour is ignored, as in Python).
    #[cfg(feature = "text")]
    pub fn draw_strings(
        &self,
        locations: &[(f64, f64)],
        texts: &[&str],
        style: &TextStyle<f32>,
    ) -> Result<()> {
        if locations.is_empty() {
            return Ok(());
        }
        for plane in &self.planes {
            plane.draw_strings(locations, texts, style)?;
        }
        Ok(())
    }

    fn push(&mut self, list: &'static str, coords: Vec<f32>, kw: Item<[f32; 3]>) {
        self.lists.entry(list).or_default().push((coords, kw));
    }
    pub fn line(&mut self, xyxy: [f32; 4], kw: Item<[f32; 3]>) {
        self.push("lines", xyxy.to_vec(), kw)
    }
    pub fn rect(&mut self, rect: [f32; 4], kw: Item<[f32; 3]>) {
        self.push("rects", rect.to_vec(), kw)
    }
    pub fn circle(&mut self, xyr: [f32; 3], kw: Item<[f32; 3]>) {
        self.push("circles", xyr.to_vec(), kw)
    }
    pub fn triangle(&mut self, t: [f32; 6], kw: Item<[f32; 3]>) {
        self.push("triangles", t.to_vec(), kw)
    }
    pub fn mesh_triangle(&mut self, t: [f32; 6], kw: Item<[f32; 3]>) {
        self.push("mesh_triangles", t.to_vec(), kw)
    }
    pub fn ellipse(&mut self, e: [f32; 4], kw: Item<[f32; 3]>) {
        self.push("ellipses", e.to_vec(), kw)
    }
    pub fn draw_string(&mut self, xy: (f64, f64), text: &str, kw: TextItem<f32>) {
        self.texts.push((xy, text.to_string(), kw));
    }

    /// `render`: KernelDraw3's own lists (through `render_immediate`, which
    /// honours per-item `width` but, as in Python, not `distance` or `alpha`),
    /// then each plane's lists.
    pub fn render(&mut self) -> Result<()> {
        for name in ORDER {
            if name == "draw_strings" {
                let texts = std::mem::take(&mut self.texts);
                if texts.is_empty() {
                    continue;
                }
                #[cfg(not(feature = "text"))]
                {
                    let _ = texts;
                    return Err(Error::Invalid("text needs the `text` feature".into()));
                }
                #[cfg(feature = "text")]
                {
                    let chunks: Vec<&[Kd3Text]> = match self.chunk_size {
                        Some(c) if c > 0 => texts.chunks(c).collect(),
                        _ => vec![&texts[..]],
                    };
                    for chunk in chunks {
                        let locations: Vec<(f64, f64)> = chunk.iter().map(|t| t.0).collect();
                        let strs: Vec<&str> = chunk.iter().map(|t| t.1.as_str()).collect();
                        let style = TextStyle::<f32> {
                            fonts: Some(chunk.iter().map(|t| t.2.font.clone()).collect()),
                            sizes: Some(chunk.iter().map(|t| t.2.size).collect()),
                            alphas: Some(chunk.iter().map(|t| t.2.alpha.clone()).collect()),
                            colors: Some(chunk.iter().map(|t| t.2.color).collect()),
                            rotations: Some(chunk.iter().map(|t| t.2.rotation).collect()),
                            ..Default::default()
                        };
                        self.draw_strings(&locations, &strs, &style)?;
                    }
                    continue;
                }
            }
            let Some(list) = self.lists.remove(name) else {
                continue;
            };
            if list.is_empty() || name == "sprites" {
                continue;
            }
            #[allow(clippy::type_complexity)]
            let chunks: Vec<&[(Vec<f32>, Item<[f32; 3]>)]> = match self.chunk_size {
                Some(c) if c > 0 => list.chunks(c).collect(),
                _ => vec![&list[..]],
            };
            for chunk in chunks {
                let coords: Vec<Vec<f32>> = chunk.iter().map(|(c, _)| c.clone()).collect();
                let colors: Vec<[f32; 3]> = chunk.iter().filter_map(|(_, kw)| kw.color).collect();
                let widths: Vec<f32> = chunk.iter().filter_map(|(_, kw)| kw.width).collect();
                let n = coords.len();
                let kw = Draw::<[f32; 3]> {
                    colors: if colors.is_empty() {
                        None
                    } else if colors.len() == n {
                        Some(colors)
                    } else {
                        return Err(Error::Invalid("color: all or none".into()));
                    },
                    widths: if widths.is_empty() {
                        None
                    } else if widths.len() == n {
                        Some(widths)
                    } else {
                        return Err(Error::Invalid("width: all or none".into()));
                    },
                    distance: if name == "lines" { Some(1.0) } else { None },
                    ..Default::default()
                };
                self.immediate(name, &coords, &kw)?;
            }
        }
        for plane in &mut self.planes {
            plane.render()?;
        }
        Ok(())
    }

    /// `read`: the `(h, w, 3)` canvas.
    pub fn read(&mut self) -> Result<Array3<f32>> {
        self.render()?;
        let (h, w) = self.shape;
        let mut out = Array3::<f32>::zeros((h, w, 3));
        for (c, plane) in self.planes.iter_mut().enumerate() {
            let p = plane.read()?;
            let p2 = p.into_dimensionality::<ndarray::Ix2>()?;
            out.index_axis_mut(Axis(2), c).assign(&p2);
        }
        Ok(out)
    }
}
