//! The fonts shipped with yapocis (`yapocis.fonts`): the Ubuntu family, under
//! the Ubuntu Font Licence (`fonts/UFL.txt`), embedded in the crate.

/// `(file name, bytes)` for every bundled font.
///
/// The order matters: `yapocis.fonts.font` resolves a partial name to the
/// first match in `os.listdir` order, which on the maintainer's machine (where
/// the parity fixtures were generated) is the order below, so `"italic"` is
/// Ubuntu-LightItalic and `"light"` is Ubuntu-LightItalic too. Pass the full
/// file name to pick a font unambiguously.
pub const FONTS: &[(&str, &[u8])] = &[
    (
        "Ubuntu-Medium.ttf",
        include_bytes!("../fonts/Ubuntu-Medium.ttf"),
    ),
    (
        "Ubuntu-LightItalic.ttf",
        include_bytes!("../fonts/Ubuntu-LightItalic.ttf"),
    ),
    (
        "Ubuntu-Regular.ttf",
        include_bytes!("../fonts/Ubuntu-Regular.ttf"),
    ),
    (
        "Ubuntu-Bold.ttf",
        include_bytes!("../fonts/Ubuntu-Bold.ttf"),
    ),
    (
        "Ubuntu-MediumItalic.ttf",
        include_bytes!("../fonts/Ubuntu-MediumItalic.ttf"),
    ),
    (
        "Ubuntu-BoldItalic.ttf",
        include_bytes!("../fonts/Ubuntu-BoldItalic.ttf"),
    ),
    (
        "Ubuntu-Italic.ttf",
        include_bytes!("../fonts/Ubuntu-Italic.ttf"),
    ),
    (
        "Ubuntu-Light.ttf",
        include_bytes!("../fonts/Ubuntu-Light.ttf"),
    ),
];

/// The font `yapocis.fonts.font()` falls back to.
pub const DEFAULT_FONT: &str = "Ubuntu-Regular.ttf";

/// The Ubuntu Font Licence text.
pub const LICENSE: &str = include_str!("../fonts/UFL.txt");

/// Names of the bundled fonts.
/// Font file names in lookup order: sorted by name, matching the Python
/// `AssetManager` (which sorts its directory listing so partial-name lookups
/// such as `"italic"` resolve the same on every machine).
pub fn fonts() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = FONTS.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    names
}

fn by_name(name: &str) -> Option<(&'static str, &'static [u8])> {
    FONTS.iter().find(|(n, _)| *n == name).copied()
}

/// Resolve a font suggestion the way `yapocis.fonts.font` does: the exact file
/// name, else the first name containing the suggestion (case-insensitive),
/// else `Ubuntu-Regular.ttf` (so `"default"` is Regular). Returns the resolved
/// name and the font bytes.
pub fn font(suggestion: &str) -> (&'static str, &'static [u8]) {
    if let Some(f) = FONTS.iter().find(|(n, _)| *n == suggestion) {
        return *f;
    }
    let lower = suggestion.to_ascii_lowercase();
    if let Some(f) = fonts()
        .into_iter()
        .find(|n| n.to_ascii_lowercase().contains(&lower))
        .and_then(by_name)
    {
        return f;
    }
    *FONTS.iter().find(|(n, _)| *n == DEFAULT_FONT).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup() {
        assert_eq!(font("default").0, "Ubuntu-Regular.ttf");
        assert_eq!(font("Ubuntu-Bold.ttf").0, "Ubuntu-Bold.ttf");
        assert_eq!(font("bold").0, "Ubuntu-Bold.ttf");
        assert_eq!(font("LIGHTITALIC").0, "Ubuntu-LightItalic.ttf");
        assert_eq!(font("italic").0, "Ubuntu-BoldItalic.ttf"); // first match in sorted order
        assert_eq!(font("Ubuntu-Italic.ttf").0, "Ubuntu-Italic.ttf");
        assert_eq!(font("nope").0, "Ubuntu-Regular.ttf");
        assert!(font("default").1.len() > 100_000);
    }
}
