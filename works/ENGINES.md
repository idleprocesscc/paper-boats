# Engines

The files in this folder are Claude's own scripts: compositions, colours, strokes, the order things are painted in, and the notes about why. None of the drawing engines they import is included here. Each engine belongs to its authors. Get it from them, then point the script at it: every script has one line that says where the engine lives (an environment variable or a relative path, marked `set to where the engine lives` for the watercolor scripts).

## watercolor (Python)

- Used by: `manarola-0929.py`, `moonglade-0929.py`, `dream-nocturne-0929.py`, `carmel-0928.py`, `tsurui-0928.py`, `lowtide-0928.py`, `carmel-2-0928.py`
- A friend's watercolour engine (`watercolor_lib`, plus `animals/animal_lib.py` for `dream-nocturne-0929.py`), shared with us privately. It is not included here and not ours to redistribute; thank you to its author.
- None of its code is copied into these scripts; they only import it. `carmel-2-0928.py` writes out one Beer-Lambert absorbance line, -log(colour / paper), which is textbook optics.
- The scripts are kept as a record of the choices: compositions, colours, strokes, the order things are painted in. `lowtide-0928.py` also needs `scipy`.

## p5.brush (JavaScript)

- Used by: the `.html` sketches.
- Upstream: <https://github.com/acamposuribe/p5.brush> (homepage <https://p5-brush.cargo.site/>), npm `p5.brush`
- License: MIT, Copyright (c) 2023-2026 Alejandro Campos Uribe
- The sketches load it from the jsDelivr CDN. You can point the `<script src>` at a local copy instead.
- Versions: most sketches use p5.brush 2.2.2. `study-night.html` uses 2.2.3, and `round-ear-dog-by-water.html` uses 1.1.1 (the older 1.x API).
- One-line local patch: most 2.2.2 sketches were rendered with a locally patched p5.brush. In `src/core/primitives.js`, `_createSpline` starts with
  `if (points && points.length === 2) curvature = 0;`
  Without this line, stock 2.2.2 can silently drop every fill after a two-point curved stroke. The patch is a change to MIT-licensed code, and the MIT notice above covers it.

## p5.js

- Upstream: <https://p5js.org/>, npm `p5`
- License: LGPL-2.1
- Loaded from the CDN by the `.html` sketches: 2.2.x for most, 2.3.4 for `study-night.html`, 1.11.3 for `round-ear-dog-by-water.html`.

## penwash and crayon (Python)

- Used by: `penwash-konglong.py` (penwash_lib), `crayon-two-kids.py` (crayon_lib)
- These engines came from a friend's shared "pencil-case" kit of hand-coded drawing tools. The copies we have contain no license file and no upstream URL.
- They are not redistributed here, and neither script copies any of their code; the scripts only import them. If you are one of the kit's authors and want a credit or a link here, it will be added.

## silkscreen (Python)

- Used by: `wanda-silkscreen-0930.py` (with `wanda_shapes.py` next to it) and `wanda-silkscreen-0930-sheet.py`
- A friend's silkscreen skill (`press.py`, `brush.py`, `inks.py`, with its riso ink card and the signature font it bundles). It is not included here. Thank you to its authors.
- None of its code is copied into these scripts; they only import it (`Press`, `Pens`, the `INK` card). Point `SILKSCREEN_ENGINE_DIR` at your copy.
- What the scripts hold is Claude's part: the traced shapes, the six plates, the ink sets and the misregistration settings. Needs `numpy`, `scipy`, `Pillow`. Re-running them reproduces the published print pixel for pixel.

## kbrush (Python) — Claude's own

- `kbrush_v1.py` and `kbrush_v2.py` are a watercolour brush, paper and water model Claude wrote from scratch for nerolette on 2026-09-29. They are included here in full.
- `kbrush-v1-sheet.py` / `kbrush-v2-sheet.py` draw the practice sheets. Needs Python 3, `numpy`, `scipy`, `Pillow`.
- Layers are combined with Kubelka-Munk; K/S values and the tide-line idea follow Curtis et al., *Computer-Generated Watercolor* (SIGGRAPH 1997).

## Music

The songs are Claude's Python scripts that generate MIDI. The released audio was played through GarageBand's built-in instruments and mixed there. No sound fonts or samples are included.

- `for-nerolette.py`, `baihua-ku-kai.py`, `ku-waterfall-2.py` play live into GarageBand over a virtual MIDI port. They need `mido` + `python-rtmidi` (both MIT) and a MIDI-receiving synth (GarageBand, or any other).
- `ku-swan-cello.py`, `doorway-guitar.py`, `lamp-is-low-5.py`, `nocturne-terminator.py` write a `.mid` file with `mido`. Load it into any DAW or synth.
- `lamp-is-low-5.py`: the melody comes from Ravel's *Pavane pour une infante défunte* (1899; Ravel died in 1937, so it is public domain in the US and EU). Nothing is taken from the lyrics or the arrangement of the 1939 popular song "The Lamp Is Low".

## Plain Python / SVG

`pi-day-2026.svg`, `moonlight-2026.svg` and `two-cats.svg` are hand-written SVG with no dependencies.

`two-cats.svg` names the handwriting font "Homemade Apple" (Google Fonts, Apache License 2.0). The font is referenced, not bundled.
