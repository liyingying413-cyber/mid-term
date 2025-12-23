# app.py
# Streamlit Interactive Generative Poster (Shapes + CSV Palette + Sliders + High-Res Export)
# Run locally: streamlit run app.py
# Deploy: push to GitHub + set Streamlit Cloud entrypoint to app.py

import io
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# -------------------------
# Utilities
# -------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def rgb01_to_rgba255(rgb01: Tuple[float, float, float], a01: float) -> Tuple[int, int, int, int]:
    r = int(clamp(rgb01[0]) * 255)
    g = int(clamp(rgb01[1]) * 255)
    b = int(clamp(rgb01[2]) * 255)
    a = int(clamp(a01) * 255)
    return (r, g, b, a)


def rotate_hue_rgb01(rgb01: Tuple[float, float, float], hue_shift: float) -> Tuple[float, float, float]:
    # hue_shift in [-0.5, 0.5] range (wrap)
    import colorsys
    r, g, b = rgb01
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + hue_shift) % 1.0
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
    return (rr, gg, bb)


def try_load_font(size: int) -> ImageFont.ImageFont:
    # Prefer DejaVu if available (common on Linux/Streamlit Cloud)
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


# -------------------------
# Palettes
# -------------------------
BUILTIN_PALETTES = {
    "dreamy": [
        (0.83, 0.73, 0.96),
        (0.76, 0.86, 0.95),
        (0.95, 0.78, 0.90),
        (0.78, 0.80, 0.95),
        (0.86, 0.92, 0.98),
        (0.85, 0.78, 0.92),
    ],
    "pastel": [
        (0.98, 0.82, 0.83),
        (0.86, 0.92, 0.84),
        (0.83, 0.88, 0.97),
        (0.99, 0.94, 0.78),
        (0.92, 0.86, 0.98),
        (0.82, 0.96, 0.95),
    ],
    "vivid": [
        (0.95, 0.25, 0.30),
        (0.98, 0.70, 0.10),
        (0.20, 0.80, 0.45),
        (0.20, 0.50, 0.95),
        (0.65, 0.25, 0.95),
        (0.10, 0.90, 0.95),
    ],
    "mono": [
        (0.15, 0.15, 0.18),
        (0.25, 0.25, 0.30),
        (0.40, 0.40, 0.48),
        (0.60, 0.60, 0.70),
        (0.78, 0.78, 0.86),
        (0.90, 0.90, 0.96),
    ],
    "neon": [
        (0.10, 0.95, 0.85),
        (0.95, 0.10, 0.75),
        (0.95, 0.90, 0.10),
        (0.20, 0.80, 0.10),
        (0.55, 0.10, 0.95),
        (0.10, 0.55, 0.95),
    ],
    "ocean": [
        (0.07, 0.20, 0.45),
        (0.12, 0.38, 0.70),
        (0.12, 0.62, 0.75),
        (0.68, 0.90, 0.95),
        (0.02, 0.50, 0.55),
        (0.55, 0.80, 0.90),
    ],
    "sunset": [
        (0.95, 0.40, 0.30),
        (0.98, 0.62, 0.22),
        (0.92, 0.25, 0.55),
        (0.55, 0.20, 0.65),
        (0.20, 0.25, 0.55),
        (0.98, 0.82, 0.55),
    ],
}


def parse_palette_csv(uploaded_file) -> Optional[List[Tuple[float, float, float]]]:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return None

    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # Accept: r,g,b in [0..1] OR [0..255]
    if not all(c in df.columns for c in ["r", "g", "b"]):
        return None

    rgb = df[["r", "g", "b"]].astype(float).values
    # Heuristic: if any value > 1.5, treat as 0..255
    if np.nanmax(rgb) > 1.5:
        rgb = np.clip(rgb / 255.0, 0, 1)

    palette = []
    for r, g, b in rgb:
        palette.append((float(clamp(r)), float(clamp(g)), float(clamp(b))))
    palette = [p for p in palette if not any(math.isnan(x) for x in p)]
    if len(palette) < 2:
        return None
    return palette


# -------------------------
# Shape geometry (normalized points around origin)
# -------------------------
def regular_polygon_points(sides: int, rotation: float = 0.0) -> List[Tuple[float, float]]:
    sides = max(3, int(sides))
    pts = []
    for i in range(sides):
        a = rotation + 2 * math.pi * i / sides
        pts.append((math.cos(a), math.sin(a)))
    return pts


def star_points(points: int = 5, inner: float = 0.5, rotation: float = -math.pi / 2) -> List[Tuple[float, float]]:
    points = max(3, int(points))
    pts = []
    for i in range(points * 2):
        r = 1.0 if i % 2 == 0 else clamp(inner, 0.1, 0.95)
        a = rotation + math.pi * i / points
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def heart_points(resolution: int = 200) -> List[Tuple[float, float]]:
    # Classic parametric heart, normalized to roughly fit in [-1,1]
    pts = []
    for i in range(resolution):
        t = 2 * math.pi * i / resolution
        x = 16 * (math.sin(t) ** 3)
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        # normalize
        x /= 18.0
        y /= 18.0
        pts.append((x, -y))  # flip y to be upright in image coords later
    return pts


def flower_points(petals: int = 6, amplitude: float = 0.35, resolution: int = 240, rotation: float = 0.0) -> List[Tuple[float, float]]:
    # Polar rose-ish: r = 1 - a*cos(k*theta)
    petals = max(3, int(petals))
    a = clamp(amplitude, 0.05, 0.8)
    pts = []
    for i in range(resolution):
        t = 2 * math.pi * i / resolution
        r = 1.0 - a * math.cos(petals * t)
        x = r * math.cos(t + rotation)
        y = r * math.sin(t + rotation)
        pts.append((x, y))
    return pts


def cloud_points(lobes: int = 5, resolution: int = 220, rotation: float = 0.0) -> List[Tuple[float, float]]:
    """
    A cute cloud silhouette: sum of a few circular bumps + flat-ish bottom.
    We approximate by sampling an implicit curve in polar form with lobe modulation.
    """
    lobes = max(3, int(lobes))
    pts = []
    for i in range(resolution):
        t = 2 * math.pi * i / resolution
        # Make top bumpier than bottom
        top_weight = 0.6 + 0.4 * math.sin(t)
        bump = 0.18 * top_weight * math.cos(lobes * t)
        squash = 0.18 * (1 - math.cos(t))  # flatten bottom
        r = 1.0 + bump - squash
        x = r * math.cos(t + rotation)
        y = r * math.sin(t + rotation)
        pts.append((x, y))
    return pts


def jitter_points(pts: List[Tuple[float, float]], wobble: float, rng: random.Random) -> List[Tuple[float, float]]:
    if wobble <= 0:
        return pts
    out = []
    for x, y in pts:
        out.append((x + rng.uniform(-wobble, wobble), y + rng.uniform(-wobble, wobble)))
    return out


def scale_translate_points(
    pts: List[Tuple[float, float]],
    cx: float,
    cy: float,
    radius: float,
    aspect_y: float = 1.0,
) -> List[Tuple[float, float]]:
    # radius in pixels
    out = []
    for x, y in pts:
        px = cx + x * radius
        py = cy + y * radius * aspect_y
        out.append((px, py))
    return out


# -------------------------
# Rendering
# -------------------------
@dataclass
class PosterParams:
    seed: int
    layers: int
    shape: str
    sides: int

    min_radius: float
    max_radius: float

    wobble: float
    spread: float  # diffusion / dispersion

    min_alpha: float
    max_alpha: float

    stroke: bool
    stroke_width: int
    stroke_alpha: float

    shadow: bool
    shadow_offset: int
    shadow_blur: int
    shadow_alpha: float

    background: str  # "Light", "Dark", "Transparent"
    base_hue: float  # [-0.5,0.5]
    palette_mode: str  # builtin keys or "csv"
    palette: List[Tuple[float, float, float]]

    title: str
    subtitle: str
    title_size: int
    subtitle_size: int
    text_x: float
    text_y: float
    text_color: str  # "Auto", "Black", "White"
    text_shadow: bool


def get_background_rgba(bg: str) -> Tuple[int, int, int, int]:
    if bg.lower().startswith("dark"):
        return (18, 18, 22, 255)
    if bg.lower().startswith("trans"):
        return (0, 0, 0, 0)
    return (245, 245, 248, 255)


def pick_text_color(bg: str, choice: str) -> Tuple[int, int, int, int]:
    if choice == "Black":
        return (10, 10, 12, 255)
    if choice == "White":
        return (250, 250, 252, 255)
    # Auto
    if bg.lower().startswith("dark"):
        return (245, 245, 248, 255)
    return (10, 10, 12, 255)


def get_shape_points(params: PosterParams, rng: random.Random) -> Tuple[str, Optional[List[Tuple[float, float]]]]:
    s = params.shape
    if s == "circle":
        return ("circle", None)

    if s == "polygon":
        rot = rng.uniform(0, 2 * math.pi)
        pts = regular_polygon_points(params.sides, rotation=rot)
        return ("poly", pts)

    if s == "star":
        rot = rng.uniform(0, 2 * math.pi)
        inner = rng.uniform(0.35, 0.65)
        pts = star_points(points=max(5, params.sides), inner=inner, rotation=rot)
        return ("poly", pts)

    if s == "heart":
        pts = heart_points(resolution=220)
        # give a small random rotation
        rot = rng.uniform(-0.35, 0.35)
        pts = [(x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)) for x, y in pts]
        return ("poly", pts)

    if s == "flower":
        petals = max(4, params.sides)
        rot = rng.uniform(0, 2 * math.pi)
        pts = flower_points(petals=petals, amplitude=rng.uniform(0.20, 0.45), resolution=260, rotation=rot)
        return ("poly", pts)

    if s == "cloud":
        lobes = max(4, min(10, params.sides))
        rot = rng.uniform(0, 2 * math.pi)
        pts = cloud_points(lobes=lobes, resolution=240, rotation=rot)
        return ("poly", pts)

    # fallback
    rot = rng.uniform(0, 2 * math.pi)
    pts = regular_polygon_points(max(3, params.sides), rotation=rot)
    return ("poly", pts)


def draw_layer_shape(
    base: Image.Image,
    params: PosterParams,
    rng: random.Random,
    i: int,
    w: int,
    h: int,
):
    # depth: 0 = back, 1 = front
    depth = 0.0 if params.layers <= 1 else i / (params.layers - 1)

    # Position diffusion: more spread for front layers looks "closer"
    # (you can flip it if you want)
    spread_px = params.spread * min(w, h)
    cx = w * 0.5 + rng.uniform(-spread_px, spread_px) * (0.35 + 0.65 * depth)
    cy = h * 0.5 + rng.uniform(-spread_px, spread_px) * (0.35 + 0.65 * depth)

    # Radius: back smaller, front larger (more depth feeling)
    r_min = params.min_radius * min(w, h)
    r_max = params.max_radius * min(w, h)
    radius = lerp(r_min, r_max, depth) * rng.uniform(0.85, 1.10)

    # Alpha: back more transparent, front more opaque
    a = lerp(params.min_alpha, params.max_alpha, depth) * rng.uniform(0.90, 1.05)
    a = clamp(a, 0.0, 1.0)

    # Color selection
    c = params.palette[rng.randrange(0, len(params.palette))]
    c = rotate_hue_rgb01(c, params.base_hue)
    fill = rgb01_to_rgba255(c, a)

    kind, pts = get_shape_points(params, rng)

    # Optional wobble on vertices
    wob = params.wobble
    # Map wobble slider ~0..1 to small vertex jitter
    wobble_amount = wob * 0.20

    # Shadow first (on separate layer then blur)
    if params.shadow:
        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow_layer, "RGBA")
        sh_alpha = clamp(params.shadow_alpha, 0.0, 1.0)
        shadow_fill = (0, 0, 0, int(sh_alpha * 255))

        ox = params.shadow_offset
        oy = params.shadow_offset

        if kind == "circle":
            bbox = [cx - radius + ox, cy - radius + oy, cx + radius + ox, cy + radius + oy]
            sd.ellipse(bbox, fill=shadow_fill)
        else:
            p2 = jitter_points(pts, wobble_amount, rng)
            poly = scale_translate_points(p2, cx + ox, cy + oy, radius, aspect_y=1.0)
            sd.polygon(poly, fill=shadow_fill)

        if params.shadow_blur > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=params.shadow_blur))

        base.alpha_composite(shadow_layer)

    # Main shape
    d = ImageDraw.Draw(base, "RGBA")

    if kind == "circle":
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        d.ellipse(bbox, fill=fill)
        if params.stroke and params.stroke_width > 0:
            sa = clamp(params.stroke_alpha, 0.0, 1.0)
            outline = (0, 0, 0, int(sa * 255))
            d.ellipse(bbox, outline=outline, width=params.stroke_width)
    else:
        p2 = jitter_points(pts, wobble_amount, rng)
        poly = scale_translate_points(p2, cx, cy, radius, aspect_y=1.0)
        d.polygon(poly, fill=fill)
        if params.stroke and params.stroke_width > 0:
            sa = clamp(params.stroke_alpha, 0.0, 1.0)
            outline = (0, 0, 0, int(sa * 255))
            # PIL polygon outline can be thin; draw as line loop for better control
            d.line(poly + [poly[0]], fill=outline, width=params.stroke_width, joint="curve")


def render_poster(params: PosterParams, size: int) -> Image.Image:
    w = h = int(size)
    bg = get_background_rgba(params.background)
    img = Image.new("RGBA", (w, h), bg)

    rng = random.Random(params.seed)

    # Draw from back -> front
    for i in range(params.layers):
        draw_layer_shape(img, params, rng, i, w, h)

    # Text
    if (params.title or params.subtitle) and params.background.lower() != "transparent":
        tx = int(params.text_x * w)
        ty = int(params.text_y * h)

        title_font = try_load_font(params.title_size)
        subtitle_font = try_load_font(params.subtitle_size)
        color = pick_text_color(params.background, params.text_color)

        td = ImageDraw.Draw(img, "RGBA")

        def draw_text_with_shadow(x, y, text, font, fill):
            if not text:
                return 0
            if params.text_shadow:
                sd = (0, 0, 0, 130) if fill[0] > 200 else (255, 255, 255, 120)
                td.text((x + 2, y + 2), text, font=font, fill=sd)
            td.text((x, y), text, font=font, fill=fill)
            bbox = td.textbbox((x, y), text, font=font)
            return bbox[3] - bbox[1]

        y = ty
        y += draw_text_with_shadow(tx, y, params.title, title_font, color) + 4
        draw_text_with_shadow(tx, y, params.subtitle, subtitle_font, color)

    return img


def png_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Generative Poster", layout="wide")

st.title("Interactive Poster Generator")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Palette CSV")
    uploaded = st.file_uploader("Drag and drop file here", type=["csv"])
    reset_palette = st.button("Reset built-in palette")

    st.markdown("---")
    st.subheader("Poster Controls")

    palette_mode = st.selectbox(
        "Palette Mode",
        ["dreamy", "pastel", "vivid", "mono", "neon", "ocean", "sunset", "csv"],
        index=0,
    )

    shape = st.selectbox(
        "Shape",
        ["polygon", "circle", "flower", "heart", "star", "cloud"],
        index=0,
    )

    layers = st.slider("Layers", 1, 80, 39, 1)
    wobble = st.slider("Wobble", 0.0, 1.0, 0.34, 0.01)

    # "Sides" meaning depends on shape:
    # polygon: sides; star: points; flower: petals; cloud: lobes
    if shape in ["polygon", "star", "flower", "cloud"]:
        sides_label = {"polygon": "Sides", "star": "Star Points", "flower": "Petals", "cloud": "Cloud Lobes"}[shape]
        sides = st.slider(sides_label, 3, 12, 9, 1)
    else:
        sides = 9

    st.markdown("### Rendering")
    min_radius = st.slider("Min Radius", 0.02, 0.40, 0.10, 0.01)
    max_radius = st.slider("Max Radius", 0.05, 0.70, 0.25, 0.01)
    if max_radius < min_radius:
        max_radius = min_radius

    spread = st.slider("Spread (Diffusion)", 0.00, 0.60, 0.28, 0.01)

    min_alpha = st.slider("Min Alpha", 0.05, 1.00, 0.74, 0.01)
    max_alpha = st.slider("Max Alpha", 0.05, 1.00, 0.85, 0.01)
    if max_alpha < min_alpha:
        max_alpha = min_alpha

    seed = st.number_input("Seed", min_value=0, max_value=999999, value=995, step=1)

    base_hue = st.slider("Base Hue (shift)", -0.50, 0.50, 0.10, 0.01)

    background = st.selectbox("Background", ["Light", "Dark", "Transparent"], index=0)

    st.markdown("### Shadow & Stroke")
    shadow = st.checkbox("Shadow", value=True)
    shadow_offset = st.slider("Shadow Offset", 0, 60, 18, 1)
    shadow_blur = st.slider("Shadow Blur", 0, 50, 16, 1)
    shadow_alpha = st.slider("Shadow Alpha", 0.0, 1.0, 0.18, 0.01)

    stroke = st.checkbox("Stroke (Outline)", value=False)
    stroke_width = st.slider("Stroke Width", 1, 12, 2, 1)
    stroke_alpha = st.slider("Stroke Alpha", 0.0, 1.0, 0.18, 0.01)

    st.markdown("### Text")
    title = st.text_input("Title", value="DREAMY GEOMETRY")
    subtitle = st.text_input("Subtitle", value="Interactive 3D Composition")
    title_size = st.slider("Title Size", 14, 64, 24, 1)
    subtitle_size = st.slider("Subtitle Size", 10, 40, 14, 1)
    text_x = st.slider("Text X", 0.00, 1.00, 0.08, 0.01)
    text_y = st.slider("Text Y", 0.00, 1.00, 0.08, 0.01)
    text_color = st.selectbox("Text Color", ["Auto", "Black", "White"], index=0)
    text_shadow = st.checkbox("Text Shadow", value=True)

# Palette selection logic
csv_palette = None
if reset_palette:
    uploaded = None

if palette_mode == "csv":
    if uploaded is not None:
        csv_palette = parse_palette_csv(uploaded)
    if not csv_palette:
        st.warning("CSV palette not loaded. Using 'dreamy' temporarily.")
        palette = BUILTIN_PALETTES["dreamy"]
    else:
        palette = csv_palette
else:
    palette = BUILTIN_PALETTES.get(palette_mode, BUILTIN_PALETTES["dreamy"])

params = PosterParams(
    seed=int(seed),
    layers=int(layers),
    shape=shape,
    sides=int(sides),
    min_radius=float(min_radius),
    max_radius=float(max_radius),
    wobble=float(wobble),
    spread=float(spread),
    min_alpha=float(min_alpha),
    max_alpha=float(max_alpha),
    stroke=bool(stroke),
    stroke_width=int(stroke_width),
    stroke_alpha=float(stroke_alpha),
    shadow=bool(shadow),
    shadow_offset=int(shadow_offset),
    shadow_blur=int(shadow_blur),
    shadow_alpha=float(shadow_alpha),
    background=background,
    base_hue=float(base_hue),
    palette_mode=palette_mode,
    palette=palette,
    title=title,
    subtitle=subtitle,
    title_size=int(title_size),
    subtitle_size=int(subtitle_size),
    text_x=float(text_x),
    text_y=float(text_y),
    text_color=text_color,
    text_shadow=bool(text_shadow),
)

with right:
    # Preview size
    preview_size = 900
    img = render_poster(params, preview_size)

    st.image(img, caption=None, use_container_width=False)

    # High-res export
    st.markdown("")
    export_col1, export_col2 = st.columns([1, 1])
    with export_col1:
        export_size = st.selectbox("Export Size (square)", [1500, 2000, 2500, 3000, 3600], index=3)
    with export_col2:
        export_dpi = st.selectbox("DPI", [150, 200, 300], index=2)

    hi = render_poster(params, int(export_size))
    st.download_button(
        label=f"Download High-Res PNG ({export_dpi} DPI)",
        data=png_bytes(hi, dpi=int(export_dpi)),
        file_name="poster.png",
        mime="image/png",
        use_container_width=True,
    )

# Footer tips (small)
st.caption(
    "Tip: For 'flower/star/cloud', the 'Sides' slider becomes Petals/Points/Lobes. "
    "Upload a CSV with columns r,g,b (0–1 or 0–255) to use your own palette."
)
