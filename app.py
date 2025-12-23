import io
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Utilities
# ----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def rgb01_to_rgba255(rgb: Tuple[float, float, float], a01: float) -> Tuple[int, int, int, int]:
    r, g, b = rgb
    return (int(clamp01(r) * 255), int(clamp01(g) * 255), int(clamp01(b) * 255), int(clamp01(a01) * 255))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix_rgb(c1: Tuple[float, float, float], c2: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    return (lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))


def darken(rgb: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    return (rgb[0] * (1 - amount), rgb[1] * (1 - amount), rgb[2] * (1 - amount))


def lighten(rgb: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    return (rgb[0] + (1 - rgb[0]) * amount, rgb[1] + (1 - rgb[1]) * amount, rgb[2] + (1 - rgb[2]) * amount)


def get_builtin_palettes() -> dict:
    return {
        "dreamy": [
            (0.86, 0.73, 0.95),
            (0.71, 0.84, 0.96),
            (0.89, 0.78, 0.86),
            (0.75, 0.73, 0.94),
            (0.82, 0.90, 0.95),
        ],
        "pastel": [
            (0.98, 0.80, 0.83),
            (0.80, 0.93, 0.86),
            (0.82, 0.86, 0.98),
            (0.98, 0.95, 0.78),
            (0.93, 0.84, 0.96),
        ],
        "vivid": [
            (0.95, 0.25, 0.40),
            (0.20, 0.70, 0.95),
            (0.15, 0.85, 0.45),
            (0.98, 0.75, 0.20),
            (0.65, 0.30, 0.95),
        ],
        "mono": [
            (0.10, 0.10, 0.12),
            (0.25, 0.25, 0.28),
            (0.45, 0.45, 0.50),
            (0.70, 0.70, 0.74),
            (0.90, 0.90, 0.92),
        ],
        "neon": [
            (0.10, 1.00, 0.75),
            (1.00, 0.15, 0.85),
            (0.20, 0.70, 1.00),
            (1.00, 0.95, 0.20),
            (0.75, 0.20, 1.00),
        ],
        "ocean": [
            (0.08, 0.28, 0.60),
            (0.10, 0.55, 0.75),
            (0.65, 0.90, 0.95),
            (0.90, 0.95, 0.98),
            (0.15, 0.40, 0.70),
        ],
        "sunset": [
            (0.98, 0.45, 0.25),
            (0.98, 0.70, 0.25),
            (0.85, 0.35, 0.75),
            (0.40, 0.25, 0.70),
            (0.95, 0.55, 0.55),
        ],
    }


def get_background_presets():
    # RGB 0~255
    return {
        "Dark": (16, 16, 18),
        "Light": (248, 248, 250),
        "Paper": (245, 241, 232),
        "Cream": (252, 246, 232),
        "Mint": (232, 248, 244),
        "Lavender": (241, 236, 252),
        "Sky": (235, 244, 255),
        "Peach": (255, 238, 232),
        "Slate": (34, 38, 45),
    }


def parse_palette_csv(file_bytes: bytes) -> List[Tuple[float, float, float]]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    required = {"r", "g", "b"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV 必须包含列：r,g,b（0~1 浮点）。可选列：name")
    colors = []
    for _, row in df.iterrows():
        colors.append((float(row["r"]), float(row["g"]), float(row["b"])))
    if len(colors) == 0:
        raise ValueError("CSV 中没有任何颜色行。")
    return colors


def polygon_points(cx: float, cy: float, radius: float, sides: int, wobble: float, rng: np.random.Generator):
    pts = []
    base_angle = rng.uniform(0, 2 * math.pi)
    for i in range(sides):
        t = i / sides
        ang = base_angle + t * 2 * math.pi
        ang += rng.normal(0, wobble * 0.25)
        r = radius * (1 + rng.normal(0, wobble * 0.35))
        x = cx + math.cos(ang) * r
        y = cy + math.sin(ang) * r
        pts.append((x, y))
    return pts


def draw_soft_shape(
    img_rgba: Image.Image,
    shape: str,
    cx: float,
    cy: float,
    radius: float,
    sides: int,
    wobble: float,
    fill_rgb: Tuple[float, float, float],
    alpha: float,
    shadow_on: bool,
    shadow_offset: Tuple[int, int],
    shadow_alpha: float,
    shadow_darkness: float,
    stroke_on: bool,
    stroke_alpha: float,
    stroke_width: int,
    stroke_darkness: float,
    gradient_steps: int,
    rng: np.random.Generator,
):
    draw = ImageDraw.Draw(img_rgba, "RGBA")

    def draw_one(pass_radius: float, color_rgb: Tuple[float, float, float], a: float, offset=(0, 0), is_stroke=False):
        ox, oy = offset
        if shape == "circle":
            bbox = [cx - pass_radius + ox, cy - pass_radius + oy, cx + pass_radius + ox, cy + pass_radius + oy]
            if is_stroke:
                draw.ellipse(bbox, outline=rgb01_to_rgba255(color_rgb, a), width=stroke_width)
            else:
                draw.ellipse(bbox, fill=rgb01_to_rgba255(color_rgb, a))
        else:
            pts = polygon_points(cx + ox, cy + oy, pass_radius, sides, wobble, rng)
            if is_stroke:
                draw.polygon(pts, outline=rgb01_to_rgba255(color_rgb, a))
                for k in range(1, max(1, stroke_width)):
                    pts2 = polygon_points(cx + ox, cy + oy, pass_radius + k * 0.35, sides, wobble * 0.2, rng)
                    draw.polygon(pts2, outline=rgb01_to_rgba255(color_rgb, a))
            else:
                draw.polygon(pts, fill=rgb01_to_rgba255(color_rgb, a))

    # Shadow
    if shadow_on:
        shadow_rgb = darken(fill_rgb, shadow_darkness)
        for s in range(max(2, gradient_steps // 2)):
            t = s / (max(2, gradient_steps // 2) - 1)
            rr = radius * (1 - 0.10 * t)
            aa = shadow_alpha * (1 - 0.7 * t)
            draw_one(rr, shadow_rgb, aa, offset=shadow_offset)

    # Fill (soft gradient-ish)
    for s in range(gradient_steps):
        t = s / (gradient_steps - 1) if gradient_steps > 1 else 1.0
        rr = radius * (1 - 0.18 * t)
        c = mix_rgb(lighten(fill_rgb, 0.35), fill_rgb, t)
        aa = alpha * (0.55 + 0.45 * t)
        draw_one(rr, c, aa)

    # Stroke
    if stroke_on and stroke_width > 0 and stroke_alpha > 0:
        stroke_rgb = darken(fill_rgb, stroke_darkness)
        draw_one(radius, stroke_rgb, stroke_alpha, is_stroke=True)


def draw_text_block(img_rgba: Image.Image, title: str, subtitle: str, dark_bg: bool):
    draw = ImageDraw.Draw(img_rgba, "RGBA")
    W, H = img_rgba.size

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, W // 26))
        sub_font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, W // 42))
    except:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    pad = int(W * 0.06)
    color = (0, 0, 0, 220) if not dark_bg else (245, 245, 245, 220)

    if title.strip():
        draw.text((pad, pad), title, font=title_font, fill=color)
    if subtitle.strip():
        draw.text((pad, pad + int(W * 0.05)), subtitle, font=sub_font, fill=color)


# ----------------------------
# Poster config
# ----------------------------
@dataclass
class PosterConfig:
    canvas_px: int
    layers: int
    shape: str
    sides: int
    wobble: float
    spread: float
    min_radius: float
    max_radius: float
    min_alpha: float
    max_alpha: float
    palette: List[Tuple[float, float, float]]

    bg_rgb: Tuple[int, int, int]
    dark_bg: bool

    base_hue_shift: float
    seed: int

    shadow_on: bool
    shadow_offset: int
    shadow_alpha: float
    shadow_darkness: float

    stroke_on: bool
    stroke_alpha: float
    stroke_width: int
    stroke_darkness: float

    gradient_steps: int


def rotate_palette(palette: List[Tuple[float, float, float]], t01: float) -> List[Tuple[float, float, float]]:
    if len(palette) < 2:
        return palette
    k = int(round(t01 * (len(palette) - 1)))
    return palette[k:] + palette[:k]


def generate_poster(cfg: PosterConfig, title: str, subtitle: str) -> Image.Image:
    rng = np.random.default_rng(cfg.seed)

    W = H = cfg.canvas_px
    bg = (cfg.bg_rgb[0], cfg.bg_rgb[1], cfg.bg_rgb[2], 255)
    img = Image.new("RGBA", (W, H), bg)

    palette = rotate_palette(cfg.palette, cfg.base_hue_shift)

    center = (W / 2, H / 2)
    max_offset = cfg.spread * (W * 0.5)

    for i in range(cfg.layers):
        depth = i / max(1, cfg.layers - 1)
        c = palette[int(rng.integers(0, len(palette)))]
        a = lerp(cfg.min_alpha, cfg.max_alpha, depth)
        r = rng.uniform(cfg.min_radius, cfg.max_radius) * W
        dx = rng.normal(0, 1) * max_offset
        dy = rng.normal(0, 1) * max_offset
        cx = center[0] + dx
        cy = center[1] + dy

        c2 = lighten(c, (1 - depth) * 0.25)

        draw_soft_shape(
            img_rgba=img,
            shape=cfg.shape,
            cx=cx,
            cy=cy,
            radius=r,
            sides=max(3, cfg.sides),
            wobble=cfg.wobble,
            fill_rgb=c2,
            alpha=a,
            shadow_on=cfg.shadow_on,
            shadow_offset=(cfg.shadow_offset, cfg.shadow_offset),
            shadow_alpha=cfg.shadow_alpha * (0.8 + 0.2 * depth),
            shadow_darkness=cfg.shadow_darkness,
            stroke_on=cfg.stroke_on,
            stroke_alpha=cfg.stroke_alpha * (0.6 + 0.4 * depth),
            stroke_width=cfg.stroke_width,
            stroke_darkness=cfg.stroke_darkness,
            gradient_steps=cfg.gradient_steps,
            rng=rng,
        )

    draw_text_block(img, title, subtitle, cfg.dark_bg)
    return img


def pil_to_png_bytes(img: Image.Image, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Interactive Poster Generator", layout="wide")
st.title("Palette CSV · Interactive Poster Generator")

palettes = get_builtin_palettes()
bg_presets = get_background_presets()

with st.sidebar:
    st.subheader("Palette CSV")
    uploaded = st.file_uploader("Drag and drop file here", type=["csv"])

    colA, colB = st.columns(2)
    with colA:
        reset_builtin = st.button("Reset built-in palette")
    with colB:
        st.write("")

    st.divider()
    st.subheader("Poster Controls")

    palette_mode = st.selectbox(
        "Palette Mode",
        options=["dreamy", "pastel", "vivid", "mono", "neon", "ocean", "sunset", "csv"],
        index=0,
    )

    shape = st.selectbox("Shape", options=["polygon", "circle"], index=0)

    layers = st.slider("Layers", 5, 80, 39, 1)
    wobble = st.slider("Wobble", 0.0, 1.0, 0.34, 0.01)
    sides = st.slider("Sides", 3, 12, 9, 1) if shape == "polygon" else 6

    st.markdown("### Rendering")
    spread = st.slider("Spread (Diffusion)", 0.05, 1.20, 0.55, 0.01)

    min_radius = st.slider("Min Radius", 0.02, 0.40, 0.10, 0.01)
    max_radius = st.slider("Max Radius", 0.05, 0.70, 0.25, 0.01)

    min_alpha = st.slider("Min Alpha", 0.05, 0.95, 0.20, 0.01)
    max_alpha = st.slider("Max Alpha", 0.05, 0.95, 0.85, 0.01)

    gradient_steps = st.slider("Gradient Steps", 2, 18, 8, 1)

    seed = st.number_input("Seed", min_value=0, max_value=999999, value=995, step=1)

    base_hue = st.slider("Base Hue (palette shift)", 0.0, 1.0, 0.60, 0.01)

    st.markdown("### Background")
    bg_choice = st.selectbox("Background", options=list(bg_presets.keys()) + ["Custom"], index=0)

    if bg_choice == "Custom":
        hex_color = st.color_picker("Pick a background color", value="#101012")
        bg_rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    else:
        bg_rgb = bg_presets[bg_choice]

    # auto dark/light text mode based on luminance
    dark_bg = (0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2]) < 120

    st.markdown("### Shadow")
    shadow_on = st.checkbox("Shadow On", value=True)
    shadow_offset = st.slider("Shadow Offset", 0, 30, 10, 1)
    shadow_alpha = st.slider("Shadow Alpha", 0.0, 0.9, 0.22, 0.01)
    shadow_darkness = st.slider("Shadow Darkness", 0.0, 0.9, 0.55, 0.01)

    st.markdown("### Stroke")
    stroke_on = st.checkbox("Stroke On", value=True)
    stroke_width = st.slider("Stroke Width", 0, 12, 2, 1)
    stroke_alpha = st.slider("Stroke Alpha", 0.0, 1.0, 0.18, 0.01)
    stroke_darkness = st.slider("Stroke Darkness", 0.0, 0.9, 0.45, 0.01)

    st.markdown("### Text")
    title = st.text_input("Title", value="DREAMY GEOMETRY")
    subtitle = st.text_input("Subtitle", value="Interactive 3D Composition")

# Determine palette
palette: Optional[List[Tuple[float, float, float]]] = None
palette_error = None

if palette_mode != "csv":
    palette = palettes[palette_mode]
else:
    if uploaded is None:
        palette_error = "你选择了 CSV 模式，但还没有上传 CSV 文件。请上传包含 r,g,b 列的 CSV。"
    else:
        try:
            palette = parse_palette_csv(uploaded.getvalue())
        except Exception as e:
            palette_error = f"CSV 读取失败：{e}"

if reset_builtin:
    palette_mode = "dreamy"
    palette = palettes["dreamy"]
    palette_error = None

# Config
cfg = PosterConfig(
    canvas_px=900,
    layers=layers,
    shape=shape,
    sides=sides,
    wobble=wobble,
    spread=spread,
    min_radius=min_radius,
    max_radius=max_radius,
    min_alpha=min_alpha,
    max_alpha=max_alpha,
    palette=palette or palettes["dreamy"],
    bg_rgb=bg_rgb,
    dark_bg=dark_bg,
    base_hue_shift=base_hue,
    seed=int(seed),
    shadow_on=shadow_on,
    shadow_offset=shadow_offset,
    shadow_alpha=shadow_alpha,
    shadow_darkness=shadow_darkness,
    stroke_on=stroke_on,
    stroke_alpha=stroke_alpha,
    stroke_width=stroke_width,
    stroke_darkness=stroke_darkness,
    gradient_steps=gradient_steps,
)

# Layout
left, right = st.columns([1, 2.2], vertical_alignment="top")

with left:
    if palette_error:
        st.warning(palette_error)
    st.caption("提示：调大 Layers + Max Alpha + Shadow 会更像“叠层 3D 纸片”质感；调大 Spread 会更“扩散”。")
    st.markdown("—")
    st.write("当前调色板颜色数量：", len(cfg.palette))
    st.write("当前背景 RGB：", cfg.bg_rgb)

with right:
    poster = generate_poster(cfg, title=title, subtitle=subtitle)
    st.image(poster, use_container_width=False)

    # 关键：去掉 alpha 通道，避免 Google Drive 预览棋盘格问题
    poster_rgb = poster.convert("RGB")
    png_bytes = pil_to_png_bytes(poster_rgb, dpi=300)

    st.download_button(
        label="Download High-Res PNG (300 DPI)",
        data=png_bytes,
        file_name="poster_300dpi.png",
        mime="image/png",
        use_container_width=True,
    )
