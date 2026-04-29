"""Generate synthetic OCR fixture images and matching ground-truth text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ocr_ground_truth"
SIZE = (800, 300)
OUTPUT_SIZE = (800, 300)  # Final rendered size after DPI scaling

# ── Realism hardening parameters ────────────────────────────────────────────
REALISM_JPEG_QUALITY = 72          # Compression artifact strength (65-85)
REALISM_BLUR_RADIUS = 1.2          # Mild camera/focus blur
REALISM_GAMMA_JITTER = 0.08        # ±8% brightness variance
REALISM_COLOR_SHIFT = 0.92         # RGB channel split factor (subpixel simulation)
REALISM_NOISE_STRENGTH = 8         # 0-12, subtle sensor noise
REALISM_WINDOW_CHROME = True       # Add title bar + borders
REALISM_SYNTAX_FRINGE = True       # Color bleed on syntax-highlighted tokens
REALISM_DPI_SCALE = True           # Down/up sample to mimic DPI scaling


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    kind: str
    lines: tuple[str, ...]
    accent: str = "#4ec9b0"
    # Realism variant profile (affects degradation pipeline)
    variant: str = "standard"  # "standard", "low_contrast", "chromatic", "tiny_text", "noisy_bg"


def _load_font(candidates: Iterable[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


MONO_FONT = _load_font(
    [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\CascadiaMono.ttf",
        r"C:\Windows\Fonts\cour.ttf",
    ],
    21,
)
UI_FONT = _load_font(
    [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ],
    20,
)
TITLE_FONT = _load_font(
    [
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ],
    22,
)


FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        "react_tsx_snippet_1",
        "code",
        (
            "export function StatusBadge({ status }: { status: 'idle' | 'done' }) {",
            "  return <span className={status === 'done' ? 'text-green-400' : 'text-zinc-400'}>{status}</span>;",
            "}",
        ),
        accent="#61dafb",
    ),
    FixtureSpec(
        "react_tsx_snippet_2",
        "code",
        (
            "const [query, setQuery] = useState('');",
            "const deferredQuery = useDeferredValue(query);",
            "const filtered = items.filter(item => item.label.includes(deferredQuery));",
        ),
        accent="#61dafb",
    ),
    FixtureSpec(
        "typescript_snippet_1",
        "code",
        (
            "type CaptureMetrics = { p50: number; p95: number; engine: 'winrt' | 'easyocr' };",
            "function formatMetrics(metrics: CaptureMetrics) {",
            "  return `${metrics.engine} ${metrics.p50}ms / ${metrics.p95}ms`;",
        ),
        accent="#3178c6",
    ),
    FixtureSpec(
        "typescript_snippet_2",
        "code",
        (
            "interface ProviderState { id: string; connected: boolean; lastCheckedAt?: string; }",
            "const ready = providers.every((provider: ProviderState) => provider.connected);",
            "setBanner(ready ? 'All providers healthy' : 'Action required');",
        ),
        accent="#3178c6",
    ),
    FixtureSpec(
        "javascript_snippet_1",
        "code",
        (
            "const grouped = messages.reduce((acc, message) => {",
            "  acc[message.role] = [...(acc[message.role] || []), message];",
            "  return acc;",
        ),
        accent="#f7df1e",
    ),
    FixtureSpec(
        "javascript_snippet_2",
        "code",
        (
            "document.querySelector('#save').addEventListener('click', async () => {",
            "  const response = await fetch('/api/settings/export');",
            "  download(await response.blob(), 'settings.json');",
        ),
        accent="#f7df1e",
    ),
    FixtureSpec(
        "frontend_css_snippet_1",
        "code",
        (
            ".settings-grid { display: grid; grid-template-columns: 220px 1fr auto; gap: 12px; }",
            ".settings-grid button { border-radius: 999px; background: linear-gradient(90deg, #0ea5e9, #22c55e); }",
            "@media (max-width: 720px) { .settings-grid { grid-template-columns: 1fr; } }",
        ),
        accent="#ec4899",
    ),
    FixtureSpec(
        "tailwind_snippet_1",
        "code",
        (
            "<div className=\"mx-auto grid max-w-5xl gap-6 px-6 py-10 md:grid-cols-[1.2fr_0.8fr]\">",
            "  <section className=\"rounded-3xl border border-white/10 bg-zinc-950/70 p-6 shadow-2xl\">",
            "    <button className=\"rounded-full bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950\">Run Benchmark</button>",
        ),
        accent="#38bdf8",
    ),
    FixtureSpec(
        "code_snippet_2",
        "code",
        (
            "def compute_total(items):",
            "    subtotal = sum(price for price in items)",
            "    return round(subtotal * 1.18, 2)",
        ),
    ),
    FixtureSpec(
        "code_snippet_3",
        "code",
        (
            "async def fetch_user(client, user_id):",
            "    response = await client.get(f'/users/{user_id}')",
            "    return response.json()",
        ),
        accent="#dcdcaa",
    ),
    FixtureSpec(
        "code_snippet_4",
        "code",
        (
            "const status = tasks.every(t => t.done) ? 'ready' : 'pending';",
            "const visible = tasks.filter(t => !t.archived);",
            "renderBoard(visible);",
        ),
        accent="#569cd6",
    ),
    FixtureSpec(
        "terminal_snippet_2",
        "terminal",
        (
            "PS C:\\OpenAssist> python -m unittest tests.test_p0_runtime",
            "Ran 30 tests in 1.31s",
            "OK",
        ),
        accent="#98c379",
    ),
    FixtureSpec(
        "terminal_snippet_3",
        "terminal",
        (
            "$ git status --short",
            "M capture/ocr.py",
            "?? benchmarks/capture_benchmark.py",
        ),
        accent="#e5c07b",
    ),
    FixtureSpec(
        "terminal_snippet_4",
        "terminal",
        (
            "[INFO] OCR engine ready: winrt",
            "[DEBUG] capture latency p50=8.2ms p95=38.2ms",
            "[WARN] ui_snippet_1 CER=0.632",
        ),
        accent="#61afef",
    ),
    FixtureSpec(
        "frontend_terminal_snippet_1",
        "terminal",
        (
            "$ npm run dev",
            "VITE v5.4.1 ready in 412 ms",
            "Local:   http://localhost:5173/",
        ),
        accent="#a855f7",
    ),
    FixtureSpec(
        "frontend_terminal_snippet_2",
        "terminal",
        (
            "$ npm run build",
            "dist/assets/index-9a3c1f.js   182.14 kB | gzip: 58.22 kB",
            "dist/assets/index-3af81c.css   21.31 kB | gzip: 4.93 kB",
        ),
        accent="#f97316",
    ),
    FixtureSpec(
        "frontend_ui_snippet_1",
        "ui",
        (
            "Design Review",
            "Hero headline       Needs stronger contrast",
            "Primary CTA         Benchmark Now",
            "Ship Preview",
        ),
        accent="#14b8a6",
    ),
    FixtureSpec(
        "ui_snippet_2",
        "ui",
        (
            "Settings",
            "OCR Engine    windows",
            "Fallback      easyocr -> tesseract",
            "Save Changes",
        ),
        accent="#0078d4",
    ),
    FixtureSpec(
        "ui_snippet_3",
        "ui",
        (
            "Provider Health",
            "Groq       Connected",
            "OpenAI     Missing Key",
            "Last Check 19:35",
        ),
        accent="#0f6cbd",
    ),
    FixtureSpec(
        "ui_snippet_4",
        "ui",
        (
            "Capture Panel",
            "Screen Interval    250 ms",
            "Smart Crop         Enabled",
            "Analyze Now",
        ),
        accent="#c239b3",
    ),
    FixtureSpec(
        "docs_snippet_1",
        "document",
        (
            "# Phase 0 Verification",
            "- Benchmark OCR latency on fixture corpus",
            "- Record p50 p95 and average CER",
        ),
        accent="#2b579a",
    ),
    FixtureSpec(
        "docs_snippet_2",
        "document",
        (
            "Implementation Notes",
            "Windows OCR is fast but code accuracy varies.",
            "Fallback engines protect against regressions.",
        ),
        accent="#0063b1",
    ),
    FixtureSpec(
        "json_snippet_1",
        "code",
        (
            '{',
            '  "engine": "winrt",',
            '  "latency_p50_ms": 8.2,',
            '  "average_cer": 0.298',
            '}',
        ),
        accent="#ce9178",
    ),
    FixtureSpec(
        "yaml_snippet_1",
        "code",
        (
            "capture:",
            "  screen:",
            "    interval_ms: 250",
            "    smart_crop: true",
        ),
        accent="#4ec9b0",
    ),
    FixtureSpec(
        "log_snippet_1",
        "terminal",
        (
            "2026-04-27 19:35:33 INFO Screen capture pipeline ready",
            "2026-04-27 19:35:33 INFO Windows Native OCR ready",
            "2026-04-27 19:35:34 INFO Saved report to benchmarks/baseline_ocr.json",
        ),
        accent="#56b6c2",
    ),
    FixtureSpec(
        "traceback_snippet_1",
        "terminal",
        (
            "Traceback (most recent call last):",
            '  File "core/app.py", line 905, in _analyze_screen_async',
            "RuntimeError: vision budget exceeded",
        ),
        accent="#e06c75",
    ),
    FixtureSpec(
        "sql_snippet_1",
        "code",
        (
            "SELECT provider, avg_latency_ms",
            "FROM benchmark_runs",
            "WHERE engine = 'winrt';",
        ),
        accent="#d7ba7d",
    ),
    FixtureSpec(
        "chat_snippet_1",
        "document",
        (
            "User: complete whats remaining of P0",
            "Assistant: benchmark runner is now in place",
            "User: sure lets expand",
        ),
        accent="#107c10",
    ),
)


def _assign_variant(spec: FixtureSpec, index: int, total: int) -> FixtureSpec:
    """Assign a realism variant based on fixture index to get a diverse set."""
    variants = [
        "standard",
        "low_contrast",      # WinRT struggles with faint text
        "chromatic",         # Subpixel confusion hits WinRT harder
        "tiny_text",         # Small font segmentation
        "noisy_bg",          # Compression + noise
    ]
    # Distribution: ~50% standard, rest distributed across stress tests
    weights = [0.5, 0.2, 0.15, 0.1, 0.05]
    # Deterministic per-fixture based on hash of name
    import hashlib
    h = int(hashlib.md5(spec.name.encode()).hexdigest()[:8], 16)
    idx = h % len(variants)
    chosen = variants[idx]
    return FixtureSpec(
        name=spec.name,
        kind=spec.kind,
        lines=spec.lines,
        accent=spec.accent,
        variant=chosen,
    )
def _assign_variant(spec: FixtureSpec, index: int, total: int) -> FixtureSpec:
    """Assign a realism variant based on spec kind to generate diverse stress tests."""
    variants = [
        "standard",
        "low_contrast",      # WinRT struggles with faint text
        "chromatic",         # Subpixel confusion hits WinRT harder
        "tiny_text",         # Small font segmentation
        "noisy_bg",          # Compression + noise
        "mixed_font",        # Mix of monospace + UI fonts — language model vs segmentation
        "syntax_highlight",  # Multiple colors in same line — segmentation hell
    ]
    # Deterministic per-fixture — rotate through variants but bias toward challenges
    import hashlib
    h = int(hashlib.md5(spec.name.encode()).hexdigest()[:8], 16)
    idx = h % len(variants)
    chosen = variants[idx]
    return FixtureSpec(
        name=spec.name,
        kind=spec.kind,
        lines=spec.lines,
        accent=spec.accent,
        variant=chosen,
    )


def _draw_window_frame(draw: ImageDraw.ImageDraw, title: str, accent: str) -> None:
    draw.rounded_rectangle((10, 10, 790, 290), radius=16, fill="#111827", outline="#2d3748", width=2)
    draw.rounded_rectangle((10, 10, 790, 46), radius=16, fill="#1f2937")
    draw.rectangle((10, 30, 790, 46), fill="#1f2937")
    draw.text((28, 18), title, fill="#f9fafb", font=TITLE_FONT)
    for idx, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = 730 + idx * 18
        draw.ellipse((x, 20, x + 10, 30), fill=color)
    draw.line((24, 54, 776, 54), fill=accent, width=2)


def _get_font_for_variant(spec: FixtureSpec, font_size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    """Return (mono_font, ui_font) appropriate for the variant."""
    mono = MONO_FONT if MONO_FONT else ImageFont.load_default()
    ui_f = UI_FONT if UI_FONT else mono
    try:
        if hasattr(mono, 'font_variant'):
            mono = mono.font_variant(size=font_size)
        if hasattr(ui_f, 'font_variant'):
            ui_f = ui_f.font_variant(size=font_size)
    except Exception:
        pass
    return mono, ui_f


def _draw_code(
    draw: ImageDraw.ImageDraw,
    spec: FixtureSpec,
    *,
    font_size: int = 21,
    text_color: str = "#e5e7eb",
) -> None:
    _draw_window_frame(draw, "VS Code - Open Assist", spec.accent)
    y = 72
    mono, ui_font = _get_font_for_variant(spec, font_size)
    line_number_x = 32
    text_start_x = 72
    max_x = 780  # right margin (800 - 20 padding)

    for index, line in enumerate(spec.lines, start=1):
        draw.text((line_number_x, y), f"{index:>2}", fill="#6b7280", font=mono)
        words = line.split()
        x = text_start_x
        for word in words:
            is_string = word.startswith(("'", '"', '`'))
            is_keyword = word.strip(":()[]{}[],;") in (
                "def", "class", "async", "await", "return", "if", "else", "const",
                "let", "var", "function", "export", "import", "from", "interface",
                "type", "public", "private", "static", "void", "int", "string",
                "boolean", "select", "where",
            )
            is_literal = word.strip(",:;()[]{}'\"") in ("None", "True", "False", "null", "undefined", "true", "false")
            is_type_name = len(word) > 1 and word[0].isupper() and word.isalnum() and not is_keyword

            chosen = mono
            if spec.variant == "mixed_font" and (is_string or is_literal or is_type_name):
                chosen = ui_font

            if spec.variant == "syntax_highlight":
                if is_string:
                    fill = "#ce9178"
                elif is_keyword:
                    fill = "#569cd6"
                elif is_type_name:
                    fill = "#4ec9b0"
                elif word.startswith(("#", "@")):
                    fill = "#d7ba7d"
                else:
                    fill = text_color
            else:
                fill = text_color

            # Measure word width
            try:
                width = chosen.getlength(word)
            except Exception:
                width = len(word) * (font_size * 0.6)

            # Word wrap if exceeds margin
            if x > text_start_x and x + width > max_x:
                x = text_start_x
                y += int(font_size * 2.2)

            draw.text((x, y), word, fill=fill, font=chosen)
            x += width + 6  # 6px gap between words
        y += int(font_size * 2.2)


def _draw_terminal(
    draw: ImageDraw.ImageDraw,
    spec: FixtureSpec,
    *,
    font_size: int = 21,
    text_color: str = "#d1d5db",
) -> None:
    _draw_window_frame(draw, "Windows PowerShell", spec.accent)
    draw.rounded_rectangle((24, 70, 776, 274), radius=12, fill="#0b1020")
    y = 88
    mono, _ = _get_font_for_variant(spec, font_size)
    max_x = 760  # right margin inside terminal

    for line in spec.lines:
        fill = text_color
        if line.startswith(("PS ", "$ ")):
            fill = "#a3e635"
        elif "WARN" in line or "RuntimeError" in line:
            fill = "#fca5a5"
        elif "INFO" in line or line == "OK":
            fill = "#86efac"

        # Simple word wrap for long terminal lines
        words = line.split()
        x = 40
        for word in words:
            try:
                width = mono.getlength(word)
            except Exception:
                width = len(word) * (font_size * 0.6)
            if x > 40 and x + width > max_x:
                x = 40
                y += int(font_size * 2.15)
            draw.text((x, y), word, fill=fill, font=mono)
            x += width + 6
        y += int(font_size * 2.15)


def _draw_ui(draw: ImageDraw.ImageDraw, spec: FixtureSpec) -> None:
    draw.rounded_rectangle((16, 16, 784, 284), radius=20, fill="#f5f7fb", outline="#d0d7e2", width=2)
    draw.rounded_rectangle((16, 16, 784, 56), radius=20, fill=spec.accent)
    draw.rectangle((16, 36, 784, 56), fill=spec.accent)
    draw.text((34, 24), spec.lines[0], fill="#ffffff", font=TITLE_FONT)
    y = 88
    for line in spec.lines[1:-1]:
        draw.rounded_rectangle((36, y - 8, 620, y + 22), radius=10, fill="#ffffff", outline="#d9e2f2")
        draw.text((52, y), line, fill="#1f2937", font=UI_FONT)
        y += 54
    draw.rounded_rectangle((560, 226, 732, 262), radius=12, fill=spec.accent)
    draw.text((590, 234), spec.lines[-1], fill="#ffffff", font=UI_FONT)


def _draw_document(draw: ImageDraw.ImageDraw, spec: FixtureSpec) -> None:
    draw.rectangle((0, 0, 800, 300), fill="#fbfbfc")
    draw.rectangle((0, 0, 800, 16), fill=spec.accent)
    y = 34
    first_font = TITLE_FONT if spec.lines[0].startswith("#") or len(spec.lines[0]) < 24 else UI_FONT
    for index, line in enumerate(spec.lines):
        font = first_font if index == 0 else UI_FONT
        fill = "#111827" if index == 0 else "#374151"
        draw.text((40, y), line, fill=fill, font=font)
        y += 54 if index == 0 else 44


# ═══════════════════════════════════════════════════════════════════════════════
# Realism Hardening Transforms  (P0 — make synthetic fixtures match real screenshots)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These transforms convert pristine vector-rendered text into something that
# looks like an actual Windows screenshot captured via mss/BitBlt. They introduce
# the same artifacts that cause WinRT to stumble: subpixel blur, compression,
# window chrome, and syntax-highlight color fringes.
#

# ── Variant-specific rendering modifiers ────────────────────────────────────
_VARIANT_MODS = {
    "standard": {
        "font_size": 21,
        "text_color": "#e5e7eb",
        "bg_color": "#0f172a",
        "contrast_boost": 0.0,
        "blur_extra": 0.0,
        "chromatic": False,
    },
    "low_contrast": {
        "font_size": 20,
        "text_color": "#a0a0a0",   # 60% gray on dark — WinRT often drops faint text
        "bg_color": "#1a1a1a",
        "contrast_boost": -0.15,
        "blur_extra": 0.3,
        "chromatic": False,
    },
    "chromatic": {
        "font_size": 21,
        "text_color": "#00ffcc",   # Cyan/magenta split prone to subpixel confusion
        "bg_color": "#0f172a",
        "contrast_boost": 0.0,
        "blur_extra": 0.0,
        "chromatic": True,
    },
    "tiny_text": {
        "font_size": 15,           # 9-11pt equivalent at 96 DPI
        "text_color": "#e5e7eb",
        "bg_color": "#0f172a",
        "contrast_boost": 0.1,
        "blur_extra": 0.2,
        "chromatic": False,
    },
    "noisy_bg": {
        "font_size": 21,
        "text_color": "#e5e7eb",
        "bg_color": "#0f172a",
        "contrast_boost": 0.0,
        "blur_extra": 0.0,
        "chromatic": False,
    },
    "mixed_font": {
        "font_size": 21,
        "text_color": "#e5e7eb",
        "bg_color": "#0f172a",
        "contrast_boost": 0.0,
        "blur_extra": 0.0,
        "chromatic": False,
    },
    "syntax_highlight": {
        "font_size": 21,
        "text_color": "#e5e7eb",
        "bg_color": "#0f172a",
        "contrast_boost": 0.0,
        "blur_extra": 0.0,
        "chromatic": False,
    },
}


def _add_window_chrome(img: Image.Image, title: str, accent: str, kind: str) -> Image.Image:
    """Overlay a semi-realistic VS Code / terminal window frame."""
    result = img.copy()
    d = ImageDraw.Draw(result)
    d.rounded_rectangle((8, 8, 792, 292), radius=16, outline="#2d3748", width=2)
    d.rounded_rectangle((10, 10, 790, 46), radius=16, fill="#1f2937")
    d.rectangle((10, 30, 790, 46), fill="#1f2937")
    d.text((28, 18), title, fill="#f9fafc", font=TITLE_FONT)
    for idx, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = 730 + idx * 18
        d.ellipse((x, 20, x + 10, 30), fill=color)
    d.line((24, 54, 776, 54), fill=accent, width=2)
    shadow = Image.new("RGBA", result.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle((11, 55, 789, 60), radius=2, fill=(0, 0, 0, 50))
    result = Image.alpha_composite(result.convert("RGBA"), shadow).convert("RGB")
    return result


def _apply_subpixel_blur(img: Image.Image, extra: float = 0.0, chromatic: bool = False) -> Image.Image:
    r, g, b = img.split()
    blur_radius = 1.0 + extra
    r = r.filter(ImageFilter.GaussianBlur(blur_radius))
    g = g.filter(ImageFilter.GaussianBlur(blur_radius + 0.2))
    b = b.filter(ImageFilter.GaussianBlur(blur_radius + 0.4))
    # Chromatic aberration: horizontal channel offset
    if chromatic:
        r = ImageChops.offset(r, -2, 0)
        b = ImageChops.offset(b, 2, 0)
    else:
        r = ImageChops.offset(r, -1, 0)
        b = ImageChops.offset(b, 1, 0)
    return Image.merge("RGB", (r, g, b))


def _add_compression_artifacts(img: Image.Image, quality: int = REALISM_JPEG_QUALITY) -> Image.Image:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(buf).convert("RGB")


def _add_gamma_jitter(img: Image.Image, jitter: float = REALISM_GAMMA_JITTER) -> Image.Image:
    import random
    gamma = 1.0 + random.uniform(-jitter, jitter)
    lut = [min(255, int(((i / 255) ** (1.0 / gamma)) * 255)) for i in range(256)]
    lut = lut * 3
    return img.point(lut)


def _add_sensor_noise(img: Image.Image, strength: int = REALISM_NOISE_STRENGTH) -> Image.Image:
    import random
    noise = Image.new("L", img.size)
    pixels = noise.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            pixels[x, y] = max(0, min(255, 128 + random.randint(-strength, strength)))
    return ImageChops.overlay(img, noise.convert("RGB"))


def _simulate_dpi_scaling(img: Image.Image) -> Image.Image:
    factor = 0.75
    small = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.BICUBIC)
    return small.resize(img.size, Image.Resampling.BICUBIC)


def apply_realism_transforms(
    img: Image.Image,
    spec: FixtureSpec,
    title: str,
    variant_mod: dict,
) -> Image.Image:
    transformed = img

    if spec.kind in ("code", "terminal"):
        transformed = _add_window_chrome(transformed, title, spec.accent, spec.kind)

    transformed = _simulate_dpi_scaling(transformed)
    transformed = _apply_subpixel_blur(
        transformed,
        extra=variant_mod.get("blur_extra", 0.0),
        chromatic=variant_mod.get("chromatic", False),
    )
    transformed = _add_gamma_jitter(transformed)

    if variant_mod.get("contrast_boost", 0.0) != 0.0:
        import random
        factor = 1.0 + variant_mod["contrast_boost"]
        lut = [min(255, max(0, int((i - 128) * factor + 128))) for i in range(256)]
        lut = lut * 3
        transformed = transformed.point(lut)

    transformed = _add_compression_artifacts(transformed)
    transformed = _add_sensor_noise(transformed)

    return transformed


def render_fixture(spec: FixtureSpec) -> Image.Image:
    variant_mod = _VARIANT_MODS.get(spec.variant, _VARIANT_MODS["standard"])
    font_size = variant_mod["font_size"]
    text_color = variant_mod["text_color"]

    bg = variant_mod["bg_color"]
    image = Image.new("RGB", SIZE, bg)
    draw = ImageDraw.Draw(image)

    if spec.kind == "code":
        _draw_code(draw, spec, font_size=font_size, text_color=text_color)
    elif spec.kind == "terminal":
        _draw_terminal(draw, spec, font_size=font_size, text_color=text_color)
    elif spec.kind == "ui":
        _draw_ui(draw, spec)
    else:
        _draw_document(draw, spec)

    # ── Apply realism hardening ─────────────────────────────────────────────
    title = "VS Code - Open Assist" if spec.kind == "code" else (
        "Windows PowerShell" if spec.kind == "terminal" else
        spec.lines[0] if spec.kind == "ui" else "Document"
    )
    image = apply_realism_transforms(image, spec, title, variant_mod)
    return image


def save_fixture(spec: FixtureSpec) -> None:
    image_path = FIXTURE_DIR / f"{spec.name}.png"
    text_path = FIXTURE_DIR / f"{spec.name}.png.txt"
    image = render_fixture(spec)
    image.save(image_path)
    text_path.write_text("\n".join(spec.lines) + "\n", encoding="utf-8")


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for idx, spec in enumerate(FIXTURES):
        variant_spec = _assign_variant(spec, idx, len(FIXTURES))
        save_fixture(variant_spec)
    print(f"Generated {len(FIXTURES)} fixtures in {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
