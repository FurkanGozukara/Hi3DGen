"""Look of the Hi3DGen app.

Gradio 6 "Origin" theme used as shipped, dark by default with a one click light/dark toggle, and the coloured action
buttons used across the SECourses apps: every button gets its own hue (btn("emerald") ...) while size, weight and
type stay the same. Written against the theme's own CSS variables, so light and dark stay in sync automatically.
"""
import html
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import gradio as gr

ROOT = Path(__file__).resolve().parent
APP_TITLE = "Hi3DGen - High-fidelity 3D from images"
FAVICON_PATH = ROOT / "ui_assets" / "hi3dgen_favicon.svg"


def app_theme() -> gr.themes.Base:
    """The stock Gradio 6 Origin theme."""
    return gr.themes.Origin()


# --------------------------------------------------------------------------------------------------
# Button colours: (deep, mid, bright). The gradient runs deep -> mid -> bright so the face has depth, the glow is built
# from the mid tone and white text stays readable on all three stops.
# --------------------------------------------------------------------------------------------------
BUTTON_HUES = {
    "emerald": ("#065f46", "#059669", "#34d399"),
    "green":   ("#166534", "#16a34a", "#4ade80"),
    "lime":    ("#3f6212", "#65a30d", "#a3e635"),
    "teal":    ("#115e59", "#0d9488", "#2dd4bf"),
    "cyan":    ("#155e75", "#0891b2", "#22d3ee"),
    "sky":     ("#075985", "#0284c7", "#38bdf8"),
    "blue":    ("#1e40af", "#2563eb", "#60a5fa"),
    "indigo":  ("#3730a3", "#4f46e5", "#818cf8"),
    "violet":  ("#5b21b6", "#7c3aed", "#a78bfa"),
    "purple":  ("#6b21a8", "#9333ea", "#c084fc"),
    "fuchsia": ("#86198f", "#c026d3", "#e879f9"),
    "pink":    ("#9d174d", "#db2777", "#f9a8d4"),
    "rose":    ("#9f1239", "#e11d48", "#fda4af"),
    "red":     ("#991b1b", "#dc2626", "#f87171"),
    "crimson": ("#4c0519", "#9f1239", "#fb7185"),
    "orange":  ("#9a3412", "#ea580c", "#fb923c"),
    "amber":   ("#92400e", "#d97706", "#fbbf24"),
    "bronze":  ("#5c3a21", "#8b5a2b", "#d4a373"),
    "slate":   ("#334155", "#475569", "#94a3b8"),
    "gray":    ("#3f3f46", "#52525b", "#a1a1aa"),
    "olive":   ("#3f4f1f", "#6b8e23", "#b5cc4a"),
    "coral":   ("#9a3b2e", "#e0573e", "#ffa08a"),
    "mint":    ("#0f5132", "#2dbd8f", "#9ff3d3"),
    "gold":    ("#713f12", "#ca8a04", "#fde047"),
}
BUTTON_COLORS = tuple(BUTTON_HUES)


def btn(color: str, *extra: str) -> list:
    """`elem_classes` of a coloured action button; `color` is one of BUTTON_COLORS."""
    if color not in BUTTON_HUES:
        raise ValueError(f"Unknown button colour: {color}")
    return ["ax", f"ax-{color}", *extra]


def _rgb(value: str) -> Tuple[int, int, int]:
    digits = value.lstrip("#")
    return tuple(int(digits[index: index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _button_palette_css() -> str:
    """Gradient, border and glow rules for every button hue (dark and light page)."""
    rules = []
    for name, (deep, mid, bright) in BUTTON_HUES.items():
        red, green, blue = _rgb(mid)
        bright_rgb = ", ".join(str(part) for part in _rgb(bright))
        rules.append(f"""
.ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 55%, {bright} 100%) !important;
  border-color: rgba({bright_rgb}, .72) !important;
  box-shadow: 0 8px 20px rgba({red}, {green}, {blue}, .30), inset 0 1px 0 rgba(255, 255, 255, .20) !important;
}}
.ax-{name}:hover:not(:disabled) {{
  border-color: rgba({bright_rgb}, .98) !important;
  box-shadow: 0 12px 26px rgba({red}, {green}, {blue}, .44), inset 0 1px 0 rgba(255, 255, 255, .28) !important;
}}
body:not(.dark) .ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 66%, {bright} 100%) !important;
  border-color: {mid} !important;
  box-shadow: 0 7px 17px rgba({red}, {green}, {blue}, .26), inset 0 1px 0 rgba(255, 255, 255, .26) !important;
}}
body:not(.dark) .ax-{name}:hover:not(:disabled) {{
  box-shadow: 0 11px 24px rgba({red}, {green}, {blue}, .38), inset 0 1px 0 rgba(255, 255, 255, .32) !important;
}}""")
    return "\n".join(rules)


_BASE_CSS = r"""
:root { --ax-ok: #047857; --ax-warn: #b45309; --ax-error: #be123c; --ax-accent: #0f766e; }
:root.dark, :root .dark { --ax-ok: #10b981; --ax-warn: #f59e0b; --ax-error: #fb7185; --ax-accent: #14b8a6; }

/* Action buttons: same height, weight and type size everywhere, only the hue changes. Covers <button> and the
   <a> that a download button may render. */
.ax {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--size-2);
  min-height: 44px;
  padding: var(--size-2) var(--size-4) !important;
  border-width: 1px !important;
  border-style: solid !important;
  border-radius: var(--radius-lg) !important;
  color: #f8fafc !important;
  font-size: var(--text-md) !important;
  font-weight: 650 !important;
  line-height: 1.25 !important;
  text-align: center;
  text-decoration: none !important;
  text-shadow: 0 1px 2px rgba(2, 6, 23, .45) !important;
  transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease;
}
.ax:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.05); }
.ax:active:not(:disabled) { transform: translateY(1px); filter: brightness(.96); }
.ax:focus-visible { outline: 2px solid #bae6fd; outline-offset: 2px; }
.ax:disabled, .ax[disabled] { filter: grayscale(.45) opacity(.62); transform: none; box-shadow: none !important; }
.row > .ax { align-self: center; }

/* Page header */
.app-header { align-items: center; flex-wrap: nowrap; gap: var(--size-4); padding-bottom: var(--size-3); border-bottom: 1px solid var(--border-color-primary); }
.app-header > :first-child { flex: 1 1 auto; min-width: 0; }
.app-header h1 { margin: 0 !important; line-height: 1.2; }
.app-header p { margin: var(--size-1) 0 0 !important; color: var(--body-text-color-subdued); }
.app-header .ax { flex: 0 0 auto; white-space: nowrap; }

/* Live status card (single image, multiple images, batch) */
.progress-shell { border: 1px solid var(--block-border-color); border-radius: var(--radius-lg); padding: var(--size-3); background: var(--background-fill-secondary); }
.progress-head { display: flex; align-items: center; gap: var(--size-2); font-weight: 650; }
.progress-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--ax-accent); flex: 0 0 auto; }
.progress-shell.busy .progress-dot { animation: ax-pulse 1.4s ease-in-out infinite; }
.progress-shell.ok .progress-dot { background: var(--ax-ok); }
.progress-shell.warn .progress-dot { background: var(--ax-warn); }
.progress-shell.error .progress-dot { background: var(--ax-error); }
.progress-track { height: 8px; margin-top: var(--size-2); border-radius: var(--radius-full); background: var(--border-color-primary); overflow: hidden; }
.progress-fill { height: 100%; border-radius: var(--radius-full); background: linear-gradient(90deg, #2563eb, #14b8a6, #22c55e); transition: width 250ms ease; }
.progress-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 128px), 1fr)); gap: var(--size-2); margin-top: var(--size-3); }
.progress-metric { min-width: 0; }
.progress-metric strong { display: block; font-size: var(--text-xs); font-weight: 600; letter-spacing: .04em; color: var(--body-text-color-subdued); overflow-wrap: anywhere; }
.progress-metric span { display: block; margin-top: 2px; font-size: var(--text-md); font-weight: 700; color: var(--body-text-color); overflow-wrap: anywhere; }
.status-ok { color: var(--ax-ok); font-weight: 700; }
.status-warn { color: var(--ax-warn); font-weight: 700; }
.status-error { color: var(--ax-error); font-weight: 700; }
.summary-strip { padding: var(--size-2) var(--size-3); border-left: 4px solid var(--ax-accent); border-radius: var(--radius-sm); background: var(--background-fill-secondary); font-size: var(--text-sm); color: var(--body-text-color-subdued); }
@keyframes ax-pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: .45; transform: scale(.8); } }

/* Parameter guide tab: an 80vh panel whose content scrolls inside it (without the scroll the guide spilled over the settings below) */
#parameter-guide-content { width: 100% !important; height: 80vh !important; max-width: none !important; max-height: none !important; }
#parameter-guide-content > div { width: 100% !important; height: 100% !important; max-width: none !important; overflow-y: auto !important; }
.parameter-guide-tab { width: 100% !important; height: 100% !important; }

.preset-status { min-height: var(--size-6); font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.section-note { color: var(--body-text-color-subdued); font-size: var(--text-sm); }
"""

APP_CSS = _BASE_CSS + _button_palette_css() + "\n"


# Dark is the default the first time the app is opened. The choice is kept in localStorage and mirrored into the
# __theme query parameter, so a reload or a bookmark restores it before Gradio paints the page.
APP_HEAD = """
<meta name="color-scheme" content="dark light">
<script>
(function () {
  var KEY = "hi3dgen.theme";
  function resolve() {
    var url = new URL(window.location.href);
    var param = url.searchParams.get("__theme");
    var stored = null;
    try { stored = window.localStorage.getItem(KEY); } catch (e) {}
    var mode = param || stored || "dark";
    if (mode !== "light") { mode = "dark"; }
    try { window.localStorage.setItem(KEY, mode); } catch (e) {}
    if (param !== mode) {
      url.searchParams.set("__theme", mode);
      window.history.replaceState(null, "", url.toString());
    }
    return mode;
  }
  function paint(mode) {
    if (!document.body) { return false; }
    document.body.classList.toggle("dark", mode === "dark");
    return true;
  }
  var mode = resolve();
  if (!paint(mode)) { document.addEventListener("DOMContentLoaded", function () { paint(mode); }); }
  // The app launches with Gradio's run history off; histories saved by earlier launches only fill the storage quota.
  try {
    Object.keys(window.localStorage).forEach(function (key) {
      if (key.indexOf("gradio:run-history:") === 0) { window.localStorage.removeItem(key); }
    });
  } catch (e) {}
})();
</script>
"""

# Switching themes only swaps the `dark` class Gradio itself keys off: instant, no reload, no server round trip.
TOGGLE_THEME_JS = """
() => {
  const dark = !document.body.classList.contains("dark");
  document.body.classList.toggle("dark", dark);
  const mode = dark ? "dark" : "light";
  try { window.localStorage.setItem("hi3dgen.theme", mode); } catch (e) {}
  const url = new URL(window.location.href);
  url.searchParams.set("__theme", mode);
  window.history.replaceState(null, "", url.toString());
}
"""


def status_card(headline: str, fraction: Optional[float] = None, metrics: Optional[Sequence[Tuple[str, str]]] = None,
                tone: str = "idle") -> str:
    """HTML of the live status card.

    `tone` is one of idle, busy, ok, warn, error; `fraction` (0..1) fills the bar and hides it when None;
    `metrics` is a list of (label, value) pairs shown under the bar. All text is escaped.
    """
    bar = ""
    if fraction is not None:
        percent = max(0.0, min(1.0, float(fraction))) * 100
        bar = f'<div class="progress-track"><div class="progress-fill" style="width: {percent:.1f}%"></div></div>'
    cells = ""
    if metrics:
        cells = '<div class="progress-metrics">' + "".join(
            f'<div class="progress-metric"><strong>{html.escape(str(label)).upper()}</strong><span>{html.escape(str(value))}</span></div>'
            for label, value in metrics
        ) + "</div>"
    return (f'<div class="progress-shell {html.escape(tone)}"><div class="progress-head"><span class="progress-dot"></span>'
            f'<span>{html.escape(headline)}</span></div>{bar}{cells}</div>')
