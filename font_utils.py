from __future__ import annotations

from pathlib import Path
from typing import Iterable

from matplotlib import font_manager, rcParams

ROOT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "fonts"
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
PREFERRED_FONTS = [
    "Pretendard",
    "Pretendard Variable",
    "Noto Sans KR",
    "Noto Sans CJK KR",
    "NanumGothic",
    "KoPubDotum",
    "KoPub돋움체 Medium",
    "Malgun Gothic",
    "AppleGothic",
]


def iter_local_font_files(extra_dirs: Iterable[Path] | None = None) -> list[Path]:
    search_dirs = [FONT_DIR, Path.cwd() / "fonts"]
    if extra_dirs:
        search_dirs.extend(extra_dirs)
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in search_dirs:
        try:
            directory = directory.resolve()
        except Exception:
            continue
        if not directory.exists() or directory in seen:
            continue
        seen.add(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS:
                found.append(path)
    return found


def register_local_fonts() -> dict[str, Path]:
    registered: dict[str, Path] = {}
    for path in iter_local_font_files():
        try:
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            registered.setdefault(name, path)
        except Exception:
            continue
    return registered


def get_available_font_names() -> list[str]:
    local = register_local_fonts()
    installed = {font.name for font in font_manager.fontManager.ttflist}
    ordered: list[str] = []
    for name in [*local.keys(), *PREFERRED_FONTS]:
        if name in installed and name not in ordered:
            ordered.append(name)
    return ordered


def set_matplotlib_korean_font(font_name: str | None = None) -> str | None:
    available = get_available_font_names()
    selected = font_name if font_name in available else (available[0] if available else None)
    if selected:
        rcParams["font.family"] = selected
        rcParams["font.sans-serif"] = [selected, *PREFERRED_FONTS, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    return selected


def local_font_assets() -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    for path in iter_local_font_files():
        try:
            assets[path.name] = path.read_bytes()
        except OSError:
            continue
    return assets


def browser_font_face_css() -> str:
    files = iter_local_font_files()
    if not files:
        return ""
    rules = []
    for index, path in enumerate(files):
        try:
            name = font_manager.FontProperties(fname=str(path)).get_name()
        except Exception:
            name = f"SurveyFont{index + 1}"
        fmt = "opentype" if path.suffix.lower() == ".otf" else "truetype"
        rules.append(
            "@font-face{"
            f"font-family:'{name}';"
            f"src:url('assets/fonts/{path.name}') format('{fmt}');"
            "font-style:normal;font-weight:100 900;font-display:swap;"
            "}"
        )
    return "\n".join(rules)


def browser_primary_font_name() -> str | None:
    registered = register_local_fonts()
    return next(iter(registered.keys()), None)
