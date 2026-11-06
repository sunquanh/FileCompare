"""界面基础件：主题、圆角控件、滚动区、对话框。

全部用 Canvas 手绘，避免 Windows 原生控件的灰色观感。
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import font as tkfont
from typing import ClassVar

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageGrab, ImageTk
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageEnhance = None
    ImageGrab = None
    ImageTk = None


# ---------------------------------------------------------------- 主题
# 两套调色板共用同一组键名。切换时既不重建窗口，也不改各处绘制代码：
#   apply_palette() 把新色值写回 Theme 类属性（之后的绘制自然用新色），
#   并返回 {旧色值: 新色值} 让 remap_tree() 把已经画好的像素级内容换掉。
DARK = {
    # 背景层次
    "BG": "#0B0D12", "BG_SOFT": "#0E1117", "SIDEBAR": "#0C0F15",
    "CARD": "#131821", "CARD_HOVER": "#18202C", "CARD_INSET": "#141A24",
    "INPUT": "#0F141C", "BORDER": "#232A38", "BORDER_SOFT": "#1B212D",
    "CHIP_BG": "#1B2434", "CHIP_ACCENT": "#1B2440", "BUBBLE_BG": "#1A2334",
    "NAV_HOVER": "#161D2B", "NAV_ACTIVE": "#111722",
    "STATUS_BG": "#101620", "TOAST_BG": "#161D29", "TOAST_BORDER": "#2A3345",
    "LOG_BG": "#0E1219", "LOG_BAR": "#11161F", "LOG_SELECT": "#26304A",
    "TILE_BG": "#0D1118", "TILE_CHECK_BG": "#131A24",
    "ART_INNER": "#111721", "RING_TRACK": "#1B2230",
    "EMPTY_ART_BG": "#102019", "EMPTY_ART_BORDER": "#1D3A2C",
    "KEEP_BADGE_BG": "#10251B", "KEEP_BADGE_BORDER": "#245039",
    # 文字
    "TEXT": "#E9EDF5", "TEXT_DIM": "#98A2B6", "MUTED": "#6B7589",
    "FAINT": "#495266", "TEXT_ON_RED": "#E7C9CC", "ON_ACCENT": "#FFFFFF",
    # 强调色
    "ACCENT": "#6C8CFF", "ACCENT_HOVER": "#7E9BFF", "ACCENT_DIM": "#232C4A",
    "VIOLET": "#A78BFA", "GREEN": "#3DDC97", "GREEN_DIM": "#14291F",
    "AMBER": "#F5B544", "AMBER_DIM": "#2C2515",
    "RED": "#E5484D", "RED_HOVER": "#F0616D", "RED_DIM": "#2A161C",
    "KIND_AUDIO": "#F472B6", "KIND_NAME": "#38BDF8",
    # 分类底色
    "KIND_EXACT_DIM": "#1A2240", "KIND_PHOTO_DIM": "#2B2415",
    "KIND_VIDEO_DIM": "#241E38", "KIND_AUDIO_DIM": "#2A1524",
    "KIND_NAME_DIM": "#12263A", "KIND_OTHER_DIM": "#1A1F2A",
    "KIND_VIDEO_CHIP": "#261E3B",
    # 控件
    "BTN_SECONDARY": "#1A2130", "BTN_SECONDARY_HOVER": "#222B3D",
    "BTN_GHOST_HOVER": "#171D28", "BTN_SOFT_HOVER": "#2B3759",
    "BTN_SUCCESS": "#173528", "BTN_SUCCESS_HOVER": "#1D4433",
    "BTN_SUCCESS_BORDER": "#26553D", "BTN_DISABLED_BG": "#171C26",
    "TRACK_OFF": "#28303F", "KNOB_ON": "#FFFFFF", "KNOB_OFF": "#B9C2D4",
    "SCROLL_THUMB": "#273045", "SCROLL_THUMB_HOVER": "#2C3546",
    "TILE_CHECK_BORDER": "#B9C3D6", "CHECK_BORDER": "#FFD9DB",
}

DAY = {
    # 背景层次：页面浅灰、卡片纯白，靠 1px 描边分层
    "BG": "#F4F6FA", "BG_SOFT": "#EDF0F6", "SIDEBAR": "#FFFFFF",
    "CARD": "#FFFFFF", "CARD_HOVER": "#F6F8FC", "CARD_INSET": "#F2F5FA",
    "INPUT": "#FFFFFF", "BORDER": "#DCE1EB", "BORDER_SOFT": "#E8ECF3",
    "CHIP_BG": "#EEF1F7", "CHIP_ACCENT": "#E9EEFB", "BUBBLE_BG": "#EEF1F7",
    "NAV_HOVER": "#F0F3F9", "NAV_ACTIVE": "#E9EEFB",
    "STATUS_BG": "#FFFFFF", "TOAST_BG": "#FFFFFF", "TOAST_BORDER": "#DCE1EB",
    "LOG_BG": "#FFFFFF", "LOG_BAR": "#F4F6FA", "LOG_SELECT": "#D9E4FF",
    "TILE_BG": "#F4F6FA", "TILE_CHECK_BG": "#FFFFFF",
    "ART_INNER": "#F7F9FC", "RING_TRACK": "#DDE3ED",
    "EMPTY_ART_BG": "#E8F5EE", "EMPTY_ART_BORDER": "#C6E7D5",
    "KEEP_BADGE_BG": "#E3F5EC", "KEEP_BADGE_BORDER": "#B3E2C9",
    # 文字
    "TEXT": "#1A1F2B", "TEXT_DIM": "#5A6474", "MUTED": "#7C8698",
    "FAINT": "#A2AAB9", "TEXT_ON_RED": "#8C5A5E", "ON_ACCENT": "#FFFFFF",
    # 强调色（在白底上压深一档，保证对比度）
    "ACCENT": "#3F63F0", "ACCENT_HOVER": "#3557DC", "ACCENT_DIM": "#E6EBFE",
    "VIOLET": "#7C5CE0", "GREEN": "#0F9D63", "GREEN_DIM": "#E3F5EC",
    "AMBER": "#B8791A", "AMBER_DIM": "#FCF1DA",
    "RED": "#D22F35", "RED_HOVER": "#B8262C", "RED_DIM": "#FBE7E8",
    "KIND_AUDIO": "#C2287C", "KIND_NAME": "#0B7FA5",
    # 分类底色
    "KIND_EXACT_DIM": "#E6EBFE", "KIND_PHOTO_DIM": "#FCF1DA",
    "KIND_VIDEO_DIM": "#F0EAFB", "KIND_AUDIO_DIM": "#FCE7F1",
    "KIND_NAME_DIM": "#E1F2F9", "KIND_OTHER_DIM": "#EEF1F6",
    "KIND_VIDEO_CHIP": "#F0EAFB",
    # 控件
    "BTN_SECONDARY": "#EEF1F7", "BTN_SECONDARY_HOVER": "#E3E8F1",
    "BTN_GHOST_HOVER": "#EDF0F6", "BTN_SOFT_HOVER": "#D7E0FF",
    "BTN_SUCCESS": "#E3F5EC", "BTN_SUCCESS_HOVER": "#D2EEDF",
    "BTN_SUCCESS_BORDER": "#B3E2C9", "BTN_DISABLED_BG": "#EDEFF4",
    "TRACK_OFF": "#C6CEDC", "KNOB_ON": "#FFFFFF", "KNOB_OFF": "#FFFFFF",
    "SCROLL_THUMB": "#C7CEDA", "SCROLL_THUMB_HOVER": "#AEB7C8",
    "TILE_CHECK_BORDER": "#8A93A6", "CHECK_BORDER": "#F0A8AE",
}

PALETTES = {"dark": DARK, "day": DAY}
_active_palette = "dark"


def palette_name() -> str:
    return _active_palette


def palette_button_styles() -> dict:
    return {
        "primary": (Theme.ACCENT, Theme.ACCENT_HOVER, Theme.ON_ACCENT, None),
        "secondary": (Theme.BTN_SECONDARY, Theme.BTN_SECONDARY_HOVER,
                      Theme.TEXT, Theme.BORDER),
        "ghost": ("", Theme.BTN_GHOST_HOVER, Theme.TEXT_DIM, None),
        "danger": (Theme.RED, Theme.RED_HOVER, Theme.ON_ACCENT, None),
        "soft": (Theme.ACCENT_DIM, Theme.BTN_SOFT_HOVER, Theme.ACCENT, None),
        "success": (Theme.BTN_SUCCESS, Theme.BTN_SUCCESS_HOVER, Theme.GREEN,
                    Theme.BTN_SUCCESS_BORDER),
    }


# 色值映射是「按值找色」，而不同角色可能用同一个色（比如卡片和侧栏都是纯白）。
# 这里定一个优先级：撞车时让底色/正文/边框这些主角色占先，避免把页面底色
# 换成了某个组件的相近色。差 1~2 级肉眼看不出来，但不会整块错色。
_REMAP_ORDER = (
    "BG", "SIDEBAR", "CARD", "INPUT", "TEXT", "TEXT_DIM", "MUTED", "FAINT",
    "BORDER", "BORDER_SOFT", "ACCENT", "ACCENT_DIM", "GREEN", "AMBER", "RED",
    "VIOLET", "KIND_AUDIO", "KIND_NAME", "ON_ACCENT", "KNOB_ON",
)


def apply_palette(name: str) -> dict:
    """切换调色板，返回 {旧色值: 新色值} 供界面就地换色。"""
    global _active_palette
    name = name if name in PALETTES else "dark"
    old, new = PALETTES[_active_palette], PALETTES[name]
    _active_palette = name
    for key, value in new.items():
        setattr(Theme, key, value)
    CanvasButton.STYLES = palette_button_styles()
    rank = {key: i for i, key in enumerate(_REMAP_ORDER)}
    mapping = {}
    for key in sorted(new, key=lambda k: rank.get(k, len(rank))):
        if key in old:
            mapping.setdefault(old[key].lower(), new[key])
    return mapping


_COLOR_OPTS = ("background", "foreground", "activebackground", "activeforeground",
               "disabledforeground", "highlightbackground", "highlightcolor",
               "insertbackground", "selectbackground", "selectforeground",
               "troughcolor", "readonlybackground")


def remap_tree(root, mapping):
    """把已经存在的窗口和画布元素按色值整体换色（含目前所有子控件）。"""
    if not mapping:
        return 0
    changed = 0
    stack = [root]
    while stack:
        widget = stack.pop()
        try:
            stack.extend(widget.winfo_children())
        except tk.TclError:
            continue
        for opt in _COLOR_OPTS:
            try:
                current = widget.cget(opt)
            except tk.TclError:
                continue
            if not isinstance(current, str):
                continue
            target = mapping.get(current.lower())
            if target and target != current:
                try:
                    widget.configure(**{opt: target})
                    changed += 1
                except tk.TclError:
                    pass
        restyle = getattr(widget, "restyle", None)
        if callable(restyle):
            try:
                restyle()
                changed += 1
            except tk.TclError:
                pass
        if isinstance(widget, tk.Canvas):
            try:
                items = widget.find_all()
            except tk.TclError:
                continue
            for item in items:
                for opt in ("fill", "outline"):
                    try:
                        current = widget.itemcget(item, opt)
                    except tk.TclError:
                        continue
                    target = mapping.get(current.lower())
                    if target and target != current:
                        try:
                            widget.itemconfigure(item, **{opt: target})
                            changed += 1
                        except tk.TclError:
                            pass
    return changed


class Theme:
    """字体与尺寸；颜色键由 apply_palette() 写在类属性上。"""

    def __init__(self, root):
        self.root = root
        try:
            self.scale = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        except Exception:
            self.scale = 1.0
        families = set(tkfont.families(root))
        self.ui = self._pick(families, "Microsoft YaHei UI", "Microsoft YaHei",
                             "Segoe UI", "Arial")
        self.icon_font = self._pick(families, "Segoe Fluent Icons",
                                    "Segoe MDL2 Assets", "Segoe UI Symbol")
        # 图标字体只在 Windows 上有；没有它时，各处图标改走矢量手绘
        self.has_icon_font = self.icon_font in families
        self.mono = self._pick(families, "Cascadia Mono", "Consolas", "Courier New")
        self._fonts = {}

    @staticmethod
    def _pick(families, *names):
        for name in names:
            if name in families:
                return name
        return names[-1]

    def s(self, value) -> int:
        return round(value * self.scale)

    def font(self, size=10, weight="normal", family=None):
        key = (size, weight, family)
        f = self._fonts.get(key)
        if f is None:
            f = tkfont.Font(root=self.root, family=family or self.ui,
                            size=size, weight=weight)
            self._fonts[key] = f
        return f

    def icon(self, size=16):
        """图标字体用像素尺寸，保证和画布坐标一致。"""
        return (self.icon_font, -self.s(size))

    def kind_color(self, kind):
        return {KIND_EXACT: self.ACCENT, KIND_PHOTO: self.AMBER,
                KIND_VIDEO: self.VIOLET, KIND_AUDIO: self.KIND_AUDIO,
                KIND_NAME: self.KIND_NAME,
                KIND_DIR: self.GREEN}.get(kind, self.TEXT_DIM)

    def kind_dim(self, kind):
        return {KIND_EXACT: self.KIND_EXACT_DIM, KIND_PHOTO: self.KIND_PHOTO_DIM,
                KIND_VIDEO: self.KIND_VIDEO_DIM, KIND_AUDIO: self.KIND_AUDIO_DIM,
                KIND_NAME: self.KIND_NAME_DIM,
                KIND_DIR: self.KIND_OTHER_DIM}.get(kind, self.KIND_OTHER_DIM)


for _key, _value in DARK.items():
    setattr(Theme, _key, _value)


KIND_EXACT = "完全重复"
KIND_PHOTO = "相似照片"
KIND_VIDEO = "重复视频"
KIND_AUDIO = "重复音乐"
KIND_NAME = "同名相似"
KIND_DIR = "重复目录"

ICON = {
    "folder": "\uE8B7",
    "search": "\uE721",
    "image": "\uE91B",
    "video": "\uE714",
    "trash": "\uE74D",
    "check": "\uE73E",
    "close": "\uE711",
    "chevron_down": "\uE70D",
    "chevron_right": "\uE76C",
    "refresh": "\uE72C",
    "open": "\uE8A7",
    "open_folder": "\uE838",
    "info": "\uE946",
    "settings": "\uE713",
    "list": "\uE8FD",
    "layers": "\uE81E",
    "shield": "\uEA18",
    "clock": "\uE823",
    "sparkle": "\uE945",
    "arrow_right": "\uE72A",
    "copy": "\uE8C8",
    "document": "\uE8A5",
    "save": "\uE74E",
    "play": "\uE768",
    "warning": "\uE7BA",
    "more": "\uE712",
    "filter": "\uE71C",
    "scan": "\uE721",
    "stop": "\uE71A",
    "link": "\uE71B",
    "history": "\uE81C",
    "photo": "\uE91B",
    "edit": "\uE70F",
    "audio": "\uE8D6",
    "compare": "\uE7C4",
    "sun": "\uE706",
    "moon": "\uE708",
    "hardlink": "\uE71B",
}

# ---------------------------------------------------------------- 矢量图标
# 图标平时走 Segoe Fluent Icons 字形；那套字体只在 Windows 上有。
# 字体缺失时（Linux / macOS）改用这里的手绘矢量，保证图标不变成方框。
def draw_icon(canvas, theme, name, cx, cy, size, color, tags=()):
    """以 (cx, cy) 为中心，在画布上画一个 size 像素的单色矢量图标。"""
    s = theme.s
    u = s(size) / 20.0

    def X(v):
        return cx + v * u

    def Y(v):
        return cy + v * u

    def pts(seq):
        out = []
        for p in seq:
            out.extend((X(p[0]), Y(p[1])))
        return out

    def line(seq, w=1.6):
        canvas.create_line(*pts(seq), fill=color, width=max(1, s(w)),
                           capstyle="round", joinstyle="round", tags=tags)

    def oval(x1, y1, x2, y2, fill="", outline=None):
        canvas.create_oval(X(x1), Y(y1), X(x2), Y(y2), fill=fill,
                           outline=color if outline is None else outline,
                           width=max(1, s(1.6)), tags=tags)

    def rect(x1, y1, x2, y2, fill="", outline=None):
        canvas.create_rectangle(X(x1), Y(y1), X(x2), Y(y2), fill=fill,
                                outline=color if outline is None else outline,
                                width=max(1, s(1.6)), tags=tags)

    def poly(seq, fill=""):
        canvas.create_polygon(*pts(seq), fill=fill, outline=color,
                              width=max(1, s(1.6)), joinstyle="round", tags=tags)

    def dot(x, y, r=1.7):
        canvas.create_oval(X(x) - r * u, Y(y) - r * u, X(x) + r * u,
                           Y(y) + r * u, fill=color, outline="", tags=tags)

    def arc(x1, y1, x2, y2, start, extent):
        canvas.create_arc(X(x1), Y(y1), X(x2), Y(y2), start=start,
                          extent=extent, style="arc", outline=color,
                          width=max(1, s(1.6)), tags=tags)

    name = {"photo": "image", "hardlink": "link",
            "open": "folder_arrow",
            "open_folder": "folder_arrow"}.get(name, name)
    if name == "search":
        oval(-7, -7, 3, 3)
        line([(2, 2), (8, 8)])
    elif name == "folder":
        poly([(-8, -3), (-4, -7), (2, -7), (4, -3), (8, -3), (8, 8), (-8, 8)])
    elif name == "folder_arrow":
        poly([(-8, -2), (-4, -6), (2, -6), (4, -2), (8, -2), (8, 8), (-8, 8)])
        line([(0, 2), (6, -4)])
        poly([(6, -4), (2, -4), (6, 0)], fill=color)
    elif name == "trash":
        line([(-7, -6), (7, -6)])
        line([(-3, -6), (-3, -9)])
        poly([(-6, -4), (6, -4), (5, 8), (-5, 8)])
        line([(-2, -1), (-2, 5)])
        line([(2, -1), (2, 5)])
    elif name == "check":
        line([(-7, 0), (-2, 6), (8, -6)], w=2.2)
    elif name == "close":
        line([(-6, -6), (6, 6)], w=2)
        line([(6, -6), (-6, 6)], w=2)
    elif name == "warning":
        poly([(0, -8), (8, 7), (-8, 7)])
        line([(0, -3), (0, 2)], w=2)
        dot(0, 5)
    elif name == "info":
        oval(-7, -7, 7, 7)
        line([(0, -2), (0, 5)], w=2)
        dot(0, -4.5)
    elif name == "shield":
        poly([(0, -8), (7, -5), (7, 2), (0, 8), (-7, 2), (-7, -5)])
    elif name == "history":
        oval(-7, -7, 7, 7)
        line([(0, 0), (0, -4)])
        line([(0, 0), (4, 2)])
        poly([(-8, -4), (-4, -8), (-3, -3)], fill=color)
    elif name == "clock":
        oval(-7, -7, 7, 7)
        line([(0, 0), (0, -5)])
        line([(0, 0), (4, 1)])
    elif name == "layers":
        poly([(0, -8), (7, -4), (0, 0), (-7, -4)])
        poly([(0, -3), (7, 1), (0, 5), (-7, 1)])
        poly([(0, 2), (7, 6), (0, 9), (-7, 6)])
    elif name == "list":
        for y in (-6, 0, 6):
            dot(-7, y, 1.5)
            line([(-3, y), (8, y)])
    elif name == "edit":
        poly([(-7, 4), (-3, 8), (8, -3), (4, -7)])
        line([(-7, 4), (-8, 8)])
        line([(-3, 8), (-8, 8)])
    elif name == "document":
        poly([(-6, -8), (4, -8), (8, -4), (8, 8), (-6, 8)])
        line([(4, -8), (4, -4), (8, -4)])
        line([(-3, -1), (4, -1)])
        line([(-3, 3), (4, 3)])
    elif name == "image":
        rect(-8, -7, 8, 7)
        oval(-4.5, -4.5, -1, -1)
        line([(-8, 7), (-2, -1), (2, 3), (5, 0), (8, 7)])
    elif name == "video":
        rect(-8, -6, 8, 6)
        poly([(-2, -3), (-2, 3), (5, 0)], fill=color)
    elif name == "audio":
        oval(-6, 3, -1, 8)
        line([(-1, 5.5), (-1, -7)])
        line([(-1, -7), (5, -4)])
    elif name == "copy":
        rect(-7, -4, 3, 8)
        rect(-3, -8, 7, 4)
    elif name == "filter":
        poly([(-8, -8), (8, -8), (2, -1), (2, 7), (-2, 7), (-2, -1)])
    elif name == "refresh":
        arc(-7, -7, 7, 7, 40, 280)
        poly([(6, -6), (8, -1), (3, -3)], fill=color)
    elif name == "save":
        poly([(-7, -8), (7, -8), (7, 8), (-7, 8)])
        rect(-4, -8, 4, -2)
        rect(-4, 1, 4, 8)
    elif name == "stop":
        rect(-6, -6, 6, 6, fill=color, outline="")
    elif name == "sun":
        oval(-3.5, -3.5, 3.5, 3.5)
        for x, y in ((0, -8), (0, 8), (-8, 0), (8, 0),
                     (-5.5, -5.5), (5.5, -5.5), (-5.5, 5.5), (5.5, 5.5)):
            line([(x * 0.75, y * 0.75), (x, y)])
    elif name == "moon":
        poly([(0, -8), (-5, -6), (-7, 0), (-5, 6), (0, 8),
              (2, 6), (4, 0), (2, -6)], fill=color)
    elif name == "play":
        poly([(-5, -7), (-5, 7), (8, 0)], fill=color)
    elif name == "link":
        oval(-8, -3, -1, 3)
        oval(1, -3, 8, 3)
        line([(-2, 0), (2, 0)])
    elif name == "chevron_down":
        line([(-6, -3), (0, 3), (6, -3)], w=2)
    elif name == "chevron_right":
        line([(-3, -6), (3, 0), (-3, 6)], w=2)
    elif name == "compare":
        line([(-7, 0), (7, 0)], w=2)
        poly([(-7, 0), (-3, -3), (-3, 3)], fill=color)
        poly([(7, 0), (3, -3), (3, 3)], fill=color)
    elif name == "settings":
        oval(-3, -3, 3, 3)
        for x, y in ((0, -8), (0, 8), (-8, 0), (8, 0),
                     (-5.5, -5.5), (5.5, -5.5), (-5.5, 5.5), (5.5, 5.5)):
            line([(x * 0.6, y * 0.6), (x, y)])
    elif name == "sparkle":
        poly([(0, -8), (2, -2), (8, 0), (2, 2), (0, 8),
              (-2, 2), (-8, 0), (-2, -2)], fill=color)
    elif name == "arrow_right":
        line([(-8, 0), (6, 0)], w=2)
        poly([(6, 0), (2, -4), (2, 4)], fill=color)
    elif name == "scan":
        line([(-8, -4), (-8, -8), (-4, -8)])
        line([(4, -8), (8, -8), (8, -4)])
        line([(-8, 4), (-8, 8), (-4, 8)])
        line([(4, 8), (8, 8), (8, 4)])
        line([(-6, 0), (6, 0)])
    elif name == "more":
        dot(-6, 0)
        dot(0, 0)
        dot(6, 0)
    else:
        rect(-7, -7, 7, 7)


def put_icon(canvas, theme, name, cx, cy, size, color, tags=()):
    """图标统一入口：有图标字体时用字形，否则用矢量手绘。"""
    if theme.has_icon_font:
        canvas.create_text(cx, cy, text=ICON.get(name, name),
                           font=theme.icon(size), fill=color, anchor="center",
                           tags=tags)
    else:
        draw_icon(canvas, theme, name, cx, cy, size, color, tags)


class Icon(tk.Canvas):
    """单色小图标（替代 Label + 字形）：字体在用字形，不在用矢量；换主题重画。"""

    def __init__(self, master, theme, name, size=13, color=None, bg=None,
                 cursor=None):
        self.theme = theme
        self.name = name
        self.size = size
        self.color = color or theme.MUTED
        self._bg = bg or theme.BG
        self._iw = theme.s(max(size, 12))
        kw = {"cursor": cursor} if cursor else {}
        super().__init__(master, width=self._iw, height=self._iw, bg=self._bg,
                         highlightthickness=0, bd=0, **kw)
        self._draw()

    def _draw(self):
        self.delete("all")
        put_icon(self, self.theme, self.name, self._iw / 2, self._iw / 2 + 0.5,
                 self.size, self.color)

    def set_color(self, color):
        self.color = color
        self._draw()

    def restyle(self):
        try:
            self.configure(bg=self.master.cget("bg"))
            self._bg = self.master.cget("bg")
        except tk.TclError:
            pass
        self._draw()


def _focus_ring(canvas, theme, w, h, radius, color=None):
    """键盘焦点描边（画在内容最上层）。"""
    round_rect(canvas, 1.5, 1.5, w - 1.5, h - 1.5, radius,
               fill="", outline=color or theme.ACCENT, width=theme.s(2))



# ---------------------------------------------------------------- 绘图基础
def round_rect(canvas, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, splinesteps=24, **kw)


def ellipsize(font, text, max_width):
    if not text:
        return ""
    if font.measure(text) <= max_width:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.measure(text[:mid] + ell) <= max_width:
            lo = mid
        else:
            hi = mid - 1
def wrap_lines(font, text, max_width):
    """按像素宽把文本折成多行（中英文混排按字折），返回行列表。"""
    if not text:
        return []
    if font.measure(text) <= max_width:
        return [text]
    lines, cur = [], ""
    for ch in text:
        if font.measure(cur + ch) <= max_width or not cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _shade(hex_color, factor):
    """把 #rrggbb 按 factor 变深（<1）或变浅（>1），非法值原样返回。"""
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    except (ValueError, TypeError):
        return hex_color
    r, g, b = (max(0, min(255, round(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------- 按钮
class CanvasButton(tk.Canvas):
    """圆角按钮，带 hover / 按下 / 禁用状态。"""

    STYLES: ClassVar[dict] = palette_button_styles()

    def __init__(self, master, theme, text, command=None, kind="primary",
                 icon=None, width=None, height=38, radius=10, size=10,
                 icon_size=14, padding=18, bg=None):
        self.theme = theme
        self.kind = kind
        self.text = text
        self.icon = icon
        self.command = command
        self.radius = theme.s(radius)
        self.font = theme.font(size, "bold" if kind in ("primary", "danger") else "normal")
        self.icon_size = icon_size
        self.padding = theme.s(padding)
        self.enabled = True
        self._hover = False
        self._pressed = False
        self._press_cancelled = False
        self._key_after_id = None
        # bg 只记显式传入的值；没传时 _draw 读当前主题，避免换主题后重画用旧色
        self._bg = bg
        natural = self.font.measure(text) + self.padding * 2
        if icon:
            natural += theme.s(icon_size) + theme.s(7)
        w = theme.s(width) if width else natural
        h = theme.s(height)
        self._btn_w, self._btn_h = w, h
        super().__init__(master, width=w, height=h, bg=self._bg or theme.BG,
                         highlightthickness=0, bd=0, cursor="hand2",
                         takefocus=1)
        self._focused = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Return>", self._on_key_activate)
        self.bind("<space>", self._on_key_activate)
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self._btn_w, self._btn_h
        fill, hover, fg, outline = self.STYLES.get(self.kind, self.STYLES["secondary"])
        if not self.enabled:
            fill, fg, outline = Theme.BTN_DISABLED_BG, Theme.FAINT, None
        elif self._pressed:
            fill = _shade(fill, 0.85) if fill else _shade(hover, 0.85)
        elif self._hover:
            fill = hover
        if not fill:
            fill = self._bg or self.theme.BG
        dy = 1 if self._pressed and self.enabled else 0
        round_rect(self, 0.5, 0.5 + dy, w - 0.5, h - 0.5 + dy, self.radius,
                   fill=fill, outline=outline or fill)
        size_px = self.theme.s(self.icon_size)
        text_w = self.font.measure(self.text)
        total = text_w + (size_px + self.theme.s(7) if self.icon else 0)
        left = (w - total) / 2
        if self.icon:
            put_icon(self, self.theme, self.icon,
                     left + size_px / 2, h / 2 + 0.5 + dy, self.icon_size, fg)
            self.create_text(left + size_px + self.theme.s(7), h / 2 + 0.5 + dy,
                             text=self.text, font=self.font, fill=fg, anchor="w")
        else:
            self.create_text(w / 2, h / 2 + 0.5 + dy, text=self.text, font=self.font,
                             fill=fg, anchor="center")
        if self._focused and self.enabled and not self._pressed:
            # 主按钮底色就是强调色，描边用白色才看得见
            ring = (Theme.ON_ACCENT if self.kind in ("primary", "danger")
                    else self.theme.ACCENT)
            _focus_ring(self, self.theme, w, h, self.radius, ring)

    def _on_focus_in(self, _=None):
        self._focused = True
        self._draw()

    def _on_focus_out(self, _=None):
        self._focused = False
        self._draw()

    def _on_key_activate(self, _=None):
        if self.enabled and self.command and self._key_after_id is None:
            self._pressed = True
            self._draw()
            self._key_after_id = self.after(80, self._key_run)
        return "break"

    def _key_run(self):
        self._key_after_id = None
        try:
            self._pressed = False
            self._draw()
        except tk.TclError:
            return
        if self.enabled and self.command:
            self.command()

    def _cancel_key_run(self):
        """取消待执行的键盘激活（鼠标点按/禁用时调用，防 80ms 内双执行）。"""
        if self._key_after_id is not None:
            try:
                self.after_cancel(self._key_after_id)
            except tk.TclError:
                pass
            self._key_after_id = None

    def _on_enter(self, _=None):
        if self.enabled:
            self._hover = True
            self._draw()

    def _on_leave(self, _=None):
        self._hover = False
        if self._pressed:
            # 松开发生在按钮外时收不到 ButtonRelease，离开即复位；
            # 同时记一笔“取消”，拖回再松开也不执行，言行一致
            self._pressed = False
            self._press_cancelled = True
        self._draw()

    def _on_press(self, _=None):
        if self.enabled and self.command:
            self._cancel_key_run()
            self._pressed = True
            self._press_cancelled = False
            self.configure(cursor="arrow")
            self._draw()

    def _on_release(self, event=None):
        if not self.enabled:
            return
        self._cancel_key_run()
        self._pressed = False
        self._draw()
        self.configure(cursor="hand2")
        cancelled = self._press_cancelled
        self._press_cancelled = False
        inside = (0 <= event.x <= self._btn_w and 0 <= event.y <= self._btn_h) \
            if event else True
        if inside and not cancelled and self.command:
            self.command()

    def restyle(self):
        """换主题后重新取底色并重画（背景色是构造时缓存的）。"""
        try:
            self._bg = self.master.cget("bg")
            self.configure(bg=self._bg)
        except tk.TclError:
            pass
        self._draw()

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        self._pressed = False
        self._cancel_key_run()
        self.configure(cursor="hand2" if enabled else "arrow",
                       takefocus=1 if enabled else 0)
        self._draw()

    def set_text(self, text, width=None):
        self.text = text
        if width:
            natural = self.font.measure(text) + self.padding * 2
            if self.icon:
                natural += self.theme.s(self.icon_size) + self.theme.s(7)
            self._btn_w = self.theme.s(width) if isinstance(width, (int, float)) else natural
            self.configure(width=self._btn_w)
        self._draw()

    def set_kind(self, kind):
        self.kind = kind
        self._draw()


# ---------------------------------------------------------------- 开关
class Switch(tk.Canvas):
    def __init__(self, master, theme, value=False, command=None, bg=None):
        self.theme = theme
        self.value = bool(value)
        self.command = command
        self.w = theme.s(42)
        self.h = theme.s(24)
        self._bg = bg or theme.CARD
        super().__init__(master, width=self.w, height=self.h, bg=self._bg,
                         highlightthickness=0, bd=0, cursor="hand2",
                         takefocus=1)
        self._pos = 1.0 if self.value else 0.0
        self._anim = None
        self.enabled = True
        self._focused = False
        self.bind("<Button-1>", self._toggle)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Return>", self._on_key_toggle)
        self.bind("<space>", self._on_key_toggle)
        self._draw()

    def _toggle(self, _=None):
        if not self.enabled:
            return
        self.set(not self.value, animate=True)
        if self.command:
            self.command(self.value)

    def _on_focus_in(self, _=None):
        self._focused = True
        self._draw()

    def _on_focus_out(self, _=None):
        self._focused = False
        self._draw()

    def _on_key_toggle(self, _=None):
        if self.enabled:
            self._toggle()
        return "break"

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        self.configure(cursor="hand2" if enabled else "arrow",
                       takefocus=1 if enabled else 0)
        self._draw()

    def set(self, value, animate=False):
        target = 1.0 if value else 0.0
        if value == self.value and not animate:
            self._draw()
            return
        self.value = bool(value)
        if animate and self.winfo_exists():
            self._animate_to(target)
        else:
            self._pos = target
            self._draw()

    def _animate_to(self, target):
        if self._anim:
            try:
                self.after_cancel(self._anim)
            except Exception:
                pass
        start = self._pos
        steps = 7

        def step(i=1):
            self._pos = start + (target - start) * (i / steps)
            self._draw()
            if i < steps:
                self._anim = self.after(14, lambda: step(i + 1))
            else:
                self._anim = None

        step()

    def _draw(self):
        self.delete("all")
        w, h = self.w, self.h
        pad = self.theme.s(2)
        track_off = (Theme.TRACK_OFF if self.enabled
                     else Theme.BTN_DISABLED_BG)
        track_on = (self.theme.ACCENT if self.enabled
                    else Theme.BTN_DISABLED_BG)
        round_rect(self, 0.5, 0.5, w - 0.5, h - 0.5, h / 2,
                   fill=track_on if self.value else track_off,
                   outline=track_on if self.value else track_off)
        r = (h - pad * 2) / 2
        left = pad + r
        right = w - pad - r
        cx = left + (right - left) * self._pos
        # 关闭状态的白滑块在白天主题下会糊在浅色轨道里，给一圈描边
        knob = (Theme.KNOB_ON if self.value else Theme.KNOB_OFF)
        if not self.enabled:
            knob = Theme.FAINT
        self.create_oval(cx - r, h / 2 - r, cx + r, h / 2 + r,
                         fill=knob,
                         outline="" if self.value else Theme.BORDER,
                         width=self.theme.s(1))
        if self._focused and self.enabled:
            # 轨道开着时就是强调色，描边用正文色才看得见
            _focus_ring(self, self.theme, w, h, h / 2, self.theme.TEXT)


# ---------------------------------------------------------------- 分段控件
class Segmented(tk.Canvas):
    def __init__(self, master, theme, options, value=None, command=None,
                 height=34, size=9, padding=13, bg=None, fill=None):
        self.theme = theme
        self.options = [(o, o) if isinstance(o, str) else tuple(o) for o in options]
        self.value = value if value is not None else self.options[0][0]
        self.command = command
        self.font = theme.font(size, "normal")
        self.h = theme.s(height)
        self.pad = theme.s(3)
        self.seg_pad = theme.s(padding)
        self._bg = bg or theme.CARD
        # fill 只记显式传入的值；没传时 _draw 读当前主题，避免换主题后重画用旧色
        self._fill = fill
        self._hover_idx = None
        widths = []
        for _, label in self.options:
            widths.append(self.font.measure(label) + self.seg_pad * 2)
        self._widths = widths
        total = sum(widths) + self.pad * 2
        super().__init__(master, width=total, height=self.h, bg=self._bg,
                         highlightthickness=0, bd=0, cursor="hand2",
                         takefocus=1)
        self.enabled = True
        self._focused = False
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Left>", self._on_key_move)
        self.bind("<Right>", self._on_key_move)
        self._draw()

    def _seg_bounds(self):
        x = self.pad
        out = []
        for w in self._widths:
            out.append((x, x + w))
            x += w
        return out

    def _draw(self):
        self.delete("all")
        w = sum(self._widths) + self.pad * 2
        round_rect(self, 0.5, 0.5, w - 0.5, self.h - 0.5, self.theme.s(10),
                   fill=self._fill or self.theme.INPUT,
                   outline=self.theme.BORDER_SOFT)
        for i, ((val, label), (x1, x2)) in enumerate(zip(self.options, self._seg_bounds())):
            active = val == self.value
            if active:
                fill = (Theme.SCROLL_THUMB if self.enabled
                        else Theme.BTN_DISABLED_BG)
                round_rect(self, x1, self.pad, x2, self.h - self.pad,
                           self.theme.s(8), fill=fill, outline="")
            if not self.enabled:
                color = self.theme.FAINT
            else:
                color = self.theme.TEXT if active else (
                    self.theme.TEXT_DIM if i == self._hover_idx else self.theme.MUTED)
            self.create_text((x1 + x2) / 2, self.h / 2 + 0.5, text=label,
                             font=self.font, fill=color,
                             anchor="center", tags=(f"seg{i}",))
            self.tag_bind(f"seg{i}", "<Button-1>", lambda e, v=val: self._choose(v))
            self.tag_bind(f"seg{i}", "<Enter>", lambda e, idx=i: self._hover(idx))
            self.tag_bind(f"seg{i}", "<Leave>", lambda e: self._hover(None))
        if self._focused and self.enabled:
            _focus_ring(self, self.theme,
                        sum(self._widths) + self.pad * 2, self.h,
                        self.theme.s(10))

    def _hover(self, idx):
        if idx != self._hover_idx:
            self._hover_idx = idx
            self._draw()

    def _choose(self, value):
        if not self.enabled:
            return
        if value != self.value:
            self.value = value
            self._draw()
            if self.command:
                self.command(value)

    def _on_focus_in(self, _=None):
        self._focused = True
        self._draw()

    def _on_focus_out(self, _=None):
        self._focused = False
        self._draw()

    def _on_key_move(self, event):
        if not self.enabled:
            return "break"
        values = [v for v, _ in self.options]
        idx = values.index(self.value) if self.value in values else 0
        step = -1 if event.keysym == "Left" else 1
        self._choose(values[(idx + step) % len(values)])
        return "break"

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        self.configure(cursor="hand2" if enabled else "arrow",
                       takefocus=1 if enabled else 0)
        self._draw()

    def set_value(self, value):
        self.value = value
        self._draw()


# ---------------------------------------------------------------- 卡片容器
class Card(tk.Canvas):
    """带圆角背景的卡片，内容放进 .body。"""

    def __init__(self, master, theme, padding=18, radius=14, bg=None,
                 fill=None, outline=None, min_height=0):
        self.theme = theme
        self.padding = theme.s(padding)
        self.radius = theme.s(radius)
        self._bg = bg or theme.BG
        # fill/outline 只记显式传入的值；没传时 _draw_bg 读当前主题
        self._fill = fill
        self._outline = outline
        self._min_height = theme.s(min_height)
        super().__init__(master, bg=self._bg, highlightthickness=0, bd=0,
                         height=self._min_height or theme.s(80))
        self.body = tk.Frame(self, bg=self._fill or theme.CARD)
        self._win = self.create_window(self.padding, self.padding,
                                       window=self.body, anchor="nw")
        self.bind("<Configure>", self._on_canvas)
        self.body.bind("<Configure>", self._on_body)
        self._pending = None

    def _on_canvas(self, event=None):
        w = self.winfo_width()
        self.coords(self._win, self.padding, self.padding)
        self.itemconfigure(self._win, width=max(1, w - self.padding * 2))
        self._draw_bg()

    def _on_body(self, _=None):
        if self._pending:
            return
        self._pending = self.after_idle(self._fit)

    def _fit(self):
        self._pending = None
        if not self.winfo_exists():
            return
        need = self.body.winfo_reqheight() + self.padding * 2
        need = max(need, self._min_height)
        if abs(self.winfo_height() - need) > 1:
            self.configure(height=need)
        self._draw_bg()

    def _draw_bg(self):
        self.delete("bg")
        w = max(self.winfo_width(), 2)
        h = max(self.winfo_height(), 2)
        round_rect(self, 0.5, 0.5, w - 0.5, h - 0.5, self.radius,
                   fill=self._fill or self.theme.CARD,
                   outline=self._outline or self.theme.BORDER_SOFT,
                   tags=("bg",))
        self.tag_lower("bg")


# ---------------------------------------------------------------- 滚动区
def _wheel_dispatch(event):
    """滚轮统一入口：从指针所在控件向上找，滚动最近的滚动区。"""
    widget = event.widget
    while isinstance(widget, tk.Misc):
        if isinstance(widget, ScrollArea):
            return widget._handle_wheel(event) if widget.winfo_exists() else None
        widget = getattr(widget, "master", None)
    return None


class ScrollArea(tk.Frame):
    def __init__(self, master, theme, bg=None, bar_width=10):
        self.theme = theme
        self._bg = bg or theme.BG
        super().__init__(master, bg=self._bg)
        self.canvas = tk.Canvas(self, bg=self._bg, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar = tk.Canvas(self, width=theme.s(bar_width), bg=self._bg,
                             highlightthickness=0, bd=0)
        self.bar.pack(side="right", fill="y")
        self.inner = tk.Frame(self.canvas, bg=self._bg)
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._thumb = None
        self._drag = None
        self._stick_top = True

        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.bar.bind("<ButtonPress-1>", self._bar_press)
        self.bar.bind("<B1-Motion>", self._bar_drag)
        self.bar.bind("<ButtonRelease-1>", self._bar_release)
        self.canvas.configure(yscrollcommand=self._on_scroll)
        top = self.winfo_toplevel()
        if not getattr(top, "_wheel_dispatcher_on", False):
            top.bind("<MouseWheel>", _wheel_dispatch, add="+")
            top._wheel_dispatcher_on = True

    # ---- 事件
    def _on_inner(self, _=None):
        at_top = self.canvas.yview()[0] <= 0.0001
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self._stick_top and at_top:
            self.canvas.yview_moveto(0)
        self._draw_bar()

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.win, width=max(1, event.width))
        self._draw_bar()

    def _handle_wheel(self, event):
        """只滚动指针所在的那个滚动区。"""
        try:
            first, last = self.canvas.yview()
        except Exception:
            return None
        if last - first >= 0.999:
            return None
        if event.delta > 0 and first <= 0.0:
            return None
        if event.delta < 0 and last >= 1.0:
            return None
        self._stick_top = False
        self.canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")
        self._draw_bar()
        return "break"

    def _on_scroll(self, first, last):
        self._draw_bar()

    def _bar_geom(self):
        h = max(self.bar.winfo_height(), 1)
        try:
            first, last = self.canvas.yview()
        except Exception:
            first, last = 0.0, 1.0
        if last - first >= 0.999:
            return None
        top = first * h
        size = max(self.theme.s(34), (last - first) * h)
        top = min(top, h - size)
        return top, size

    def _draw_bar(self):
        self.bar.delete("all")
        geom = self._bar_geom()
        if not geom:
            return
        top, size = geom
        w = self.bar.winfo_width() or self.theme.s(10)
        bw = self.theme.s(4)
        x = (w - bw) / 2
        round_rect(self.bar, x, top + 2, x + bw, top + size - 2, bw / 2,
                   fill=Theme.SCROLL_THUMB_HOVER, outline="")

    def _bar_press(self, event):
        geom = self._bar_geom()
        if not geom:
            return
        top, size = geom
        if top <= event.y <= top + size:
            self._drag = (event.y, self.canvas.yview()[0])
            self._stick_top = False
        else:
            self._stick_top = False
            self.canvas.yview_scroll(-1 if event.y < top else 1, "pages")

    def _bar_drag(self, event):
        if not self._drag:
            return
        start_y, start_first = self._drag
        h = max(self.bar.winfo_height(), 1)
        span = self.canvas.yview()[1] - self.canvas.yview()[0]
        delta = (event.y - start_y) / h
        self.canvas.yview_moveto(max(0.0, min(1.0 - span, start_first + delta)))

    def _bar_release(self, _=None):
        self._drag = None

    # ---- 对外
    def scroll_to_top(self):
        self._stick_top = True
        self.canvas.yview_moveto(0)
        self._draw_bar()

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.scroll_to_top()


# ---------------------------------------------------------------- 小标签
class Chip(tk.Canvas):
    def __init__(self, master, theme, text, icon=None, color=None, value=None,
                 bg=None, size=9, height=28):
        self.theme = theme
        self.text = text
        self.value = value
        self.icon = icon
        self.color = color or theme.TEXT_DIM
        self.font = theme.font(size, "normal")
        self.value_font = theme.font(size, "bold")
        self._bg = bg or theme.BG
        self.h = theme.s(height)
        self.pad = theme.s(10)
        self._measure()
        super().__init__(master, width=self.w, height=self.h, bg=self._bg,
                         highlightthickness=0, bd=0)
        self._draw()

    def _measure(self):
        tw = self.font.measure(self.text)
        vw = self.value_font.measure(self.value) if self.value else 0
        iw = self.theme.s(13) + self.theme.s(6) if self.icon else 0
        gap = self.theme.s(6) if self.value and self.text else 0
        self.w = self.pad * 2 + tw + vw + iw + gap

    def _draw(self):
        self.delete("all")
        round_rect(self, 0.5, 0.5, self.w - 0.5, self.h - 0.5, self.theme.s(9),
                   fill=Theme.CARD_INSET, outline=self.theme.BORDER_SOFT)
        x = self.pad
        color = self.color
        if self.icon:
            put_icon(self, self.theme, self.icon,
                     x + self.theme.s(6.5), self.h / 2 + 0.5, 13, color)
            x += self.theme.s(13) + self.theme.s(6)
        if self.text:
            self.create_text(x, self.h / 2 + 0.5, text=self.text,
                             font=self.font, fill=self.theme.TEXT_DIM, anchor="w")
            x += self.font.measure(self.text)
        if self.value:
            x += self.theme.s(6)
            self.create_text(x, self.h / 2 + 0.5, text=self.value,
                             font=self.value_font, fill=self.color, anchor="w")

    def set_value(self, value, text=None):
        self.value = value
        if text is not None:
            self.text = text
        self._measure()
        self.configure(width=self.w)
        self._draw()


# ---------------------------------------------------------------- 对话框
def _center_on(parent, top, w, h, offset=0.34):
    parent = parent.winfo_toplevel()
    px, py = parent.winfo_rootx(), parent.winfo_rooty()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    x = px + (pw - w) // 2
    y = py + max(40, int((ph - h) * offset))
    sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
    x = max(20, min(sw - w - 20, x))
    y = max(20, min(sh - h - 40, y))
    top.geometry(f"{w}x{h}+{x}+{y}")


def _make_dim_overlay(host, theme):
    """在窗口上盖一层压暗的主界面截图（Tk 画布不支持真透明）。"""
    if ImageGrab is None or ImageEnhance is None or ImageTk is None:
        return None
    try:
        host.update_idletasks()
        x, y = host.winfo_rootx(), host.winfo_rooty()
        w, h = host.winfo_width(), host.winfo_height()
        if w <= 1 or h <= 1:
            return None
        shot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        dark = ImageEnhance.Brightness(shot.convert("RGB")).enhance(0.42)
        overlay = tk.Canvas(host, width=w, height=h, bg=theme.BG,
                           highlightthickness=0, bd=0)
        photo = ImageTk.PhotoImage(dark)
        overlay._dim_photo = photo
        overlay.create_image(0, 0, image=photo, anchor="nw")
        overlay.place(x=0, y=0, width=w, height=h)
        overlay.tk.call("raise", overlay._w)
        return overlay
    except Exception:
        return None


def confirm(parent, theme, title, message, ok_text="确定", cancel_text="取消",
            danger=False, extra_text=None, note=None, icon="trash",
            subtitle="请确认以下操作"):
    """深色确认框。返回 'ok' / 'extra' / None。"""
    result = {"value": None}
    host = parent.winfo_toplevel()
    overlay = _make_dim_overlay(host, theme)
    if overlay is not None:
        overlay.bind("<Button-1>", lambda e: top.focus_force())
    top = tk.Toplevel(parent)
    top.withdraw()
    top.overrideredirect(True)
    top.configure(bg=theme.BORDER)
    top.transient(parent.winfo_toplevel())
    try:
        top.attributes("-topmost", True)
    except Exception:
        pass

    shell = tk.Frame(top, bg=theme.BORDER)
    shell.pack(fill="both", expand=True)
    card = tk.Frame(shell, bg=theme.CARD)
    card.pack(fill="both", expand=True, padx=1, pady=1)

    inner = tk.Frame(card, bg=theme.CARD)
    inner.pack(fill="both", expand=True, padx=theme.s(24), pady=theme.s(20))
    tk.Frame(inner, bg=theme.CARD, width=theme.s(400), height=1).pack()

    head = tk.Frame(inner, bg=theme.CARD)
    head.pack(fill="x")
    bubble = tk.Canvas(head, width=theme.s(40), height=theme.s(40), bg=theme.CARD,
                       highlightthickness=0, bd=0)
    bubble.pack(side="left", pady=(1, 0))
    round_rect(bubble, 1, 1, theme.s(39), theme.s(39), theme.s(12),
               fill=theme.RED_DIM if danger else Theme.CHIP_BG, outline="")
    put_icon(bubble, theme, icon, theme.s(20), theme.s(20) + 1, 18,
             theme.RED if danger else theme.ACCENT)

    title_box = tk.Frame(head, bg=theme.CARD)
    title_box.pack(side="left", fill="x", expand=True, padx=(theme.s(12), 0))
    tk.Label(title_box, text=title, bg=theme.CARD, fg=theme.TEXT,
             font=theme.font(13, "bold"), anchor="w").pack(fill="x", pady=(theme.s(2), 0))
    if subtitle:
        tk.Label(title_box, text=subtitle, bg=theme.CARD, fg=theme.MUTED,
                 font=theme.font(9), anchor="w").pack(fill="x", pady=(theme.s(3), 0))

    close = tk.Canvas(head, width=theme.s(26), height=theme.s(26), bg=theme.CARD,
                      highlightthickness=0, bd=0, cursor="hand2")
    close.pack(side="right", anchor="n")
    put_icon(close, theme, "close", theme.s(13), theme.s(13), 12,
             theme.MUTED)
    close.bind("<Button-1>", lambda e: _close())

    body = tk.Label(inner, text=message, bg=theme.CARD, fg=theme.TEXT_DIM,
                    font=theme.font(10), justify="left", anchor="w",
                    wraplength=theme.s(396))
    body.pack(fill="x", pady=(theme.s(16), 0))

    if note:
        note_box = tk.Frame(inner, bg=theme.CARD)
        note_box.pack(fill="x", pady=(theme.s(10), 0))
        tk.Label(note_box, text=note, bg=theme.CARD, fg=theme.AMBER,
                 font=theme.font(9), justify="left", anchor="w",
                 wraplength=theme.s(396)).pack(fill="x")

    buttons = tk.Frame(inner, bg=theme.CARD)
    buttons.pack(fill="x", pady=(theme.s(22), 0))
    ok = CanvasButton(buttons, theme, ok_text, command=lambda: _close("ok"),
                      kind="danger" if danger else "primary", height=36,
                      size=9, padding=16, bg=theme.CARD)
    ok.pack(side="right")
    if cancel_text:
        cancel = CanvasButton(buttons, theme, cancel_text, command=lambda: _close(),
                              kind="ghost", height=36, size=9, padding=14,
                              bg=theme.CARD)
        cancel.pack(side="right", padx=(0, theme.s(6)))
    if extra_text:
        extra = CanvasButton(buttons, theme, extra_text,
                             command=lambda: _close("extra"), kind="ghost",
                             height=36, size=9, padding=12, bg=theme.CARD)
        extra.pack(side="left")

    def _close(value=None):
        result["value"] = value
        try:
            top.grab_release()
        except Exception:
            pass
        try:
            overlay.destroy()
        except Exception:
            pass
        top.destroy()

    def _drag_start(event):
        top._drag_xy = (event.x_root - top.winfo_x(), event.y_root - top.winfo_y())

    def _drag_move(event):
        if not hasattr(top, "_drag_xy"):
            return
        dx, dy = top._drag_xy
        top.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    for w in (head, title_box, body):
        w.bind("<ButtonPress-1>", _drag_start)
        w.bind("<B1-Motion>", _drag_move)

    def _cleanup_overlay(event):
        if event.widget is not top:
            return
        try:
            overlay.destroy()
        except Exception:
            pass

    top.bind("<Escape>", lambda e: _close())
    if danger:
        # 危险框回车不再确认，必须点红色按钮
        top.bind("<Return>", lambda e: _close())
    else:
        top.bind("<Return>", lambda e: _close("ok"))
    top.bind("<Destroy>", _cleanup_overlay, add="+")

    top.update_idletasks()
    w = max(theme.s(460), top.winfo_reqwidth())
    h = top.winfo_reqheight()
    _center_on(parent, top, w, h)
    if overlay is not None:
        try:
            overlay.tk.call("raise", overlay._w)
        except Exception:
            pass
    top.deiconify()
    top.lift()
    top.focus_force()
    try:
        top.grab_set()
    except Exception:
        pass
    top.wait_window()
    return result["value"]


def alert(parent, theme, title, message, ok_text="知道了", icon="info"):
    return confirm(parent, theme, title, message, ok_text=ok_text,
                   cancel_text=None, icon=icon, subtitle="")


def prompt(parent, theme, title, message, value="", ok_text="保存",
           cancel_text="取消", icon="save", placeholder=""):
    """深色单行输入框。返回输入的文字；取消返回 None。"""
    result = {"value": None}
    host = parent.winfo_toplevel()
    overlay = _make_dim_overlay(host, theme)
    if overlay is not None:
        overlay.bind("<Button-1>", lambda e: top.focus_force())
    top = tk.Toplevel(parent)
    top.withdraw()
    top.overrideredirect(True)
    top.configure(bg=theme.BORDER)
    top.transient(parent.winfo_toplevel())
    try:
        top.attributes("-topmost", True)
    except Exception:
        pass

    shell = tk.Frame(top, bg=theme.BORDER)
    shell.pack(fill="both", expand=True)
    card = tk.Frame(shell, bg=theme.CARD)
    card.pack(fill="both", expand=True, padx=1, pady=1)
    inner = tk.Frame(card, bg=theme.CARD)
    inner.pack(fill="both", expand=True, padx=theme.s(24), pady=theme.s(20))
    tk.Frame(inner, bg=theme.CARD, width=theme.s(400), height=1).pack()

    head = tk.Frame(inner, bg=theme.CARD)
    head.pack(fill="x")
    bubble = tk.Canvas(head, width=theme.s(40), height=theme.s(40),
                       bg=theme.CARD, highlightthickness=0, bd=0)
    bubble.pack(side="left", pady=(1, 0))
    round_rect(bubble, 1, 1, theme.s(39), theme.s(39), theme.s(12),
               fill=Theme.CHIP_BG, outline="")
    put_icon(bubble, theme, icon, theme.s(20), theme.s(20) + 1, 18,
             theme.ACCENT)
    box = tk.Frame(head, bg=theme.CARD)
    box.pack(side="left", fill="x", expand=True, padx=(theme.s(12), 0))
    tk.Label(box, text=title, bg=theme.CARD, fg=theme.TEXT,
             font=theme.font(13, "bold"), anchor="w").pack(fill="x",
                                                          pady=(theme.s(2), 0))
    tk.Label(box, text=message, bg=theme.CARD, fg=theme.MUTED,
             font=theme.font(9), anchor="w").pack(fill="x", pady=(theme.s(3), 0))
    close = tk.Canvas(head, width=theme.s(26), height=theme.s(26),
                      bg=theme.CARD, highlightthickness=0, bd=0, cursor="hand2")
    close.pack(side="right", anchor="n")
    put_icon(close, theme, "close", theme.s(13), theme.s(13), 12,
             theme.MUTED)
    close.bind("<Button-1>", lambda e: _close())

    var = tk.StringVar(value=value)
    field = tk.Frame(inner, bg=theme.INPUT, highlightthickness=1,
                     highlightbackground=theme.BORDER,
                     highlightcolor=theme.ACCENT)
    field.pack(fill="x", pady=(theme.s(18), 0))
    entry = tk.Entry(field, textvariable=var, bg=theme.INPUT, fg=theme.TEXT,
                     insertbackground=theme.TEXT, relief="flat", bd=0,
                     highlightthickness=0, font=theme.font(10))
    entry.pack(fill="x", padx=theme.s(10), ipady=theme.s(6))
    if placeholder:
        ph = tk.Label(field, text=placeholder, bg=theme.INPUT, fg=theme.FAINT,
                      font=theme.font(9), cursor="xterm")
        ph.place(x=theme.s(10), rely=0.5, anchor="w")
        ph.bind("<Button-1>", lambda e: entry.focus_set())

        def sync(*_):
            if var.get():
                ph.place_forget()
            elif entry.focus_get() is not entry:
                ph.place(x=theme.s(10), rely=0.5, anchor="w")
        var.trace_add("write", sync)
        entry.bind("<FocusIn>", lambda e: ph.place_forget(), add="+")
        entry.bind("<FocusOut>", lambda e: sync(), add="+")

    buttons = tk.Frame(inner, bg=theme.CARD)
    buttons.pack(fill="x", pady=(theme.s(20), 0))
    ok = CanvasButton(buttons, theme, ok_text, command=lambda: _close(True),
                      kind="primary", height=36, size=9, padding=16,
                      bg=theme.CARD)
    ok.pack(side="right")
    if cancel_text:
        cancel = CanvasButton(buttons, theme, cancel_text,
                              command=lambda: _close(), kind="ghost",
                              height=36, size=9, padding=14, bg=theme.CARD)
        cancel.pack(side="right", padx=(0, theme.s(6)))

    def _close(submit=False):
        if submit:
            text = var.get().strip()
            if text:
                result["value"] = text
        try:
            top.grab_release()
        except Exception:
            pass
        try:
            overlay.destroy()
        except Exception:
            pass
        top.destroy()

    def _drag_start(event):
        top._drag_xy = (event.x_root - top.winfo_x(),
                        event.y_root - top.winfo_y())

    def _drag_move(event):
        if hasattr(top, "_drag_xy"):
            dx, dy = top._drag_xy
            top.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    for w in (head, box):
        w.bind("<ButtonPress-1>", _drag_start)
        w.bind("<B1-Motion>", _drag_move)
    top.bind("<Escape>", lambda e: _close())
    top.bind("<Return>", lambda e: _close(True))
    top.bind("<Destroy>", lambda e: overlay.destroy()
             if e.widget is top and overlay is not None else None, add="+")

    top.update_idletasks()
    w = max(theme.s(460), top.winfo_reqwidth())
    h = top.winfo_reqheight()
    _center_on(parent, top, w, h)
    if overlay is not None:
        try:
            overlay.tk.call("raise", overlay._w)
        except Exception:
            pass
    top.deiconify()
    top.lift()
    entry.focus_set()
    entry.icursor("end")
    entry.select_range(0, "end")
    try:
        top.grab_set()
    except Exception:
        pass
    top.wait_window()
    return result["value"]


# ---------------------------------------------------------------- 弹出菜单
class PopupMenu(tk.Toplevel):
    def __init__(self, parent, theme, items, x=None, y=None):
        super().__init__(parent)
        self.theme = theme
        self.withdraw()
        self.overrideredirect(True)
        self.configure(bg=theme.BORDER)
        self.transient(parent.winfo_toplevel())
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass
        self.items = items
        shell = tk.Frame(self, bg=theme.BORDER)
        shell.pack(fill="both", expand=True)
        body = tk.Frame(shell, bg=theme.CARD)
        body.pack(fill="both", expand=True, padx=1, pady=1)

        pad = theme.s(4)
        for item in items:
            if item is None:
                tk.Frame(body, bg=theme.BORDER_SOFT, height=1).pack(fill="x", padx=pad, pady=pad)
                continue
            label, icon, command = (item + (None,))[:3] if isinstance(item, tuple) else (item, None, None)
            row = tk.Frame(body, bg=theme.CARD, cursor="hand2")
            row.pack(fill="x", padx=pad, pady=1)
            if icon:
                ic = Icon(row, theme, icon, size=13, color=theme.MUTED,
                          bg=theme.CARD)
                ic.pack(side="left", padx=(theme.s(6), 0), pady=theme.s(6))
            txt = tk.Label(row, text=label, bg=theme.CARD,
                           fg=theme.RED if label.startswith("删除") else theme.TEXT,
                           font=theme.font(9), anchor="w")
            txt.pack(side="left", fill="x", expand=True,
                     padx=(theme.s(6) if not icon else 0, theme.s(18)),
                     pady=theme.s(7))
            for w in (row, txt) + ((ic,) if icon else ()):
                w.bind("<Enter>", lambda e, r=row: self._hover(r, True))
                w.bind("<Leave>", lambda e, r=row: self._hover(r, False))
                w.bind("<ButtonRelease-1>", lambda e, c=command: self._run(c))

        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        if x is None or y is None:
            x = parent.winfo_pointerx() + 2
            y = parent.winfo_pointery() + 2
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = max(8, min(sw - w - 8, x))
        y = max(8, min(sh - h - 8, y))
        self.geometry(f"+{x}+{y}")
        self.deiconify()
        self.lift()
        self.focus_force()
        try:
            self.grab_set()
        except Exception:
            pass
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Button-1>", self._outside_click)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _outside_click(self, event):
        """点到菜单外面就关掉（配合 grab，外部点击也会送到这里）。"""
        if not (0 <= event.x <= self.winfo_width()
                and 0 <= event.y <= self.winfo_height()):
            self.destroy()

    def _on_destroy(self, event):
        if event.widget is not self:
            return
        try:
            self.grab_release()
        except Exception:
            pass

    def _hover(self, row, on):
        color = self.theme.CARD_HOVER if on else self.theme.CARD
        row.configure(bg=color)
        for child in row.winfo_children():
            child.configure(bg=color)

    def _run(self, command):
        self.destroy()
        if command:
            command()


# ---------------------------------------------------------------- 标题栏
def enable_dark_titlebar(root, dark=True):
    """让系统标题栏跟随主题；白天主题交还给系统默认的浅色边框。"""
    if os.name != "nt":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
        if not dark:
            return
        caption = ctypes.c_int(0x00120D0B)   # COLORREF: BBGGRR of #0B0D12
        text = ctypes.c_int(0x00F5EDE9)
        border = ctypes.c_int(0x00120D0B)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption), 4)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 36, ctypes.byref(text), 4)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(border), 4)
    except Exception:
        pass


# ---------------------------------------------------------------- LOGO
_LOGO_CACHE = {}


def make_logo(theme, size=40):
    key = (size, theme.scale)
    if key in _LOGO_CACHE:
        return _LOGO_CACHE[key]
    if Image is None or ImageTk is None:
        return None
    px = theme.s(size)
    S = px * 4
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gradient = Image.new("RGBA", (S, S))
    top = (108, 140, 255)
    bottom = (167, 139, 250)
    gd = ImageDraw.Draw(gradient)
    for y in range(S):
        t = y / max(1, S - 1)
        gd.line([(0, y), (S, y)], fill=(
            int(top[0] + (bottom[0] - top[0]) * t),
            int(top[1] + (bottom[1] - top[1]) * t),
            int(top[2] + (bottom[2] - top[2]) * t), 255))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=S * 0.28, fill=255)
    img.paste(gradient, (0, 0), mask)

    d = ImageDraw.Draw(img)
    lw = max(2, int(S * 0.045))
    # 后面那张卡片
    d.rounded_rectangle((S * 0.24, S * 0.28, S * 0.60, S * 0.70),
                        radius=S * 0.085, outline=(255, 255, 255, 170), width=lw)
    # 前面那张卡片
    d.rounded_rectangle((S * 0.40, S * 0.38, S * 0.76, S * 0.80),
                        radius=S * 0.085, fill=(255, 255, 255, 42),
                        outline=(255, 255, 255, 240), width=lw)
    # 对勾
    d.line((S * 0.50, S * 0.60, S * 0.57, S * 0.67), fill=(255, 255, 255, 255),
           width=lw, joint="curve")
    d.line((S * 0.57, S * 0.67, S * 0.69, S * 0.51), fill=(255, 255, 255, 255),
           width=lw, joint="curve")

    img = img.resize((px, px), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    _LOGO_CACHE[key] = photo
    return photo
