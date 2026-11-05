"""Dedup · 重复文件清理器 —— 图形界面。

运行：双击 run_gui.bat，或 python dedup_gui.py
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import threading
import tkinter as tk
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

try:
    from PIL import Image, ImageChops, ImageGrab, ImageOps, ImageTk
except Exception:  # pragma: no cover
    Image = None
    ImageChops = None
    ImageGrab = None
    ImageTk = None

try:
    import dedup_core as core
    import dedup_tools as tools
    from ui_kit import (
        ICON,
        PALETTES,
        apply_palette,
        remap_tree,
        KIND_AUDIO,
        KIND_EXACT,
        KIND_NAME,
        KIND_PHOTO,
        KIND_VIDEO,
        CanvasButton,
        Card,
        Chip,
        PopupMenu,
        ScrollArea,
        Segmented,
        Switch,
        Theme,
        alert,
        confirm,
        ellipsize,
        enable_dark_titlebar,
        make_logo,
        prompt,
        round_rect,
    )
except Exception as _exc:  # pragma: no cover
    import traceback as _tb
    try:
        Path(__file__).resolve().with_name("dedup_error.log").write_text(
            _tb.format_exc(), encoding="utf-8")
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, f"启动失败：{_exc}\n\n请查看 dedup_error.log", "Dedup", 0x10)
    except Exception:
        pass
    raise

APP_NAME = "Dedup"
APP_CN = "重复文件清理器"
VERSION = "2.5"
SETTINGS_PATH = core.app_dir() / "dedup_settings.json"
LEGACY_SETTINGS = Path(__file__).resolve().with_name("dedup_settings.json")

DEFAULTS = {
    "last_dir": "",
    "strictness": "standard",
    "min_size": 0,
    "exclude": "",
    "recursive": True,
    "exact": True,
    "photos": True,
    "videos": True,
    "audio": True,
    "names": True,
    "names_fuzzy": False,
    "protect": "",
    "cache": True,
    "keep_rule": "best",
    "sort_key": "wasted",
    "theme": "dark",
    "scan_presets": {},
    "geometry": "",
    # 整理工具
    "org_dir": "",
    "org_mode": "type",
    "org_granularity": "month",
    "org_exif": True,
    "org_recursive": True,
    "org_two_level": True,
    "org_action": "move",
    # 重命名工具
    "ren_dir": "",
    "ren_recursive": False,
    "ren_find": "",
    "ren_replace": "",
    "ren_use_replace": False,
    "ren_icase": True,
    "ren_prefix": "",
    "ren_suffix": "",
    "ren_use_template": False,
    "ren_template": "{原名}_{n}",
    "ren_start": "1",
    "ren_width": 2,
    "ren_case": "none",
}


# ---------------------------------------------------------------- 设置
def load_settings():
    data = dict(DEFAULTS)
    if not SETTINGS_PATH.exists() and LEGACY_SETTINGS.exists():
        try:
            SETTINGS_PATH.write_text(
                LEGACY_SETTINGS.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data.update({k: raw[k] for k in DEFAULTS if k in raw})
    except Exception:
        pass
    return data


def save_settings(data):
    try:
        SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------- 小工具
def reveal(path):
    """在资源管理器中定位文件。

    用列表参数调用，不经过 cmd.exe，文件名里的引号和 & 不会变成命令。
    """
    path = str(path)
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer", f"/select,{path}"])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except Exception:
        try:
            os.startfile(os.path.dirname(path))
        except Exception:
            pass


def open_path(path):
    try:
        os.startfile(str(path))
    except Exception:
        pass


def short_root(path):
    """结果页副标题用短目录名，避免长路径把右侧按钮挤扁。"""
    try:
        p = Path(path)
        return p.name or str(p)
    except Exception:
        return str(path)


# ---------------------------------------------------------------- 侧栏导航
class NavItem(tk.Canvas):
    def __init__(self, master, theme, label, icon, width, command):
        self.theme = theme
        self.label = label
        self.icon = icon
        self.command = command
        self.w = width
        self.h = theme.s(40)
        self.active = False
        self.hover = False
        self.badge = 0
        super().__init__(master, width=width, height=self.h, bg=theme.SIDEBAR,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", lambda e: self.command())
        self._draw()

    def _on_enter(self, _=None):
        self.hover = True
        self._draw()

    def _on_leave(self, _=None):
        self.hover = False
        self._draw()

    def restyle(self):
        try:
            self.configure(bg=self.master.cget("bg"))
        except tk.TclError:
            pass
        self._draw()

    def _draw(self):
        t = self.theme
        self.delete("all")
        w, h = self.w, self.h
        if self.active:
            round_rect(self, 0, 1, w, h - 1, t.s(9), fill=t.NAV_HOVER, outline="")
            round_rect(self, 0, h / 2 - t.s(8), t.s(3), h / 2 + t.s(8), t.s(1.5),
                       fill=t.ACCENT, outline="")
        elif self.hover:
            round_rect(self, 0, 1, w, h - 1, t.s(9), fill=t.NAV_ACTIVE, outline="")
        color = t.TEXT if self.active else (t.TEXT_DIM if self.hover else t.MUTED)
        self.create_text(t.s(18), h / 2 + 0.5, text=ICON.get(self.icon, self.icon),
                         font=t.icon(15), fill=t.ACCENT if self.active else color,
                         anchor="center")
        self.create_text(t.s(38), h / 2 + 0.5, text=self.label, font=t.font(9, "bold" if self.active else "normal"),
                         fill=color, anchor="w")
        if self.badge:
            text = str(self.badge) if self.badge < 1000 else "999+"
            f = t.font(8, "bold")
            bw = f.measure(text) + t.s(14)
            x2 = w - t.s(10)
            x1 = x2 - bw
            round_rect(self, x1, h / 2 - t.s(9), x2, h / 2 + t.s(9), t.s(9),
                       fill=t.ACCENT_DIM, outline="")
            self.create_text((x1 + x2) / 2, h / 2 + 0.5, text=text, font=f,
                             fill=t.ACCENT, anchor="center")

    def set_active(self, value):
        self.active = bool(value)
        self._draw()

    def set_badge(self, n):
        if n != self.badge:
            self.badge = int(n or 0)
            self._draw()


# ---------------------------------------------------------------- 搜索框
class SearchBox(tk.Frame):
    def __init__(self, master, theme, width=210, on_change=None):
        self.theme = theme
        self.on_change = on_change
        self._job = None
        super().__init__(master, bg=theme.INPUT, width=theme.s(width),
                         height=theme.s(34), highlightthickness=1,
                         highlightbackground=theme.BORDER,
                         highlightcolor=theme.ACCENT)
        self.pack_propagate(False)
        self.var = tk.StringVar()
        icon = tk.Label(self, text=ICON["search"], bg=theme.INPUT, fg=theme.MUTED,
                        font=theme.icon(12))
        icon.pack(side="left", padx=(theme.s(10), theme.s(4)))
        self.entry = tk.Entry(self, textvariable=self.var, bg=theme.INPUT,
                              fg=theme.TEXT, insertbackground=theme.TEXT,
                              relief="flat", bd=0, highlightthickness=0,
                              font=theme.font(9))
        self.entry.pack(side="left", fill="both", expand=True, pady=theme.s(5))
        self.clear = tk.Label(self, text=ICON["close"], bg=theme.INPUT,
                              fg=theme.MUTED, font=theme.icon(10), cursor="hand2")
        self.placeholder = tk.Label(self, text="搜索文件名", bg=theme.INPUT,
                                    fg=theme.FAINT, font=theme.font(9))
        self.placeholder.place(x=theme.s(30), rely=0.5, anchor="w")
        self.placeholder.bind("<Button-1>", lambda e: self.entry.focus_set())
        self.entry.bind("<FocusIn>", lambda e: self._update_placeholder())
        self.entry.bind("<FocusOut>", lambda e: self._update_placeholder())
        self.clear.bind("<Button-1>", self._clear)
        self.var.trace_add("write", lambda *a: self._changed())

    def _update_placeholder(self):
        show = not self.var.get() and self.focus_get() is not self.entry
        if show:
            self.placeholder.place(x=self.theme.s(30), rely=0.5, anchor="w")
        else:
            self.placeholder.place_forget()
        if self.var.get():
            self.clear.pack(side="right", padx=(0, self.theme.s(8)))
        else:
            self.clear.pack_forget()

    def _clear(self, _=None):
        self.var.set("")
        self.entry.focus_set()

    def _changed(self):
        self._update_placeholder()
        if self._job:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
        self._job = self.after(240, self._fire)

    def _fire(self):
        self._job = None
        if self.on_change:
            self.on_change(self.var.get().strip())


# ---------------------------------------------------------------- 缩略图服务
class ThumbnailService:
    def __init__(self, app):
        self.app = app
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="thumb")
        self.cache = {}
        self.failed = set()
        self.pending = set()
        self.waiters = {}

    def request(self, path, mtime, box, callback):
        key = f"{path}|{int(mtime)}|{box[0]}x{box[1]}"
        if key in self.cache:
            callback(self.cache[key])
            return
        if key in self.failed:
            callback(None)
            return
        self.waiters.setdefault(key, []).append(callback)
        if key in self.pending:
            return
        self.pending.add(key)
        self.executor.submit(self._work, key, str(path), box)

    def _work(self, key, path, box):
        try:
            pil = core.make_thumbnail(path, box)
        except Exception:
            pil = None
        try:
            self.app.post(("thumb", (key, pil)))
        except Exception:
            pass

    def deliver(self, key, pil):
        self.pending.discard(key)
        callbacks = self.waiters.pop(key, [])
        if pil is None:
            self.failed.add(key)
            for cb in callbacks:
                try:
                    cb(None)
                except Exception:
                    pass
            return
        if ImageTk is None:
            return
        photo = ImageTk.PhotoImage(pil)
        self.cache[key] = photo
        if len(self.cache) > 1500:
            for k in list(self.cache.keys())[:250]:
                self.cache.pop(k, None)
        for cb in callbacks:
            try:
                cb(photo)
            except Exception:
                pass

    def stop(self):
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


# ---------------------------------------------------------------- 应用
class App(tk.Tk):
    def __init__(self, initial_dir=None, start_page="scan"):
        super().__init__()
        self.settings = load_settings()
        self.theme_name = self.settings.get("theme")
        if self.theme_name not in PALETTES:
            self.theme_name = "dark"
        apply_palette(self.theme_name)   # 必须在建任何控件之前定下配色
        self.theme = Theme(self)
        t = self.theme
        self.title(f"{APP_NAME} · {APP_CN}")
        self.configure(bg=t.BG)
        self.minsize(t.s(1020), t.s(680))
        geo = self._safe_geometry(self.settings.get("geometry") or "")
        self.geometry(geo)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.scanning = False
        self.result = None
        self.thumb_service = ThumbnailService(self)
        self._toast = None
        self._closed = False

        logo = make_logo(t, 64)
        if logo:
            self._logo_ref = logo
            try:
                self.iconphoto(True, logo)
            except Exception:
                pass

        self._build_shell()
        self._build_pages()
        self.show_page(start_page)

        self.scan_page.set_path(initial_dir or self.settings.get("last_dir", ""))

        self.log("界面已启动，引擎就绪", "ok")
        self.log(f"依赖：numpy {self._dep('numpy')} · Pillow {self._dep('PIL')} · "
                 f"OpenCV {self._dep('cv2')}", "info")

        self.after(80, self._poll)
        self.after(120, lambda: enable_dark_titlebar(self, dark=(self.theme_name == "dark")))
        self.bind("<Control-o>", lambda e: self.show_page("scan"))
        self.bind("<Control-Key-1>", lambda e: self.show_page("scan"))
        self.bind("<Control-Key-2>", lambda e: self.show_page("results"))
        self.bind("<Control-Key-3>", lambda e: self.show_page("logs"))
        self.bind("<Control-Key-4>", lambda e: self.show_page("organize"))
        self.bind("<Control-Key-5>", lambda e: self.show_page("rename"))

    def _safe_geometry(self, geo):
        """把记住的窗口位置收进当前屏幕，分辨率/缩放变了也不会出现半截窗口。"""
        t = self.theme
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        match = re.match(r"^(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$", geo or "")
        if match:
            w, h, x, y = (int(v) for v in match.groups())
            if (t.s(1020) <= w <= screen_w and t.s(680) <= h <= screen_h
                    and 0 <= x <= screen_w - t.s(120)
                    and 0 <= y <= screen_h - t.s(120)):
                return f"{w}x{h}+{x}+{y}"
        w = min(t.s(1240), screen_w - t.s(40))
        h = min(t.s(820), screen_h - t.s(80))
        x = max(20, (screen_w - w) // 2)
        y = max(20, (screen_h - h) // 2 - t.s(16))
        return f"{w}x{h}+{x}+{y}"

    @staticmethod
    def _dep(name):
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "ok")
        except Exception:
            return "缺失"

    # ------------------------------------------------------------ 外壳
    def _build_shell(self):
        t = self.theme
        self.sidebar = tk.Frame(self, bg=t.SIDEBAR, width=t.s(232))
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # 品牌
        brand = tk.Frame(self.sidebar, bg=t.SIDEBAR)
        brand.pack(fill="x", padx=t.s(20), pady=(t.s(22), t.s(20)))
        logo = make_logo(t, 38)
        if logo:
            self._brand_logo = logo
            tk.Label(brand, image=logo, bg=t.SIDEBAR).pack(side="left")
        text_box = tk.Frame(brand, bg=t.SIDEBAR)
        text_box.pack(side="left", padx=(t.s(11), 0))
        tk.Label(text_box, text=APP_NAME, bg=t.SIDEBAR, fg=t.TEXT,
                 font=t.font(14, "bold"), anchor="w").pack(fill="x")
        tk.Label(text_box, text=APP_CN, bg=t.SIDEBAR, fg=t.MUTED,
                 font=t.font(8), anchor="w").pack(fill="x", pady=(t.s(1), 0))

        tk.Frame(self.sidebar, bg=t.BORDER_SOFT, height=1).pack(fill="x",
                                                                padx=t.s(18))
        self.nav = {}
        self._nav_header("工 作 区")
        for key, label, icon in (("scan", "扫描文件夹", "search"),
                                 ("results", "扫描结果", "layers"),
                                 ("logs", "运行日志", "list")):
            self._nav_item(key, label, icon)
        self._nav_header("工 具 箱")
        for key, label, icon in (("organize", "文件整理", "clock"),
                                 ("rename", "批量重命名", "edit")):
            self._nav_item(key, label, icon)

        # 底部
        tk.Label(self.sidebar, text=f"v{VERSION} · 本地离线", bg=t.SIDEBAR,
                 fg=t.FAINT, font=t.font(8), anchor="w").pack(
            side="bottom", fill="x", padx=t.s(24), pady=(0, t.s(14)))

        self.status_canvas = tk.Canvas(self.sidebar, width=t.s(204),
                                       height=t.s(66), bg=t.SIDEBAR,
                                       highlightthickness=0, bd=0)
        self.status_canvas.pack(side="bottom", padx=t.s(14), pady=(0, t.s(10)))
        self._draw_status("引擎就绪", t.GREEN, "本地离线处理 · 不上传文件")

        self.help_nav = NavItem(self.sidebar, t, "使用说明", "info", t.s(204),
                                command=self._show_help)
        self.help_nav.pack(side="bottom", fill="x", padx=t.s(14), pady=(0, t.s(4)))

        self.theme_btn = CanvasButton(self.sidebar, t, "", command=self.toggle_theme,
                                      kind="ghost", width=t.s(204),
                                      height=32, size=8, padding=12)
        self.theme_btn.pack(side="bottom", fill="x", padx=t.s(14), pady=(0, t.s(2)))
        self._sync_theme_button()

        # 主区域
        self.main = tk.Frame(self, bg=t.BG)
        self.main.pack(side="left", fill="both", expand=True)

    def _nav_header(self, text):
        t = self.theme
        tk.Label(self.sidebar, text=text, bg=t.SIDEBAR, fg=t.FAINT,
                 font=t.font(8, "bold"), anchor="w").pack(
            fill="x", padx=t.s(24), pady=(t.s(15), t.s(7)))

    def _nav_item(self, key, label, icon):
        t = self.theme
        item = NavItem(self.sidebar, t, label, icon, t.s(204),
                       command=lambda k=key: self.show_page(k))
        item.pack(fill="x", padx=t.s(14), pady=(0, t.s(2)))
        self.nav[key] = item

    def _draw_status(self, text, color, sub):
        self._status = (text, color, sub)
        t = self.theme
        c = self.status_canvas
        c.delete("all")
        w, h = t.s(204), t.s(66)
        round_rect(c, 0.5, 0.5, w - 0.5, h - 0.5, t.s(11),
                   fill=t.STATUS_BG, outline=t.BORDER_SOFT)
        c.create_oval(t.s(15), t.s(18), t.s(21), t.s(24), fill=color, outline="")
        c.create_text(t.s(28), t.s(21), text=text, font=t.font(8, "bold"),
                      fill=t.TEXT, anchor="w")
        c.create_text(t.s(16), t.s(41), text=sub, font=t.font(8),
                      fill=t.MUTED, anchor="w")

    def set_status(self, text, color, sub="本地离线处理 · 不上传文件"):
        self._status = (text, color, sub)
        self._draw_status(text, color, sub)

    # ------------------------------------------------------------ 主题
    def _sync_theme_button(self):
        """按钮上显示“切过去会变成什么”，而不是当前状态。"""
        if self.theme_name == "dark":
            self.theme_btn.text, self.theme_btn.icon = "白天主题", "sun"
        else:
            self.theme_btn.text, self.theme_btn.icon = "深色主题", "moon"
        self.theme_btn._btn_w = self.theme.s(204)
        self.theme_btn.configure(width=self.theme_btn._btn_w)
        self.theme_btn._draw()

    def toggle_theme(self):
        self.set_theme("day" if self.theme_name == "dark" else "dark")

    def set_theme(self, name):
        if name not in PALETTES or name == self.theme_name:
            return
        self.theme_name = name
        mapping = apply_palette(name)
        remap_tree(self, mapping)
        self._sync_theme_button()
        if getattr(self, "_status", None):
            self._draw_status(*self._status)
        enable_dark_titlebar(self, dark=(name == "dark"))
        for card in self._visible_cards():
            try:
                card.refresh()
            except Exception:
                pass
        self.settings["theme"] = name
        save_settings(self.settings)
        self.log(f"已切换到{'白天' if name == 'day' else '深色'}主题", "info")

    def _visible_cards(self):
        page = getattr(self, "results_page", None)
        cards = getattr(page, "cards", None) or []
        return [c for c in cards if c.winfo_exists()]

    def _build_pages(self):
        self.pages = {}
        self.scan_page = ScanPage(self.main, self)
        self.results_page = ResultsPage(self.main, self)
        self.log_page = LogPage(self.main, self)
        self.organize_page = OrganizePage(self.main, self)
        self.rename_page = RenamePage(self.main, self)
        self.pages = {"scan": self.scan_page, "results": self.results_page,
                      "logs": self.log_page, "organize": self.organize_page,
                      "rename": self.rename_page}
        self.results = self.results_page

    def show_page(self, name):
        for key, page in self.pages.items():
            if key != name:
                page.pack_forget()
                self.nav[key].set_active(False)
        page = self.pages[name]
        page.pack(fill="both", expand=True)
        self.nav[name].set_active(True)
        if hasattr(page, "on_show"):
            page.on_show()

    def set_result_count(self, n):
        self.nav["results"].set_badge(n)

    # ------------------------------------------------------------ 事件
    def post(self, event):
        if not self._closed:
            self.queue.put(event)

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                event = self.queue.get_nowait()
                try:
                    self._handle_event(event)
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.after(45, self._poll)

    def _handle_event(self, event):
        kind = event[0]
        if kind == "progress":
            self.results_page.on_progress(*event[1])
        elif kind == "log":
            self.log(event[1], event[2] if len(event) > 2 else "info")
        elif kind == "scan_done":
            self.results_page.on_done(event[1][0])
        elif kind == "scan_cancelled":
            self.results_page.on_cancelled()
        elif kind == "scan_error":
            self.results_page.on_error(*event[1])
        elif kind == "deleted":
            self.results_page.on_deleted(*event[1])
        elif kind == "org_plan":
            self.organize_page.on_plan(*event[1])
        elif kind == "org_done":
            self.organize_page.on_done(*event[1])
        elif kind == "org_undo_done":
            self.organize_page.on_undo_done(*event[1])
        elif kind == "ren_done":
            self.rename_page.on_done(*event[1])
        elif kind == "ren_undo_done":
            self.rename_page.on_undo_done(*event[1])
        elif kind == "ren_plan":
            self.rename_page.on_plan(*event[1])
        elif kind == "cleanup_restored":
            self.results_page.on_cleanup_restored(*event[1])
        elif kind == "thumb":
            key, pil = event[1]
            self.thumb_service.deliver(key, pil)

    # ------------------------------------------------------------ 日志
    def log(self, message, level="info"):
        if hasattr(self, "log_page"):
            self.log_page.add(message, level)

    # ------------------------------------------------------------ 提示条
    def toast(self, text, kind="info", duration=2600):
        t = self.theme
        if self._toast is not None:
            try:
                self._toast.destroy()
            except Exception:
                pass
        colors = {"info": t.ACCENT, "ok": t.GREEN, "warn": t.AMBER,
                  "err": t.RED}
        icons = {"info": "info", "ok": "check", "warn": "warning", "err": "warning"}
        color = colors.get(kind, t.ACCENT)
        f = t.font(9)
        w = f.measure(text) + t.s(54)
        h = t.s(38)
        c = tk.Canvas(self, width=w, height=h, bg=t.BG, highlightthickness=0, bd=0)
        round_rect(c, 0.5, 0.5, w - 0.5, h - 0.5, t.s(11),
                   fill=t.TOAST_BG, outline=t.TOAST_BORDER)
        c.create_text(t.s(19), h / 2 + 0.5, text=ICON[icons.get(kind, "info")],
                      font=t.icon(13), fill=color, anchor="center")
        c.create_text(t.s(34), h / 2 + 0.5, text=text, font=f, fill=t.TEXT,
                      anchor="w")
        c.place(relx=0.5, rely=1.0, anchor="s", y=-t.s(86))
        c.tk.call("raise", c._w)
        self._toast = c
        self.after(duration, lambda: self._hide_toast(c))

    def _hide_toast(self, widget):
        if self._toast is widget:
            try:
                widget.destroy()
            except Exception:
                pass
            self._toast = None

    def _show_help(self):
        t = self.theme
        msg = ("1. 选择要检查的文件夹，勾选需要检测的重复类型。\n"
               "   完全重复 / 相似照片 / 重复视频 / 同名相似，可同时检测。\n"
               "2. 扫描只读取文件，不会修改或删除任何内容。\n"
               "   重复扫描同一批文件时会复用指纹缓存，速度更快。\n"
               "3. 结果按「组」展示，同组是内容相同或高度相似的文件。\n"
               "   每组默认保留清晰度/体积最合适的一个，其余会被勾选。\n"
               "4. 点击缩略图可切换是否清理；双击可直接打开文件；\n"
               "   右键可「设为保留项」或在资源管理器中定位。\n"
               "5. 确认后移入回收站，随时可以还原。\n\n"
               "工具箱\n"
               "· 文件整理：按类型或日期把文件归到子文件夹，\n"
               "  支持预览、复制/移动、清理空目录，并可一键撤销。\n"
               "· 批量重命名：查找替换、前后缀、序号模板、大小写，\n"
               "  先实时预览再执行，也可一键撤销。\n\n"
               "提示：照片相似度检测最实用；视频检测需要解码关键帧，较慢。")
        alert(self, t, "使用说明", msg, icon="info")

    def _on_close(self):
        if self.scanning:
            t = self.theme
            if confirm(self, t, "扫描还在进行", "现在退出会中断当前扫描。确定要退出吗？",
                       ok_text="退出", cancel_text="继续扫描", icon="warning") != "ok":
                return
            self.cancel_event.set()
        busy_tool = (getattr(self, "organize_page", None)
                     and self.organize_page.busy) or \
                    (getattr(self, "rename_page", None)
                     and self.rename_page.busy)
        if busy_tool:
            t = self.theme
            if confirm(self, t, "文件操作还在进行",
                       "现在退出可能中断当前整理或重命名。\n"
                       "已经处理的文件会保留操作记录，下次可以撤销。\n\n"
                       "确定要退出吗？",
                       ok_text="退出", cancel_text="继续等待",
                       icon="warning") != "ok":
                return
        try:
            geo = self.geometry()
            if re.match(r"^\d+x\d+\+-?\d+\+-?\d+$", geo):
                self.settings["geometry"] = geo
                save_settings(self.settings)
        except Exception:
            pass
        self._closed = True
        self.thumb_service.stop()
        self.destroy()

    def open_file(self, path):
        open_path(path)

    def reveal_file(self, path):
        reveal(path)

    def copy_text(self, text):
        self.clipboard_clear()
        self.clipboard_append(str(text))
        self.toast("路径已复制", "ok")


# ---------------------------------------------------------------- 扫描页
class ScanPage(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=app.theme.BG)
        self.app = app
        self.theme = app.theme
        self.settings = app.settings
        self._build()

    def _build(self):
        t = self.theme
        s = t.s
        outer = tk.Frame(self, bg=t.BG)
        outer.pack(fill="both", expand=True)

        self.content_scroll = ScrollArea(outer, t, bg=t.BG)
        self.content_scroll.pack(fill="both", expand=True,
                                 padx=(s(24), s(8)), pady=(s(20), 0))
        page = tk.Frame(self.content_scroll.inner, bg=t.BG)
        page.pack(fill="both", expand=True, padx=(s(10), s(18)))

        # 标题
        header = tk.Frame(page, bg=t.BG)
        header.pack(fill="x")
        left = tk.Frame(header, bg=t.BG)
        left.pack(side="left")
        tk.Label(left, text="扫描文件夹", bg=t.BG, fg=t.TEXT,
                 font=t.font(21, "bold"), anchor="w").pack(fill="x")
        tk.Label(left, text="在本地比对文件内容。先看清楚，再决定清理哪些。",
                 bg=t.BG, fg=t.TEXT_DIM, font=t.font(10), anchor="w").pack(
            fill="x", pady=(s(5), 0))
        Chip(header, t, text="本机离线处理", icon="shield", color=t.GREEN,
             bg=t.BG).pack(side="right", pady=(s(7), 0))

        # 文件夹
        folder_card = Card(page, t, padding=16)
        folder_card.pack(fill="x", pady=(s(20), 0))
        body = folder_card.body

        bubble = tk.Canvas(body, width=s(42), height=s(42), bg=t.CARD,
                           highlightthickness=0, bd=0)
        bubble.pack(side="left")
        round_rect(bubble, 0, 0, s(42), s(42), s(11), fill=t.BUBBLE_BG, outline="")
        bubble.create_text(s(21), s(21) + 1, text=ICON["folder"],
                           font=t.icon(19), fill=t.ACCENT, anchor="center")

        text_box = tk.Frame(body, bg=t.CARD)
        text_box.pack(side="left", fill="x", expand=True, padx=(s(13), s(14)))
        tk.Label(text_box, text="扫描目录", bg=t.CARD, fg=t.MUTED,
                 font=t.font(8), anchor="w").pack(fill="x")
        field = tk.Frame(text_box, bg=t.INPUT, highlightthickness=1,
                         highlightbackground=t.BORDER, highlightcolor=t.ACCENT)
        field.pack(fill="x", pady=(s(5), 0))
        self.path_var = tk.StringVar()
        self.entry = tk.Entry(field, textvariable=self.path_var, bg=t.INPUT,
                              fg=t.TEXT, insertbackground=t.TEXT, relief="flat",
                              bd=0, highlightthickness=0, font=t.font(10))
        self.entry.pack(fill="x", padx=s(10), ipady=s(5))
        self.entry.bind("<Return>", lambda e: self.start_scan())
        tk.Label(text_box, text="可直接输入或粘贴路径，回车开始扫描",
                 bg=t.CARD, fg=t.FAINT, font=t.font(8), anchor="w").pack(
            fill="x", pady=(s(4), 0))

        CanvasButton(body, t, "选择文件夹", command=self.browse, kind="secondary",
                     icon="folder", height=38, size=9, padding=14, bg=t.CARD
                     ).pack(side="right")

        # 选项区
        options = tk.Frame(page, bg=t.BG)
        options.pack(fill="x", pady=(s(16), 0))
        options.grid_columnconfigure(0, weight=1, uniform="opt")
        options.grid_columnconfigure(1, weight=1, uniform="opt")

        # 左侧：检测内容
        detect_card = Card(options, t, padding=20)
        detect_card.grid(row=0, column=0, sticky="n", padx=(0, s(8)))
        d = detect_card.body
        self._section_title(d, "检测内容", "layers", "勾选需要查找的重复类型")
        self.sw_exact = self._detect_row(
            d, "完全重复", "字节完全一致的文件，零误判", "document",
            t.ACCENT, t.KIND_EXACT_DIM, bool(self.settings.get("exact", True)))
        self.sw_photo = self._detect_row(
            d, "相似照片", "同一张图的不同压缩、格式版本", "image",
            t.AMBER, t.AMBER_DIM, bool(self.settings.get("photos", True)))
        self.sw_video = self._detect_row(
            d, "重复视频", "时长与关键帧接近的视频，速度较慢", "video",
            t.VIOLET, t.KIND_VIDEO_CHIP, bool(self.settings.get("videos", True)))
        if not core.video_available():
            self.sw_video.set(False)
            self.sw_video.configure(cursor="arrow")
            self.sw_video.bind("<Button-1>", lambda e: None)
        self.sw_audio = self._detect_row(
            d, "重复音乐", "歌手 + 标题 + 时长相同，支持 MP3 / FLAC / M4A", "audio",
            t.KIND_AUDIO, t.KIND_AUDIO_DIM, bool(self.settings.get("audio", True)))
        self.sw_name = self._detect_row(
            d, "同名相似", "去掉「副本 / Copy(2)」标记后同名，可能是旧版本", "copy",
            t.KIND_NAME, t.KIND_NAME_DIM, bool(self.settings.get("names", True)))
        self.sw_fuzzy = self._detect_row(
            d, "文件名模糊匹配", "名称相近且大小接近，适合找同一文档的多个版本", "filter",
            t.KIND_NAME, t.KIND_NAME_DIM, bool(self.settings.get("names_fuzzy", False)))

        note = tk.Frame(d, bg=t.CARD)
        note.pack(fill="x", pady=(s(14), 0))
        tk.Frame(note, bg=t.BORDER_SOFT, height=1).pack(fill="x", pady=(0, s(12)))
        row = tk.Frame(note, bg=t.CARD)
        row.pack(fill="x")
        tk.Label(row, text=ICON["info"], bg=t.CARD, fg=t.MUTED,
                 font=t.icon(11)).pack(side="left")
        tk.Label(row, text="照片检测最常用；视频需要逐帧解码，数量多时会慢一些。",
                 bg=t.CARD, fg=t.MUTED, font=t.font(8), anchor="w").pack(
            side="left", padx=(s(6), 0))

        # 右侧：扫描选项
        opt_card = Card(options, t, padding=20)
        opt_card.grid(row=0, column=1, sticky="n", padx=(s(8), 0))
        o = opt_card.body
        self._section_title(o, "扫描选项", "settings", None)

        preset_row = tk.Frame(o, bg=t.CARD)
        preset_row.pack(fill="x", pady=(s(12), 0))
        tk.Label(preset_row, text="扫描方案", bg=t.CARD, fg=t.TEXT_DIM,
                 font=t.font(9), anchor="w").pack(side="left")
        self.btn_save_preset = CanvasButton(
            preset_row, t, "保存当前", command=self.save_preset, kind="ghost",
            height=30, size=8, padding=10, bg=t.CARD)
        self.btn_save_preset.pack(side="right")
        self.btn_load_preset = CanvasButton(
            preset_row, t, "载入方案", command=self._preset_menu, kind="ghost",
            height=30, size=8, padding=10, bg=t.CARD)
        self.btn_load_preset.pack(side="right", padx=(0, s(6)))

        tk.Label(o, text="识别严格程度", bg=t.CARD, fg=t.TEXT_DIM,
                 font=t.font(9), anchor="w").pack(fill="x", pady=(s(12), s(6)))
        self.seg_strict = Segmented(
            o, t, [("relaxed", "宽松"), ("standard", "标准"), ("strict", "严格")],
            value=self.settings.get("strictness", "standard"),
            command=self._strict_changed, bg=t.CARD)
        self.seg_strict.pack(anchor="w")
        self.strict_hint = tk.Label(o, text="", bg=t.CARD, fg=t.MUTED,
                                    font=t.font(8), anchor="w")
        self.strict_hint.pack(fill="x", pady=(s(5), 0))
        self._strict_changed(self.seg_strict.value)

        tk.Label(o, text="最小文件大小", bg=t.CARD, fg=t.TEXT_DIM,
                 font=t.font(9), anchor="w").pack(fill="x", pady=(s(14), s(6)))
        self.seg_size = Segmented(
            o, t, [(0, "不限"), (1 << 20, "1 MB"), (10 << 20, "10 MB"),
                   (100 << 20, "100 MB")],
            value=int(self.settings.get("min_size", 0)),
            bg=t.CARD, padding=11)
        self.seg_size.pack(anchor="w")

        tk.Label(o, text="排除目录", bg=t.CARD, fg=t.TEXT_DIM,
                 font=t.font(9), anchor="w").pack(fill="x", pady=(s(14), s(6)))
        ex_field = tk.Frame(o, bg=t.INPUT, highlightthickness=1,
                            highlightbackground=t.BORDER, highlightcolor=t.ACCENT)
        ex_field.pack(fill="x")
        self.exclude_var = tk.StringVar(value=self.settings.get("exclude", ""))
        ex_entry = tk.Entry(ex_field, textvariable=self.exclude_var, bg=t.INPUT,
                            fg=t.TEXT, insertbackground=t.TEXT, relief="flat",
                            bd=0, highlightthickness=0, font=t.font(9))
        ex_entry.pack(fill="x", padx=s(10), ipady=s(5))
        self._placeholder(ex_entry, self.exclude_var, "例如：缓存, 备份（逗号分隔）")

        tk.Label(o, text="保护文件夹（永不清理）", bg=t.CARD, fg=t.TEXT_DIM,
                 font=t.font(9), anchor="w").pack(fill="x", pady=(s(14), s(6)))
        pr_field = tk.Frame(o, bg=t.INPUT, highlightthickness=1,
                            highlightbackground=t.BORDER,
                            highlightcolor=t.ACCENT)
        pr_field.pack(fill="x")
        self.protect_var = tk.StringVar(value=self.settings.get("protect", ""))
        pr_entry = tk.Entry(pr_field, textvariable=self.protect_var, bg=t.INPUT,
                            fg=t.TEXT, insertbackground=t.TEXT, relief="flat",
                            bd=0, highlightthickness=0, font=t.font(9))
        pr_entry.pack(fill="x", padx=s(10), ipady=s(5))
        self._placeholder(pr_entry, self.protect_var,
                          "例如：D:\\照片原图（多个用逗号分隔）")
        tk.Label(o, text="保护文件夹里的文件会标记为「保护」，不会被勾选或删除",
                 bg=t.CARD, fg=t.FAINT, font=t.font(8), anchor="w").pack(
            fill="x", pady=(s(4), 0))

        sub_row = tk.Frame(o, bg=t.CARD)
        sub_row.pack(fill="x", pady=(s(15), 0))
        sub_text = tk.Frame(sub_row, bg=t.CARD)
        sub_text.pack(side="left", fill="x", expand=True)
        tk.Label(sub_text, text="包含子文件夹", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), anchor="w").pack(fill="x")
        tk.Label(sub_text, text="关闭后只扫描当前目录", bg=t.CARD, fg=t.MUTED,
                 font=t.font(8), anchor="w").pack(fill="x", pady=(s(2), 0))
        self.sw_recursive = Switch(sub_row, t,
                                   value=bool(self.settings.get("recursive", True)),
                                   bg=t.CARD)
        self.sw_recursive.pack(side="right", pady=(s(4), 0))

        cache_row = tk.Frame(o, bg=t.CARD)
        cache_row.pack(fill="x", pady=(s(13), 0))
        cache_text = tk.Frame(cache_row, bg=t.CARD)
        cache_text.pack(side="left", fill="x", expand=True)
        tk.Label(cache_text, text="指纹缓存", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), anchor="w").pack(fill="x")
        tk.Label(cache_text, text="重复扫描同一批文件时更快", bg=t.CARD,
                 fg=t.MUTED, font=t.font(8), anchor="w").pack(
            fill="x", pady=(s(2), 0))
        self.sw_cache = Switch(cache_row, t,
                               value=bool(self.settings.get("cache", True)),
                               bg=t.CARD)
        self.sw_cache.pack(side="right", pady=(s(4), 0))

        # 底栏（固定在窗口底部，不随内容滚动）
        footer = tk.Frame(outer, bg=t.BG)
        footer.pack(side="bottom", fill="x", padx=s(32), pady=(s(14), s(16)),
                    before=self.content_scroll)
        hint = tk.Frame(footer, bg=t.BG)
        hint.pack(side="left")
        tk.Label(hint, text=ICON["shield"], bg=t.BG, fg=t.GREEN,
                 font=t.icon(12)).pack(side="left")
        tk.Label(hint, text="扫描只读取文件内容，不会修改或删除任何文件",
                 bg=t.BG, fg=t.TEXT_DIM, font=t.font(9)).pack(side="left",
                                                               padx=(s(7), 0))
        self.start_btn = CanvasButton(footer, t, "开始扫描",
                                      command=self.start_scan, kind="primary",
                                      icon="search", width=150, height=44,
                                      size=10, bg=t.BG)
        self.start_btn.pack(side="right")

        # 流程提示（窗口太矮时自动隐藏）
        self.tips_card = Card(page, t, padding=16)
        tips = self.tips_card.body
        tips.grid_columnconfigure(0, weight=1, uniform="tip")
        tips.grid_columnconfigure(1, weight=1, uniform="tip")
        tips.grid_columnconfigure(2, weight=1, uniform="tip")
        tip_data = [
            ("layers", t.ACCENT, t.CHIP_ACCENT, "分组浏览",
             "内容相同或相似的文件聚成一组"),
            ("check", t.GREEN, t.GREEN_DIM, "勾选清理",
             "想删的点一下，每组至少留一个"),
            ("shield", t.AMBER, t.AMBER_DIM, "可还原",
             "清理进回收站，误删也能找回"),
        ]
        for col, (icon, color, dim, title, desc) in enumerate(tip_data):
            item = tk.Frame(tips, bg=t.CARD)
            item.grid(row=0, column=col, sticky="w",
                      padx=(0 if col == 0 else s(18), s(10)))
            bubble = tk.Canvas(item, width=s(30), height=s(30), bg=t.CARD,
                               highlightthickness=0, bd=0)
            bubble.pack(side="left", pady=(s(1), 0))
            round_rect(bubble, 0, 0, s(30), s(30), s(9), fill=dim, outline="")
            bubble.create_text(s(15), s(15) + 1, text=ICON[icon],
                               font=t.icon(13), fill=color, anchor="center")
            box = tk.Frame(item, bg=t.CARD)
            box.pack(side="left", padx=(s(9), 0))
            tk.Label(box, text=title, bg=t.CARD, fg=t.TEXT,
                     font=t.font(9, "bold"), anchor="w").pack(fill="x")
            tk.Label(box, text=desc, bg=t.CARD, fg=t.MUTED,
                     font=t.font(8), anchor="w").pack(fill="x", pady=(s(2), 0))
        self.tips_card.pack(fill="x", pady=(s(14), s(6)))
        self._tips_visible = True
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        if event.widget is not self:
            return
        want = event.height >= self.theme.s(742)
        if want and not self._tips_visible:
            self.tips_card.pack(fill="x",
                                pady=(self.theme.s(14), self.theme.s(6)))
            self._tips_visible = True
        elif not want and self._tips_visible:
            self.tips_card.pack_forget()
            self._tips_visible = False

    # ---- 组件
    def _section_title(self, parent, text, icon, desc):
        t = self.theme
        row = tk.Frame(parent, bg=t.CARD)
        row.pack(fill="x")
        tk.Label(row, text=ICON.get(icon, icon), bg=t.CARD, fg=t.ACCENT,
                 font=t.icon(13)).pack(side="left")
        tk.Label(row, text=text, bg=t.CARD, fg=t.TEXT, font=t.font(11, "bold"),
                 anchor="w").pack(side="left", padx=(self.theme.s(7), 0))
        if desc:
            tk.Label(parent, text=desc, bg=t.CARD, fg=t.MUTED,
                     font=t.font(8), anchor="w").pack(fill="x",
                                                      pady=(self.theme.s(4), 0))

    def _detect_row(self, parent, title, desc, icon, color, dim, value):
        t = self.theme
        s = t.s
        row = tk.Frame(parent, bg=t.CARD)
        row.pack(fill="x", pady=(s(13), 0))
        bubble = tk.Canvas(row, width=s(34), height=s(34), bg=t.CARD,
                           highlightthickness=0, bd=0)
        bubble.pack(side="left", pady=(s(2), 0))
        round_rect(bubble, 0, 0, s(34), s(34), s(9), fill=dim, outline="")
        bubble.create_text(s(17), s(17) + 1, text=ICON.get(icon, icon),
                           font=t.icon(15), fill=color, anchor="center")
        box = tk.Frame(row, bg=t.CARD)
        box.pack(side="left", fill="x", expand=True, padx=(s(10), s(4)))
        tk.Label(box, text=title, bg=t.CARD, fg=t.TEXT, font=t.font(10, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(box, text=desc, bg=t.CARD, fg=t.MUTED, font=t.font(8),
                 anchor="w").pack(fill="x", pady=(s(2), 0))
        sw = Switch(row, t, value=value, bg=t.CARD)
        sw.pack(side="right", pady=(s(4), 0))
        return sw

    def _placeholder(self, entry, var, text):
        t = self.theme
        ph = tk.Label(entry.master, text=text, bg=t.INPUT, fg=t.FAINT,
                      font=t.font(9), cursor="xterm")
        ph.place(x=t.s(10), rely=0.5, anchor="w")
        ph.bind("<Button-1>", lambda e: entry.focus_set())

        def sync(*_):
            if var.get():
                ph.place_forget()
            elif entry.focus_get() is not entry:
                ph.place(x=t.s(10), rely=0.5, anchor="w")

        var.trace_add("write", sync)
        entry.bind("<FocusIn>", lambda e: ph.place_forget(), add="+")
        entry.bind("<FocusOut>", lambda e: sync(), add="+")
        sync()

    def _strict_changed(self, value):
        hints = {
            "relaxed": "宽松：更容易找到相似文件，可能混入少量误判",
            "standard": "标准：在漏检和误判之间取平衡（推荐）",
            "strict": "严格：只保留高度相似的结果，更保守",
        }
        self.strict_hint.configure(text=hints.get(value, ""))

    # ---- 扫描方案
    def _current_preset(self):
        return {
            "path": self.path_var.get().strip(),
            "strictness": self.seg_strict.value,
            "min_size": int(self.seg_size.value),
            "exclude": self.exclude_var.get().strip(),
            "protect": self.protect_var.get().strip(),
            "recursive": bool(self.sw_recursive.value),
            "cache": bool(self.sw_cache.value),
            "exact": bool(self.sw_exact.value),
            "photos": bool(self.sw_photo.value),
            "videos": bool(self.sw_video.value),
            "audio": bool(self.sw_audio.value),
            "names": bool(self.sw_name.value),
            "names_fuzzy": bool(self.sw_fuzzy.value),
        }

    def _save_presets(self):
        settings = self.settings
        settings["scan_presets"] = {
            k: dict(v) for k, v in settings.get("scan_presets", {}).items()
        }
        save_settings(settings)

    def save_preset(self):
        t = self.theme
        name = prompt(self.app, t, "保存扫描方案",
                      "把当前目录和扫描选项存成一个方案，下次一键载入。",
                      value=(self.path_var.get().strip() and
                             Path(self.path_var.get().strip()).name) or "我的方案",
                      ok_text="保存", icon="save")
        if not name:
            return
        presets = self.settings.setdefault("scan_presets", {})
        if name in presets and confirm(self.app, t, "覆盖已有方案",
                   f"已经存在名为「{name}」的方案，要覆盖它吗？",
                   ok_text="覆盖", cancel_text="取消",
                   icon="warning") != "ok":
            return
        presets[name] = self._current_preset()
        self._save_presets()
        self.app.log(f"已保存扫描方案：{name}", "ok")
        self.app.toast(f"已保存方案「{name}」", "ok")

    def _preset_menu(self):
        t = self.theme
        presets = self.settings.get("scan_presets", {})
        if not presets:
            alert(self.app, t, "还没有保存的方案",
                  "先设置好扫描目录和选项，再点「保存当前」即可。",
                  icon="save")
            return
        items = [(name, "history", lambda n=name: self.apply_preset(n))
                 for name in sorted(presets)]
        items += [None, ("删除方案…", "trash", self._delete_preset_menu)]
        x = self.btn_load_preset.winfo_rootx()
        y = self.btn_load_preset.winfo_rooty() + self.btn_load_preset.winfo_height() + 4
        PopupMenu(self.app, t, items, x=x, y=y)

    def _delete_preset_menu(self):
        t = self.theme
        presets = self.settings.get("scan_presets", {})
        items = [(f"删除：{name}", "trash",
                  lambda n=name: self._delete_preset(n))
                 for name in sorted(presets)]
        x = self.btn_load_preset.winfo_rootx()
        y = self.btn_load_preset.winfo_rooty() + self.btn_load_preset.winfo_height() + 4
        PopupMenu(self.app, t, items, x=x, y=y)

    def _delete_preset(self, name):
        presets = self.settings.setdefault("scan_presets", {})
        presets.pop(name, None)
        self._save_presets()
        self.app.toast(f"已删除方案「{name}」", "info")

    def apply_preset(self, name):
        data = self.settings.get("scan_presets", {}).get(name)
        if not data:
            return
        if data.get("path"):
            self.path_var.set(data["path"])
        self.seg_strict.set_value(data.get("strictness", "standard"))
        self._strict_changed(self.seg_strict.value)
        self.seg_size.set_value(int(data.get("min_size", 0)))
        self.exclude_var.set(data.get("exclude", ""))
        self.protect_var.set(data.get("protect", ""))
        for attr, key in (("sw_recursive", "recursive"), ("sw_cache", "cache"),
                          ("sw_exact", "exact"), ("sw_photo", "photos"),
                          ("sw_video", "videos"), ("sw_audio", "audio"),
                          ("sw_name", "names"), ("sw_fuzzy", "names_fuzzy")):
            getattr(self, attr).set(bool(data.get(key, True)))
        if not core.video_available():
            self.sw_video.set(False)
        self.app.log(f"已载入扫描方案：{name}", "info")
        self.app.toast(f"已载入方案「{name}」", "ok")

    def set_path(self, path):
        self.path_var.set(str(path or ""))

    def browse(self):
        current = self.path_var.get().strip().strip('"')
        initial = current if current and os.path.isdir(current) else str(Path.home())
        chosen = filedialog.askdirectory(parent=self, title="选择要扫描的文件夹",
                                         initialdir=initial, mustexist=True)
        if chosen:
            self.path_var.set(os.path.normpath(chosen))
            self.entry.icursor("end")
            self.entry.focus_set()

    def on_show(self):
        if not self.path_var.get():
            self.entry.focus_set()

    def start_scan(self):
        t = self.theme
        if self.app.scanning:
            self.app.show_page("results")
            return
        raw = self.path_var.get().strip().strip('"')
        raw = os.path.expandvars(os.path.expanduser(raw))
        if not raw:
            alert(self.app, t, "还没有选择文件夹", "请先选择一个要扫描的目录。", icon="folder")
            return
        root = Path(raw)
        if not root.is_dir():
            alert(self.app, t, "目录不存在", f"找不到这个文件夹：\n{root}",
                  icon="warning")
            return
        if core.is_filesystem_root(root):
            if confirm(self.app, t, "扫描整个盘符？",
                       "你选择的是盘符根目录，扫描可能耗时很久。确定要继续吗？",
                       ok_text="继续扫描", cancel_text="取消",
                       icon="warning") != "ok":
                return

        preset = self.seg_strict.value
        preset_vals = core.PRESETS.get(preset, core.PRESETS["standard"])
        exclude = tuple(re.split(r"[,，;；\s]+", self.exclude_var.get().strip()))
        exclude = tuple(x for x in exclude if x)
        protect = tuple(re.split(r"[,，;；\n]+",
                                 self.protect_var.get().strip()))
        protect = tuple(os.path.expandvars(os.path.expanduser(x))
                        for x in protect if x)
        opts = core.ScanOptions(
            root=root,
            exact=bool(self.sw_exact.value),
            photos=bool(self.sw_photo.value),
            videos=bool(self.sw_video.value),
            audio=bool(self.sw_audio.value),
            names=bool(self.sw_name.value),
            names_fuzzy=bool(self.sw_fuzzy.value),
            cache=bool(self.sw_cache.value),
            img_dist=preset_vals["img_dist"],
            img_fine=preset_vals["img_fine"],
            img_hist=preset_vals["img_hist"],
            vid_dist=preset_vals["vid_dist"],
            vid_fine=preset_vals["vid_fine"],
            vid_ratio=preset_vals["vid_ratio"],
            name_ratio=preset_vals["name_ratio"],
            min_size=int(self.seg_size.value),
            exclude=exclude,
            protect=protect,
            recursive=bool(self.sw_recursive.value),
            keep_rule=self.settings.get("keep_rule", "best"),
        )
        if not (opts.exact or opts.photos or opts.videos or opts.audio
                or opts.names):
            alert(self.app, t, "没有选择检测类型",
                  "至少勾选一种检测内容，或者直接按「开始扫描」使用默认选项。",
                  icon="warning")
            return

        self.settings.update({
            "last_dir": str(root),
            "strictness": preset,
            "min_size": int(self.seg_size.value),
            "exclude": self.exclude_var.get().strip(),
            "protect": self.protect_var.get().strip(),
            "recursive": bool(self.sw_recursive.value),
            "exact": opts.exact,
            "photos": opts.photos,
            "videos": opts.videos,
            "audio": opts.audio,
            "names": opts.names,
            "names_fuzzy": opts.names_fuzzy,
            "cache": opts.cache,
            "keep_rule": opts.keep_rule,
        })
        save_settings(self.settings)
        self.app.show_page("results")
        self.app.results_page.start_scan(opts)


# ---------------------------------------------------------------- 日志页
class LogPage(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=app.theme.BG)
        self.app = app
        self.theme = app.theme
        self._build()

    def _build(self):
        t = self.theme
        s = t.s
        page = tk.Frame(self, bg=t.BG)
        page.pack(fill="both", expand=True, padx=s(32), pady=(s(26), s(22)))
        header = tk.Frame(page, bg=t.BG)
        header.pack(fill="x")
        left = tk.Frame(header, bg=t.BG)
        left.pack(side="left")
        tk.Label(left, text="运行日志", bg=t.BG, fg=t.TEXT,
                 font=t.font(21, "bold"), anchor="w").pack(fill="x")
        tk.Label(left, text="扫描过程中的步骤、跳过项和清理记录都在这里。",
                 bg=t.BG, fg=t.TEXT_DIM, font=t.font(10), anchor="w").pack(
            fill="x", pady=(s(5), 0))
        CanvasButton(header, t, "清空日志", command=self.clear, kind="ghost",
                     icon="trash", height=34, size=9, bg=t.BG).pack(
            side="right", pady=(s(7), 0))

        panel = tk.Frame(page, bg=t.LOG_BG, highlightthickness=1,
                         highlightbackground=t.BORDER)
        panel.pack(fill="both", expand=True, pady=(s(20), 0))
        bar = tk.Frame(panel, bg=t.LOG_BAR, height=s(34))
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="●", bg=t.LOG_BAR, fg=t.GREEN,
                 font=t.font(7)).pack(side="left", padx=(s(14), s(6)))
        tk.Label(bar, text="dedup · session", bg=t.LOG_BAR, fg=t.MUTED,
                 font=t.font(8, family=t.mono)).pack(side="left")
        tk.Label(bar, text="自动滚动", bg=t.LOG_BAR, fg=t.FAINT,
                 font=t.font(8)).pack(side="right", padx=s(14))
        tk.Frame(panel, bg=t.BORDER_SOFT, height=1).pack(fill="x")

        body = tk.Frame(panel, bg=t.LOG_BG)
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, bg=t.LOG_BG, fg=t.TEXT_DIM, relief="flat",
                           bd=0, highlightthickness=0, wrap="word",
                           font=t.font(9, family=t.mono), padx=s(14), pady=s(12),
                           insertbackground=t.TEXT, spacing1=s(2), spacing3=s(2),
                           selectbackground=t.LOG_SELECT)
        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar = tk.Canvas(body, width=s(11), bg=t.LOG_BG,
                                   highlightthickness=0, bd=0)
        self.scrollbar.pack(side="right", fill="y")
        self._bar_first, self._bar_last = 0.0, 1.0
        self._drag = None
        self.text.configure(yscrollcommand=self._on_text_scroll)
        self.scrollbar.bind("<Configure>", lambda e: self._draw_scrollbar())
        self.scrollbar.bind("<ButtonPress-1>", self._bar_press)
        self.scrollbar.bind("<B1-Motion>", self._bar_drag)
        self.scrollbar.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.text.bind("<MouseWheel>", self._on_wheel)
        self.scrollbar.bind("<MouseWheel>", self._on_wheel)
        self.text.tag_configure("time", foreground=t.FAINT)
        self.text.tag_configure("info", foreground=t.TEXT_DIM)
        self.text.tag_configure("ok", foreground=t.GREEN)
        self.text.tag_configure("warn", foreground=t.AMBER)
        self.text.tag_configure("err", foreground=t.RED)
        self.text.configure(state="disabled")
        self._lines = 0

    def add(self, message, level="info"):
        if not hasattr(self, "text"):
            return
        at_bottom = True
        try:
            at_bottom = self.text.yview()[1] > 0.995
        except Exception:
            pass
        self.text.configure(state="normal")
        self.text.insert("end", f"{datetime.now():%H:%M:%S}  ", "time")
        self.text.insert("end", str(message) + "\n", level)
        self._lines += 1
        if self._lines > 3000:
            self.text.delete("1.0", "400.0")
            self._lines -= 400
        self.text.configure(state="disabled")
        if at_bottom:
            self.text.see("end")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._lines = 0
        self._bar_first, self._bar_last = 0.0, 1.0
        self._draw_scrollbar()

    # ---- 自绘滚动条
    def _on_text_scroll(self, first, last):
        try:
            self._bar_first, self._bar_last = float(first), float(last)
        except Exception:
            return
        self._draw_scrollbar()

    def _bar_geom(self):
        h = max(self.scrollbar.winfo_height(), 1)
        span = self._bar_last - self._bar_first
        if span >= 0.999:
            return None
        top = self._bar_first * h
        size = max(self.theme.s(30), span * h)
        top = min(top, h - size)
        return top, size

    def _draw_scrollbar(self):
        if not hasattr(self, "scrollbar") or not self.scrollbar.winfo_exists():
            return
        self.scrollbar.delete("all")
        geom = self._bar_geom()
        if not geom:
            return
        top, size = geom
        w = self.scrollbar.winfo_width() or self.theme.s(11)
        bw = self.theme.s(4)
        x1 = (w - bw) / 2
        round_rect(self.scrollbar, x1, top + 2, x1 + bw, top + size - 2,
                   bw / 2, fill=self.theme.SCROLL_THUMB_HOVER, outline="")

    def _bar_press(self, event):
        geom = self._bar_geom()
        if not geom:
            return
        top, size = geom
        if top <= event.y <= top + size:
            self._drag = (event.y, self._bar_first)
        else:
            self.text.yview_scroll(-1 if event.y < top else 1, "pages")

    def _bar_drag(self, event):
        if not self._drag:
            return
        start_y, start_first = self._drag
        h = max(self.scrollbar.winfo_height(), 1)
        span = self._bar_last - self._bar_first
        delta = (event.y - start_y) / h
        self.text.yview_moveto(max(0.0, min(1.0 - span, start_first + delta)))

    def _on_wheel(self, event):
        self.text.yview_scroll(-3 if event.delta > 0 else 3, "units")
        self._draw_scrollbar()
        return "break"

    def on_show(self):
        self._draw_scrollbar()


# ---------------------------------------------------------------- 结果页
class ResultsPage(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=app.theme.BG)
        self.app = app
        self.theme = app.theme
        self.result = None
        self.options = None
        self.scanning = False
        self.busy = False
        self.selection = set()
        self.kind = "全部"
        self.query = ""
        self.cards = []
        self.sort_key = "wasted"
        self.keep_rule = "best"
        self._protected = {}
        self._last_cleanup = None
        self.settings = app.settings
        self._render_id = 0
        self._progress = 0.0
        self._target = 0.0
        self._tick_id = None
        self._size_map = {}
        self._build()

    # ------------------------------------------------------------ 构建
    def _build(self):
        t = self.theme
        s = t.s
        self.header = tk.Frame(self, bg=t.BG)
        self.header.pack(fill="x", padx=s(32), pady=(s(26), 0))
        left = tk.Frame(self.header, bg=t.BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="扫描结果", bg=t.BG, fg=t.TEXT,
                 font=t.font(21, "bold"), anchor="w").pack(fill="x")
        self.subtitle = tk.Label(left, text="还没有扫描结果", bg=t.BG,
                                 fg=t.TEXT_DIM, font=t.font(10), anchor="w")
        self.subtitle.pack(fill="x", pady=(s(5), 0))
        self.btn_export = CanvasButton(self.header, t, "导出报告",
                                       command=self.export, kind="ghost",
                                       icon="save", height=38, size=9,
                                       bg=t.BG)
        self.btn_export.pack(side="right", padx=(s(8), 0))
        self.btn_export.set_enabled(False)
        self.btn_rescan = CanvasButton(self.header, t, "重新扫描",
                                       command=lambda: self.app.show_page("scan"),
                                       kind="secondary", icon="refresh",
                                       height=38, size=9, bg=t.BG)
        self.btn_rescan.pack(side="right")

        self.body = tk.Frame(self, bg=t.BG)
        self.body.pack(fill="both", expand=True, pady=(s(16), 0))
        self._build_empty()
        self._build_scanning()
        self._build_result()
        self._show_state("empty")

    def _build_empty(self):
        t = self.theme
        self.empty_view = tk.Frame(self.body, bg=t.BG)
        self._fill_empty("default")

    def _fill_empty(self, mode="default", result=None):
        """重建空状态：default=还没扫描，clean=没有重复。"""
        t = self.theme
        for w in self.empty_view.winfo_children():
            w.destroy()
        center = tk.Frame(self.empty_view, bg=t.BG)
        center.place(relx=0.5, rely=0.44, anchor="center")
        s = t.s
        if mode == "clean":
            art = tk.Canvas(center, width=s(200), height=s(140), bg=t.BG,
                            highlightthickness=0, bd=0)
            art.pack()
            art.create_oval(s(29), s(9), s(171), s(151), fill=t.EMPTY_ART_BG,
                            outline=t.EMPTY_ART_BORDER)
            art.create_text(s(100), s(78), text=ICON["check"],
                            font=t.icon(42), fill=t.GREEN)
            tk.Label(center, text="很干净，没有发现重复文件", bg=t.BG, fg=t.TEXT,
                     font=t.font(14, "bold")).pack(pady=(s(4), 0))
            if result is not None:
                tk.Label(center, text=f"检查了 {result.file_count:,} 个文件，"
                                      f"共 {core.human_size(result.total_size)}",
                         bg=t.BG, fg=t.MUTED, font=t.font(9)).pack(pady=(s(7), 0))
            CanvasButton(center, t, "再次扫描",
                         command=lambda: self.app.show_page("scan"),
                         kind="secondary", icon="refresh", width=140, height=40,
                         size=9, bg=t.BG).pack(pady=(s(18), 0))
            return

        art = tk.Canvas(center, width=s(260), height=s(150), bg=t.BG,
                        highlightthickness=0, bd=0)
        art.pack()
        self._draw_placeholder_art(art, "layers", t.ACCENT)
        tk.Label(center, text="还没有扫描结果", bg=t.BG, fg=t.TEXT,
                 font=t.font(14, "bold")).pack(pady=(s(8), 0))
        tk.Label(center, text="选择一个文件夹，看看里面有哪些重复的照片、视频和文件。",
                 bg=t.BG, fg=t.MUTED, font=t.font(9)).pack(pady=(s(7), 0))
        CanvasButton(center, t, "去选择文件夹",
                     command=lambda: self.app.show_page("scan"),
                     kind="primary", icon="folder", width=160, height=40,
                     size=9, bg=t.BG).pack(pady=(s(20), 0))

    def _draw_placeholder_art(self, art, icon, color):
        t = self.theme
        s = t.s
        cx, cy = s(130), s(72)
        art.create_oval(cx - s(54), cy - s(54), cx + s(54), cy + s(54),
                        fill=t.STATUS_BG, outline=t.BORDER_SOFT)
        art.create_oval(cx - s(40), cy - s(40), cx + s(40), cy + s(40),
                        outline=t.RING_TRACK)
        art.create_text(cx, cy + s(1), text=ICON.get(icon, icon), font=t.icon(38),
                        fill=color)
        for dx, dy, ic, col in ((-s(76), -s(26), "image", t.AMBER),
                                (s(76), -s(10), "video", t.VIOLET),
                                (-s(62), s(44), "document", t.TEXT_DIM),
                                (s(66), s(48), "copy", t.ACCENT)):
            art.create_oval(cx + dx - s(14), cy + dy - s(14),
                            cx + dx + s(14), cy + dy + s(14),
                            fill=t.ART_INNER, outline=t.BORDER_SOFT)
            art.create_text(cx + dx, cy + dy + 1, text=ICON[ic], font=t.icon(13),
                            fill=col)

    def _build_scanning(self):
        t = self.theme
        s = t.s
        self.scan_view = tk.Frame(self.body, bg=t.BG)
        center = tk.Frame(self.scan_view, bg=t.BG)
        center.place(relx=0.5, rely=0.44, anchor="center")
        self.ring = tk.Canvas(center, width=s(156), height=s(156), bg=t.BG,
                              highlightthickness=0, bd=0)
        self.ring.pack()
        self.scan_title = tk.Label(center, text="正在扫描", bg=t.BG, fg=t.TEXT,
                                   font=t.font(14, "bold"))
        self.scan_title.pack(pady=(s(12), 0))
        self.scan_detail = tk.Label(center, text="准备中…", bg=t.BG, fg=t.TEXT_DIM,
                                    font=t.font(9), wraplength=s(520))
        self.scan_detail.pack(pady=(s(6), 0))
        self.steps = tk.Frame(center, bg=t.BG)
        self.steps.pack(pady=(s(16), 0))
        self.step_labels = {}
        self.cancel_btn = CanvasButton(center, t, "取消扫描", command=self.cancel_scan,
                                       kind="ghost", icon="stop", height=36,
                                       size=9, bg=t.BG)
        self.cancel_btn.pack(pady=(s(18), 0))
        self._draw_ring(0)

    def _build_result(self):
        t = self.theme
        s = t.s
        self.result_view = tk.Frame(self.body, bg=t.BG)

        self.chips = tk.Frame(self.result_view, bg=t.BG)
        self.chips.pack(fill="x", padx=s(32), pady=(0, s(13)))

        bar = tk.Frame(self.result_view, bg=t.BG)
        bar.pack(fill="x", padx=s(32), pady=(0, s(12)))
        self.filter_box = tk.Frame(bar, bg=t.BG)
        self.filter_box.pack(side="left")
        self.btn_sort = CanvasButton(bar, t, "排序", command=self._sort_menu,
                                     kind="ghost", icon="filter", height=34,
                                     size=8, padding=10, bg=t.BG)
        self.btn_sort.pack(side="left", padx=(s(12), 0))
        self.btn_keep = CanvasButton(bar, t, "保留策略", command=self._keep_menu,
                                     kind="ghost", icon="shield", height=34,
                                     size=8, padding=10, bg=t.BG)
        self.btn_keep.pack(side="left", padx=(s(6), 0))
        self.search = SearchBox(bar, t, width=214, on_change=self._on_search)
        self.search.pack(side="right")

        self.actionbar = tk.Frame(self.result_view, bg=t.BG_SOFT)
        self.actionbar.pack(side="bottom", fill="x")
        tk.Frame(self.actionbar, bg=t.BORDER_SOFT, height=1).pack(fill="x")
        inner = tk.Frame(self.actionbar, bg=t.BG_SOFT)
        inner.pack(fill="x", padx=s(32), pady=s(12))
        self.sel_icon = tk.Label(inner, text=ICON["trash"], bg=t.BG_SOFT,
                                 fg=t.MUTED, font=t.icon(14))
        self.sel_icon.pack(side="left")
        self.sel_label = tk.Label(inner, text="尚未选择要清理的文件", bg=t.BG_SOFT,
                                  fg=t.TEXT_DIM, font=t.font(9))
        self.sel_label.pack(side="left", padx=(s(7), s(10)))
        self.sel_size = tk.Label(inner, text="", bg=t.BG_SOFT, fg=t.GREEN,
                                 font=t.font(9, "bold"))
        self.sel_size.pack(side="left")
        self.btn_delete = CanvasButton(inner, t, "移入回收站",
                                       command=self.ask_delete, kind="danger",
                                       icon="trash", width=142, height=38,
                                       size=9, bg=t.BG_SOFT)
        self.btn_delete.pack(side="right")
        self.btn_clear = CanvasButton(inner, t, "取消选择", command=self.clear_selection,
                                      kind="ghost", height=38, size=9,
                                      bg=t.BG_SOFT, padding=12)
        self.btn_clear.pack(side="right", padx=(0, s(8)))
        self.btn_pick = CanvasButton(inner, t, "快捷选择", command=self._pick_menu,
                                     kind="ghost", icon="layers", height=38,
                                     size=9, bg=t.BG_SOFT, padding=12)
        self.btn_pick.pack(side="right", padx=(0, s(8)))
        self.btn_undo_clean = CanvasButton(inner, t, "还原上次清理",
                                           command=self.undo_clean, kind="ghost",
                                           icon="history", height=38, size=9,
                                           bg=t.BG_SOFT, padding=12)
        self.btn_undo_clean.pack(side="right", padx=(0, s(8)))
        self.btn_delete.set_enabled(False)
        self.btn_clear.set_enabled(False)
        self.btn_undo_clean.set_enabled(False)
        self.hint_label = tk.Label(inner, text="移入回收站后可随时还原", bg=t.BG_SOFT,
                                   fg=t.MUTED, font=t.font(8))
        self.hint_label.pack(side="right", padx=(0, s(16)))

        self.scroll = ScrollArea(self.result_view, t, bg=t.BG)
        self.scroll.pack(fill="both", expand=True, padx=(s(24), s(10)),
                         pady=(0, s(6)))

    # ------------------------------------------------------------ 状态切换
    def _show_state(self, name):
        for f in (self.empty_view, self.scan_view, self.result_view):
            f.pack_forget()
        if name == "scan":
            self.scan_view.pack(fill="both", expand=True)
        elif name == "result":
            self.result_view.pack(fill="both", expand=True)
        else:
            self.empty_view.pack(fill="both", expand=True)

    def on_show(self):
        pass

    # ------------------------------------------------------------ 扫描控制
    def start_scan(self, options):
        t = self.theme
        self.options = options
        self.scanning = True
        self.result = None
        self.busy = False
        self.selection.clear()
        self._size_map = {}
        self.app.scanning = True
        self.app.cancel_event.clear()
        self.app.set_result_count(0)
        self.app.set_status("正在扫描…", t.AMBER, "只读模式 · 不会修改文件")
        self.subtitle.configure(text=f"正在扫描  {short_root(options.root)}")
        self.btn_export.set_enabled(False)
        self._progress = 0.0
        self._target = 0.0
        self._draw_ring(0)
        self.scan_title.configure(text="正在扫描")
        self.scan_detail.configure(text="准备中…")
        self.cancel_btn.set_text("取消扫描")
        self.cancel_btn.set_enabled(True)
        self._build_steps()
        self._show_state("scan")
        self._start_tick()
        self.search.var.set("")
        self.query = ""
        self.kind = "全部"
        self.app.log(f"开始扫描：{options.root}", "info")
        if options.exclude:
            self.app.log(f"排除目录：{', '.join(options.exclude)}", "info")

        cancel = self.app.cancel_event

        def progress(stage, done, total, detail):
            self.app.post(("progress", (stage, done, total, detail)))

        def log(msg):
            self.app.post(("log", msg, self._guess_level(msg)))

        def work():
            try:
                result = core.scan_folder(options, on_progress=progress,
                                          on_log=log, cancel=cancel)
                self.app.post(("scan_done", (result,)))
            except core.ScanCancelled:
                self.app.post(("scan_cancelled", ()))
            except Exception as exc:
                self.app.post(("scan_error", (str(exc), traceback.format_exc())))

        threading.Thread(target=work, daemon=True, name="scan").start()

    @staticmethod
    def _guess_level(msg):
        if any(k in msg for k in ("失败", "错误", "无法", "未安装", "跳过")):
            return "warn"
        if "完成" in msg:
            return "ok"
        return "info"

    def _build_steps(self):
        t = self.theme
        for w in self.steps.winfo_children():
            w.destroy()
        self.step_labels = {}
        order = [("collect", "收集文件", True),
                 ("exact", "完全重复", bool(self.options.exact)),
                 ("photo", "相似照片", bool(self.options.photos)),
                 ("video", "重复视频", bool(self.options.videos)),
                 ("audio", "重复音乐", bool(getattr(self.options, "audio", False))),
                 ("name", "同名相似", bool(self.options.names))]
        for i, (key, label, enabled) in enumerate(order):
            if i:
                tk.Label(self.steps, text="›", bg=t.BG, fg=t.FAINT,
                         font=t.font(9)).pack(side="left", padx=t.s(6))
            lbl = tk.Label(self.steps, text=f"●  {label}", bg=t.BG,
                           fg=t.MUTED if enabled else t.FAINT,
                           font=t.font(8))
            lbl.pack(side="left")
            self.step_labels[key] = (lbl, enabled)

    def _start_tick(self):
        if self._tick_id:
            try:
                self.after_cancel(self._tick_id)
            except Exception:
                pass
        self._tick_id = self.after(40, self._tick)

    def _tick(self):
        self._tick_id = None
        if not self.scanning:
            return
        self._progress += (self._target - self._progress) * 0.18
        if abs(self._target - self._progress) < 0.001:
            self._progress = self._target
        self._draw_ring(self._progress)
        self._tick_id = self.after(40, self._tick)

    def _draw_ring(self, pct):
        t = self.theme
        s = t.s
        c = self.ring
        c.delete("all")
        size = s(156)
        w = s(9)
        pad = w / 2 + s(2)
        c.create_oval(pad, pad, size - pad, size - pad, outline=t.RING_TRACK, width=w)
        pct = max(0.0, min(1.0, pct))
        if pct > 0.002:
            c.create_arc(pad, pad, size - pad, size - pad, start=90,
                         extent=-359.9 * pct, style="arc", outline=t.ACCENT,
                         width=w)
        c.create_text(size / 2, size / 2 - s(8), text=f"{round(pct * 100)}%",
                      font=t.font(20, "bold"), fill=t.TEXT)
        c.create_text(size / 2, size / 2 + s(18), text="扫描进度",
                      font=t.font(8), fill=t.MUTED)

    def on_progress(self, stage, done, total, detail):
        if not self.scanning:
            return
        self.scan_detail.configure(text=detail)
        order = ["collect", "exact", "photo", "video", "audio", "name"]
        weights = {"collect": 7, "exact": 25, "photo": 27, "video": 28,
                   "audio": 10, "name": 3}
        enabled = ["collect"]
        if self.options.exact:
            enabled.append("exact")
        if self.options.photos:
            enabled.append("photo")
        if self.options.videos:
            enabled.append("video")
        if getattr(self.options, "audio", False):
            enabled.append("audio")
        if getattr(self.options, "names", False):
            enabled.append("name")
        total_w = sum(weights[k] for k in enabled) or 1
        base = 0
        for k in enabled:
            if k == stage:
                break
            base += weights[k]
        frac = (done / total) if total else 0.35
        frac = max(0.0, min(1.0, frac))
        target = (base + weights.get(stage, 0) * frac) / total_w
        self._target = max(self._target, target)
        names = {"collect": "收集文件", "exact": "完全重复", "photo": "相似照片",
                 "video": "重复视频", "audio": "重复音乐", "name": "同名相似"}
        self.scan_title.configure(text=names.get(stage, "正在扫描"))
        # 步骤状态
        current = order.index(stage) if stage in order else 0
        for key, (lbl, on) in self.step_labels.items():
            if not on:
                lbl.configure(fg=self.theme.FAINT)
            elif order.index(key) < current:
                lbl.configure(fg=self.theme.GREEN)
            elif key == stage:
                lbl.configure(fg=self.theme.ACCENT)
            else:
                lbl.configure(fg=self.theme.MUTED)

    def cancel_scan(self):
        if not self.scanning:
            return
        self.app.cancel_event.set()
        self.cancel_btn.set_text("正在停止…")
        self.cancel_btn.set_enabled(False)
        self.scan_title.configure(text="正在停止扫描…")
        self.scan_detail.configure(text="正在收尾，请稍候")
        self.app.log("用户取消了扫描", "warn")

    def on_cancelled(self):
        self.scanning = False
        self.app.scanning = False
        self.app.set_status("引擎就绪", self.theme.GREEN)
        self.btn_export.set_enabled(False)
        self.app.set_result_count(0)
        self.subtitle.configure(text="扫描已取消")
        self._fill_empty("default")
        self._show_state("empty")
        self.app.toast("已取消扫描", "warn")

    def on_error(self, message, tb):
        self.scanning = False
        self.app.scanning = False
        self.app.set_status("引擎就绪", self.theme.GREEN)
        self.app.log(f"扫描失败：{message}", "err")
        self.app.log(tb, "err")
        self._show_state("empty")
        self.app.toast("扫描失败，请查看日志", "err")
        alert(self.app, self.theme, "扫描失败", message[:800], icon="warning")

    def on_done(self, result):
        t = self.theme
        self.scanning = False
        self.app.scanning = False
        self.result = result
        self.app.result = result
        self.selection.clear()
        self._size_map = {}
        self._protected = {}
        self.keep_rule = self.settings.get("keep_rule", "best")
        self.sort_key = self.settings.get("sort_key", "wasted")
        for g in result.groups:
            g.keep = core._pick_keep(g.kind, g.items, self.keep_rule,
                                     result.root)
            self.selection |= g.default_selection()
            for item in g.items:
                self._size_map[str(item.path)] = item.size
                if item.protected:
                    self._protected[str(item.path)] = True
        try:
            self._last_cleanup = tools.last_restorable_cleanup()
            self.btn_undo_clean.set_enabled(self._last_cleanup is not None)
        except Exception:
            pass
        self.app.set_status("引擎就绪", t.GREEN)
        self.app.set_result_count(result.total_groups)
        self.subtitle.configure(
            text=f"{short_root(result.root)}    ·    {result.file_count:,} 个文件"
                 f"    ·    {result.elapsed:.1f} 秒")
        self.app.log(f"结果已就绪：{result.total_groups} 组 · "
                     f"{result.total_files} 个文件", "ok")
        if result.bad_photos or result.bad_videos:
            self.app.log(f"有 {result.bad_photos + result.bad_videos} 个文件无法解析，已跳过",
                         "warn")

        if not result.groups:
            self._show_state("empty")
            self.subtitle.configure(
                text=f"{short_root(result.root)}    ·    {result.file_count:,} 个文件")
            self._show_clean(result)
            self.app.toast("扫描完成，没有发现重复文件", "ok")
            return

        self._show_state("result")
        self.kind = "全部"
        self.query = ""
        self.search.var.set("")
        self.btn_export.set_enabled(True)
        self._rebuild_filters()
        self._update_summary()
        self._render_groups()
        self._update_footer()
        self.scroll.scroll_to_top()
        self.app.toast(f"扫描完成 · 找到 {result.total_groups} 组重复", "ok")

    def _show_clean(self, result):
        self._fill_empty("clean", result)

    # ------------------------------------------------------------ 汇总 / 过滤
    def _update_summary(self):
        t = self.theme
        if not self.result:
            return
        for w in self.chips.winfo_children():
            w.destroy()
        data = [
            ("layers", f"{self.result.total_groups} 组重复", t.ACCENT),
            ("document", f"{self.result.total_files} 个文件", t.TEXT),
            ("shield", f"可释放 {core.human_size(self.result.total_wasted)}", t.GREEN),
            ("clock", f"{self.result.elapsed:.1f} 秒", t.MUTED),
        ]
        for icon, text, color in data:
            Chip(self.chips, t, text=text, icon=icon, color=color,
                 bg=t.BG).pack(side="left", padx=(0, t.s(8)))
        if self.result.bad_photos or self.result.bad_videos:
            skipped = self.result.bad_photos + self.result.bad_videos
            Chip(self.chips, t, text=f"{skipped} 个无法解析", icon="warning",
                 color=t.AMBER, bg=t.BG).pack(side="left")

    def _rebuild_filters(self):
        t = self.theme
        for w in self.filter_box.winfo_children():
            w.destroy()
        if not self.result:
            return
        counts = self.result.counts()
        total = len(self.result.groups)
        options = [("全部", f"全部 {total}")]
        for kind in core.KIND_ORDER:
            n = counts[kind]["groups"]
            if n:
                options.append((kind, f"{kind} {n}"))
        values = [o[0] for o in options]
        if self.kind not in values:
            self.kind = "全部"
        self.filter_seg = Segmented(self.filter_box, t, options, value=self.kind,
                                    command=self._on_filter, bg=t.BG, size=8,
                                    padding=10)
        self.filter_seg.pack()

    def _on_filter(self, value):
        self.kind = value
        self._render_groups()
        self.scroll.scroll_to_top()

    def _on_search(self, value):
        self.query = value
        self._render_groups()
        self.scroll.scroll_to_top()

    def _visible_groups(self):
        if not self.result:
            return []
        q = self.query.lower().strip()
        out = []
        for g in self.result.groups:
            if self.kind != "全部" and g.kind != self.kind:
                continue
            if q and not any(q in item.name.lower() for item in g.items):
                continue
            out.append(g)
        if self.sort_key == "similarity":
            out.sort(key=lambda g: (-g.similarity, -g.wasted))
        elif self.sort_key == "files":
            out.sort(key=lambda g: (-len(g.items), -g.wasted))
        elif self.sort_key == "name":
            out.sort(key=lambda g: g.items[0].name.lower())
        else:
            out.sort(key=lambda g: -g.wasted)
        return out

    # ------------------------------------------------------------ 渲染
    def _render_groups(self):
        t = self.theme
        self._render_id += 1
        token = self._render_id
        self.scroll.clear()
        self.cards = []
        groups = self._visible_groups()
        if not groups:
            box = tk.Frame(self.scroll.inner, bg=t.BG)
            box.pack(pady=t.s(70))
            tk.Label(box, text=ICON["filter"], bg=t.BG, fg=t.FAINT,
                     font=t.icon(30)).pack()
            tk.Label(box, text="没有匹配的结果", bg=t.BG, fg=t.TEXT_DIM,
                     font=t.font(11, "bold")).pack(pady=(t.s(10), 0))
            tk.Label(box, text="换个分类，或者清空搜索关键词试试。", bg=t.BG,
                     fg=t.MUTED, font=t.font(9)).pack(pady=(t.s(4), 0))
            return
        self._render_batch(groups, 0, token)

    def _render_batch(self, groups, start, token):
        if token != self._render_id:
            return
        t = self.theme
        end = min(start + 6, len(groups))
        for g in groups[start:end]:
            card = GroupCard(self.scroll.inner, self.app, g)
            card.pack(fill="x", pady=(0, t.s(10)))
            self.cards.append(card)
        if end < len(groups):
            self.after(1, lambda: self._render_batch(groups, end, token))

    # ------------------------------------------------------------ 选择
    def _selected_count_in(self, group):
        return sum(1 for item in group.items if str(item.path) in self.selection)

    def toggle_item(self, card, index):
        group = card.group
        item = group.items[index]
        key = str(item.path)
        if index == group.keep:
            # 保留项不能勾选：否则「每组至少留一个」只是看起来成立，
            # 实际会把指标指向的保留文件一起删掉。要删它先改保留项。
            self.app.toast("这是当前保留的文件；要删它，先把别的文件设为保留",
                           "warn", 2600)
            return
        if key in self.selection:
            self.selection.discard(key)
        else:
            if item.protected or self._protected.get(key):
                self.app.toast("该文件在保护文件夹内，不会被清理", "warn")
                return
            if self._selected_count_in(group) >= len(group.items) - 1:
                self.app.toast("每组至少保留一个文件", "warn")
                return
            self.selection.add(key)
        card.refresh()
        self._update_footer()

    def set_keep(self, card, index):
        group = card.group
        group.keep = index
        self.selection -= {str(i.path) for i in group.items}
        self.selection |= group.default_selection()
        card.refresh()
        self._update_footer()
        self.app.toast(f"已保留「{group.items[index].name}」", "ok", 1800)

    def reset_group(self, card):
        group = card.group
        self.selection -= {str(i.path) for i in group.items}
        self.selection |= group.default_selection()
        card.refresh()
        self._update_footer()

    def clear_selection(self):
        self.selection.clear()
        for card in self.cards:
            card.refresh()
        self._update_footer()

    def _update_footer(self):
        t = self.theme
        selected = [p for p in self.selection if p in self._size_map]
        n = len(selected)
        size = sum(self._size_map.get(p, 0) for p in selected)
        if n:
            self.sel_icon.configure(fg=t.RED)
            self.sel_label.configure(text=f"已选 {n} 个文件", fg=t.TEXT)
            self.sel_size.configure(text=f"可释放 {core.human_size(size)}")
            self.hint_label.pack(side="right", padx=(0, t.s(16)))
        else:
            self.sel_icon.configure(fg=t.MUTED)
            self.sel_label.configure(text="尚未选择要清理的文件", fg=t.TEXT_DIM)
            self.sel_size.configure(text="")
            self.hint_label.pack_forget()
        self.btn_delete.set_enabled(n > 0 and not self.busy)
        self.btn_clear.set_enabled(n > 0 and not self.busy)

    # ------------------------------------------------------------ 排序 / 保留策略
    def _sort_menu(self):
        t = self.theme
        options = [("按可释放空间", "wasted"), ("按相似度", "similarity"),
                   ("按文件数", "files"), ("按文件名", "name")]
        items = [(label, "filter", lambda k=key: self._set_sort(k))
                 for label, key in options]
        x = self.btn_sort.winfo_rootx()
        y = self.btn_sort.winfo_rooty() + self.btn_sort.winfo_height() + 4
        PopupMenu(self.app, t, items, x=x, y=y)

    def _set_sort(self, key):
        self.sort_key = key
        self.settings["sort_key"] = key
        save_settings(self.settings)
        self._render_groups()
        self.scroll.scroll_to_top()
        self.app.toast("已重新排序", "info", 1400)

    def _keep_menu(self):
        t = self.theme
        items = [(label, "check", lambda k=key: self._set_keep_rule(k))
                 for key, label in core.KEEP_RULE_LABELS.items()]
        x = self.btn_keep.winfo_rootx()
        y = self.btn_keep.winfo_rooty() + self.btn_keep.winfo_height() + 4
        PopupMenu(self.app, t, items, x=x, y=y)

    def _set_keep_rule(self, rule):
        self.keep_rule = rule
        self.settings["keep_rule"] = rule
        save_settings(self.settings)
        if self.result:
            for group in self.result.groups:
                group.keep = core._pick_keep(group.kind, group.items, rule,
                                             self.result.root)
            self.selection.clear()
            for group in self.result.groups:
                self.selection |= group.default_selection()
            for card in self.cards:
                card.refresh()
            self._update_footer()
        self.app.toast(
            f"保留策略：{core.KEEP_RULE_LABELS.get(rule, rule)}", "info")

    # ------------------------------------------------------------ 快捷选择
    def _pick_menu(self):
        items = [
            ("全选除保留项", "check", self.pick_all_but_keep),
            ("只选相似度 ≥ 90%", "layers", lambda: self.pick_by_similarity(90)),
            ("只选完全重复", "document",
             lambda: self.pick_by_kind(core.KIND_EXACT)),
            None,
            ("取消选择", "refresh", self.clear_selection),
        ]
        PopupMenu(self.app, self.theme, items)

    def _select_visible(self, predicate):
        self.selection.clear()
        for group in self._visible_groups():
            if not predicate(group):
                continue
            for index, item in enumerate(group.items):
                if index == group.keep or item.protected:
                    continue
                self.selection.add(str(item.path))
        for card in self.cards:
            card.refresh()
        self._update_footer()

    def pick_all_but_keep(self):
        self._select_visible(lambda g: True)

    def pick_by_similarity(self, threshold):
        self._select_visible(
            lambda g: g.kind != core.KIND_NAME and g.similarity >= threshold)

    def pick_by_kind(self, kind):
        self._select_visible(lambda g: g.kind == kind)

    # ------------------------------------------------------------ 对比 / 还原
    def open_compare(self, card):
        group = getattr(card, "group", None)
        if not group or len(group.items) < 2:
            return None
        dialog = CompareDialog(self.app, group)
        self.app.log(f"对比：{group.kind} · {len(group.items)} 个文件", "info")
        return dialog

    def undo_clean(self):
        if self.busy:
            return
        path = self._last_cleanup or tools.last_restorable_cleanup()
        if not path or not Path(path).exists():
            self.app.toast("没有可以撤销的清理记录", "info")
            self.btn_undo_clean.set_enabled(False)
            return
        try:
            data = tools.read_manifest(path)
        except Exception:
            self.app.toast("清理记录损坏，无法还原", "err")
            return
        if not isinstance(data, dict):
            self.app.toast("清理记录损坏，无法还原", "err")
            return
        items = data.get("items", [])
        if not isinstance(items, list):
            self.app.toast("清理记录损坏，无法还原", "err")
            return
        paths = [r.get("src") for r in items
                 if isinstance(r, dict) and r.get("src")]
        if not paths:
            self.app.toast("这次清理没有可还原的文件", "info")
            return
        t = self.theme
        if confirm(self.app, t, "撤销上次清理",
                   f"把 {len(paths)} 个文件从回收站还原到原位置。\n"
                   "如果原位置已有同名文件，会跳过而不是覆盖。",
                   ok_text="还原", cancel_text="取消",
                   icon="history") != "ok":
            return
        self.busy = True
        self.btn_undo_clean.set_enabled(False)
        self.app.set_status("正在还原…", t.AMBER, "从回收站还原文件")

        def work():
            try:
                restored, failed = core.restore_from_recycle(paths)
            except Exception as exc:
                self.app.post(("log", f"还原失败：{exc}", "err"))
                restored, failed = [], paths
            self.app.post(("cleanup_restored", (restored, failed)))

        threading.Thread(target=work, daemon=True, name="cleanup-undo").start()

    def on_cleanup_restored(self, restored, failed):
        t = self.theme
        self.busy = False
        self.app.set_status("引擎就绪", t.GREEN)
        self.app.log(f"还原完成：{len(restored)} 个成功，{len(failed)} 个失败",
                     "ok" if not failed else "warn")
        if restored:
            self._last_cleanup = None
            self.app.toast(f"已还原 {len(restored)} 个文件，重新扫描即可看到",
                           "ok")
        else:
            self.btn_undo_clean.set_enabled(False)
        if failed:
            names = "\n".join(Path(p).name for p in failed[:6])
            alert(self.app, t, f"{len(failed)} 个文件未还原",
                  f"可能已被移出回收站，或原位置已有同名文件。\n\n{names}",
                  icon="warning")

    # ------------------------------------------------------------ 导出
    def export(self):
        if not self.result or not self.result.groups:
            return
        initial = f"重复文件报告_{datetime.now():%Y%m%d_%H%M}.csv"
        path = filedialog.asksaveasfilename(
            parent=self.app, title="导出检测报告", initialfile=initial,
            defaultextension=".csv", filetypes=[("CSV 表格", "*.csv")])
        if not path:
            return
        try:
            core.export_report(path, self.result.groups, self.result.root)
            self.app.toast("报告已导出", "ok")
            self.app.log(f"报告已导出：{path}", "ok")
        except Exception as exc:
            alert(self.app, self.theme, "导出失败", str(exc), icon="warning")

    # ------------------------------------------------------------ 清理
    def ask_delete(self):
        if self.busy:
            return
        paths = [p for p in self.selection if p in self._size_map]
        if not paths:
            self.app.toast("还没有选择要清理的文件", "warn")
            return
        size = sum(self._size_map[p] for p in paths)
        t = self.theme
        choice = confirm(
            self.app, t, "清理这些重复文件？",
            f"即将处理 {len(paths)} 个文件，预计释放 {core.human_size(size)}。\n"
            f"每组都会至少留下一个文件。",
            ok_text="移入回收站", cancel_text="取消", extra_text="永久删除…",
            danger=True, icon="trash",
            note="移到回收站的文件可以随时还原；不会影响未勾选的文件。")
        if choice == "ok":
            self._delete(paths, recycle=True)
        elif choice == "extra":
            again = confirm(
                self.app, t, "永久删除？",
                f"这 {len(paths)} 个文件将被直接删除，不会进入回收站，也无法恢复。",
                ok_text="永久删除", cancel_text="返回", danger=True,
                icon="warning",
                note="建议优先使用「移入回收站」，确认无误后再清空回收站。")
            if again == "ok":
                self._delete(paths, recycle=False)

    def _delete(self, paths, recycle):
        # 最后一道闸：保护目录里的文件、以及各组当前保留的文件都不进入清理表
        keep = {str(grp.items[grp.keep].path)
                for grp in (self.result.groups if self.result else [])
                if 0 <= grp.keep < len(grp.items)}
        blocked = [p for p in paths if self._protected.get(p) or p in keep]
        paths = [p for p in paths if p not in keep and
                 not self._protected.get(p)]
        if not paths:
            self.app.toast("所选文件都在保护文件夹内或属于保留项，已取消",
                           "warn")
            return
        if blocked:
            self.app.log(f"已跳过 {len(blocked)} 个保护/保留文件", "warn")
        # 先按计划写一份记录：删一半被杀也能撤销已进回收站的部分
        try:
            planned = [{"act": "recycle" if recycle else "delete",
                        "src": p, "size": self._size_map.get(p, 0)}
                       for p in paths]
            self._cleanup_partial = tools.save_manifest(
                "cleanup", self.result.root if self.result else Path("."),
                planned)
        except Exception:
            self._cleanup_partial = None
        t = self.theme
        self._last_recycle = recycle
        self.busy = True
        self.btn_delete.set_text("正在处理…")
        self.btn_delete.set_enabled(False)
        self.btn_clear.set_enabled(False)
        self.app.set_status("正在清理…", t.AMBER, "移入回收站" if recycle else "永久删除")
        self.app.log(("移入回收站" if recycle else "永久删除") +
                     f"：{len(paths)} 个文件，{core.human_size(sum(self._size_map.get(p, 0) for p in paths))}",
                     "warn")

        def work():
            try:
                if recycle:
                    deleted, failed = core.recycle_paths(paths)
                else:
                    deleted, failed = core.permanent_delete(paths)
                self.app.post(("deleted", (deleted, failed, recycle)))
            except Exception as exc:
                self.app.post(("deleted", ([], paths, recycle)))
                self.app.post(("log", f"清理失败：{exc}", "err"))

        threading.Thread(target=work, daemon=True, name="delete").start()

    def on_deleted(self, deleted, failed, recycle):
        t = self.theme
        self.busy = False
        self.btn_delete.set_text("移入回收站")
        self.app.set_status("引擎就绪", t.GREEN)
        deleted_set = set(deleted)
        freed = sum(self._size_map.get(p, 0) for p in deleted_set)
        if deleted:
            try:
                records = [{"act": "recycle" if recycle else "delete",
                            "src": p, "size": self._size_map.get(p, 0)}
                           for p in deleted]
                if getattr(self, "_cleanup_partial", None) is not None:
                    tools.rewrite_manifest(self._cleanup_partial, records)
                    self._last_cleanup = self._cleanup_partial
                else:
                    self._last_cleanup = tools.save_manifest(
                        "cleanup", self.result.root if self.result else Path("."),
                        records)
                self.btn_undo_clean.set_enabled(bool(recycle))
            except Exception:
                pass
        self._cleanup_partial = None
        self.app.log(f"清理完成：成功 {len(deleted)} 个，失败 {len(failed)} 个",
                     "ok" if not failed else "warn")
        if deleted:
            remain = []
            for group in self.result.groups:
                keep_path = (str(group.items[group.keep].path)
                             if 0 <= group.keep < len(group.items) else None)
                group.items = [i for i in group.items if str(i.path) not in deleted_set]
                if len(group.items) < 2:
                    continue
                idx = next((i for i, it in enumerate(group.items)
                            if str(it.path) == keep_path), None)
                group.keep = idx if idx is not None else core._pick_keep(group.kind,
                                                                         group.items)
                remain.append(group)
            self.result.groups = remain
            self.selection -= deleted_set
            for group in remain:
                if 0 <= group.keep < len(group.items):
                    self.selection.discard(str(group.items[group.keep].path))
            for item in list(self._size_map):
                if item in deleted_set:
                    self._size_map.pop(item, None)
            self.app.set_result_count(len(remain))
            self._rebuild_filters()
            self._update_summary()
            self._render_groups()
            self._update_footer()
            if not remain:
                self._show_state("empty")
                self._show_clean(self.result)
                self.subtitle.configure(text=f"{short_root(self.result.root)}    ·    已清理完毕")
                self.app.toast(f"已清理 {len(deleted)} 个文件", "ok")
                return
            self.app.toast(f"已清理 {len(deleted)} 个文件，释放 "
                           f"{core.human_size(freed)}", "ok")
        else:
            self._update_footer()

        if failed:
            names = "\n".join(Path(p).name for p in failed[:8])
            more = f"\n… 还有 {len(failed) - 8} 个" if len(failed) > 8 else ""
            alert(self.app, self.theme, f"{len(failed)} 个文件未能处理",
                  f"可能是权限、只读或文件被占用。\n\n{names}{more}",
                  icon="warning")


class CompareDialog(tk.Toplevel):
    """并排对比同一组里的两个文件：同步缩放、拖拽平移、可选差异热图。"""

    def __init__(self, app, group):
        super().__init__(app)
        t = app.theme
        self.app = app
        self.theme = t
        self.group = group
        others = [i for i in range(len(group.items)) if i != group.keep]
        self.right_index = others[0] if others else max(0, len(group.items) - 1)
        self.zoom = 0.0
        self.pan = [[0.0, 0.0], [0.0, 0.0]]
        self._drag = None
        self._diff = tk.BooleanVar(value=False)
        self._raw = [None, None]
        self._photos = [None, None]
        self._retries = 0
        self.title(f"对比 · {group.kind} · {len(group.items)} 个文件")
        self.configure(bg=t.BG)
        self.minsize(t.s(720), t.s(520))
        self.geometry(f"{t.s(1120)}x{t.s(780)}")
        self.transient(app.winfo_toplevel())
        self._build()
        self.after(60, lambda: enable_dark_titlebar(self, dark=(self.theme_name == "dark")))
        self.bind("<Escape>", lambda e: self.close())
        self.bind("<Left>", lambda e: self._cycle(-1))
        self.bind("<Right>", lambda e: self._cycle(1))
        self._load()

    # -------------------------------------------------- 构建
    def pair(self):
        left = self.group.items[self.group.keep]
        right = self.group.items[min(self.right_index, len(self.group.items) - 1)]
        return left, right

    def _build(self):
        t = self.theme
        s = t.s
        head = tk.Frame(self, bg=t.BG)
        head.pack(fill="x", padx=s(16), pady=(s(14), s(6)))
        self.title_left = tk.Label(head, bg=t.BG, fg=t.TEXT, font=t.font(10, "bold"),
                                   anchor="w")
        self.title_left.pack(side="left")
        self.title_right = tk.Label(head, bg=t.BG, fg=t.TEXT, font=t.font(10, "bold"),
                                    anchor="e")
        self.title_right.pack(side="right")

        body = tk.Frame(self, bg=t.BORDER_SOFT)
        body.pack(fill="both", expand=True, padx=s(16), pady=s(8))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1, uniform="col")
        body.grid_columnconfigure(1, weight=1, uniform="col")
        self.canvases = []
        for side in (0, 1):
            canvas = tk.Canvas(body, bg=t.CARD, highlightthickness=0, bd=0)
            canvas.grid(row=0, column=side, sticky="nsew",
                        padx=(0 if side == 0 else 1, 1 if side == 0 else 0),
                        pady=0)
            canvas.bind("<Configure>", lambda e: self._draw())
            canvas.bind("<MouseWheel>", self._wheel)
            canvas.bind("<ButtonPress-1>", self._press)
            canvas.bind("<B1-Motion>", self._move)
            canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
            self.canvases.append(canvas)

        foot = tk.Frame(self, bg=t.BG)
        foot.pack(fill="x", padx=s(16), pady=(s(4), s(14)))
        self.meta = tk.Label(foot, bg=t.BG, fg=t.MUTED, font=t.font(8), anchor="w")
        self.meta.pack(side="left")
        for text, command, kind in (
                ("关闭", self.close, "ghost"),
                ("定位", self._reveal, "ghost"),
                ("打开", self._open, "ghost"),
                ("100%", lambda: self._set_zoom(1.0), "ghost"),
                ("适应窗口", lambda: self._set_zoom(0.0), "ghost")):
            btn = CanvasButton(foot, t, text, command=command, kind=kind,
                               height=34, size=8, bg=t.BG, padding=12)
            btn.pack(side="right", padx=(s(6), 0))
        diff = tk.Checkbutton(foot, text="差异热图", variable=self._diff,
                              command=self._draw, bg=t.BG, fg=t.MUTED,
                              activebackground=t.BG, activeforeground=t.TEXT,
                              selectcolor=t.CARD, font=t.font(8), bd=0,
                              highlightthickness=0, cursor="hand2")
        diff.pack(side="right", padx=(s(10), s(6)))

    # -------------------------------------------------- 载入
    def _load(self):
        left, right = self.pair()
        self.title_left.configure(text=left.name)
        self.title_right.configure(text=right.name)
        self._raw[0] = self._read(left)
        self._raw[1] = self._read(right)
        same = (left.size == right.size)
        self.meta.configure(
            text=f"{core.human_size(left.size)} vs {core.human_size(right.size)}"
                 + (" · 大小相同" if same else "")
                 + " · 滚轮缩放，拖动平移，左右键切换版本")
        self.zoom = 0.0
        self.pan = [[0.0, 0.0], [0.0, 0.0]]
        self._draw()

    @staticmethod
    def _read(item):
        try:
            if core.is_photo(item.path):
                with Image.open(item.path) as source:
                    try:
                        source.draft("RGB", (2400, 2400))
                    except Exception:
                        pass
                    image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
                return image
            if core.is_video(item.path):
                frame = core.make_thumbnail(item.path, (960, 600), 0)
                if frame is not None:
                    return frame.convert("RGB")
        except Exception:
            return None
        return None

    # -------------------------------------------------- 绘制
    def _fit_scale(self, side):
        """适应窗口时的缩放比例；画布还没完成布局时返回 0。"""
        image = self._raw[side]
        canvas = self.canvases[side]
        if image is None:
            return 0.0
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 40 or ch < 40:
            return 0.0
        return min(cw / max(image.width, 1), ch / max(image.height, 1), 4.0)

    def _draw(self):
        """两边用同一个比例重绘；布局未就绪时稍后重试。"""
        if not self.winfo_ismapped() or any(
                c.winfo_width() < 40 or c.winfo_height() < 40
                for c in self.canvases):
            self._retries += 1
            if self._retries < 40:
                self.after(80, self._draw)
            return
        self._retries = 0
        if self.zoom > 0:
            scale = self.zoom
        else:
            fits = [s for s in (self._fit_scale(0), self._fit_scale(1)) if s > 0]
            if not fits:
                self.after(80, self._draw)
                return
            scale = min(fits)
        for side in (0, 1):
            self._draw_side(side, scale)

    def _draw_side(self, side, scale=None):
        t = self.theme
        canvas = self.canvases[side]
        canvas.delete("all")
        cw = max(canvas.winfo_width(), 20)
        ch = max(canvas.winfo_height(), 20)
        image = self._raw[side]
        if image is None:
            item = self.pair()[side]
            canvas.create_text(
                cw / 2, ch / 2,
                text=("这个格式无法直接预览\n" if not core.is_photo(item.path)
                      else "读取失败\n") + "可用“打开”按钮查看原文件",
                fill=t.MUTED, font=t.font(9), justify="center")
            return
        if side == 1 and self._diff.get():
            image = self._difference() or image
        if scale is None:
            scale = self.zoom if self.zoom > 0 else (self._fit_scale(side) or 1.0)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        shown = image if size == image.size else image.resize(
            size, Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST)
        try:
            photo = ImageTk.PhotoImage(shown)
        except Exception:
            return
        self._photos[side] = photo
        canvas.create_image(cw / 2 + self.pan[side][0], ch / 2 + self.pan[side][1],
                            image=photo, anchor="center")

    def _difference(self):
        if not (Image and ImageChops) or self._raw[0] is None or self._raw[1] is None:
            return None
        left = self._raw[0]
        try:
            right = self._raw[1]
            if right.size != left.size:
                # 等比缩放后居中放到左图尺寸的黑底上，不拉伸变形
                ratio = min(left.width / right.width,
                            left.height / right.height)
                scaled = right.resize(
                    (max(1, int(right.width * ratio)),
                     max(1, int(right.height * ratio))),
                    Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", left.size, (0, 0, 0))
                canvas.paste(scaled, ((left.width - scaled.width) // 2,
                                      (left.height - scaled.height) // 2))
                right = canvas
            diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
            gray = diff.convert("L")
            # 重压缩造成的差异本身很淡，按 98% 分位归一化后映射成热图：
            # 黑（无差异）→ 红 → 黄（差异很大），一眼能看出改动位置
            hist = gray.histogram()
            total = sum(hist) or 1
            acc, scale = 0, 8
            for value, count in enumerate(hist):
                acc += count
                if acc >= total * 0.98:
                    scale = max(value, 8)
                    break
            red, green, blue = [], [], []
            for value in range(256):
                ratio = min(1.0, value / scale)
                red.append(int(255 * min(1.0, ratio * 2.2)))
                green.append(int(255 * max(0.0, ratio - 0.45) ** 1.4))
                blue.append(int(255 * max(0.0, ratio - 0.75) ** 1.4))
            return Image.merge("RGB", (gray.point(red), gray.point(green),
                                       gray.point(blue)))
        except Exception:
            return None

    # -------------------------------------------------- 交互
    def _set_zoom(self, value):
        self.zoom = value
        if value == 0.0:
            self.pan = [[0.0, 0.0], [0.0, 0.0]]
        self._draw()

    def _wheel(self, event):
        side = self.canvases.index(event.widget) if event.widget in self.canvases else 0
        base = self.zoom if self.zoom > 0 else (self._fit_scale(side) or 1.0)
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.zoom = max(0.05, min(12.0, base * factor))
        self._draw()
        return "break"

    def _press(self, event):
        self._drag = (event.x, event.y,
                      [list(self.pan[0]), list(self.pan[1])])

    def _move(self, event):
        if not self._drag:
            return
        x0, y0, base = self._drag
        dx, dy = event.x - x0, event.y - y0
        self.pan = [[base[0][0] + dx, base[0][1] + dy],
                    [base[1][0] + dx, base[1][1] + dy]]
        self._draw()

    def _cycle(self, step):
        others = [i for i in range(len(self.group.items)) if i != self.group.keep]
        if len(others) < 2:
            return
        try:
            pos = others.index(self.right_index)
        except ValueError:
            pos = 0
        self.right_index = others[(pos + step) % len(others)]
        self._load()

    def _open(self, index=None):
        item = self.pair()[1 if index is None else index]
        self.app.open_file(item.path)

    def _reveal(self):
        self.app.reveal_file(self.pair()[0].path)

    def close(self):
        try:
            if self.grab_current() == self:
                self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


# ---------------------------------------------------------------- 分组卡片
class GroupCard(tk.Canvas):
    def __init__(self, master, app, group):
        t = app.theme
        super().__init__(master, bg=t.BG, highlightthickness=0, bd=0,
                         height=t.s(210))
        self.app = app
        self.theme = t
        self.group = group
        self._width = 0
        self._tiles = {}
        self._thumbs = {}
        self._boxes = {}
        self._click_job = None
        self._rendered = False
        self.bind("<Configure>", self._on_configure)
        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, _=None):
        if self._click_job:
            try:
                self.after_cancel(self._click_job)
            except Exception:
                pass

    def _on_configure(self, event):
        if abs(event.width - self._width) > 2:
            self._width = event.width
            self._render()

    def refresh(self):
        self._render()

    def _render(self):
        t = self.theme
        s = t.s
        w = self._width or self.winfo_width()
        if w < 60:
            return
        self._rendered = True
        pad = s(18)
        gap = s(12)
        tw = s(148)
        th = s(96)
        label_h = s(42)
        head_h = s(30)
        n = len(self.group.items)
        per = max(1, (w - pad * 2 + gap) // (tw + gap))
        rows = max(1, math.ceil(n / per))
        h = pad * 2 + head_h + s(10) + rows * (th + label_h + gap) - gap
        self.configure(height=h)

        self.delete("all")
        round_rect(self, 0.5, 0.5, w - 0.5, h - 0.5, s(14),
                   fill=t.CARD, outline=t.BORDER_SOFT)
        self._draw_header(w, pad, head_h)

        base_y = pad + head_h + s(10)
        self._tiles = {}
        self._boxes = {}
        for i, item in enumerate(self.group.items):
            col = i % per
            row = i // per
            x = pad + col * (tw + gap)
            y = base_y + row * (th + label_h + gap)
            self._tiles[i] = item
            self._boxes[i] = (x, y)
            self._draw_tile(i, item, x, y, tw, th)
        self._rendered = True

    def _draw_header(self, w, pad, head_h):
        t = self.theme
        s = t.s
        g = self.group
        kind_color = t.kind_color(g.kind)
        dim = t.kind_dim(g.kind)
        icons = {KIND_EXACT: "document", KIND_PHOTO: "image",
                 KIND_VIDEO: "video", KIND_AUDIO: "audio", KIND_NAME: "copy"}
        f = t.font(8, "bold")
        bw = f.measure(g.kind) + s(28)
        by = pad
        round_rect(self, pad, by, pad + bw, by + s(24), s(8),
                   fill=dim, outline="")
        self.create_text(pad + s(11), by + s(12) + 0.5,
                         text=ICON.get(icons.get(g.kind, "document"), ""),
                         font=t.icon(11), fill=kind_color, anchor="center")
        self.create_text(pad + s(20), by + s(12) + 0.5, text=g.kind,
                         font=f, fill=kind_color, anchor="w")

        f9 = t.font(9)
        right = w - pad
        open_text = "打开位置"
        ow = f9.measure(open_text)
        self.create_text(right, by + s(12) + 0.5, text=open_text, font=f9,
                         fill=t.MUTED, anchor="e", tags=("act_open",))
        self.tag_bind("act_open", "<Button-1>", lambda e: self._open_folder())
        self.tag_bind("act_open", "<Enter>", lambda e: self._action_hover("act_open", True))
        self.tag_bind("act_open", "<Leave>", lambda e: self._action_hover("act_open", False))

        right -= ow + s(14)
        compare_text = "对比"
        cw = f9.measure(compare_text)
        self.create_text(right, by + s(12) + 0.5, text=compare_text, font=f9,
                         fill=t.MUTED, anchor="e", tags=("act_compare",))
        self.tag_bind("act_compare", "<Button-1>",
                      lambda e: self.app.results_page.open_compare(self))
        self.tag_bind("act_compare", "<Enter>",
                      lambda e: self._action_hover("act_compare", True))
        self.tag_bind("act_compare", "<Leave>",
                      lambda e: self._action_hover("act_compare", False))

        right -= cw + s(14)
        reset_text = "重置"
        rw = f9.measure(reset_text)
        self.create_text(right, by + s(12) + 0.5, text=reset_text, font=f9,
                         fill=t.MUTED, anchor="e", tags=("act_reset",))
        self.tag_bind("act_reset", "<Button-1>",
                      lambda e: self.app.results_page.reset_group(self))
        self.tag_bind("act_reset", "<Enter>", lambda e: self._action_hover("act_reset", True))
        self.tag_bind("act_reset", "<Leave>", lambda e: self._action_hover("act_reset", False))

        right -= rw + s(16)
        wasted_text = f"可释放 {core.human_size(g.wasted)}"
        self.create_text(right, by + s(12) + 0.5, text=wasted_text,
                         font=t.font(8, "bold"), fill=t.GREEN, anchor="e")
        right -= t.font(8, "bold").measure(wasted_text) + s(16)

        info = f"{len(g.items)} 个版本"
        if g.note:
            info += f" · {g.note}"
        avail = max(s(40), right - (pad + bw + s(10)))
        info = ellipsize(t.font(8), info, avail)
        self.create_text(pad + bw + s(10), by + s(12) + 0.5, text=info,
                         font=t.font(8), fill=t.MUTED, anchor="w")

    def _draw_tile(self, i, item, x, y, tw, th):
        t = self.theme
        s = t.s
        g = self.group
        selected = str(item.path) in self.app.results_page.selection
        keep = i == g.keep
        frame_tag = f"frame{i}"
        tile_tag = f"tile{i}"

        outline = t.RED if selected else (t.BORDER if keep else t.BORDER_SOFT)
        width = s(2) if selected else s(1)
        round_rect(self, x, y, x + tw, y + th, s(10), fill=t.TILE_BG,
                   outline=outline, width=width, tags=(frame_tag,))

        # 占位
        ph = f"ph{i}"
        is_vid = core.is_video(item.path)
        if is_vid:
            ic, col = "play", t.VIOLET
        elif core.is_photo(item.path):
            ic, col = "image", t.AMBER
        else:
            ic, col = "document", t.TEXT_DIM
        self.create_text(x + tw / 2, y + th / 2 - (s(8) if not is_vid else 0),
                         text=ICON[ic], font=t.icon(22), fill=col,
                         tags=(ph, tile_tag))
        if not core.is_photo(item.path) and not is_vid:
            ext = item.path.suffix.replace(".", "").upper() or "FILE"
            self.create_text(x + tw / 2, y + th / 2 + s(17), text=ext[:6],
                             font=t.font(7, "bold"), fill=t.MUTED,
                             tags=(ph, tile_tag))

        # 勾选圈
        cx, cy, r = x + s(11), y + s(11), s(9)
        chk_tag = f"chk{i}"
        if selected:
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=t.RED,
                             outline=t.CHECK_BORDER, width=s(1), tags=(chk_tag, tile_tag))
            self.create_text(cx, cy + 0.5, text=ICON["check"], font=t.icon(10),
                             fill="#FFFFFF", tags=(chk_tag, tile_tag))
        else:
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=t.TILE_CHECK_BG,
                             outline=t.TILE_CHECK_BORDER, width=s(1), tags=(chk_tag, tile_tag))

        # 推荐保留
        if keep:
            label = "保留"
            f = t.font(7, "bold")
            bw = f.measure(label) + s(14)
            bx2 = x + tw - s(8)
            bx1 = bx2 - bw
            by = y + s(8)
            kb = f"keepbadge{i}"
            round_rect(self, bx1, by, bx2, by + s(20), s(7),
                       fill=t.KEEP_BADGE_BG, outline=t.KEEP_BADGE_BORDER, tags=(kb,))
            self.create_text((bx1 + bx2) / 2, by + s(10) + 0.5, text=label,
                             font=f, fill=t.GREEN, tags=(kb,))
            self.tag_bind(kb, "<Button-1>",
                          lambda e, idx=i: self.app.results_page.set_keep(self, idx))
            self.tag_bind(kb, "<Enter>", lambda e: self.configure(cursor="hand2"))
            self.tag_bind(kb, "<Leave>", lambda e: self.configure(cursor="arrow"))

        # 文件名 / 信息
        fname = ellipsize(t.font(9), item.name, tw - s(4))
        self.create_text(x, y + th + s(8), text=fname, font=t.font(9),
                         fill=t.TEXT if not selected else t.TEXT_ON_RED,
                         anchor="nw", tags=(tile_tag,))
        meta = self._meta_text(item)
        meta = ellipsize(t.font(8), meta, tw - s(4))
        self.create_text(x, y + th + s(25), text=meta, font=t.font(8),
                         fill=t.MUTED, anchor="nw", tags=(tile_tag,))

        # 事件
        for tag in (tile_tag, frame_tag):
            self.tag_bind(tag, "<Button-1>", lambda e, idx=i: self._single(idx))
            self.tag_bind(tag, "<Double-Button-1>", lambda e, idx=i: self._double(idx))
            self.tag_bind(tag, "<Button-3>", lambda e, idx=i: self._menu(e, idx))
            self.tag_bind(tag, "<Enter>", lambda e: self.configure(cursor="hand2"))
            self.tag_bind(tag, "<Leave>", lambda e: self.configure(cursor="arrow"))

        self._request_thumb(i, tw, th)

    def _meta_text(self, item):
        parts = [core.human_size(item.size)]
        if item.duration:
            parts.append(core.format_duration(item.duration))
        if item.dims:
            parts.append(item.dims)
        if len(parts) == 1:
            parts.append(core.format_date(item.mtime))
        return " · ".join(parts)

    # ---- 缩略图
    def _request_thumb(self, i, tw, th):
        if ImageTk is None:
            return
        item = self.group.items[i]
        self.app.thumb_service.request(
            item.path, item.mtime, (tw, th),
            lambda photo, idx=i: self._set_thumb(idx, photo))

    def _set_thumb(self, i, photo):
        if not self.winfo_exists() or photo is None or i not in self._boxes:
            return
        t = self.theme
        s = t.s
        x, y = self._boxes[i]
        tw, th = s(148), s(96)
        self.delete(f"ph{i}")
        self.delete(f"img{i}")
        self._thumbs[i] = photo
        self.create_image(x + tw / 2, y + th / 2, image=photo,
                          tags=(f"img{i}", f"tile{i}", f"frame{i}"))
        self.tag_raise(f"chk{i}")
        if i == self.group.keep:
            self.tag_raise(f"keepbadge{i}")

    # ---- 交互
    def _single(self, index):
        if self._click_job:
            try:
                self.after_cancel(self._click_job)
            except Exception:
                pass
        self._click_job = self.after(
            220, lambda: self._fire_toggle(index))

    def _fire_toggle(self, index):
        self._click_job = None
        self.app.results_page.toggle_item(self, index)

    def _double(self, index):
        if self._click_job:
            try:
                self.after_cancel(self._click_job)
            except Exception:
                pass
            self._click_job = None
        self.app.open_file(self.group.items[index].path)

    def _menu(self, event, index):
        item = self.group.items[index]
        menu = [
            ("设为保留项", "check", lambda: self.app.results_page.set_keep(self, index)),
            ("对比这个文件", "compare",
             lambda: self.app.results_page.open_compare(self)),
            ("取消本组选择", "refresh", lambda: self.app.results_page.reset_group(self)),
            None,
            ("打开文件", "open", lambda: self.app.open_file(item.path)),
            ("在资源管理器中显示", "open_folder", lambda: self.app.reveal_file(item.path)),
            ("复制完整路径", "copy", lambda: self.app.copy_text(item.path)),
        ]
        PopupMenu(self.app, self.theme, menu, x=event.x_root, y=event.y_root)

    def _open_folder(self):
        if self.group.items:
            self.app.reveal_file(self.group.items[0].path)

    def _action_hover(self, tag, on):
        self.configure(cursor="hand2" if on else "arrow")
        for item in self.find_withtag(tag):
            self.itemconfigure(item, fill=self.theme.TEXT if on else self.theme.MUTED)


# ---------------------------------------------------------------- 工具页基类
class ToolPage(tk.Frame):
    """整理 / 重命名页共用的小组件。"""

    def __init__(self, master, app):
        self.app = app
        self.theme = app.theme
        self.settings = app.settings
        super().__init__(master, bg=self.theme.BG)
        self.busy = False
        self._pending = None

    def _section_title(self, parent, text, icon, desc=None):
        t = self.theme
        row = tk.Frame(parent, bg=t.CARD)
        row.pack(fill="x")
        tk.Label(row, text=ICON.get(icon, icon), bg=t.CARD, fg=t.ACCENT,
                 font=t.icon(13)).pack(side="left")
        tk.Label(row, text=text, bg=t.CARD, fg=t.TEXT,
                 font=t.font(11, "bold"), anchor="w").pack(
            side="left", padx=(t.s(7), 0))
        if desc:
            tk.Label(parent, text=desc, bg=t.CARD, fg=t.MUTED,
                     font=t.font(8), anchor="w").pack(
                fill="x", pady=(t.s(4), 0))

    def _field(self, parent, var, hint=None, font=9, width=None):
        t = self.theme
        field = tk.Frame(parent, bg=t.INPUT, highlightthickness=1,
                         highlightbackground=t.BORDER,
                         highlightcolor=t.ACCENT)
        kw = {"width": width} if width else {}
        entry = tk.Entry(field, textvariable=var, bg=t.INPUT, fg=t.TEXT,
                         insertbackground=t.TEXT, relief="flat", bd=0,
                         highlightthickness=0, font=t.font(font), **kw)
        entry.pack(fill="x", padx=t.s(10), ipady=t.s(5))
        field.entry = entry
        if hint:
            ph = tk.Label(field, text=hint, bg=t.INPUT, fg=t.FAINT,
                          font=t.font(8), cursor="xterm")
            ph.place(x=t.s(10), rely=0.5, anchor="w")
            ph.bind("<Button-1>", lambda e: entry.focus_set())

            def sync(*_):
                if var.get():
                    ph.place_forget()
                elif entry.focus_get() is not entry:
                    ph.place(x=t.s(10), rely=0.5, anchor="w")

            var.trace_add("write", sync)
            entry.bind("<FocusIn>", lambda e: ph.place_forget(), add="+")
            entry.bind("<FocusOut>", lambda e: sync(), add="+")
            sync()
        return field

    def _path_card(self, parent, title, var, command, hint):
        t = self.theme
        s = t.s
        card = Card(parent, t, padding=16)
        card.pack(fill="x", pady=(s(18), 0))
        body = card.body
        bubble = tk.Canvas(body, width=s(42), height=s(42), bg=t.CARD,
                           highlightthickness=0, bd=0)
        bubble.pack(side="left")
        round_rect(bubble, 0, 0, s(42), s(42), s(11), fill=t.BUBBLE_BG,
                   outline="")
        bubble.create_text(s(21), s(21) + 1, text=ICON["folder"],
                           font=t.icon(19), fill=t.ACCENT, anchor="center")
        box = tk.Frame(body, bg=t.CARD)
        box.pack(side="left", fill="x", expand=True, padx=(s(13), s(14)))
        tk.Label(box, text=title, bg=t.CARD, fg=t.MUTED, font=t.font(8),
                 anchor="w").pack(fill="x")
        field = self._field(box, var, hint, font=10)
        field.pack(fill="x", pady=(s(5), 0))
        tk.Label(box, text="可直接输入或粘贴路径", bg=t.CARD, fg=t.FAINT,
                 font=t.font(8), anchor="w").pack(fill="x", pady=(s(4), 0))
        CanvasButton(body, t, "选择文件夹", command=command, kind="secondary",
                     icon="folder", height=38, size=9, padding=14,
                     bg=t.CARD).pack(side="right")
        return field.entry

    def _preview_row(self, parent, left, right, tone="normal", note=""):
        t = self.theme
        s = t.s
        row = tk.Frame(parent, bg=t.CARD)
        row.pack(fill="x", pady=s(1))
        f = t.font(9)
        colors = {"normal": t.TEXT, "muted": t.MUTED, "warn": t.AMBER,
                  "ok": t.GREEN}
        tk.Label(row, text=ellipsize(f, left, s(280)), bg=t.CARD, fg=t.TEXT,
                 font=f, anchor="w").pack(side="left")
        tk.Label(row, text="→", bg=t.CARD, fg=t.FAINT,
                 font=t.font(9)).pack(side="left", padx=s(8))
        tk.Label(row, text=ellipsize(f, right, s(320)), bg=t.CARD,
                 fg=colors.get(tone, t.TEXT), font=f, anchor="w").pack(
            side="left", fill="x", expand=True)
        if note:
            tk.Label(row, text=note, bg=t.CARD, fg=t.MUTED,
                     font=t.font(8)).pack(side="right", padx=s(4))

    def _browse_into(self, var, title, entry=None):
        cur = var.get().strip().strip('"')
        initial = cur if cur and os.path.isdir(cur) else str(Path.home())
        chosen = filedialog.askdirectory(parent=self, title=title,
                                          initialdir=initial, mustexist=True)
        if chosen:
            var.set(os.path.normpath(chosen))
            if entry is not None:
                entry.icursor("end")
                entry.focus_set()

    def _page_setup(self, subtitle):
        t = self.theme
        s = t.s
        self.outer = tk.Frame(self, bg=t.BG)
        self.outer.pack(fill="both", expand=True)
        self.scroll = ScrollArea(self.outer, t, bg=t.BG)
        self.scroll.pack(fill="both", expand=True, padx=(s(24), s(8)),
                         pady=(s(20), 0))
        self.page = tk.Frame(self.scroll.inner, bg=t.BG)
        self.page.pack(fill="x", padx=(s(10), s(18)))
        header = tk.Frame(self.page, bg=t.BG)
        header.pack(fill="x")
        left = tk.Frame(header, bg=t.BG)
        left.pack(side="left")
        tk.Label(left, text=subtitle[0], bg=t.BG, fg=t.TEXT,
                 font=t.font(21, "bold"), anchor="w").pack(fill="x")
        tk.Label(left, text=subtitle[1], bg=t.BG, fg=t.TEXT_DIM,
                 font=t.font(10), anchor="w").pack(fill="x",
                                                   pady=(s(5), 0))
        self.header_right = tk.Frame(header, bg=t.BG)
        self.header_right.pack(side="right", pady=(s(4), 0))
        return self.page


# ---------------------------------------------------------------- 文件整理页
class OrganizePage(ToolPage):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.plan_rows = []
        self.summary = {}
        self.root = None
        self._build()

    def _build(self):
        t = self.theme
        s = t.s
        self._page_setup(("文件整理", "按类型或日期把文件归到子文件夹，可随时撤销。"))
        self.undo_btn = CanvasButton(self.header_right, t, "撤销上次整理",
                                     command=self.undo, kind="ghost",
                                     icon="history", height=38, size=9,
                                     bg=t.BG)
        self.undo_btn.pack()
        self.undo_btn.set_enabled(tools.last_manifest_multi("organize", "finder") is not None)

        self.path_var = tk.StringVar(
            value=self.settings.get("org_dir")
            or self.settings.get("last_dir", ""))
        entry = self._path_card(self.page, "要整理的文件夹", self.path_var,
                                self._browse, "整理只影响这个文件夹内部")
        entry.bind("<Return>", lambda e: self.preview())

        options = tk.Frame(self.page, bg=t.BG)
        options.pack(fill="x", pady=(s(16), 0))
        options.grid_columnconfigure(0, weight=1, uniform="org")
        options.grid_columnconfigure(1, weight=1, uniform="org")

        left_card = Card(options, t, padding=20)
        left_card.grid(row=0, column=0, sticky="n", padx=(0, s(8)))
        d = left_card.body
        self._section_title(d, "整理方式", "layers",
                            "照片 / 视频 / 文档… 或 年 / 月 / 日")
        self.seg_mode = Segmented(
            d, t, [("type", "按类型"), ("date", "按日期")],
            value=self.settings.get("org_mode", "type"),
            command=self._mode_changed, bg=t.CARD)
        self.seg_mode.pack(anchor="w", pady=(s(14), 0))

        self.type_box = tk.Frame(d, bg=t.CARD)
        row = tk.Frame(self.type_box, bg=t.CARD)
        row.pack(fill="x")
        box = tk.Frame(row, bg=t.CARD)
        box.pack(side="left", fill="x", expand=True)
        tk.Label(box, text="保留两级目录", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), anchor="w").pack(fill="x")
        tk.Label(box, text="照片/JPG、文档/PDF… 关闭后只分到「照片 / 文档」",
                 bg=t.CARD, fg=t.MUTED, font=t.font(8), anchor="w").pack(
            fill="x", pady=(s(2), 0))
        self.sw_two_level = Switch(
            row, t, value=bool(self.settings.get("org_two_level", True)),
            bg=t.CARD)
        self.sw_two_level.pack(side="right", pady=(s(4), 0))

        self.date_box = tk.Frame(d, bg=t.CARD)
        tk.Label(self.date_box, text="日期粒度", bg=t.CARD, fg=t.TEXT_DIM,
                 font=t.font(9), anchor="w").pack(fill="x", pady=(0, s(6)))
        self.seg_gran = Segmented(
            self.date_box, t,
            [("year", "年"), ("month", "年-月"), ("day", "年-月-日")],
            value=self.settings.get("org_granularity", "month"),
            bg=t.CARD, padding=10)
        self.seg_gran.pack(anchor="w")
        exif_row = tk.Frame(self.date_box, bg=t.CARD)
        exif_row.pack(fill="x", pady=(s(12), 0))
        exif_box = tk.Frame(exif_row, bg=t.CARD)
        exif_box.pack(side="left", fill="x", expand=True)
        tk.Label(exif_box, text="优先拍摄日期", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), anchor="w").pack(fill="x")
        tk.Label(exif_box, text="读取照片 EXIF，失败时用修改日期",
                 bg=t.CARD, fg=t.MUTED, font=t.font(8), anchor="w").pack(
            fill="x", pady=(s(2), 0))
        self.sw_exif = Switch(exif_row, t,
                              value=bool(self.settings.get("org_exif", True)),
                              bg=t.CARD)
        self.sw_exif.pack(side="right", pady=(s(4), 0))

        right_card = Card(options, t, padding=20)
        right_card.grid(row=0, column=1, sticky="n", padx=(s(8), 0))
        o = right_card.body
        self._section_title(o, "执行方式", "settings", "建议先用「预览」确认目标位置")
        self.seg_action = Segmented(
            o, t, [("move", "移动"), ("copy", "复制")],
            value=self.settings.get("org_action", "move"), bg=t.CARD)
        self.seg_action.pack(anchor="w", pady=(s(14), 0))
        for attr, title, desc, key in (
                ("sw_recursive", "包含子文件夹", "关闭后只整理当前目录",
                 "org_recursive"),
                ("sw_cleanup", "删除空文件夹", "移动完成后清理变空的旧目录",
                 None)):
            row2 = tk.Frame(o, bg=t.CARD)
            row2.pack(fill="x", pady=(s(13), 0))
            box2 = tk.Frame(row2, bg=t.CARD)
            box2.pack(side="left", fill="x", expand=True)
            tk.Label(box2, text=title, bg=t.CARD, fg=t.TEXT,
                     font=t.font(9), anchor="w").pack(fill="x")
            tk.Label(box2, text=desc, bg=t.CARD, fg=t.MUTED,
                     font=t.font(8), anchor="w").pack(
                fill="x", pady=(s(2), 0))
            sw = Switch(row2, t,
                        value=bool(self.settings.get(key, True))
                        if key else True, bg=t.CARD)
            sw.pack(side="right", pady=(s(4), 0))
            setattr(self, attr, sw)

        note = tk.Frame(o, bg=t.CARD)
        note.pack(fill="x", pady=(s(15), 0))
        tk.Frame(note, bg=t.BORDER_SOFT, height=1).pack(fill="x",
                                                        pady=(0, s(10)))
        tk.Label(note, text="每次整理都会生成操作记录，\n可点右上角「撤销上次整理」还原。",
                 bg=t.CARD, fg=t.MUTED, font=t.font(8), anchor="w",
                 justify="left").pack(fill="x")

        # 预览区
        prev_head = tk.Frame(self.page, bg=t.BG)
        prev_head.pack(fill="x", pady=(s(20), s(8)))
        tk.Label(prev_head, text="整理预览", bg=t.BG, fg=t.TEXT,
                 font=t.font(12, "bold"), anchor="w").pack(side="left")
        self.preview_meta = tk.Label(prev_head, text="还没有生成计划", bg=t.BG,
                                     fg=t.MUTED, font=t.font(8), anchor="e")
        self.preview_meta.pack(side="right")
        self.preview_card = Card(self.page, t, padding=16)
        self.preview_card.pack(fill="x", pady=(0, s(8)))
        self.preview_box = self.preview_card.body
        self._placeholder_preview("选好文件夹后点右下角「预览整理」，\n"
                                  "这里会列出每个文件将去的位置。")

        # 底部操作栏
        footer = tk.Frame(self.outer, bg=t.BG)
        footer.pack(side="bottom", fill="x", padx=s(32),
                    pady=(s(12), s(16)), before=self.scroll)
        hint = tk.Frame(footer, bg=t.BG)
        hint.pack(side="left")
        tk.Label(hint, text=ICON["shield"], bg=t.BG, fg=t.GREEN,
                 font=t.icon(12)).pack(side="left")
        tk.Label(hint, text="预览只读；执行前会再次确认",
                 bg=t.BG, fg=t.TEXT_DIM, font=t.font(9)).pack(
            side="left", padx=(s(7), 0))
        self.btn_run = CanvasButton(footer, t, "执行整理", command=self.execute,
                                    kind="primary", icon="check", width=140,
                                    height=44, size=10, bg=t.BG)
        self.btn_run.pack(side="right")
        self.btn_preview = CanvasButton(footer, t, "预览整理",
                                        command=self.preview, kind="secondary",
                                        icon="layers", width=132, height=44,
                                        size=9, bg=t.BG)
        self.btn_preview.pack(side="right", padx=(0, s(10)))
        self.btn_run.set_enabled(False)
        self._mode_changed(self.seg_mode.value)

    # ---- 交互
    def _mode_changed(self, value):
        t = self.theme
        if value == "date":
            self.type_box.pack_forget()
            self.date_box.pack(fill="x", pady=(t.s(12), 0))
        else:
            self.date_box.pack_forget()
            self.type_box.pack(fill="x", pady=(t.s(12), 0))

    def _browse(self):
        self._browse_into(self.path_var, "选择要整理的文件夹")

    def _save_settings(self):
        self.settings.update({
            "org_dir": self.path_var.get().strip(),
            "org_mode": self.seg_mode.value,
            "org_granularity": self.seg_gran.value,
            "org_exif": bool(self.sw_exif.value),
            "org_recursive": bool(self.sw_recursive.value),
            "org_two_level": bool(self.sw_two_level.value),
            "org_action": self.seg_action.value,
        })
        save_settings(self.settings)

    def _placeholder_preview(self, text):
        for w in self.preview_box.winfo_children():
            w.destroy()
        tk.Label(self.preview_box, text=text, bg=self.theme.CARD,
                 fg=self.theme.MUTED, font=self.theme.font(9),
                 justify="left", anchor="w").pack(fill="x", pady=self.theme.s(16))

    def preview(self):
        if self.busy:
            return
        t = self.theme
        raw = os.path.expandvars(os.path.expanduser(
            self.path_var.get().strip().strip('"')))
        if not raw:
            alert(self.app, t, "还没有选择文件夹", "请先选择一个要整理的目录。",
                  icon="folder")
            return
        root = Path(raw)
        if not root.is_dir():
            alert(self.app, t, "目录不存在", f"找不到这个文件夹：\n{root}",
                  icon="warning")
            return
        self._save_settings()
        self.root = root
        self.busy = True
        self.btn_preview.set_enabled(False)
        self.btn_preview.set_text("正在预览…")
        self.btn_run.set_enabled(False)
        self.app.set_status("正在生成整理计划…", t.AMBER, "只读预览 · 不会移动文件")
        self.app.log(f"生成整理计划：{root}", "info")
        mode = self.seg_mode.value
        gran = self.seg_gran.value
        exif = bool(self.sw_exif.value)
        recursive = bool(self.sw_recursive.value)
        two = bool(self.sw_two_level.value)

        def work():
            try:
                rows, summary, skipped = tools.build_organize_plan(
                    root, mode=mode, granularity=gran, use_exif=exif,
                    recursive=recursive, two_level=two)
                self.app.post(("org_plan", (rows, summary, skipped, "")))
            except Exception as exc:
                self.app.post(("org_plan", ([], {}, 0, str(exc))))

        threading.Thread(target=work, daemon=True, name="org-plan").start()

    def on_plan(self, rows, summary, skipped, error=""):
        t = self.theme
        self.busy = False
        self.btn_preview.set_enabled(True)
        self.btn_preview.set_text("预览整理")
        self.app.set_status("引擎就绪", t.GREEN)
        if error:
            self.app.log(f"整理计划失败：{error}", "err")
            self.app.toast("整理计划生成失败", "err")
            self._placeholder_preview(f"生成计划失败：{error}")
            self.preview_meta.configure(text="失败")
            self.btn_run.set_enabled(False)
            return
        self.plan_rows = rows
        self.summary = summary
        for w in self.preview_box.winfo_children():
            w.destroy()
        if not rows:
            self.preview_meta.configure(text="没有需要整理的文件")
            self._placeholder_preview(
                "没有需要整理的文件。\n"
                "如果文件已经在目标结构里，会自动跳过。")
            self.btn_run.set_enabled(False)
            self.app.toast("没有需要整理的文件", "info")
            return
        total_size = sum(v[1] for v in summary.values())
        self.preview_meta.configure(
            text=f"{len(rows):,} 个文件 · {len(summary)} 个目标文件夹 · "
                 f"{core.human_size(total_size)}")
        chips = tk.Frame(self.preview_box, bg=t.CARD)
        chips.pack(fill="x", pady=(0, self.theme.s(10)))
        for label, (n, size) in sorted(
                summary.items(), key=lambda kv: -kv[1][0])[:6]:
            Chip(chips, t, text=label, value=str(n), color=t.ACCENT,
                 bg=t.CARD, size=8, height=26).pack(
                side="left", padx=(0, self.theme.s(6)))
        if len(summary) > 6:
            tk.Label(chips, text=f"+{len(summary) - 6} 类", bg=t.CARD,
                     fg=t.MUTED, font=t.font(8)).pack(side="left",
                                                      padx=(self.theme.s(4), 0))
        shown = rows[:250]
        for label, src, dst_dir, name in shown:
            try:
                rel = os.path.relpath(dst_dir, str(self.root))
            except Exception:
                rel = dst_dir
            self._preview_row(self.preview_box, name, rel)
        if len(rows) > len(shown):
            tk.Label(self.preview_box,
                     text=f"还有 {len(rows) - len(shown):,} 个文件未显示",
                     bg=t.CARD, fg=t.MUTED, font=self.theme.font(8),
                     anchor="w").pack(fill="x", pady=(self.theme.s(8), 0))
        skipped_note = f"，已跳过 {skipped} 个已在目标位置的文件" if skipped else ""
        self.app.log(f"整理计划：{len(rows)} 个文件，{len(summary)} 个目标目录"
                     f"{skipped_note}", "ok")
        self.btn_run.set_enabled(True)

    def execute(self):
        if self.busy:
            return
        if not self.plan_rows:
            self.app.toast("请先点「预览整理」", "warn")
            return
        t = self.theme
        action = self.seg_action.value
        action_cn = "移动" if action == "move" else "复制"
        msg = (f"将{action_cn} {len(self.plan_rows):,} 个文件到分类子文件夹。\n\n"
               f"目标位置：{short_root(str(self.root))}\n"
               f"{'原文件会保留。' if action == 'copy' else '移动后可点「撤销上次整理」还原。'}")
        if confirm(self.app, t, f"确认{action_cn}文件", msg,
                   ok_text="开始整理", cancel_text="再看看",
                   icon="warning") != "ok":
            return
        self._save_settings()
        rows = list(self.plan_rows)
        root = self.root
        cleanup = bool(self.sw_cleanup.value) and action == "move"
        self.busy = True
        self.btn_run.set_enabled(False)
        self.btn_preview.set_enabled(False)
        self.btn_run.set_text("正在整理…")
        self.app.set_status("正在整理文件…", t.AMBER, f"{action_cn}中 · 请稍候")
        self.app.log(f"开始整理：{len(rows)} 个文件（{action_cn}）", "info")

        def prog(done, total, name):
            if done % 200 == 0 or done == total:
                self.app.post(("log", f"整理进度 {done:,}/{total:,} · {name}",
                               "info"))

        def work():
            partial = {"path": None}

            def journal(records):
                """边执行边写记录：进程被中断时已移动的文件也能撤销。"""
                try:
                    if partial["path"] is None:
                        partial["path"] = tools.save_manifest(
                            "organize", root, records)
                    else:
                        tools.rewrite_manifest(partial["path"], records)
                except Exception:
                    pass

            try:
                done, failed, records, _reasons = tools.apply_organize_plan(
                    rows, action=action, on_progress=prog, on_record=journal)
                removed = []
                if cleanup and records:
                    dirs = {str(Path(r["src"]).parent) for r in records}
                    tools._remove_empty_dir_chains(
                        dirs, [str(root)], on_removed=removed.append)
                mp = ""
                if records:
                    if partial["path"] is not None:
                        tools.rewrite_manifest(partial["path"], records)
                        mp = str(partial["path"])
                    else:
                        mp = str(tools.save_manifest("organize", root,
                                                     records))
                self.app.post(("org_done", (done, failed, mp, len(removed),
                                           action_cn, len(rows))))
            except Exception as exc:
                self.app.post(("log", f"整理失败：{exc}", "err"))
                self.app.post(("org_done", (0, len(rows), "", 0, action_cn,
                                           len(rows))))

        threading.Thread(target=work, daemon=True, name="org-apply").start()

    def on_done(self, done, failed, manifest, removed, action_cn, total):
        t = self.theme
        self.busy = False
        self.plan_rows = []
        self.btn_run.set_text("执行整理")
        self.btn_run.set_enabled(False)
        self.btn_preview.set_enabled(True)
        self.app.set_status("引擎就绪", t.GREEN)
        self.undo_btn.set_enabled(tools.last_manifest_multi("organize", "finder") is not None)
        if failed:
            self.app.log(f"整理完成：成功 {done} 个，失败 {failed} 个", "warn")
            self.app.toast(f"完成 {done} 个，{failed} 个失败", "warn")
        else:
            self.app.log(f"整理完成：{done} 个文件已{action_cn}，"
                         f"清理空文件夹 {removed} 个", "ok")
            self.app.toast(f"已{action_cn} {done} 个文件", "ok")
        self.preview_meta.configure(text="整理完成")
        self._placeholder_preview(
            f"本次已{action_cn} {done} 个文件"
            + (f"，失败 {failed} 个（详见运行日志）" if failed else "")
            + (f"，清理空文件夹 {removed} 个" if removed else "")
            + "。\n点右下角「预览整理」可以整理下一批。")

    def undo(self):
        if self.busy:
            return
        mp = tools.last_manifest_multi("organize", "finder")
        if not mp:
            self.app.toast("没有可以撤销的整理/移动记录", "info")
            return
        t = self.theme
        try:
            data = tools.read_manifest(mp)
        except Exception:
            alert(self.app, t, "操作记录损坏", "无法读取上一次整理记录。",
                  icon="warning")
            return
        if not isinstance(data, dict):
            alert(self.app, t, "操作记录损坏", "无法读取上一次整理记录。",
                  icon="warning")
            return
        kind_label = "移动" if data.get("kind") == "finder" else "整理"
        msg = (f"将撤销 {data['time']} 的{kind_label}（{data.get('count', '?')} 个文件）：\n"
               f"把文件放回原来的位置。\n\n如果原位置已有同名文件，会跳过。")
        if confirm(self.app, t, f"撤销上次{kind_label}", msg, ok_text="撤销",
                   cancel_text="取消", icon="warning") != "ok":
            return
        self.busy = True
        self.undo_btn.set_enabled(False)
        self.btn_preview.set_enabled(False)
        self.btn_run.set_enabled(False)
        self.app.set_status(f"正在撤销{kind_label}…", t.AMBER, "把文件放回原位")
        self.app.log(f"撤销{kind_label}：{mp.name}", "info")

        def work():
            try:
                done, failed = tools.undo_manifest(
                    mp, on_progress=lambda d, tt, n: None)
                if not failed:
                    try:
                        mp.unlink()
                    except Exception:
                        pass
                self.app.post(("org_undo_done", (done, failed)))
            except Exception as exc:
                self.app.post(("log", f"撤销失败：{exc}", "err"))
                self.app.post(("org_undo_done", (0, -1)))

        threading.Thread(target=work, daemon=True, name="org-undo").start()

    def on_undo_done(self, done, failed):
        t = self.theme
        self.busy = False
        self.btn_preview.set_enabled(True)
        self.btn_run.set_enabled(False)
        self.undo_btn.set_enabled(tools.last_manifest_multi("organize", "finder") is not None)
        self.app.set_status("引擎就绪", t.GREEN)
        if failed < 0:
            self.app.toast("撤销失败，请查看日志", "err")
            return
        self.app.log(f"撤销完成：{done} 个文件已放回原位"
                     + (f"，{failed} 个失败" if failed else ""),
                     "warn" if failed else "ok")
        self.app.toast(f"已撤销 {done} 个文件", "ok" if not failed else "warn")
        self.preview_meta.configure(text="已撤销")
        self._placeholder_preview("已撤销上一次整理/移动，文件已放回原位。")

    def on_show(self):
        self.undo_btn.set_enabled(tools.last_manifest_multi("organize", "finder") is not None)
        if not self.path_var.get():
            self.path_var.set(self.settings.get("last_dir", ""))


# ---------------------------------------------------------------- 批量重命名页
class RenamePage(ToolPage):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.rows = []
        self._plan_gen = 0
        self._build()

    def _build(self):
        t = self.theme
        s = t.s
        st = self.settings
        self._page_setup(("批量重命名", "查找替换、加前后缀、序号模板，先预览再执行。"))
        self.undo_btn = CanvasButton(self.header_right, t, "撤销上次改名",
                                     command=self.undo, kind="ghost",
                                     icon="history", height=38, size=9,
                                     bg=t.BG)
        self.undo_btn.pack()
        self.undo_btn.set_enabled(tools.last_manifest("rename") is not None)

        self.path_var = tk.StringVar(value=st.get("ren_dir")
                                     or st.get("last_dir", ""))
        entry = self._path_card(self.page, "要重命名的文件夹", self.path_var,
                                self._browse, "只改文件名，不动文件内容")
        entry.bind("<Return>", lambda e: self._refresh())

        card = Card(self.page, t, padding=20)
        card.pack(fill="x", pady=(s(16), 0))
        body = card.body
        self._section_title(body, "重命名规则", "edit",
                            "多个规则会按顺序叠加，扩展名始终保留")

        # 范围
        scope = tk.Frame(body, bg=t.CARD)
        scope.pack(fill="x", pady=(s(13), 0))
        scope_box = tk.Frame(scope, bg=t.CARD)
        scope_box.pack(side="left", fill="x", expand=True)
        tk.Label(scope_box, text="包含子文件夹", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), anchor="w").pack(fill="x")
        tk.Label(scope_box, text="关闭后只列出当前目录的文件",
                 bg=t.CARD, fg=t.MUTED, font=t.font(8), anchor="w").pack(
            fill="x", pady=(s(2), 0))
        self.sw_recursive = Switch(scope, t, value=bool(st.get("ren_recursive")),
                                   command=lambda v: self._changed(), bg=t.CARD)
        self.sw_recursive.pack(side="right", pady=(s(4), 0))

        # 查找替换
        replace_row = tk.Frame(body, bg=t.CARD)
        replace_row.pack(fill="x", pady=(s(14), 0))
        self.sw_replace = Switch(
            replace_row, t, value=bool(st.get("ren_use_replace")),
            command=lambda v: self._changed(), bg=t.CARD)
        self.sw_replace.pack(side="left", pady=(s(4), 0))
        tk.Label(replace_row, text="查找替换", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), width=7, anchor="w").pack(
            side="left", padx=(s(8), s(6)))
        self.var_find = tk.StringVar(value=st.get("ren_find", ""))
        self.var_replace = tk.StringVar(value=st.get("ren_replace", ""))
        f_find = self._field(replace_row, self.var_find, "要查找的文字")
        f_find.pack(side="left", fill="x", expand=True)
        tk.Label(replace_row, text="→", bg=t.CARD, fg=t.MUTED,
                 font=t.font(9)).pack(side="left", padx=s(6))
        f_repl = self._field(replace_row, self.var_replace, "替换为")
        f_repl.pack(side="left", fill="x", expand=True)
        tk.Label(replace_row, text="忽略大小写", bg=t.CARD, fg=t.MUTED,
                 font=t.font(8)).pack(side="left", padx=(s(10), s(5)))
        self.sw_icase = Switch(replace_row, t, value=bool(st.get("ren_icase")),
                               command=lambda v: self._changed(), bg=t.CARD)
        self.sw_icase.pack(side="left", pady=(s(4), 0))

        # 前后缀
        affix_row = tk.Frame(body, bg=t.CARD)
        affix_row.pack(fill="x", pady=(s(11), 0))
        tk.Label(affix_row, text="添加前后缀", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), width=9, anchor="w").pack(side="left")
        self.var_prefix = tk.StringVar(value=st.get("ren_prefix", ""))
        self.var_suffix = tk.StringVar(value=st.get("ren_suffix", ""))
        f_prefix = self._field(affix_row, self.var_prefix, "前缀，如 2025_")
        f_prefix.pack(side="left", fill="x", expand=True)
        f_suffix = self._field(affix_row, self.var_suffix, "后缀，如 _已整理")
        f_suffix.pack(side="left", fill="x", expand=True, padx=(s(8), 0))

        # 模板 + 序号
        tpl_row = tk.Frame(body, bg=t.CARD)
        tpl_row.pack(fill="x", pady=(s(11), 0))
        self.sw_template = Switch(
            tpl_row, t, value=bool(st.get("ren_use_template")),
            command=lambda v: self._changed(), bg=t.CARD)
        self.sw_template.pack(side="left", pady=(s(4), 0))
        tk.Label(tpl_row, text="命名模板", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), width=7, anchor="w").pack(
            side="left", padx=(s(8), s(6)))
        self.var_template = tk.StringVar(value=st.get("ren_template",
                                                      "{原名}_{n}"))
        f_tpl = self._field(tpl_row, self.var_template,
                            "例如 旅行_{n} 或 {日期}_{原名}")
        f_tpl.pack(side="left", fill="x", expand=True)
        tk.Label(tpl_row, text="起始", bg=t.CARD, fg=t.MUTED,
                 font=t.font(8)).pack(side="left", padx=(s(10), s(4)))
        self.var_start = tk.StringVar(value=str(st.get("ren_start", "1")))
        f_start = self._field(tpl_row, self.var_start, None, width=4)
        f_start.pack(side="left")
        tk.Label(tpl_row, text="位数", bg=t.CARD, fg=t.MUTED,
                 font=t.font(8)).pack(side="left", padx=(s(10), s(5)))
        self.seg_width = Segmented(
            tpl_row, t, [(1, "1"), (2, "2"), (3, "3")],
            value=int(st.get("ren_width", 2)),
            command=lambda v: self._changed(), bg=t.CARD, padding=8)
        self.seg_width.pack(side="left")
        tk.Label(body, text="可用变量：{原名} 原主名 · {n} 序号 · {日期} 修改日期",
                 bg=t.CARD, fg=t.FAINT, font=t.font(8), anchor="w").pack(
            fill="x", pady=(s(7), 0))

        # 大小写
        case_row = tk.Frame(body, bg=t.CARD)
        case_row.pack(fill="x", pady=(s(12), 0))
        tk.Label(case_row, text="英文大小写", bg=t.CARD, fg=t.TEXT,
                 font=t.font(9), anchor="w").pack(side="left")
        self.seg_case = Segmented(
            case_row, t, [("none", "保持不变"), ("upper", "全部大写"),
                          ("lower", "全部小写"), ("title", "首字母大写")],
            value=st.get("ren_case", "none"),
            command=lambda v: self._changed(), bg=t.CARD, padding=10)
        self.seg_case.pack(side="left", padx=(s(12), 0))

        # 预览
        prev_head = tk.Frame(self.page, bg=t.BG)
        prev_head.pack(fill="x", pady=(s(20), s(8)))
        tk.Label(prev_head, text="重命名预览", bg=t.BG, fg=t.TEXT,
                 font=t.font(12, "bold"), anchor="w").pack(side="left")
        self.preview_meta = tk.Label(prev_head, text="请选择文件夹", bg=t.BG,
                                     fg=t.MUTED, font=t.font(8), anchor="e")
        self.preview_meta.pack(side="right")
        self.preview_card = Card(self.page, t, padding=16)
        self.preview_card.pack(fill="x", pady=(0, s(8)))
        self.preview_box = self.preview_card.body
        self._placeholder_preview("选择文件夹并设置规则后，这里会实时显示 "
                                  "「旧文件名 → 新文件名」。")

        # 底栏
        footer = tk.Frame(self.outer, bg=t.BG)
        footer.pack(side="bottom", fill="x", padx=s(32),
                    pady=(s(12), s(16)), before=self.scroll)
        hint = tk.Frame(footer, bg=t.BG)
        hint.pack(side="left")
        tk.Label(hint, text=ICON["warning"], bg=t.BG, fg=t.AMBER,
                 font=t.icon(12)).pack(side="left")
        tk.Label(hint, text="改名有风险，执行前请先确认预览",
                 bg=t.BG, fg=t.TEXT_DIM, font=t.font(9)).pack(
            side="left", padx=(s(7), 0))
        self.btn_run = CanvasButton(footer, t, "执行重命名",
                                    command=self.execute, kind="primary",
                                    icon="edit", width=150, height=44,
                                    size=10, bg=t.BG)
        self.btn_run.pack(side="right")
        self.btn_run.set_enabled(False)

        for var in (self.path_var, self.var_find, self.var_replace,
                    self.var_prefix, self.var_suffix, self.var_template,
                    self.var_start):
            var.trace_add("write", lambda *a: self._changed())

    # ---- 交互
    def _rename_options(self):
        try:
            start = int(self.var_start.get().strip() or "1")
        except ValueError:
            start = 1
        return tools.RenameOptions(
            find=self.var_find.get(),
            replace=self.var_replace.get(),
            use_replace=bool(self.sw_replace.value),
            case_insensitive=bool(self.sw_icase.value),
            prefix=self.var_prefix.get(),
            suffix=self.var_suffix.get(),
            use_template=bool(self.sw_template.value),
            template=self.var_template.get() or "{原名}",
            num_start=start,
            num_step=1,
            num_width=int(self.seg_width.value),
            letter_case=self.seg_case.value,
            recursive=bool(self.sw_recursive.value),
        )

    def _browse(self):
        self._browse_into(self.path_var, "选择要重命名的文件夹")

    def _save_settings(self):
        self.settings.update({
            "ren_dir": self.path_var.get().strip(),
            "ren_recursive": bool(self.sw_recursive.value),
            "ren_find": self.var_find.get(),
            "ren_replace": self.var_replace.get(),
            "ren_use_replace": bool(self.sw_replace.value),
            "ren_icase": bool(self.sw_icase.value),
            "ren_prefix": self.var_prefix.get(),
            "ren_suffix": self.var_suffix.get(),
            "ren_use_template": bool(self.sw_template.value),
            "ren_template": self.var_template.get(),
            "ren_start": self.var_start.get(),
            "ren_width": int(self.seg_width.value),
            "ren_case": self.seg_case.value,
        })
        save_settings(self.settings)

    def _placeholder_preview(self, text):
        for w in self.preview_box.winfo_children():
            w.destroy()
        tk.Label(self.preview_box, text=text, bg=self.theme.CARD,
                 fg=self.theme.MUTED, font=self.theme.font(9),
                 justify="left", anchor="w").pack(fill="x",
                                                  pady=self.theme.s(16))

    def _changed(self, *_):
        if self._pending:
            try:
                self.after_cancel(self._pending)
            except Exception:
                pass
        self._pending = self.after(350, self._refresh)

    def _refresh(self):
        self._pending = None
        if self.busy:
            return
        raw = os.path.expandvars(os.path.expanduser(
            self.path_var.get().strip().strip('"')))
        if not raw or not os.path.isdir(raw):
            self.rows = []
            self.preview_meta.configure(text="请选择文件夹")
            self.btn_run.set_enabled(False)
            self._placeholder_preview("选择一个文件夹，规则会实时预览。")
            return
        self._plan_gen += 1
        token = self._plan_gen
        options = self._rename_options()
        self.preview_meta.configure(text="正在生成预览…")
        self.btn_run.set_enabled(False)

        def work():
            try:
                rows, summary = tools.build_rename_plan(Path(raw), options)
                self.app.post(("ren_plan", (token, rows, summary, "")))
            except Exception as exc:
                self.app.post(("ren_plan", (token, [], {}, str(exc))))

        threading.Thread(target=work, daemon=True, name="ren-plan").start()

    def on_plan(self, token, rows, summary, error=""):
        if token != self._plan_gen or self.busy:
            return
        t = self.theme
        if error:
            self.app.log(f"预览失败：{error}", "err")
            self.rows = []
            self.preview_meta.configure(text="预览失败")
            self.btn_run.set_enabled(False)
            self._placeholder_preview(f"读取文件夹失败：{error}")
            return
        self.rows = rows
        changes = [r for r in rows if r[3] != "same"]
        auto = sum(1 for r in rows if r[3] == "auto")
        self.preview_meta.configure(
            text=f"{len(rows):,} 个文件 · {len(changes):,} 个将改名"
                 + (f" · {auto} 个自动编号" if auto else ""))
        for w in self.preview_box.winfo_children():
            w.destroy()
        if not changes:
            self._placeholder_preview(
                "没有文件需要改名。\n调整规则后再看看，或确认文件是否已符合命名。")
            self.btn_run.set_enabled(False)
            return
        for src, old, new, status in changes[:250]:
            tone = "warn" if status == "auto" else "normal"
            note = "自动编号" if status == "auto" else ""
            self._preview_row(self.preview_box, old, new, tone=tone, note=note)
        if len(changes) > 250:
            tk.Label(self.preview_box,
                     text=f"还有 {len(changes) - 250:,} 个文件未显示",
                     bg=t.CARD, fg=t.MUTED, font=t.font(8), anchor="w").pack(
                fill="x", pady=(t.s(8), 0))
        self.btn_run.set_enabled(bool(changes))

    def execute(self):
        if self.busy:
            return
        changes = [r for r in self.rows if r[3] != "same"]
        if not changes:
            self.app.toast("没有文件需要改名", "info")
            return
        t = self.theme
        raw = self.path_var.get().strip().strip('"')
        msg = (f"将重命名 {len(changes):,} 个文件。\n\n"
               f"示例：{changes[0][1]}  →  {changes[0][2]}\n\n"
               "重命名记录会保存，可点右上角「撤销上次改名」还原。")
        if confirm(self.app, t, "确认批量重命名", msg, ok_text="开始改名",
                   cancel_text="再看看", icon="warning") != "ok":
            return
        self._save_settings()
        rows = list(self.rows)
        root = Path(raw)
        self.busy = True
        self.btn_run.set_enabled(False)
        self.btn_run.set_text("正在改名…")
        self.app.set_status("正在重命名…", t.AMBER, "请稍候")
        self.app.log(f"开始批量重命名：{len(changes)} 个文件", "info")

        def prog(done, total, name):
            if done % 200 == 0 or done == total:
                self.app.post(("log", f"改名进度 {done:,}/{total:,} · {name}",
                               "info"))

        def work():
            partial = {"path": None}

            def journal(records):
                """边执行边写记录：进程被中断时已改名的文件也能撤销。"""
                try:
                    if partial["path"] is None:
                        partial["path"] = tools.save_manifest(
                            "rename", root, records)
                    else:
                        tools.rewrite_manifest(partial["path"], records)
                except Exception:
                    pass

            try:
                done, failed, records = tools.apply_rename_plan(
                    rows, on_progress=prog, on_record=journal)
                mp = ""
                if records:
                    if partial["path"] is not None:
                        tools.rewrite_manifest(partial["path"], records)
                        mp = str(partial["path"])
                    else:
                        mp = str(tools.save_manifest("rename", root, records))
                self.app.post(("ren_done", (done, failed, mp, len(changes))))
            except Exception as exc:
                self.app.post(("log", f"重命名失败：{exc}", "err"))
                self.app.post(("ren_done", (0, len(changes), "", len(changes))))

        threading.Thread(target=work, daemon=True, name="ren-apply").start()

    def on_done(self, done, failed, manifest, total):
        t = self.theme
        self.busy = False
        self.btn_run.set_text("执行重命名")
        self.app.set_status("引擎就绪", t.GREEN)
        self.undo_btn.set_enabled(tools.last_manifest("rename") is not None)
        self.app.log(f"重命名完成：成功 {done} 个"
                     + (f"，失败 {failed} 个" if failed else ""),
                     "warn" if failed else "ok")
        self.app.toast(f"已重命名 {done} 个文件",
                       "ok" if not failed else "warn")
        self._refresh()

    def undo(self):
        if self.busy:
            return
        mp = tools.last_manifest("rename")
        if not mp:
            self.app.toast("没有可以撤销的改名记录", "info")
            return
        t = self.theme
        try:
            data = tools.read_manifest(mp)
        except Exception:
            alert(self.app, t, "操作记录损坏", "无法读取上一次改名记录。",
                  icon="warning")
            return
        if not isinstance(data, dict):
            alert(self.app, t, "操作记录损坏", "无法读取上一次改名记录。",
                  icon="warning")
            return
        msg = (f"将撤销 {data['time']} 的改名（{data.get('count', '?')} 个文件）：\n"
               "把文件名改回原来的样子。")
        if confirm(self.app, t, "撤销上次改名", msg, ok_text="撤销",
                   cancel_text="取消", icon="warning") != "ok":
            return
        self.busy = True
        self.undo_btn.set_enabled(False)
        self.btn_run.set_enabled(False)
        self.app.set_status("正在撤销改名…", t.AMBER, "恢复原文件名")
        self.app.log(f"撤销改名：{mp.name}", "info")

        def work():
            try:
                done, failed = tools.undo_manifest(mp)
                if not failed:
                    try:
                        mp.unlink()
                    except Exception:
                        pass
                self.app.post(("ren_undo_done", (done, failed)))
            except Exception as exc:
                self.app.post(("log", f"撤销失败：{exc}", "err"))
                self.app.post(("ren_undo_done", (0, -1)))

        threading.Thread(target=work, daemon=True, name="ren-undo").start()

    def on_undo_done(self, done, failed):
        t = self.theme
        self.busy = False
        self.undo_btn.set_enabled(tools.last_manifest("rename") is not None)
        self.app.set_status("引擎就绪", t.GREEN)
        if failed < 0:
            self.app.toast("撤销失败，请查看日志", "err")
            return
        self.app.log(f"撤销改名完成：{done} 个文件"
                     + (f"，{failed} 个失败" if failed else ""),
                     "warn" if failed else "ok")
        self.app.toast(f"已撤销 {done} 个文件", "ok" if not failed else "warn")
        self._refresh()

    def on_show(self):
        self.undo_btn.set_enabled(tools.last_manifest("rename") is not None)
        if not self.path_var.get():
            self.path_var.set(self.settings.get("last_dir", ""))
        self._changed()


# ---------------------------------------------------------------- 启动
def set_dpi_awareness():
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def print_window_to_file(hwnd: int, path) -> bool:
    """用 PrintWindow 抓整个窗口的像素并存盘。

    比 ImageGrab 可靠：不受窗口遮挡、也不受 DPI 缩放影响（150% 缩放下
    ImageGrab 的 bbox 会抓偏，截出来是空白桌面）。
    """
    try:
        import ctypes
        from ctypes import wintypes

        class Header(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        class Info(ctypes.Structure):
            _fields_ = [("bmiHeader", Header),
                        ("bmiColors", wintypes.DWORD * 3)]

        if Image is None:
            return False
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        # 窗口还没完成映射时 rect 可能只有标题栏那么大，不能存成图
        if width < 400 or height < 300:
            return False
        hdc = user32.GetWindowDC(hwnd)
        mem = gdi32.CreateCompatibleDC(hdc)
        bmp = gdi32.CreateCompatibleBitmap(hdc, width, height)
        old = gdi32.SelectObject(mem, bmp)
        user32.PrintWindow(hwnd, mem, 2)
        info = Info()
        info.bmiHeader.biSize = ctypes.sizeof(Header)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(mem, bmp, 0, height, buffer, ctypes.byref(info), 0)
        image = Image.frombuffer("RGBA", (width, height), buffer, "raw",
                                 "BGRA", 0, 1).convert("RGB")
        gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, hdc)
        # 与重绘抢跑时会抓到全黑帧，交给调用方重试
        if image.getbbox() is None:
            return False
        image.save(path)
        return True
    except Exception:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重复文件清理器（图形界面）")
    parser.add_argument("dir", nargs="?", default=None, help="启动后预填的文件夹")
    parser.add_argument("--page", default="scan",
                        choices=["scan", "results", "logs", "organize", "rename"])
    parser.add_argument("--scan", default=None, help="启动后自动扫描该目录（调试用）")
    parser.add_argument("--screenshot", default=None, help="截图后退出（调试用）")
    parser.add_argument("--shot-delay", type=int, default=2200)
    args = parser.parse_args()

    set_dpi_awareness()
    app = App(initial_dir=args.dir, start_page=args.page)

    if args.dir:
        app.organize_page.path_var.set(args.dir)
        app.rename_page.path_var.set(args.dir)

    if args.page == "organize" and args.dir:
        app.after(500, app.organize_page.preview)
    elif args.page == "rename" and args.dir:
        app.after(500, app.rename_page._refresh)

    if args.scan:
        from pathlib import Path
        root = Path(args.scan)
        if root.is_dir():
            app.scan_page.set_path(str(root))
            app.after(300, app.scan_page.start_scan)

    if args.screenshot:
        attempts = [0]

        def capture():
            attempts[0] += 1
            ok = False
            try:
                app.lift()
                app.attributes("-topmost", True)
                app.update()
                ok = print_window_to_file(int(app.wm_frame(), 16),
                                          args.screenshot)
                if not ok and attempts[0] > 3 and ImageGrab is not None:
                    x, y = app.winfo_rootx(), app.winfo_rooty()
                    w, h = app.winfo_width(), app.winfo_height()
                    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                    if img.getbbox() is not None:  # 别把全黑屏当成截图
                        img.save(args.screenshot)
                        ok = True
            except Exception:
                traceback.print_exc()
            if not ok and attempts[0] <= 6:
                app.after(400, capture)  # 窗口还没画好或被抓成全黑，稍后重试
                return
            app._closed = True
            app.destroy()
        app.after(args.shot_delay, capture)

    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            crash = Path(__file__).with_name("dedup_error.log")
            crash.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        try:
            input("\n出错了，按回车退出…")
        except Exception:
            pass
