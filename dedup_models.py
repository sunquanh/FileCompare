"""数据模型、常量、阈值预设与小工具。

这一层不依赖界面，也不做任何文件写入，供扫描各阶段共享。
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- 可选依赖
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image, ImageDraw, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageOps = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# ---------------------------------------------------------------- 子进程
CREATE_NO_WINDOW = 0x08000000  # Windows: CREATE_NO_WINDOW


def hidden_subprocess_kwargs() -> dict:
    """让外部命令（ffmpeg/ffprobe/fpcalc）安静执行，不弹控制台窗口。

    不加这些标志时，每调用一次子进程就会在屏幕上闪过一个黑色 cmd 窗口；
    扫一个含几十个视频的目录就会闪几十次。
    """
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE
    return {"creationflags": CREATE_NO_WINDOW, "startupinfo": startupinfo}


def find_tool(name: str, extra_dirs=()) -> str | None:
    """先找 PATH，再找程序所在目录（打包后可以就把 exe 放在旁边）。"""
    import shutil

    found = shutil.which(name)
    if found:
        return found
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else None
    roots = [Path(x) for x in extra_dirs]
    if exe_dir is not None:
        roots.append(exe_dir)
    roots.append(app_dir())
    for root in roots:
        for suffix in ("", ".exe"):
            candidate = root / f"{name}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


# ---------------------------------------------------------------- 忽略记录
_IGNORE_LOG = logging.getLogger("dedup.ignored")


def ignore(where: str, exc: BaseException | None = None) -> None:
    """记录被主动忽略的异常。

    默认不出声（不影响界面），但不再是完全静默：把 dedup.ignored 调到
    DEBUG 就能看到每一步为什么被跳过，排查「文件明明存在却没扫描到」这类问题。
    """
    if _IGNORE_LOG.isEnabledFor(logging.DEBUG):
        _IGNORE_LOG.debug("%s 已忽略: %r", where, exc)


# ---------------------------------------------------------------- 图片插件
_HEIF_STATE = None


def enable_heif() -> bool:
    """如果装了 pillow-heif，就注册 HEIC/HEIF 解码器（可选依赖）。"""
    global _HEIF_STATE
    if _HEIF_STATE is not None:
        return _HEIF_STATE
    _HEIF_STATE = False
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        _HEIF_STATE = True
    except Exception:
        _HEIF_STATE = False
    return _HEIF_STATE


# ---------------------------------------------------------------- 常量
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp",
             ".tif", ".tiff", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv",
             ".flv", ".ts", ".webm", ".mpg", ".mpeg"}
AUDIO_EXT = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg", ".wma",
             ".ape", ".opus", ".aiff", ".alac"}

SKIP_DIRS = {
    "$RECYCLE.BIN", "System Volume Information", ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".idea", ".vscode", ".vs",
    "AppData", "Windows", "Program Files", "Program Files (x86)",
}
SKIP_PREFIX = ("_重复检测", "_重复待删", "~$")

KIND_EXACT = "完全重复"
KIND_PHOTO = "相似照片"
KIND_VIDEO = "重复视频"
KIND_AUDIO = "重复音乐"
KIND_NAME = "同名相似"
KIND_DIR = "重复目录"
KIND_ORDER = (KIND_EXACT, KIND_PHOTO, KIND_VIDEO, KIND_AUDIO, KIND_NAME,
              KIND_DIR)

# 保留策略
KEEP_RULES = ("best", "oldest", "newest", "largest", "shortest", "folder")
KEEP_RULE_LABELS = {
    "best": "智能推荐（清晰度/体积/时间）",
    "oldest": "保留最旧的",
    "newest": "保留最新的",
    "largest": "保留体积最大的",
    "shortest": "保留路径最短的",
    "folder": "保留扫描目录顶层的",
}

# 严格程度预设。
#   img_dist  照片粗筛：8x8 dHash（64 位）允许的位差
#   img_fine  照片复核：16x16 pHash（256 位）允许的位差
#   img_hist  颜色直方图 Bhattacharyya 距离（0 = 完全相同，1 = 完全不符）
#   vid_dist / vid_fine  单帧 dHash / 16x16 位差
#   vid_ratio 匹配帧占比下限
PRESETS = {
    "relaxed": {"img_dist": 12, "img_fine": 62, "img_hist": 0.38,
                "vid_dist": 14, "vid_fine": 120, "vid_ratio": 0.50,
                "name_ratio": 0.78},
    "standard": {"img_dist": 8, "img_fine": 42, "img_hist": 0.26,
                 "vid_dist": 10, "vid_fine": 96, "vid_ratio": 0.60,
                 "name_ratio": 0.88},
    "strict": {"img_dist": 5, "img_fine": 26, "img_hist": 0.16,
               "vid_dist": 7, "vid_fine": 72, "vid_ratio": 0.75,
               "name_ratio": 0.94},
}

CACHE_TAGS = {"quick": "q3", "sample": "s3", "full": "f3",
              "photo": "p4", "video": "v3", "audio": "a3"}


class ScanCancelled(Exception):
    """用户取消了扫描。"""


# ---------------------------------------------------------------- 小工具
def human_size(n: float) -> str:
    n = float(n or 0)
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB", "PB"):
        n /= 1024.0
        if n < 1024 or unit == "PB":
            if n >= 100:
                return f"{n:.0f} {unit}"
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def format_duration(seconds: float) -> str:
    seconds = round(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_date(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


def is_photo(path) -> bool:
    return Path(path).suffix.lower() in PHOTO_EXT


def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXT


def is_audio(path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXT


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise ScanCancelled


def app_dir() -> Path:
    """可写的数据目录：优先 %APPDATA%\\Dedup，失败时退回脚本目录。"""
    base = (os.environ.get("APPDATA")
            or os.environ.get("XDG_CONFIG_HOME")
            or str(Path.home()))
    candidate = Path(base) / "Dedup"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return candidate
    except Exception:
        return Path(__file__).resolve().parent


def norm_path(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


# ---------------------------------------------------------------- 数据模型
@dataclass
class FileItem:
    path: Path
    size: int
    mtime: float = 0.0
    mtime_ns: int = 0
    dev: int = 0
    ino: int = 0
    is_symlink: bool = False
    width: int = 0
    height: int = 0
    duration: float = 0.0
    protected: bool = False
    bitrate: float = 0.0
    audio_artist: str = ""
    audio_title: str = ""
    audio_album: str = ""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def dims(self) -> str:
        if self.width and self.height:
            return f"{self.width} × {self.height}"
        return ""


@dataclass
class DupGroup:
    kind: str
    items: list[FileItem]
    similarity: int = 100
    note: str = ""
    keep: int = 0
    select_by_default: bool = True

    @property
    def wasted(self) -> int:
        if (self.kind == KIND_NAME or len(self.items) < 2
                or not self.select_by_default):
            # 同名组内容未必一致、无声学比对的音频组默认不删，
            # 都不计入可释放空间
            return 0
        return sum(i.size for i in self.items) - max(i.size for i in self.items)

    def default_selection(self) -> set[str]:
        if not self.select_by_default:
            return set()
        return {str(i.path) for idx, i in enumerate(self.items)
                if idx != self.keep and not i.protected}


@dataclass
class ScanOptions:
    root: Path
    exact: bool = True
    photos: bool = True
    videos: bool = True
    audio: bool = True
    names: bool = True
    names_fuzzy: bool = False
    cache: bool = True
    img_dist: int = 8
    img_fine: int = 42
    img_hist: float = 0.26
    vid_dist: int = 10
    vid_fine: int = 96
    vid_ratio: float = 0.60
    name_ratio: float = 0.88
    min_size: int = 0
    exclude: tuple = ()
    protect: tuple = ()
    reference_dirs: tuple = ()
    quick_scan: bool = False
    recursive: bool = True
    workers: int = 0
    keep_rule: str = "best"
    max_image_edge: int = 1024

    def __post_init__(self):
        self.root = Path(self.root)
        if not self.workers:
            self.workers = min(8, os.cpu_count() or 4)
        self.exclude = tuple(x for x in (self.exclude or ()) if x)
        self.protect = tuple(x for x in (self.protect or ()) if x)
        self.reference_dirs = tuple(x for x in (self.reference_dirs or ())
                                    if x)
        if self.keep_rule not in KEEP_RULES:
            self.keep_rule = "best"
        if not self.max_image_edge:
            self.max_image_edge = 1024

    @classmethod
    def from_preset(cls, root, preset="standard", **kw):
        opts = cls(root=root, **PRESETS.get(preset, PRESETS["standard"]))
        for k, v in kw.items():
            setattr(opts, k, v)
        return opts


@dataclass
class ScanResult:
    root: Path
    groups: list[DupGroup] = field(default_factory=list)
    file_count: int = 0
    total_size: int = 0
    elapsed: float = 0.0
    bad_photos: int = 0
    bad_videos: int = 0
    bad_audio: int = 0
    skipped_inodes: int = 0
    skipped_unreadable: int = 0
    skipped_empty: int = 0
    tiny_photos: list[str] = field(default_factory=list)
    cancelled: bool = False
    video_backend: str = ""
    notes: list[str] = field(default_factory=list)

    def counts(self):
        out = {}
        for kind in KIND_ORDER:
            gs = [g for g in self.groups if g.kind == kind]
            out[kind] = {
                "groups": len(gs),
                "files": sum(len(g.items) for g in gs),
                "wasted": sum(g.wasted for g in gs),
            }
        return out

    @property
    def total_groups(self) -> int:
        return len(self.groups)

    @property
    def total_files(self) -> int:
        return sum(len(g.items) for g in self.groups)

    @property
    def total_wasted(self) -> int:
        return sum(g.wasted for g in self.groups)


# ---------------------------------------------------------------- 保留策略
def pick_keep(kind: str, items: list[FileItem], rule: str = "best",
              root=None) -> int:
    """推荐保留哪一项。rule 见 KEEP_RULE_LABELS。"""
    if not items:
        return 0
    # 符号链接只剩指针：组里有真实文件时，不选链接作保留项
    cands = [i for i, it in enumerate(items) if not it.is_symlink]
    if not cands:
        cands = list(range(len(items)))

    def root_depth(item):
        if root is None:
            return len(item.path.parts)
        try:
            return len(item.path.relative_to(Path(root)).parts)
        except Exception:
            return len(item.path.parts)

    if rule == "oldest":
        key = lambda x: (x.protected, -x.mtime)
    elif rule == "newest":
        key = lambda x: (x.protected, x.mtime)
    elif rule == "largest":
        key = lambda x: (x.protected, x.size)
    elif rule == "shortest":
        key = lambda x: (x.protected, -len(str(x.path)))
    elif rule == "folder":
        key = lambda x: (x.protected, -root_depth(x), x.size)
    elif kind == KIND_NAME:
        key = lambda x: (x.protected, x.mtime, -len(str(x.path)))
    elif kind == KIND_AUDIO:
        key = lambda x: (x.protected, x.bitrate, x.size, x.mtime)
    elif kind == KIND_EXACT:
        key = lambda x: (x.protected, x.mtime, -len(str(x.path)))
    else:
        key = lambda x: (x.protected, x.width * x.height, x.size, x.mtime)
    return max(cands, key=lambda i: key(items[i]))


# ---------------------------------------------------------------- 聚类
class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def clusters_union(n, pairs):
    """简单传递闭包聚类（完全重复用，边全部等价，无链式膨胀问题）。"""
    d = _DSU(n)
    for a, b, *_ in pairs:
        d.union(a, b)
    buckets = {}
    for i in range(n):
        buckets.setdefault(d.find(i), []).append(i)
    return [sorted(v) for v in buckets.values() if len(v) > 1]


def clusters_complete_link(n, edges):
    """完全链接聚类：新成员必须与组内**每一个**成员都通过阈值。

    edges: [(i, j, distance)]，只包含已通过所有阈值检验的文件对。
    相比并查集，能避免 A~B、B~C 但 A 与 C 完全不像时被硬并成一组。
    """
    dist = {}
    for a, b, d in edges:
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        old = dist.get(key)
        if old is None or d < old:
            dist[key] = d

    groups: list[list[int] | None] = []
    assign: dict[int, int] = {}
    for key, d in sorted(dist.items(), key=lambda kv: kv[1]):
        i, j = key
        gi, gj = assign.get(i), assign.get(j)
        if gi is None and gj is None:
            groups.append([i, j])
            idx = len(groups) - 1
            assign[i] = assign[j] = idx
        elif gi is not None and gj is not None:
            if gi == gj or groups[gi] is None or groups[gj] is None:
                continue
            a_group = groups[gi]
            b_group = groups[gj]
            if all((min(x, y), max(x, y)) in dist
                   for x in a_group for y in b_group):
                a_group.extend(b_group)
                for x in b_group:
                    assign[x] = gi
                groups[gj] = None
        else:
            idx = gi if gi is not None else gj
            new = j if gi is not None else i
            group = groups[idx]
            if group is None:
                continue
            if all((min(x, new), max(x, new)) in dist for x in group):
                group.append(new)
                assign[new] = idx
    return [sorted(g) for g in groups if g]


# ---------------------------------------------------------------- 线程同步
class Counter:
    """线程安全的计数器，用于缓存命中统计。"""

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def add(self, n=1):
        with self._lock:
            self.value += n


def clean_stem_regex():
    return re.compile(r"\s+")


__all__ = [
    "AUDIO_EXT",
    "CACHE_TAGS",
    "KEEP_RULES",
    "KEEP_RULE_LABELS",
    "KIND_AUDIO",
    "KIND_EXACT",
    "KIND_NAME",
    "KIND_ORDER",
    "KIND_PHOTO",
    "KIND_VIDEO",
    "PHOTO_EXT",
    "PRESETS",
    "SKIP_DIRS",
    "SKIP_PREFIX",
    "VIDEO_EXT",
    "Counter",
    "DupGroup",
    "FileItem",
    "Image",
    "ImageDraw",
    "ImageOps",
    "ScanCancelled",
    "ScanOptions",
    "ScanResult",
    "app_dir",
    "check_cancel",
    "clusters_complete_link",
    "clusters_union",
    "cv2",
    "format_date",
    "format_duration",
    "find_tool",
    "hidden_subprocess_kwargs",
    "human_size",
    "ignore",
    "is_audio",
    "is_photo",
    "is_video",
    "norm_path",
    "np",
    "pick_keep",
]
