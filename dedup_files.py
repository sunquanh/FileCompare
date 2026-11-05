"""文件系统操作：收集、回收站删除与还原、硬链接、缩略图、CSV 报告。

安全约定：
  * 扫描阶段只读；
  * 删除默认走回收站，失败时逐个重试；
  * 还原时**绝不覆盖**原位置已有文件；
  * 硬链接替换失败时保留原文件。
"""
from __future__ import annotations

import csv
import os
import shutil
import uuid
from pathlib import Path

import dedup_video
from dedup_models import (
    FileItem,
    ScanOptions,
    check_cancel,
    enable_heif,
    format_duration,
    ignore,
    is_video,
    norm_path,
)

SCAN_STAGE_LABELS = {
    "collect": "收集文件",
    "exact": "完全重复",
    "photo": "相似照片",
    "video": "重复视频",
    "audio": "重复音乐",
    "name": "同名相似",
}


# ---------------------------------------------------------------- 收集
def collect_files(options: ScanOptions, cancel=None, on_progress=None):
    """递归收集文件，返回 (FileItem 列表, 跳过的硬链接数, 无法读取的文件数,
    0 字节文件数)。

    同一个物理文件（NTFS 硬链接、目录联接、重复挂载）只保留第一次出现的路径，
    否则会把「删除后并不释放空间」的硬链接当成重复文件。
    """
    root = Path(options.root).resolve()
    skip = set(options.exclude)
    protect_roots = [norm_path(Path(x).expanduser()) .rstrip("\\/")
                     for x in options.protect]

    def is_protected(candidate: Path) -> bool:
        text = norm_path(candidate)
        return any(text == q or text.startswith((q + os.sep, q + "/")) for q in protect_roots)

    found: list[FileItem] = []
    seen_inodes: set[tuple] = set()
    skipped_inodes = 0
    skipped_unreadable = 0
    skipped_empty = 0
    checked = 0

    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                followlinks=False):
        check_cancel(cancel)
        if options.recursive:
            dirnames[:] = [d for d in dirnames
                           if d not in skip and d not in _SKIP and
                           not d.startswith(("_重复检测", "_重复待删", "~$"))]
        else:
            dirnames[:] = []
        directory = Path(dirpath)
        for name in filenames:
            if name.startswith(("_重复检测", "_重复待删", "~$")):
                continue
            path = directory / name
            try:
                stat = path.stat()
            except OSError:
                skipped_unreadable += 1
                continue
            checked += 1
            if stat.st_size <= 0:
                skipped_empty += 1
                continue
            if stat.st_size < options.min_size:
                continue
            if not path.is_file():
                continue
            if stat.st_ino:
                key = (stat.st_dev, stat.st_ino)
                if key in seen_inodes:
                    skipped_inodes += 1
                    continue
                seen_inodes.add(key)
            found.append(FileItem(
                path=path, size=stat.st_size, mtime=stat.st_mtime,
                mtime_ns=getattr(stat, "st_mtime_ns", 0),
                dev=stat.st_dev, ino=stat.st_ino,
                is_symlink=path.is_symlink(),
                protected=bool(protect_roots) and is_protected(path)))
            if on_progress and (len(found) % 400 == 0):
                on_progress(len(found), 0, f"已发现 {len(found):,} 个文件")
        if on_progress and checked and checked % 1200 == 0:
            on_progress(len(found), 0, f"正在遍历… 已发现 {len(found):,} 个文件")

    if on_progress:
        on_progress(len(found), len(found), f"共发现 {len(found):,} 个文件")
    return found, skipped_inodes, skipped_unreadable, skipped_empty


_SKIP = {
    "$RECYCLE.BIN", "System Volume Information", ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".idea", ".vscode", ".vs",
    "AppData", "Windows", "Program Files", "Program Files (x86)",
}


# ---------------------------------------------------------------- 删除
def recycle_paths(paths):
    """移到 Windows 回收站。返回 (成功路径, 失败路径)。"""
    paths = [str(Path(p)) for p in paths]
    if not paths:
        return [], []
    existing = [p for p in paths if os.path.exists(p)]
    missing = [p for p in paths if not os.path.exists(p)]
    if not existing:
        return [], missing
    if os.name != "nt":
        return [], existing + missing

    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_uint16),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE = 3
    flags = 0x0040 | 0x0010 | 0x0004 | 0x0400 | 0x0200

    def run(batch):
        buffer = "\0".join(batch) + "\0\0"
        op = SHFILEOPSTRUCTW(None, FO_DELETE, buffer, None, flags, False,
                             None, None)
        try:
            result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        except Exception:
            return 1
        if result != 0 or op.fAnyOperationsAborted:
            return 1
        return 0

    def split_after(batch):
        deleted = [p for p in batch if not os.path.exists(p)]
        failed = [p for p in batch if os.path.exists(p)]
        return deleted, failed

    if run(existing) == 0:
        deleted, failed = split_after(existing)
        if not failed:
            return deleted, missing

    deleted, failed = [], []
    for path in existing:
        if run([path]) == 0 and not os.path.exists(path):
            deleted.append(path)
        else:
            failed.append(path)
    return deleted, failed + missing


def permanent_delete(paths):
    deleted, failed = [], []
    for path in paths:
        try:
            os.remove(path)
            deleted.append(str(path))
        except OSError:
            failed.append(str(path))
    return deleted, failed


# ---------------------------------------------------------------- 回收站还原
def _read_recycle_meta(index_file: Path):
    """读取 $I 文件里记录的原路径。

    格式（Vista 及以后，version 2）：
      0  version(8) | 8 原大小(8) | 16 删除时间(8) | 24 路径长度(4，字符数)
      28 原路径（UTF-16LE，以 NUL 结尾）
    XP 的 version 1 没有长度字段，路径直接在 24。旧代码一律从 24 解码，
    于是路径前面多出一个长度字节，永远对不上——还原因此总是失败。
    """
    try:
        data = index_file.read_bytes()
    except OSError:
        return None
    if len(data) < 28:
        return None
    version = int.from_bytes(data[:8], "little")
    offset = 28 if version >= 2 else 24
    try:
        text = data[offset:].decode("utf-16-le", "ignore")
    except Exception:
        return None
    cut = text.find("\x00")
    if cut >= 0:
        text = text[:cut]
    text = text.replace("\x00", "").strip()
    # 相对路径说明解析偏了，宁可报找不到也不要乱移动
    if not text or not os.path.isabs(text):
        return None
    return text


def _recycle_index(drives):
    """建立 {原路径(normcase): ($R 路径, $I 路径)} 索引。"""
    index = {}
    for drive in drives:
        base = Path(f"{drive}\\$Recycle.Bin")
        if not base.exists():
            continue
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for sid_dir in entries:
            if not sid_dir.is_dir():
                continue
            try:
                indexes = list(sid_dir.glob("$I*"))
            except OSError:
                continue
            for index_file in indexes:
                original = _read_recycle_meta(index_file)
                if not original:
                    continue
                data_file = index_file.with_name("$R" + index_file.name[2:])
                if not data_file.exists():
                    continue
                index[norm_path(original)] = (data_file, index_file, original)
    return index


def restore_from_recycle(paths):
    """把回收站里的文件还原到原位置。返回 (成功, 失败)。

    原位置已有同名文件时跳过，绝不覆盖。
    """
    paths = [str(p) for p in paths]
    restored, failed = [], []
    if os.name != "nt" or not paths:
        return restored, paths
    drives = sorted({os.path.splitdrive(p)[0] for p in paths if
                     os.path.splitdrive(p)[0]})
    index = _recycle_index(drives)
    for path in paths:
        entry = index.get(norm_path(path))
        if not entry:
            failed.append(path)
            continue
        data_file, index_file, original = entry
        destination = Path(original)
        if destination.exists():
            failed.append(path)
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(data_file), str(destination))
            try:
                index_file.unlink()
            except OSError:
                pass
            restored.append(path)
        except Exception:
            failed.append(path)
    return restored, failed


# ---------------------------------------------------------------- 硬链接
def hardlink_replace(keep_path, dup_paths):
    """把重复文件替换为指向 keep 的硬链接（不占额外空间）。"""
    keep = Path(keep_path)
    kept, failed = [], []
    try:
        keep_stat = keep.stat()
    except OSError:
        return [], [str(p) for p in dup_paths]
    for raw in dup_paths:
        path = Path(raw)
        if not path.exists():
            failed.append(str(path))
            continue
        try:
            if os.path.samefile(keep, path):
                kept.append(str(path))
                continue
            if path.stat().st_dev != keep_stat.st_dev:
                failed.append(str(path))
                continue
            temp = path.with_name(
                path.name + f".deduplink-{uuid.uuid4().hex[:8]}.tmp")
            try:
                os.link(keep, temp)
                os.replace(temp, path)
                kept.append(str(path))
            finally:
                if temp.exists():
                    try:
                        temp.unlink()
                    except OSError:
                        pass
        except Exception:
            failed.append(str(path))
    return kept, failed


# ---------------------------------------------------------------- 缩略图
def make_thumbnail(path, box=(150, 100), radius=10):
    """生成圆角缩略图（PIL Image, RGBA）。失败返回 None。"""
    from PIL import Image, ImageOps
    enable_heif()
    path = Path(path)
    width, height = int(box[0]), int(box[1])
    if width <= 0 or height <= 0:
        return None
    try:
        image = None
        if is_video(path):
            image = dedup_video.grab_frame(path, 0.15, max(width * 2, 320))
        if image is None:
            with Image.open(path) as source:
                try:
                    source.draft("RGB", (width * 2, height * 2))
                except Exception as exc:
                    ignore(__name__, exc)
                image = ImageOps.exif_transpose(source).convert("RGB")
        return _cover_round(image, width, height, radius)
    except Exception:
        return None


def _cover_round(image, width, height, radius):
    from PIL import Image, ImageDraw
    ratio = max(width / image.width, height / image.height)
    new_w = max(width, int(image.width * ratio + 0.5))
    new_h = max(height, int(image.height * ratio + 0.5))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    image = image.crop((left, top, left + width, top + height)).convert("RGBA")
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=min(radius, height // 2, width // 2), fill=255)
    image.putalpha(mask)
    return image


# ---------------------------------------------------------------- 报告
def export_report(path, groups, root=None):
    """导出 CSV（Excel 可直接打开）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["类型", "组号", "文件名", "大小MB", "尺寸/时长",
                         "音频标签", "相似度", "是否推荐保留", "受保护",
                         "完整路径"])
        for group_index, group in enumerate(groups, 1):
            for index, item in enumerate(group.items):
                info = item.dims or format_duration(item.duration)
                if item.duration and item.dims:
                    info = f"{format_duration(item.duration)} · {item.dims}"
                tag = ""
                if item.audio_title or item.audio_artist:
                    tag = f"{item.audio_artist} - {item.audio_title}".strip(" -")
                writer.writerow([
                    group.kind, group_index, item.name,
                    f"{item.size / 1048576:.2f}", info, tag,
                    f"{group.similarity}%", "是" if index == group.keep else "",
                    "是" if item.protected else "", str(item.path),
                ])
    return path
