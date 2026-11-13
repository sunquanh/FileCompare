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
import time
import uuid
from pathlib import Path

import dedup_video
from dedup_models import (
    SKIP_PREFIX,
    FileItem,
    ScanOptions,
    check_cancel,
    enable_heif,
    format_duration,
    ignore,
    is_video,
    norm_path,
    should_skip_dir,
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

    参考目录允许位于扫描根之外：它们同样会被收集并标记为 protected，
    这样它们只作为「保留基准」参与比对，永远不进清理表（pick_keep 优先保留
    protected 项，default_selection 也排除 protected 项）。
    """
    root = Path(options.root).resolve()
    skip = {str(x).strip().lower() for x in options.exclude if str(x).strip()}
    # R23 忽略列表：完整路径（文件或目录），扫描时跳过。目录按前缀匹配。
    _ignore = [norm_path(Path(str(x)).expanduser()).rstrip("/" + os.sep)
               for x in getattr(options, "ignore_paths", ()) or ()
               if str(x).strip()]

    def _is_ignored(candidate: Path) -> bool:
        """R23：该路径（文件或目录）是否在忽略列表里（精确匹配或位于被忽略的目录下）。"""
        if not _ignore:
            return False
        text = norm_path(candidate).rstrip("/" + os.sep)
        return any(text == q or text.startswith((q + os.sep, q + "/"))
                   for q in _ignore)
    protect_roots = [norm_path(Path(x).expanduser()).rstrip("/" + os.sep)
                     for x in list(options.protect)
                     + list(getattr(options, "reference_dirs", ()))]

    def is_protected(candidate: Path) -> bool:
        text = norm_path(candidate)
        return any(text == q or text.startswith((q + os.sep, q + "/"))
                   for q in protect_roots)

    found: list[FileItem] = []
    seen_inodes: set[tuple] = set()
    skipped_inodes = 0
    skipped_unreadable = 0
    skipped_empty = 0
    checked = 0

    def walk(base: Path, forced_protected: bool):
        nonlocal skipped_inodes, skipped_unreadable, skipped_empty, checked
        base_depth = len(base.parts)
        for dirpath, dirnames, filenames in os.walk(base, topdown=True,
                                                    followlinks=False):
            check_cancel(cancel)
            directory = Path(dirpath)
            depth = len(directory.parts) - base_depth + 1
            if options.recursive:
                dirnames[:] = [d for d in dirnames
                               if d.lower() not in skip
                               and not should_skip_dir(d, depth)
                               and not d.startswith(SKIP_PREFIX)
                               and not _is_ignored(directory / d)]
            else:
                dirnames[:] = []
            for name in filenames:
                if name.startswith(SKIP_PREFIX):
                    continue
                path = directory / name
                if _is_ignored(path):
                    continue
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
                    protected=forced_protected
                    or (bool(protect_roots) and is_protected(path))))
                if on_progress and (len(found) % 400 == 0):
                    on_progress(len(found), 0, f"已发现 {len(found):,} 个文件")
            if on_progress and checked and checked % 1200 == 0:
                on_progress(len(found), 0,
                            f"正在遍历… 已发现 {len(found):,} 个文件")

    walk(root, False)

    # 扫描根之外的参考目录：一并收集，作为「保留基准」
    root_key = norm_path(root).rstrip("/" + os.sep)
    walked = [root_key]
    for raw in getattr(options, "reference_dirs", ()) or ():
        try:
            ref = Path(os.path.expandvars(
                os.path.expanduser(str(raw)))).resolve()
        except (OSError, ValueError):
            continue
        key = norm_path(ref).rstrip("/" + os.sep)
        if not key or key in walked:
            continue
        if key.startswith((root_key + os.sep, root_key + "/")):
            continue                      # 已在扫描根内，上面的遍历覆盖过
        if any(key.startswith((w + os.sep, w + "/")) for w in walked):
            continue                      # 已被另一个参考目录包含
        if not ref.is_dir():
            continue
        walked.append(key)
        walk(ref, True)

    if on_progress:
        on_progress(len(found), len(found), f"共发现 {len(found):,} 个文件")
    return found, skipped_inodes, skipped_unreadable, skipped_empty


def find_duplicate_dirs(files):
    """找内容成组相同的目录：同一目录下直接文件的 (文件名, 大小) 集合完全一致。

    返回 [[FileItem, ...], ...]，每个内层列表是一组重复目录（FileItem 的 path
    指向目录本身，size 是目录内直接文件的总大小）。只按文件名+大小判断，
    内容未校验，调用方应默认不勾选。单个文件的目录也可成组（签名只有一个
    (文件名, 大小)，同样默认不勾选、不直接清理）。"""
    by_dir: dict[str, list[FileItem]] = {}
    for f in files:
        by_dir.setdefault(str(Path(f.path).parent), []).append(f)
    sigs: dict[tuple, list[Path]] = {}
    totals: dict[str, int] = {}
    for directory, items in by_dir.items():
        sig = tuple(sorted((it.name, it.size) for it in items))
        sigs.setdefault(sig, []).append(Path(directory))
        totals[directory] = sum(it.size for it in items)
    groups = []
    for sig, dirs in sigs.items():
        if len(dirs) < 2:
            continue
        items = sorted(
            (FileItem(path=d, size=totals[str(d)],
                      mtime=max((f.mtime for f in by_dir[str(d)]), default=0.0))
             for d in dirs),
            key=lambda x: str(x.path))
        groups.append(items)
    groups.sort(key=lambda g: -g[0].size)
    return groups


# ---------------------------------------------------------------- 删除
def _drive_root(path) -> str:
    """返回卷根（本地盘盘符根或 UNC 共享根），取不到返回空串。"""
    text = str(path)
    drive = os.path.splitdrive(text)[0]
    if not drive:
        return ""
    if drive.startswith("\\\\"):
        parts = [p for p in drive.replace("/", "\\").split("\\") if p]
        return "\\\\" + "\\".join(parts[:2]) + "\\"
    return drive + "\\"


def _mb(n: float) -> str:
    return f"{max(0.0, float(n)) / 1048576:.0f} MB"


def _volume_fs_type(root: str) -> str:
    if os.name != "nt" or not root:
        return ""
    import ctypes
    fs = ctypes.create_unicode_buffer(64)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(root), None, 0, None, None, None, fs, 64)
    return fs.value.upper() if ok else ""


def _recycle_bin_exists(root: str) -> bool:
    return bool(root) and (Path(root) / "$Recycle.Bin").is_dir()


def _recycle_usage(root: str):
    """(回收站已用字节, 卷总字节)；取不到返回 None。"""
    if os.name != "nt" or not root:
        return None
    import ctypes
    from ctypes import wintypes

    class SHQUERYRBINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD),
                    ("i64Size", ctypes.c_int64),
                    ("i64NumItems", ctypes.c_int64)]

    info = SHQUERYRBINFO(ctypes.sizeof(SHQUERYRBINFO), 0, 0)
    try:
        used = 0
        if ctypes.windll.shell32.SHQueryRecycleBinW(
                ctypes.c_wchar_p(root), ctypes.byref(info)) == 0:
            used = int(info.i64Size)
        free = ctypes.c_ulonglong()
        total = ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(root), ctypes.byref(free),
                ctypes.byref(total), None):
            return None
        return used, int(total.value)
    except Exception:
        return None


# Windows 回收站默认上限是卷容量的 5%，超过就"删进"永久删除
_RECYCLE_QUOTA_SHARE = 0.05
# 小批量不查配额：这点空间不可能触发永久删除
_RECYCLE_MIN_CHECK = 16 << 20
_RECYCLE_FS_OK = {"NTFS", "REFS"}


def recycle_capable(paths):
    """筛出真正能进回收站的文件，返回 (ok, blocked)。

    blocked 是 [(路径, 原因)]。Windows 在回收站被关闭、卷不是 NTFS/ReFS
    （网络盘、FAT32 U 盘）或超出回收站配额时会静默永久删除，调用方必须
    先让用户知道，而不是当成「可还原」。
    """
    paths = [str(p) for p in paths]
    if os.name != "nt":
        return [], [(p, "当前系统不支持回收站") for p in paths]
    ok, blocked = [], []
    for path in paths:
        if not os.path.exists(path):
            ok.append(path)          # 已不存在：交给删除流程按失败处理
            continue
        root = _drive_root(path)
        fs = _volume_fs_type(root)
        if fs not in _RECYCLE_FS_OK:
            blocked.append((path, f"{root or path} 不是 NTFS/ReFS 卷"
                                  f"（{fs or '类型未知'}），没有回收站"))
            continue
        if not _recycle_bin_exists(root):
            blocked.append((path, f"{root} 的回收站不可用（可能已被关闭）"))
            continue
        ok.append(path)

    pending = {}
    for path in ok:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        root = _drive_root(path)
        pending[root] = pending.get(root, 0) + size
    for root, need in pending.items():
        if need <= _RECYCLE_MIN_CHECK:
            continue
        usage = _recycle_usage(root)
        if not usage:
            continue
        used, total = usage
        budget = max(0, int(total * _RECYCLE_QUOTA_SHARE) - used)
        if need > budget:
            reason = (f"{root} 回收站配额不足：这批需 {_mb(need)}，"
                      f"剩余约 {_mb(budget)}")
            blocked.extend((p, reason) for p in ok if _drive_root(p) == root)
    if blocked:
        blocked_set = {p for p, _ in blocked}
        ok = [p for p in ok if p not in blocked_set]
    return ok, blocked


def verify_recycled(paths):
    """核对哪些文件真的进了回收站，返回 (已进回收站, 被永久删除)。

    SHFileOperationW 返回 0 只代表「删除成功」，不代表「进了回收站」。
    清理之后核对一遍，别把永久删除报成可还原。
    """
    paths = [str(p) for p in paths]
    if os.name != "nt" or not paths:
        return set(), set(paths)
    roots = sorted({_drive_root(p) for p in paths if _drive_root(p)})
    index = _recycle_index(roots)
    recycled, gone = set(), set()
    for path in paths:
        (recycled if norm_path(path) in index else gone).add(path)
    return recycled, gone


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
    # FOF_WANTNUKEWARNING：万一仍会变成永久删除，让系统弹警告而不是静默处理
    flags = 0x0040 | 0x0010 | 0x0004 | 0x0400 | 0x0200 | 0x4000

    def run(batch):
        buffer = "\0".join(batch) + "\0\0"
        op = SHFILEOPSTRUCTW(None, FO_DELETE, buffer, None, flags, False,
                             None, None)
        try:
            result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        except Exception:
            return "error"
        if op.fAnyOperationsAborted:
            return "aborted"
        return "error" if result != 0 else "ok"

    def split_after(batch):
        deleted = [p for p in batch if not os.path.exists(p)]
        failed = [p for p in batch if os.path.exists(p)]
        return deleted, failed

    status = run(existing)
    if status == "ok":
        deleted, failed = split_after(existing)
        if not failed:
            return deleted, missing
    elif status == "aborted":
        # 用户在「永久删除」警告里点了取消：不要逐文件重试（会反复弹窗）
        deleted, failed = split_after(existing)
        return deleted, failed + missing

    deleted, failed = [], []
    for path in existing:
        if run([path]) == "ok" and not os.path.exists(path):
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
            try:
                # 只读文件 os.remove 会失败，去掉只读位再试一次
                import stat as _stat
                os.chmod(path, _stat.S_IWRITE)
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


def _recycle_index(roots):
    """建立 {原路径(normcase): ($R 路径, $I 路径)} 索引。

    roots 是卷根（盘符根或 UNC 共享根），内部再拼 $Recycle.Bin。
    """
    index = {}
    for root in roots:
        base = Path(root) / "$Recycle.Bin"
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
    roots = sorted({_drive_root(p) for p in paths if _drive_root(p)})
    index = _recycle_index(roots)
    for path in paths:
        entry = index.get(norm_path(path))
        if _restore_entry(path, entry):
            restored.append(path)
        else:
            failed.append(path)
    return restored, failed


_RESTORE_ATTEMPTS = 3
_RESTORE_DELAY = 0.08


def _restore_entry(path, entry):
    """把一条回收站条目搬回原位置。

    刚被删除的文件常被杀毒软件或索引服务短暂占用，搬回去会报「另一个程序
    正在使用」；回收站元数据也可能晚一两帧才写全。这些都是几十毫秒的窗口，
    所以按小步退避重试，而不是把用户点「撤销」的一次落空当成永久失败。
    """
    destination = Path(entry[2]) if entry else Path(str(path))
    for attempt in range(_RESTORE_ATTEMPTS):
        if not entry:
            time.sleep(_RESTORE_DELAY * (attempt + 1))
            entry = _recycle_index([_drive_root(str(path))]).get(
                norm_path(str(path)))
            continue
        data_file, index_file, _original = entry
        if destination.exists():
            return False
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(data_file), str(destination))
        except OSError:
            # 跨卷搬移是「复制再删除」：复制完了、删 $R 失败也算还原成功
            if destination.exists() and not data_file.exists():
                try:
                    index_file.unlink()
                except OSError:
                    pass
                return True
            if attempt + 1 < _RESTORE_ATTEMPTS:
                time.sleep(_RESTORE_DELAY * (attempt + 1))
                continue
            return False
        try:
            index_file.unlink()
        except OSError:
            pass
        return True
    return False


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


def restore_hardlinks(records):
    """撤销硬链接替换：副本恢复成普通文件（内容与保留文件相同）。

    records: [{"act": "hardlink", "src": 副本路径, "dst": 保留文件路径}]。
    只有当副本仍是保留文件的硬链接时才恢复，否则跳过（不覆盖、不猜测）。
    返回 (restored, failed)。
    """
    restored, failed = [], []
    for record in records:
        src, dst = Path(record["src"]), Path(record["dst"])
        try:
            if not src.exists() or not dst.exists():
                failed.append(str(src))
                continue
            if not os.path.samefile(src, dst):
                failed.append(str(src))
                continue
            temp = src.with_name(
                src.name + f".deduprestore-{uuid.uuid4().hex[:8]}.tmp")
            try:
                shutil.copy2(str(dst), str(temp))
                os.replace(temp, src)
                restored.append(str(src))
            finally:
                if temp.exists():
                    try:
                        temp.unlink()
                    except OSError:
                        pass
        except Exception:
            failed.append(str(src))
    return restored, failed


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


def export_report_json(path, groups, root=None):
    """导出 JSON 报告（供脚本和 CLI 使用）。"""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "root": str(root) if root else "",
        "groups": [
            {
                "kind": group.kind,
                "similarity": group.similarity,
                "note": group.note,
                "keep": group.keep,
                "wasted": group.wasted,
                "items": [
                    {"path": str(item.path), "name": item.name,
                     "size": item.size, "mtime": item.mtime,
                     "dims": item.dims,
                     "duration": item.duration,
                     "protected": bool(item.protected)}
                    for item in group.items
                ],
            }
            for group in groups
        ],
        "total_groups": len(groups),
        "total_wasted": sum(g.wasted for g in groups),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
