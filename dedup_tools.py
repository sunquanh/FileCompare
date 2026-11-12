"""文件工具引擎：整理归类 + 批量重命名。

修复要点：
  * 撤销整理时**绝不覆盖**原位置已有文件，冲突一律跳过并统计；
  * 重命名改成两阶段（先改临时名再改目标名），支持 A↔B 互换与仅大小写改动；
  * 撞名检查包含目录名与所有同目录文件（不受扩展名过滤影响）；
  * 每次操作写记录，长任务边执行边写日志，进程中断也能撤销已完成部分。

所有整理/重命名都会写操作记录（%APPDATA%/Dedup/_操作记录/*.json），可一键撤销。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dedup_models import app_dir, ignore, should_skip_dir

# ---------------------------------------------------------------- 类别表
CATEGORY = {
    "照片": {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp", ".gif",
             ".tif", ".tiff", ".raw", ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2"},
    "视频": {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv", ".flv",
             ".ts", ".webm", ".mpg", ".mpeg", ".rmvb", ".vob", ".mts"},
    "音频": {".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg", ".wma", ".ape", ".opus"},
    "文档": {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt",
             ".md", ".rtf", ".csv", ".epub", ".mobi", ".azw3", ".odt", ".ods"},
    "压缩包": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".iso", ".cab"},
    "安装包": {".exe", ".msi", ".apk", ".dmg", ".pkg", ".deb", ".rpm", ".appx", ".msix"},
    "代码": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go",
             ".rs", ".php", ".rb", ".sh", ".bat", ".ps1", ".html", ".css", ".json",
             ".xml", ".yml", ".yaml", ".toml", ".sql", ".ini", ".cfg", ".conf"},
}

FORMAT_ALIAS = {
    ".jpeg": "JPG", ".tif": "TIFF", ".tiff": "TIFF", ".mpeg": "MPG",
    ".m4v": "M4V", ".heif": "HEIC", ".opus": "OPUS", ".azw3": "AZW3",
    ".msix": "MSIX", ".appx": "APPX",
}

UNCATEGORIZED = "其他"

SKIP_DIRS = {
    "$RECYCLE.BIN", "System Volume Information", ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".idea", ".vscode", ".vs",
}

MANIFEST_DIR = app_dir() / "_操作记录"
MANIFEST_KEEP = 12
_LOG = logging.getLogger("dedup.tools")


class Cancelled(Exception):
    """用户取消。"""


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled


def human_size(n: float) -> str:
    n = float(n or 0)
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if n >= 100 else f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


def norm(name: str) -> str:
    return os.path.normcase(name)


# ---------------------------------------------------------------- 分类
def category_of(suffix: str) -> str:
    s = suffix.lower()
    for cat, exts in CATEGORY.items():
        if s in exts:
            return cat
    return UNCATEGORIZED


def format_of(suffix: str) -> str:
    s = suffix.lower()
    if not s:
        return "无扩展名"
    if s in FORMAT_ALIAS:
        return FORMAT_ALIAS[s]
    return s[1:].upper()


def _exif_datetime_raw(image):
    """读拍摄日期原串。

    旧实现用私有 API `_getexif()`，Pillow 10 起已移除，于是「优先拍摄日期」
    静默失效、全部退回文件修改时间。这里用公开的 getexif()。
    """
    for getter in ("getexif", "_getexif"):
        try:
            accessor = getattr(image, getter, None)
            data = accessor() if callable(accessor) else None
        except Exception:
            data = None
        if not data:
            continue
        for tag in (36867, 306):
            try:
                value = data.get(tag)
            except Exception:
                value = None
            if value:
                return value
    return None


def _exif_date(path: Path):
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        if path.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff",
                                       ".png", ".webp", ".heic", ".heif"):
            return None
        with Image.open(path) as image:
            raw = _exif_datetime_raw(image)
            if not raw:
                return None
            return datetime.strptime(str(raw)[:10], "%Y:%m:%d")
    except Exception:
        return None


# ---------------------------------------------------------------- 收集
def collect_all(root: Path, recursive: bool = True, cancel=None,
                with_dirs: bool = False):
    """收集 root 下的文件（可选目录名），跳过系统目录。"""
    root = Path(root)
    files, dirs = [], []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                followlinks=False):
        _check_cancel(cancel)
        directory = Path(dirpath)
        if recursive:
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(("~$", "."))
                           and not should_skip_dir(
                               d, len(directory.parts) - len(root.parts) + 1)]
        else:
            dirnames[:] = []
        if with_dirs:
            # 目录名也要带完整路径：改名撞名按（目录，文件名）成对算
            dirs.extend(directory / d for d in dirnames)
        for name in filenames:
            path = directory / name
            try:
                path.stat()
            except OSError:
                continue
            files.append(path)
    return (files, dirs) if with_dirs else files


# ---------------------------------------------------------------- 整理
def build_organize_plan(root: Path, mode: str = "type", granularity: str = "month",
                        use_exif: bool = True, recursive: bool = True,
                        two_level: bool = True, cancel=None):
    """生成整理计划。返回 (rows, summary, skipped)。"""
    root = Path(root)
    files = collect_all(root, recursive=recursive, cancel=cancel)
    rows, summary, skipped = [], {}, 0
    seen = set()
    for path in sorted(files, key=lambda x: str(x).lower()):
        try:
            stat = path.stat()
        except OSError:
            continue
        _check_cancel(cancel)
        if mode == "type":
            cat = category_of(path.suffix)
            fmt = format_of(path.suffix)
            label = f"{cat} / {fmt}" if two_level else cat
            dst_dir = root / cat / fmt if two_level else root / cat
        else:
            moment = _exif_date(path) if use_exif else None
            if moment is None:
                try:
                    moment = datetime.fromtimestamp(stat.st_mtime)
                except (OSError, OverflowError, ValueError) as exc:
                    _LOG.debug("时间戳无效 %s: %s", path, exc)
                    continue
            if granularity == "year":
                label = f"{moment:%Y}"
                dst_dir = root / f"{moment:%Y}"
            elif granularity == "month":
                label = f"{moment:%Y}年{moment:%m}月"
                dst_dir = root / f"{moment:%Y}" / f"{moment:%Y-%m}"
            else:
                label = f"{moment:%Y-%m-%d}"
                dst_dir = (root / f"{moment:%Y}" / f"{moment:%Y-%m}"
                           / f"{moment:%Y-%m-%d}")
        if path.parent == dst_dir:
            skipped += 1
            continue
        key = (str(path), str(dst_dir))
        if key in seen:
            continue
        seen.add(key)
        rows.append([label, str(path), str(dst_dir), path.name])
        entry = summary.setdefault(label, [0, 0])
        entry[0] += 1
        entry[1] += stat.st_size
    return rows, summary, skipped


def apply_organize_plan(rows, action: str = "move", on_progress=None,
                        cancel=None, on_record=None):
    """按计划移动/复制。返回 (done, failed, records, reasons)。

    on_record(records) 每 100 个文件回调一次，用于写日志式操作记录。
    """
    done, failed, records = 0, 0, []
    total = len(rows)
    used_dirs: dict[str, set] = {}
    reasons: dict[str, str] = {}
    for index, (label, src, dst_dir, name) in enumerate(rows):
        _check_cancel(cancel)
        try:
            source = Path(src)
            destination = Path(dst_dir)
            destination.mkdir(parents=True, exist_ok=True)
            used = used_dirs.setdefault(
                dst_dir, {norm(x.name) for x in destination.iterdir()
                          if x.is_file()})
            target = name
            if norm(target) in used:
                stem, suffix = source.stem, source.suffix
                k = 1
                while norm(f"{stem}_{k}{suffix}") in used:
                    k += 1
                target = f"{stem}_{k}{suffix}"
            used.add(norm(target))
            target_path = destination / target
            if norm(str(source)) == norm(str(target_path)):
                continue
            if action == "move":
                shutil.move(str(source), str(target_path))
            else:
                shutil.copy2(str(source), str(target_path))
            done += 1
            record = {"act": action, "src": str(source),
                      "dst": str(target_path)}
            if action != "move":
                # 记下复制件的大小/时间，撤销时据此确认没被改动过
                try:
                    st = target_path.stat()
                    record["size"] = int(st.st_size)
                    record["mtime"] = float(st.st_mtime)
                except OSError as exc:
                    ignore(__name__, exc)
            records.append(record)
            if on_record is not None and done % 100 == 0:
                on_record(records)
        except Exception as exc:
            failed += 1
            reasons[src] = str(exc)
        if on_progress:
            on_progress(index + 1, total, name)
    return done, failed, records, reasons


def _remove_empty_dir_chains(paths, protect=(), on_removed=None):
    """自底向上删除真正空了的目录，protect 中的目录及上级不会被删。"""
    protected = [Path(p).resolve() for p in protect]
    for start in list(paths):
        current = Path(start)
        while current != current.parent and current.exists() and current.is_dir():
            if any(str(current.resolve()) == str(p) for p in protected):
                break
            try:
                if any(current.iterdir()):
                    break
                current.rmdir()
                if on_removed:
                    on_removed(current)
            except OSError:
                break
            current = current.parent


# ---------------------------------------------------------------- 重命名
@dataclass
class RenameOptions:
    find: str = ""
    replace: str = ""
    use_replace: bool = False
    case_insensitive: bool = True
    prefix: str = ""
    suffix: str = ""
    use_template: bool = False
    template: str = "{原名}_{n}"
    num_start: int = 1
    num_step: int = 1
    num_width: int = 2
    letter_case: str = "none"    # none / upper / lower / title
    recursive: bool = False
    ext: str = ""


TOKENS = "{原名} 原文件主名   {n} 序号   {日期} 修改日期(如 20250807)"


def _case_fold_stem(stem: str, mode: str) -> str:
    if mode == "upper":
        return stem.upper()
    if mode == "lower":
        return stem.lower()
    if mode == "title":
        return re.sub(r"[a-zA-Z]+",
                      lambda m: m.group(0)[:1].upper() + m.group(0)[1:].lower(),
                      stem)
    return stem


def build_rename_plan(root: Path, opts: RenameOptions, cancel=None):
    """生成重命名计划。返回 (rows, summary)。

    rows 每行 [源路径, 原名, 新名, 状态]  状态: rename / auto / same
    """
    root = Path(root)
    files = collect_all(root, recursive=opts.recursive, cancel=cancel)
    exts = {x.strip().lower() if x.strip().startswith(".")
            else "." + x.strip()
            for x in re.split(r"[,，;；\s]+", opts.ext or "") if x.strip()}
    selected = [p for p in files if not exts or p.suffix.lower() in exts]
    selected.sort(key=lambda p: str(p).lower())
    selected_keys = {norm(str(p)) for p in selected}

    # 撞名集合：按（目录，文件名）成对算，只防同目录撞名；
    # 包含所有不参与本次改名的文件 + 所有目录
    _all, dir_names = collect_all(root, recursive=True, with_dirs=True)
    occupied = {(norm(str(p.parent)), norm(p.name)) for p in _all
                if norm(str(p)) not in selected_keys}
    occupied |= {(norm(str(Path(d).parent)), norm(Path(d).name))
                 for d in dir_names}

    pattern = None
    if opts.use_replace and opts.find:
        flag = re.IGNORECASE if opts.case_insensitive else 0
        try:
            pattern = re.compile(re.escape(opts.find), flag)
        except re.error:
            pattern = None

    counter = opts.num_start
    rows = []
    planned_new: set[tuple] = set()
    for path in selected:
        _check_cancel(cancel)
        stem, ext = path.stem, path.suffix
        if pattern is not None:
            stem = pattern.sub(opts.replace, stem)
        stem = _case_fold_stem(stem, opts.letter_case)
        if opts.prefix:
            stem = opts.prefix + stem
        if opts.suffix:
            stem = stem + opts.suffix

        if opts.use_template and opts.template:
            template = opts.template
            number = ""
            if "{n}" in template:
                number = str(counter).zfill(max(1, opts.num_width))
                counter += opts.num_step
            name = (template
                    .replace("{原名}", stem)
                    .replace("{n}", number)
                    .replace("{日期}",
                             datetime.fromtimestamp(path.stat().st_mtime)
                             .strftime("%Y%m%d")))
            new_stem = name.strip() or stem
        else:
            new_stem = stem

        if new_stem == path.stem:
            rows.append([str(path), path.name, path.name, "same"])
            continue

        status = "rename"
        base = new_stem
        k = 2
        here = norm(str(path.parent))
        while ((here, norm(base + ext)) in occupied
               or (here, norm(base + ext)) in planned_new):
            base = f"{new_stem} ({k})"
            k += 1
            status = "auto"
        planned_new.add((here, norm(base + ext)))
        rows.append([str(path), path.name, base + ext, status])

    summary = {}
    for _, _, _, state in rows:
        summary[state] = summary.get(state, 0) + 1
    return rows, summary


def apply_rename_plan(rows, on_progress=None, cancel=None, on_record=None):
    """执行重命名（两阶段，支持互换与仅大小写改动）。返回 (done, failed, records)。

    on_record(records) 在阶段一完成后与阶段二每 100 个文件回调一次。
    records 始终反映文件当前所在位置（阶段一是临时名，阶段二改成目标名），
    供调用方边执行边写撤销记录，进程中断也能撤销已完成部分。
    """
    todo = [row for row in rows if row[3] != "same"]
    total = len(todo)
    done, failed = 0, 0
    staged: list[tuple[Path, Path, str]] = []
    state: dict[str, str] = {}

    def snapshot():
        return [{"act": "rename", "src": src, "dst": dst}
                for src, dst in state.items()]

    # 阶段一：全部改成临时名，避免 A→B、B→A 互相占用
    for src, _old, new, _status in todo:
        _check_cancel(cancel)
        source = Path(src)
        if not source.exists():
            failed += 1
            continue
        temp = source.with_name(f".dedupren-{uuid.uuid4().hex[:10]}")
        try:
            os.rename(source, temp)
        except OSError:
            failed += 1
            continue
        staged.append((source, temp, new))
        state[str(source)] = str(temp)
    if on_record is not None and state:
        on_record(snapshot())

    # 阶段二：临时名 → 目标名
    for step, (source, temp, new) in enumerate(staged, 1):
        _check_cancel(cancel)
        target = source.with_name(new)
        try:
            if target.exists():
                raise FileExistsError("目标已存在")
            os.rename(temp, target)
            done += 1
            state[str(source)] = str(target)
            if on_record is not None and done % 100 == 0:
                on_record(snapshot())
        except Exception:
            try:
                os.rename(temp, source)
                state.pop(str(source), None)
            except OSError:
                pass
            failed += 1
        if on_progress:
            on_progress(step, total, new)
    return done, failed, snapshot()


# ---------------------------------------------------------------- 操作记录
def _cleanup_stale_temps():
    """删掉一小时前被杀留下的 .tmp 临时记录。

    save/rewrite 都用“临时文件 + os.replace”写记录，中途被杀会留一个
    .tmp；它不进 *.json glob，不影响撤销，只是攒灰。只删一小时前的，
    不碰正在写的临时文件。"""
    try:
        now = datetime.now()
        for temp in MANIFEST_DIR.glob("*.tmp"):
            try:
                age = (now - datetime.fromtimestamp(
                    temp.stat().st_mtime)).total_seconds()
                if age > 3600:
                    temp.unlink()
            except OSError:
                pass
    except OSError:
        pass


def save_manifest(kind: str, root: Path, records) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_temps()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = MANIFEST_DIR / f"{kind}-{stamp}.json"
    data = {
        "kind": kind,
        "root": str(Path(root)),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(records),
        "items": list(records),
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    os.replace(temp, path)
    for old in sorted(MANIFEST_DIR.glob(f"{kind}-*.json"))[:-MANIFEST_KEEP]:
        try:
            old.unlink()
        except Exception as exc:
            ignore(__name__, exc)
    return path


def rewrite_manifest(path: Path, records) -> Path:
    """把已执行的部分记录写回（用于中断后仍可撤销）。

    原子写（临时文件 + os.replace）：改写中途被杀也不会留下截断的记录。"""
    try:
        _cleanup_stale_temps()
        data = read_manifest(path)
        data["items"] = list(records)
        data["count"] = len(records)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        os.replace(temp, path)
    except Exception as exc:
        ignore(__name__, exc)
    return path


def _manifest_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def last_manifest(kind: str | None = None):
    if not MANIFEST_DIR.exists():
        return None
    files = sorted(MANIFEST_DIR.glob("*.json"), key=_manifest_mtime)
    if kind:
        files = [f for f in files if f.name.startswith(kind + "-")]
    return files[-1] if files else None


def last_manifest_multi(*kinds: str):
    """取多个种类里最新的一份操作记录（按修改时间）。"""
    best = None
    for kind in kinds:
        path = last_manifest(kind)
        if path is not None and (best is None
                                 or _manifest_mtime(path) > _manifest_mtime(best)):
            best = path
    return best


def last_restorable_cleanup():
    """最新一份清理记录里有可还原（移入回收站/硬链接）的文件时返回它，否则 None。

    永久删除的记录不可还原，撤销按钮不应为它启用。"""
    path = last_manifest("cleanup")
    if path is None:
        return None
    try:
        data = read_manifest(path)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("items", [])
    if not isinstance(items, list):
        return None
    if any(isinstance(r, dict) and r.get("act") in ("recycle", "hardlink")
           for r in items):
        return path
    return None


def read_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _copy_still_ours(record) -> bool:
    """撤销「复制」前确认目标还是当初复制的那份。

    旧实现直接 unlink：用户整理后又编辑过目标文件，撤销会把编辑结果删掉。
    """
    try:
        st = Path(record["dst"]).stat()
    except (OSError, KeyError):
        return False
    size = record.get("size")
    mtime = record.get("mtime")
    if size is None:
        return True          # 旧记录没有校验信息，保持原行为
    if int(st.st_size) != int(size):
        return False
    if mtime is not None and abs(float(st.st_mtime) - float(mtime)) > 2.0:
        return False
    return True


def _undo_renames(records, cancel=None):
    """两阶段撤销改名：支持 A↔B 互换与仅大小写改动。

    单阶段判断「目标存在且源不存在」在 Windows 上必然失败：大小写改名时
    新旧路径指向同一个文件，互换时两个位置都有文件——实测撤销一直是静默
    跳过（0 成功、0 失败、N 跳过），用户以为撤销了其实没有。
    """
    done = failed = skipped = 0
    staged = []
    for record in records:
        _check_cancel(cancel)
        try:
            source = Path(record["src"])
            destination = Path(record["dst"])
        except Exception:
            failed += 1
            continue
        if not destination.exists():
            skipped += 1
            continue
        temp = destination.with_name(f".dedupundo-{uuid.uuid4().hex[:10]}")
        try:
            os.rename(destination, temp)
        except OSError:
            failed += 1
            continue
        staged.append((source, destination, temp))
    for source, destination, temp in staged:
        _check_cancel(cancel)
        try:
            if source.exists():
                # 原位置被别的文件占了：放回改名后的位置，绝不覆盖
                os.rename(temp, destination)
                skipped += 1
                continue
            source.parent.mkdir(parents=True, exist_ok=True)
            os.rename(temp, source)
            done += 1
        except Exception:
            try:
                if temp.exists() and not destination.exists():
                    os.rename(temp, destination)
            except OSError as exc:
                ignore(__name__, exc)
            failed += 1
    return done, failed, skipped


def undo_manifest(path: Path, on_progress=None, cancel=None):
    """撤销一次整理/重命名/清理记录。返回 (done, failed, skipped)。

    原位置已有同名文件时跳过（绝不覆盖）；改名走两阶段执行。
    """
    data = read_manifest(path)
    if not isinstance(data, dict):
        raise ValueError(f"操作记录损坏：{path.name} 不是对象")
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"操作记录损坏：{path.name} 的 items 不是列表")
    items = list(items)
    items.reverse()
    total = len(items)
    rename_records = [r for r in items
                      if isinstance(r, dict) and r.get("act") == "rename"]
    rest = [r for r in items
            if not (isinstance(r, dict) and r.get("act") == "rename")]
    done, failed, skipped = _undo_renames(rename_records, cancel)
    touched: set = set()
    root = data.get("root")
    for index, record in enumerate(rest):
        _check_cancel(cancel)
        try:
            source = Path(record["src"])
            destination = Path(record["dst"])
            touched.add(destination.parent)
            if record["act"] == "move":
                if not destination.exists() or source.exists():
                    skipped += 1
                else:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
                    done += 1
            elif record["act"] == "copy":
                if not destination.exists() or not _copy_still_ours(record):
                    skipped += 1
                else:
                    destination.unlink()
                    done += 1
            else:
                skipped += 1
        except Exception:
            failed += 1
        if on_progress:
            dst = record.get("dst", "") if isinstance(record, dict) else ""
            if not isinstance(dst, str):
                dst = ""
            on_progress(index + 1, total, Path(dst).name)
    if data.get("kind") == "organize":
        _remove_empty_dir_chains(touched, [root] if root else [])
    return done, failed, skipped


def undo_cleanup_manifest(path):
    """撤销一次清理记录：回收站还原 + 硬链接恢复。返回 (done, failed, skipped)。"""
    from dedup_files import restore_from_recycle, restore_hardlinks
    data = read_manifest(path)
    if not isinstance(data, dict):
        raise ValueError(f"操作记录损坏：{path.name} 不是对象")
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"操作记录损坏：{path.name} 的 items 不是列表")
    recycle = [r.get("src") for r in items
               if isinstance(r, dict) and r.get("act") == "recycle" and r.get("src")]
    hardlinks = [r for r in items
                 if isinstance(r, dict) and r.get("act") == "hardlink"
                 and r.get("src") and r.get("dst")]
    done = failed = 0
    if recycle:
        restored, missing = restore_from_recycle(recycle)
        done += len(restored)
        failed += len(missing)
    if hardlinks:
        restored, missing = restore_hardlinks(hardlinks)
        done += len(restored)
        failed += len(missing)
    skipped = len(items) - len(recycle) - len(hardlinks)
    return done, failed, skipped


def scan_result_to_dict(result):
    """ScanResult 转 JSON 友好结构（CLI 与脚本共用）。"""
    return {
        "root": str(result.root),
        "file_count": result.file_count,
        "total_size": result.total_size,
        "elapsed": round(result.elapsed, 1),
        "total_groups": result.total_groups,
        "total_wasted": result.total_wasted,
        "groups": [
            {"kind": group.kind, "similarity": group.similarity,
             "note": group.note, "keep": group.keep,
             "items": [{"path": str(item.path), "size": item.size}
                       for item in group.items]}
            for group in result.groups
        ],
    }


def _cli_scan(args):
    import dedup_core as core
    root = Path(args.dir)
    if not root.is_dir():
        print(f"目录不存在：{root}")
        return 1
    if args.dry_run:
        print("dry-run：只扫描，不修改任何文件")
    opts = core.ScanOptions.from_preset(root, quick_scan=args.quick_scan)
    result = core.scan_folder(opts)
    print(f"扫描完成：{result.file_count:,} 个文件，{result.total_groups} 组，"
          f"可释放 {human_size(result.total_wasted)}，耗时 {result.elapsed:.1f} 秒")
    for kind in core.KIND_ORDER:
        count = result.counts()[kind]
        if count["groups"]:
            print("  " + kind + "：" + str(count["groups"]) + " 组，" + str(count["files"]) + " 个文件")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scan_result_to_dict(result),
                                  ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"报告已写入：{out}")


# ---------------------------------------------------------------- 命令行
def _cli_organize(args):
    root = Path(args.dir)
    rows, summary, skipped = build_organize_plan(
        root, mode=args.mode, granularity=args.granularity,
        use_exif=not args.no_exif, recursive=not args.top)
    if not rows:
        print("没有可以整理的文件")
        return
    for label, (count, size) in sorted(summary.items(), key=lambda kv: -kv[1][0]):
        print(f"  {label}: {count} 个, {human_size(size)}")
    print(f"待整理 {len(rows)} 个文件, 已在目标位置跳过 {skipped}")
    if not args.yes and input(
            "确认执行? (y/回车=取消) ").strip().lower() not in ("y", "yes"):
        print("已取消")
        return 1
    done, failed, records, _ = apply_organize_plan(
        rows, action="copy" if args.copy else "move")
    path = save_manifest("organize", root, records)
    print(f"完成: {done} 个, 失败 {failed}。操作记录: {path}")


def _cli_rename(args):
    opts = RenameOptions(
        use_replace=bool(args.find), find=args.find or "",
        replace=args.replace or "", prefix=args.prefix or "",
        suffix=args.suffix or "", use_template=bool(args.template),
        template=args.template or "{原名}_{n}", letter_case=args.case,
        recursive=args.recursive)
    rows, summary = build_rename_plan(Path(args.dir), opts)
    if summary.get("rename", 0) + summary.get("auto", 0) == 0:
        print("没有需要改名的文件")
        return
    print(f"计划重命名 {summary.get('rename', 0)} 个, "
          f"自动加序号 {summary.get('auto', 0)} 个")
    for row in rows[:15]:
        if row[3] != "same":
            print(f"  {row[1]}  ->  {row[2]}")
    if not args.yes and input(
            "确认执行? (y/回车=取消) ").strip().lower() not in ("y", "yes"):
        print("已取消")
        return 1
    partial = {"path": None}

    def journal(records):
        """边执行边写记录：进程被中断时已改名的文件也能撤销。"""
        try:
            if partial["path"] is None:
                partial["path"] = save_manifest("rename", Path(args.dir),
                                               records)
            else:
                rewrite_manifest(partial["path"], records)
        except Exception:
            pass

    done, failed, records = apply_rename_plan(rows, on_record=journal)
    if partial["path"] is not None:
        rewrite_manifest(partial["path"], records)
        path = partial["path"]
    else:
        path = save_manifest("rename", Path(args.dir), records)
    print(f"完成: {done} 个, 失败 {failed}。操作记录: {path}")


def _cli_undo(args):
    path = last_manifest(args.kind)
    if not path:
        print("没有可以撤销的操作")
        return
    data = read_manifest(path)
    if not isinstance(data, dict):
        print(f"操作记录损坏：{path.name} 不是对象")
        return 1
    print(f"撤销 {data.get('time', '未知时间')} 的 {data.get('kind', '未知操作')} 操作，共 "
          f"{data.get('count', '?')} 条: {path.name}")
    if not args.yes and input("确认? (y/回车=取消) ").strip().lower() != "y":
        print("已取消")
        return 1
    if data.get("kind") == "cleanup":
        done, failed, skipped = undo_cleanup_manifest(path)
    else:
        done, failed, skipped = undo_manifest(path)
    print(f"撤销完成: {done} 个, 跳过 {skipped} 个, 失败 {failed} 个")


def main():
    parser = argparse.ArgumentParser(description="文件整理 + 批量重命名")
    sub = parser.add_subparsers(dest="cmd", required=True)

    organize = sub.add_parser("organize", help="整理归类")
    organize.add_argument("dir")
    organize.add_argument("--mode", choices=["type", "date"], default="type")
    organize.add_argument("--granularity", choices=["year", "month", "day"],
                          default="month")
    organize.add_argument("--copy", action="store_true")
    organize.add_argument("--top", action="store_true")
    organize.add_argument("--no-exif", action="store_true")
    organize.add_argument("--yes", action="store_true")
    organize.set_defaults(func=_cli_organize)

    rename = sub.add_parser("rename", help="批量重命名")
    rename.add_argument("dir")
    rename.add_argument("--find", default="")
    rename.add_argument("--replace", default="")
    rename.add_argument("--prefix", default="")
    rename.add_argument("--suffix", default="")
    rename.add_argument("--template", default="")
    rename.add_argument("--case", choices=["none", "upper", "lower", "title"],
                        default="none")
    rename.add_argument("--recursive", action="store_true")
    rename.add_argument("--yes", action="store_true")
    rename.set_defaults(func=_cli_rename)

    scan = sub.add_parser("scan", help="扫描重复文件（只读，不修改任何文件）")
    scan.add_argument("dir")
    scan.add_argument("--json", default="", help="把分组结果写成 JSON 报告")
    scan.add_argument("--quick-scan", action="store_true",
                      help="快速扫描：只找完全重复")
    scan.add_argument("--dry-run", action="store_true",
                      help="试运行：只扫描，不修改任何文件（扫描本身就是只读的）")
    scan.set_defaults(func=_cli_scan)

    undo = sub.add_parser("undo", help="撤销最近一次操作")
    undo.add_argument("--kind", choices=["organize", "rename", "finder", "cleanup"],
                      default=None)
    undo.add_argument("--yes", action="store_true")
    undo.set_defaults(func=_cli_undo)

    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
