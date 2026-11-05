"""重复内容检测工具 —— 命令行入口（薄封装）。

真正的检测逻辑都在 dedup_core 里，本文件只负责：
  * 把命令行参数翻译成 ScanOptions；
  * 把每组重复文件集中到结果的「团NN」文件夹（默认硬链接，不占空间）；
  * 可选按类型归档（--organize），并支持撤销记录之外的整理行为与旧版一致。

用法：
  双击 run.bat                   扫描脚本所在目录，重复内容硬链接到结果目录
  python dedup_finder.py -h      查看全部参数
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import dedup_core as core
import dedup_tools as tools

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BANNER = "=" * 62


def log(message=""):
    print(message, flush=True)


def confirm(message, args):
    """移动原文件前的确认。--yes 跳过；读不到输入或回答不是 y 一律取消。"""
    if args.yes:
        return True
    try:
        answer = input(message).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("已取消")
        return False
    if answer not in ("y", "yes"):
        print("已取消")
        return False
    return True


def safe_name(path: Path, used: set, index: int) -> str:
    candidate = f"{index:02d}_{path.name}"
    stem, suffix = path.stem, path.suffix
    k = 2
    while candidate.lower() in used:
        candidate = f"{index:02d}_{stem}_{k}{suffix}"
        k += 1
    used.add(candidate.lower())
    return candidate


def materialize(groups, out_dir: Path, kind: str, mode: str):
    """把每组文件放进 团NN 子目录。返回记录行。"""
    rows = []
    base = out_dir / kind
    for index, group in enumerate(groups, 1):
        group_dir = base / f"团{index:02d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        used: set = set()
        for order, item in enumerate(group, 1):
            target = group_dir / safe_name(item.path, used, order)
            try:
                if mode == "move":
                    shutil.move(str(item.path), str(target))
                elif mode == "copy":
                    shutil.copy2(str(item.path), str(target))
                else:
                    try:
                        os.link(str(item.path), str(target))
                    except OSError:
                        shutil.copy2(str(item.path), str(target))
            except Exception as exc:
                log(f"    [警告] {item.name} 处理失败: {exc}")
                continue
            rows.append({
                "类型": kind, "组号": index, "文件名": item.name,
                "大小MB": round(item.size / 1048576, 2),
                "原始路径": str(item.path), "放入": str(target),
            })
    return rows


def collect_files_in(root: Path, out_dir: Path):
    """收集文件（跳过输出目录与系统目录）。返回 [(Path, size)]。"""
    result = []
    out_key = core.norm_path(out_dir) if hasattr(core, "norm_path") else str(out_dir)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                followlinks=False):
        directory = Path(dirpath)
        dirnames[:] = [d for d in dirnames
                       if d not in core.SKIP_DIRS and not d.startswith("_重复")]
        if str(directory).lower().startswith(out_key.lower()):
            continue
        for name in filenames:
            if name.startswith(core.SKIP_PREFIX):
                continue
            path = directory / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0:
                continue
            result.append((path, size))
    return result


def main():
    parser = argparse.ArgumentParser(description="扫描目录，把重复内容集中到结果文件夹")
    parser.add_argument("--dir", default=None, help="要扫描的目录，默认脚本所在目录")
    parser.add_argument("--out", default=None, help="结果输出目录")
    parser.add_argument("--mode", choices=["link", "copy", "move"],
                        default="link", help="link=硬链接(默认) copy=复制 move=移动")
    parser.add_argument("--img-dist", type=int, default=8,
                        help="照片 dHash 阈值（64 位里允许的位差，默认 8）")
    parser.add_argument("--img-fine", type=int, default=42,
                        help="照片 pHash 复核阈值（256 位，默认 42）")
    parser.add_argument("--img-confirm", type=int, default=None,
                        help="同 --img-fine，保留旧参数名")
    parser.add_argument("--vid-dist", type=int, default=10,
                        help="视频单帧阈值（默认 10）")
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--no-photo", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--exclude", default="", help="额外排除的目录名，逗号分隔")
    parser.add_argument("--organize", action="store_true",
                        help="顺便把不重复的文件按 类别/格式 归档")
    parser.add_argument("--organize-mode", choices=["link", "copy", "move"],
                        default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--clean-empty", action="store_true")
    parser.add_argument("target", nargs="?", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    chosen = args.dir or args.target
    root = Path(chosen).resolve() if chosen else script_dir
    if not root.is_dir():
        log(f"目录不存在: {root}")
        return 1
    if core.is_filesystem_root(root):
        log("警告：正在扫描盘符根目录，文件很多时会很慢。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).resolve() if args.out else root / f"_重复检测_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(BANNER)
    log("重复内容检测" + ("  +  按类型归档" if args.organize else ""))
    log(BANNER)
    log(f"扫描目录: {root}")
    log(f"输出目录: {out_dir}")
    log(f"模式: {args.mode}")
    log()

    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    options = core.ScanOptions(
        root=root,
        exact=not args.no_exact,
        photos=not args.no_photo,
        videos=not args.no_video,
        audio=False,
        names=False,
        cache=True,
        img_dist=args.img_dist,
        img_fine=args.img_fine or args.img_confirm or 42,
        vid_dist=args.vid_dist,
        min_size=args.min_size,
        exclude=exclude,
        workers=args.workers or 0,
    )

    try:
        result = core.scan_folder(
            options,
            on_progress=lambda stage, done, total, detail:
                print(f"\r  {detail:<40}", end="", flush=True),
            on_log=lambda message: None)
    except RuntimeError as exc:
        print()
        log(str(exc))
        return 1
    print()

    if args.mode == "move" and result.groups:
        move_files = sum(len(g.items) for g in result.groups
                         if g.kind in (core.KIND_EXACT, core.KIND_PHOTO,
                                       core.KIND_VIDEO))
        if move_files:
            log(f"将移动 {move_files} 个重复文件到 {out_dir}，原位置的文件会被移走。")
            if not confirm("确认执行? (y/回车=取消) ", args):
                return 1

    summary = {}
    rows = []
    for kind in (core.KIND_EXACT, core.KIND_PHOTO, core.KIND_VIDEO):
        groups = [g for g in result.groups if g.kind == kind]
        if not groups:
            continue
        summary[kind] = (len(groups), sum(len(g.items) for g in groups),
                         sum(g.wasted for g in groups))
        rows += materialize([g.items for g in groups], out_dir, kind, args.mode)

    log()
    for kind, (groups, files, wasted) in summary.items():
        log(f"  {kind}: {groups} 组 / {files} 个文件 / 可释放 "
            f"{core.human_size(wasted)}")

    organize_records = []
    if args.organize:
        moved = 0
        mode = args.organize_mode or args.mode
        pending = collect_files_in(root, out_dir)
        if mode == "move" and pending:
            log(f"归档将移动 {len(pending)} 个文件到 {out_dir / '_归档'}，原位置的文件会被移走。")
            if not confirm("确认执行? (y/回车=取消) ", args):
                pending = []
        for path, size in pending:
            category = "其他"
            for name, exts in core_category().items():
                if path.suffix.lower() in exts:
                    category = name
                    break
            target_dir = out_dir / "_归档" / category
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / path.name
            if target.exists():
                stem, suffix = path.stem, path.suffix
                k = 1
                while (target_dir / f"{stem}_{k}{suffix}").exists():
                    k += 1
                target = target_dir / f"{stem}_{k}{suffix}"
                log(f"  {path.name} 已存在，改名为 {target.name}")
            try:
                if mode == "move":
                    shutil.move(str(path), str(target))
                    organize_records.append({"act": "move", "src": str(path),
                                             "dst": str(target)})
                elif mode == "copy":
                    shutil.copy2(str(path), str(target))
                else:
                    os.link(str(path), str(target))
                moved += 1
            except Exception as exc:
                log(f"  跳过 {path.name}: {exc}")
                continue
        log(f"  归档: {moved} 个文件")

    if rows:
        report = out_dir / "报告.csv"
        with open(report, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log(f"  报告: {report}")

    move_records = list(organize_records)
    if args.mode == "move":
        move_records += [{"act": "move", "src": r["原始路径"], "dst": r["放入"]}
                         for r in rows]
    if move_records:
        try:
            manifest = tools.save_manifest("finder", root, move_records)
            log(f"  操作记录: {manifest}")
        except Exception as exc:
            fallback = out_dir / f"finder-{stamp}.json"
            try:
                data = {"kind": "finder", "root": str(root),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "count": len(move_records), "items": move_records}
                temp = fallback.with_suffix(".tmp")
                temp.write_text(
                    json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8")
                os.replace(temp, fallback)
                log(f"  [警告] 撤销记录写入失败({exc})，已改存到 {fallback}，"
                    "可拷入 _操作记录 后用 undo --kind finder 撤销")
            except Exception as exc2:
                log(f"  [警告] 撤销记录写入失败({exc})，备用记录也写失败"
                    f"({exc2})，请自行保留报告.csv")

    if args.clean_empty and move_records:
        dirs = {str(Path(r["src"]).parent) for r in move_records}
        removed = []
        tools._remove_empty_dir_chains(
            dirs, [str(root), str(out_dir)], on_removed=removed.append)
        if removed:
            log(f"  清理空目录: {len(removed)} 个")

    if result.skipped_unreadable:
        log(f"  注意：{result.skipped_unreadable} 个文件无法读取，已跳过")
    if result.skipped_empty:
        log(f"  注意：{result.skipped_empty} 个 0 字节文件，已跳过")
    log(f"\n完成：{result.total_groups} 组，可释放 "
        f"{core.human_size(result.total_wasted)}，耗时 {result.elapsed:.1f} 秒")
    return 0


def core_category():
    """按扩展名归类的表（与 dedup_tools 保持一致的最小集合）。"""
    return {
        "照片": {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp",
                 ".gif", ".tif", ".tiff", ".raw", ".cr2", ".nef", ".arw", ".dng"},
        "视频": {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv",
                 ".flv", ".ts", ".webm", ".mpg", ".mpeg"},
        "音频": {".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg", ".wma",
                 ".ape", ".opus"},
        "文档": {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                 ".txt", ".md", ".rtf", ".csv", ".epub"},
        "压缩包": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".iso"},
        "安装包": {".exe", ".msi", ".apk", ".dmg", ".pkg", ".deb", ".rpm"},
    }


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
