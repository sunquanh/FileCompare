"""dedup_cli.py — FileCompare 命令行入口（R23）。

用法：
    python dedup_cli.py <目录> [--json out.json] [--csv out.csv]
    python dedup_cli.py <目录> [--exact-only] [--no-photos] [--min-sim 90]

不依赖图形界面：扫一遍目录，把重复组打印到终端，可选导出 CSV/JSON 报告。
退出码：0 有重复组或无重复都正常；1 参数错误；2 扫描出错。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dedup_core import scan_folder
from dedup_files import export_report, export_report_json
from dedup_models import ScanOptions


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="FileCompare 命令行：扫描目录，列出重复文件组。")
    parser.add_argument("dir", help="要扫描的文件夹")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="导出 JSON 报告到该路径")
    parser.add_argument("--csv", dest="csv_out", default=None,
                        help="导出 CSV 报告到该路径")
    parser.add_argument("--min-sim", type=int, default=0,
                        help="只显示相似度 >= 该值的组（0–100，默认 0 全部显示）")
    parser.add_argument("--exact-only", action="store_true",
                        help="只做字节完全一致检测，不做照片/视频相似检测")
    parser.add_argument("--no-photos", action="store_true",
                        help="跳过照片相似检测")
    parser.add_argument("--no-videos", action="store_true",
                        help="跳过视频相似检测")
    parser.add_argument("--no-names", action="store_true",
                        help="跳过同名文件检测")
    parser.add_argument("--min-size", type=int, default=0,
                        help="只扫描 >= 该字节数的文件（默认 0 不限）")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    if not root.is_dir():
        print(f"错误：{root} 不是一个存在的文件夹", file=sys.stderr)
        return 1
    if not 0 <= args.min_sim <= 100:
        print("错误：--min-sim 须在 0–100 之间", file=sys.stderr)
        return 1

    opts = ScanOptions(
        root=root,
        exact=True,
        photos=not args.no_photos and not args.exact_only,
        videos=not args.no_videos and not args.exact_only,
        audio=not args.exact_only,
        names=not args.no_names and not args.exact_only,
        min_size=args.min_size,
    )

    def _progress(stage, done, total, detail):
        if total:
            print(f"\r[{stage}] {done}/{total} {detail}", end="",
                  flush=True)
        else:
            print(f"\r[{stage}] {detail}", end="", flush=True)

    try:
        result = scan_folder(opts, on_progress=_progress)
    except Exception as exc:
        print(f"\n扫描出错：{exc}", file=sys.stderr)
        return 2
    print()

    groups = [g for g in result.groups if g.similarity >= args.min_sim]
    total_wasted = sum(g.wasted for g in groups)
    print(f"扫描完成：{root}")
    print(f"重复组：{len(groups)} 组，可释放约 {_human_size(total_wasted)}")
    for i, group in enumerate(groups, 1):
        print(f"\n[{i}] {group.kind} · 相似度 {group.similarity}% · "
              f"可释放 {_human_size(group.wasted)}"
              + (f" · {group.note}" if group.note else ""))
        for j, item in enumerate(group.items):
            mark = "（保留）" if j == group.keep else ""
            print(f"    {item.path}{mark}")

    if args.json_out:
        export_report_json(args.json_out, groups, root)
        print(f"\nJSON 报告已写到：{args.json_out}")
    if args.csv_out:
        export_report(args.csv_out, groups, root)
        print(f"CSV 报告已写到：{args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
