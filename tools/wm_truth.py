# -*- coding: utf-8 -*-
"""真实样本真值清单：给本地素材打指纹，让 hit / damage 从此可比。

为什么要有它：bench/real/ 下的真图和人工掩膜是本地素材（.gitignore 排除），
换机器、清理目录或手滑改了一笔掩膜，基准里的 hit / damage 就会变；报表只
会显示"数值变了"，看不出是算法改了还是真值改了。上一轮 real_screen_ide
从检测器初稿换成人工真值，hit 就从 0.88 掉到 0.33，正是这种混淆。

所以把每个用例的尺寸、掩膜像素数和两份文件的 sha256 写进
bench/real/manifest.json（入库，只存指纹不存像素）：

    python tools/wm_truth.py --verify      # 校验本地素材与清单一致
    python tools/wm_truth.py --update      # 素材确有意更新时重写清单

--verify 发现缺失、尺寸变化、掩膜被改、图片被换，都以退出码 1 结束；真值
口径（每个用例的真值从哪来、画到哪算）写在 bench/real/TRUTH.md。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BENCH = Path(__file__).resolve().parent.parent / "bench" / "real"
MANIFEST = BENCH / "manifest.json"
MANIFEST_NAME = "manifest.json"

# 读掩膜的口径必须和 wm_bench 一致（> 127 算水印像素）
MASK_THRESHOLD = 127


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mask_stats(path):
    """返回 (水印像素数, bbox[含左不含右] 或 None)。"""
    arr = np.asarray(Image.open(path).convert("L"))
    hit = arr > MASK_THRESHOLD
    count = int(hit.sum())
    if count == 0:
        return 0, None
    ys, xs = np.nonzero(hit)
    return count, [int(xs.min()), int(ys.min()),
                   int(xs.max()) + 1, int(ys.max()) + 1]


def _truth_label(meta, mask_pixels):
    """真值来源的口径标签，缺省按手工/修订掩膜记。"""
    if mask_pixels == 0:
        return "empty"
    return meta.get("truth") or "manual-mask"


def collect(root=None):
    """扫描 <case>/input.png，生成清单主体。"""
    root = Path(root) if root else BENCH
    cases = {}
    for input_path in sorted(root.glob("*/input.png")):
        case_dir = input_path.parent
        name = case_dir.name
        if name.startswith("_"):
            continue
        mask_path = case_dir / "mask.png"
        if not mask_path.is_file():
            continue
        width, height = Image.open(input_path).size
        mask_pixels, bbox = _mask_stats(mask_path)
        meta = {}
        meta_path = case_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        cases[name] = {
            "kind": meta.get("kind"),
            "truth": _truth_label(meta, mask_pixels),
            "size": [int(width), int(height)],
            "mask_pixels": mask_pixels,
            "mask_bbox": bbox,
            "input_sha256": _sha256(input_path),
            "mask_sha256": _sha256(mask_path),
        }
    return cases


def manifest_drift(root=None):
    """对比本地素材与 manifest，返回问题描述列表（空 = 一致）。

    没有 manifest（比如合成样本目录）时返回空列表：这项检查只对真实样本
    有意义，缺清单不应拦下普通评测。
    """
    root = Path(root) if root else BENCH
    path = root / MANIFEST_NAME
    if not path.is_file():
        return []
    try:
        recorded = json.loads(path.read_text(encoding="utf-8")).get("cases", {})
    except (OSError, ValueError):
        return [f"{path.name} 无法解析"]
    current = collect(root)
    problems = []
    for name in sorted(recorded):
        want = recorded[name]
        got = current.get(name)
        if got is None:
            problems.append(f"{name}: 本地缺 input.png 或 mask.png")
            continue
        if want.get("size") != got["size"]:
            problems.append(f"{name}: 尺寸 {want.get('size')} -> {got['size']}")
        if want.get("mask_pixels") != got["mask_pixels"]:
            problems.append(f"{name}: 掩膜像素 {want.get('mask_pixels')}"
                            f" -> {got['mask_pixels']}")
        if want.get("mask_bbox") != got["mask_bbox"]:
            problems.append(f"{name}: 掩膜范围 {want.get('mask_bbox')}"
                            f" -> {got['mask_bbox']}")
        if want.get("input_sha256") != got["input_sha256"]:
            problems.append(f"{name}: input.png 已换（sha256 不同）")
        if want.get("mask_sha256") != got["mask_sha256"]:
            problems.append(f"{name}: mask.png 已改（sha256 不同）")
    for name in sorted(set(current) - set(recorded)):
        problems.append(f"{name}: 清单里没有这个用例")
    return problems


def _write_manifest(cases, path=None):
    path = Path(path) if path else MANIFEST
    payload = {
        "version": 1,
        "note": "bench/real 本地素材指纹清单，只存哈希与统计，不存像素；"
                "由 tools/wm_truth.py --update 生成，真值口径见 TRUTH.md",
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="真实样本真值清单")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify", action="store_true",
                        help="校验本地素材与清单一致（不一致退出码 1）")
    action.add_argument("--update", action="store_true",
                        help="按当前本地素材重写清单")
    parser.add_argument("--root", metavar="DIR",
                        help="样本目录（默认 bench/real）")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else BENCH
    if args.update:
        cases = collect(root)
        if not cases:
            print(f"{root} 里没有找到用例（需要 <case>/input.png 和 mask.png）")
            return 2
        path = _write_manifest(cases, root / MANIFEST_NAME)
        print(f"已写入 {path}，共 {len(cases)} 例：")
        for name, info in cases.items():
            print(f"  {name:<18} {info['truth']:<12}"
                  f" 掩膜 {info['mask_pixels']:>7} px  {info['size'][0]}x"
                  f"{info['size'][1]}")
        return 0

    problems = manifest_drift(root)
    if not problems:
        print(f"{root} 的 {len(collect(root))} 例本地素材与清单一致")
        return 0
    print("本地素材与清单不一致：")
    for msg in problems:
        print(f"  [!] {msg}")
    print("\n真值确有意改动就用 --update 重写清单；否则请恢复素材。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
