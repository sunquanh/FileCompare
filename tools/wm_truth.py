# -*- coding: utf-8 -*-
"""真实样本真值清单：给本地素材打指纹，让 hit / damage 从此可比。

为什么要有它：bench/real/ 下的真图和人工掩膜是本地素材（.gitignore 排除），
换机器、清理目录或手滑改了一笔掩膜，基准里的 hit / damage 就会变；报表只
会显示"数值变了"，看不出是算法改了还是真值改了。上一轮 real_screen_ide
从检测器初稿换成人工真值，hit 就从 0.88 掉到 0.33，正是这种混淆。

所以把每个用例的掩膜像素数和两份文件的 sha256 写进 bench/real/manifest.json
（入库，只存指纹不存像素）：

    python tools/wm_truth.py --verify      # 校验本地素材与清单一致
    python tools/wm_truth.py --update      # 素材确有意更新时重写清单

--verify 发现缺失、掩膜被改、图片被换，都以退出码 1 结束；真值口径（每个
用例的真值从哪来、画到哪算）写在 bench/real/TRUTH.md。
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
MANIFEST_NAME = "manifest.json"

# 读真值掩膜的唯一口径：两份工具都走 load_mask，别再各写一份阈值。
MASK_THRESHOLD = 127


def load_mask(path):
    """读真值掩膜（255 / 任何 > 127 的像素算水印）。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > MASK_THRESHOLD


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root=None):
    """扫描 <case>/input.png，生成清单主体：掩膜像素数 + 两份文件指纹。"""
    root = Path(root) if root else BENCH
    cases = {}
    for input_path in sorted(root.glob("*/input.png")):
        case_dir = input_path.parent
        if case_dir.name.startswith("_"):
            continue
        mask_path = case_dir / "mask.png"
        if not mask_path.is_file():
            continue
        cases[case_dir.name] = {
            "mask_pixels": int(load_mask(mask_path).sum()),
            "input_sha256": _sha256(input_path),
            "mask_sha256": _sha256(mask_path),
        }
    return cases


def manifest_drift(root=None):
    """对比本地素材与清单，返回问题描述列表（空 = 一致或没有清单）。

    没有 manifest（比如合成样本目录）时返回空列表：这项检查只对真实样本
    有意义，缺清单不该拦下普通评测。
    """
    root = Path(root) if root else BENCH
    path = root / MANIFEST_NAME
    if not path.is_file():
        return []
    recorded = json.loads(path.read_text(encoding="utf-8")).get("cases", {})
    current = collect(root)
    problems = []
    for name in sorted(recorded):
        want = recorded[name]
        got = current.get(name)
        if got is None:
            problems.append(f"{name}: 本地缺 input.png 或 mask.png")
            continue
        if want.get("mask_pixels") != got["mask_pixels"]:
            problems.append(f"{name}: 掩膜像素 {want.get('mask_pixels')}"
                            f" -> {got['mask_pixels']}")
        if want.get("input_sha256") != got["input_sha256"]:
            problems.append(f"{name}: input.png 已换（sha256 不同）")
        if want.get("mask_sha256") != got["mask_sha256"]:
            problems.append(f"{name}: mask.png 已改（sha256 不同）")
    for name in sorted(set(current) - set(recorded)):
        problems.append(f"{name}: 清单里没有这个用例")
    return problems


def _write_manifest(cases, path):
    payload = {
        "note": "bench/real 本地素材指纹清单，只存告警用的最小事实（掩膜像素数、"
                "两份文件的 sha256）；由 tools/wm_truth.py --update 生成，"
                "真值口径见 TRUTH.md",
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
            print(f"  {name:<18} 掩膜 {info['mask_pixels']:>7} px")
        return 0

    if not (root / MANIFEST_NAME).is_file():
        print(f"{root} 没有 {MANIFEST_NAME}，先跑 --update")
        return 2
    problems = manifest_drift(root)
    if not problems:
        print(f"{root} 的本地素材与清单一致")
        return 0
    print("本地素材与清单不一致：")
    for msg in problems:
        print(f"  [!] {msg}")
    print("\n真值确有意改动就用 --update 重写清单；否则请恢复素材。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
