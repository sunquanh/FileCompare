# -*- coding: utf-8 -*-
"""真值样本库脚手架：把一批真图变成 tools/wm_bench.py 能跑的用例。

去水印这条线此前的样本全是合成的（白纸文档 / 屏摄 / 草地照片），阈值调得
再漂亮也不知道在真图上如何。真实样本按 bench 的目录格式丢进来就能一起跑，
缺的只是“从零标注一批掩膜”的起步成本——一页一页用画图工具描水印笔画，
20 张就要一下午。

这里把起步成本压到“改初稿”：先跑一遍现成检测当草稿，人工只补漏、删多余
像素，然后在图上叠一层红色半透明看得见对没对上。

    python tools/wm_label.py export 真图目录 bench/real          # 生成用例初稿
    python tools/wm_label.py overlay bench/real                  # 看红色掩膜对不对
    python tools/wm_label.py check bench/real                     # 交卷前校验格式
    python tools/wm_bench.py bench/real --baseline bench/shipped.json

用例目录格式（与 wm_bench 一致）：

    <case>/input.png    带水印的输入（export 统一转成 PNG，避免二次压缩）
    <case>/mask.png     水印真值掩膜（255 = 水印像素，人工修这一张）
    <case>/clean.png    干净参考（可选，没有就只算 hit/damage/fp）
    <case>/meta.json    {"kind": "doc|photo|screen", "tiled": bool, "note": ...}
    <case>/_overlay.png 红色叠加预览（工具生成，不用手改）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import dedup_watermark as wm  # noqa: E402
import dedup_watermark_template as tpl  # noqa: E402

KINDS = ("doc", "photo", "screen")
SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _images(src):
    src = Path(src)
    if src.is_file():
        return [src] if src.suffix.lower() in SUFFIXES else []
    return sorted(p for p in src.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUFFIXES)


def _kind_of(rgb, forced=None):
    if forced and forced in KINDS:
        return forced
    if wm.looks_like_photo(rgb):
        return "photo"
    # 屏摄页：文档结构 + 明显的低频亮度梯度（暗角）。文档截图的中位色度
    # 接近 0、四角与中心亮度差很小。
    luma, _chroma = wm._luma_chroma(rgb)
    height, width = luma.shape
    corners = np.median([luma[:height // 8, :width // 8].mean(),
                         luma[:height // 8, -width // 8:].mean(),
                         luma[-height // 8:, :width // 8].mean(),
                         luma[-height // 8:, -width // 8:].mean()])
    middle = float(np.median(luma[height // 3:2 * height // 3,
                                   width // 3:2 * width // 3]))
    return "screen" if middle - corners > 12.0 else "doc"


def _draft_mask(rgb, kind):
    """检测初稿：文档走全局掩膜，照片走网格，平铺另给周期提示。"""
    if kind == "photo":
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for det in wm._grid_light_boxes(rgb):
            if det.mask is not None:
                mask |= det.mask
        return mask, None
    tint = wm.detect_tint(rgb)
    if not tint:
        return np.zeros(rgb.shape[:2], dtype=bool), None
    mask = wm._detect_mask(rgb, tint)
    if mask.any():
        luma, chroma = wm._luma_chroma(rgb)
        mask = wm._recall_mask(rgb, mask, tint, stats=(luma, chroma))
    # 平铺水印整页印满：检测多半只认出一部分，用“按周期折叠”的掩膜当地盘
    # 更完整（重叠部分交给人工删）。先只估周期（自相关，便宜）；估不出就不
    # 做整页重建——那是每张图一次全尺寸折叠，人工标注时没必要付这个钱。
    period = tpl.tiled_period(rgb)
    if period is None:
        return mask, None
    _fixed, info = tpl.remove_tiled_watermark(rgb, period=period)
    if info is not None and "mask" in info:
        mask = mask | info["mask"]
    return mask, (period[0], period[1])


def _overlay(rgb, mask):
    """红色半透明叠加：掩膜位置、笔画粗细、有没有吃掉正文，一眼能看。"""
    out = (rgb.astype(np.float32) * 0.45 + 255.0 * 0.55).astype(np.uint8)
    out[mask] = (np.array(rgb, dtype=np.float32)[mask] * 0.25
                 + np.array([255, 40, 40], dtype=np.float32) * 0.75)
    return np.clip(out, 0, 255).astype(np.uint8)


def export(sources, out_dir, kind=None, limit=None):
    """为每张图建一个用例目录，返回目录列表。

    idempotent：同名用例已存在就跳过（不覆盖人工改过的 mask.png）。
    """
    out_dir = Path(out_dir)
    images = _images(sources)
    if limit:
        images = images[:int(limit)]
    made = []
    for index, path in enumerate(images, 1):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        name = path.stem
        case = out_dir / name
        suffix = 2
        while (case / "input.png").exists():
            case = out_dir / f"{name}_{suffix}"
            suffix += 1
        case.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(case / "input.png")
        found_kind = _kind_of(rgb, kind)
        mask, period = _draft_mask(rgb, found_kind)
        Image.fromarray((mask.astype(np.uint8)) * 255).save(case / "mask.png")
        Image.fromarray(_overlay(rgb, mask)).save(case / "_overlay.png")
        note = "初稿：检测掩膜，人工补漏/删多余"
        if period:
            note += f"；平铺周期 {period[1]}×{period[0]}"
        (case / "meta.json").write_text(json.dumps({
            "kind": found_kind, "tiled": bool(period), "note": note,
            "source": str(path)}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        made.append(case)
        print(f"  [{index}/{len(images)}] {case.name}: {found_kind}"
              f"、掩膜 {int(mask.sum())} 像素")
    print(f"生成 {len(made)} 个用例初稿 -> {out_dir}")
    print("下一步：在画图工具里改 mask.png（白色=水印），"
          "再跑 overlay 看红色叠加确认")
    return made


def overlay(cases, force=False):
    """重新生成 _overlay.png（mask.png 改过之后跑这一步）。"""
    cases = Path(cases)
    made = []
    for case in sorted(p for p in cases.iterdir() if p.is_dir()):
        src = case / "input.png"
        mask_path = case / "mask.png"
        if not src.is_file() or not mask_path.is_file():
            continue
        target = case / "_overlay.png"
        if target.exists() and not force:
            continue
        with Image.open(src) as im:
            rgb = np.asarray(im.convert("RGB"))
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L")) > 127
        if mask.shape != rgb.shape[:2]:
            print(f"  [!] {case.name}: mask 尺寸 {mask.shape} != "
                  f"图 {rgb.shape[:2]}")
            continue
        Image.fromarray(_overlay(rgb, mask)).save(target)
        made.append(target)
        print(f"  {case.name}: 掩膜 {int(mask.sum())} 像素")
    print(f"生成 {len(made)} 张叠加预览")
    return made


def check(cases):
    """校验用例库：格式不对的当场说清楚（bench 只会默默跳过）。"""
    cases = Path(cases)
    problems = []
    count = 0
    clean = 0
    for case in sorted(p for p in cases.iterdir() if p.is_dir()):
        if not (case / "input.png").is_file():
            continue
        count += 1
        with Image.open(case / "input.png") as im:
            shape = np.asarray(im).shape[:2]
        mask_path = case / "mask.png"
        if not mask_path.is_file():
            problems.append(f"{case.name}: 缺 mask.png（干净样本也要给全黑一张）")
        else:
            with Image.open(mask_path) as im:
                mask = np.asarray(im.convert("L"))
            if mask.shape != shape:
                problems.append(f"{case.name}: mask {mask.shape} 与图 "
                                f"{shape} 尺寸不一致")
            elif not (mask > 127).any():
                # 全黑是合法的：干净样本就该一张全黑掩膜，bench 拿它量假阳性
                clean += 1
        meta_path = case / "meta.json"
        if not meta_path.is_file():
            problems.append(f"{case.name}: 缺 meta.json")
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                problems.append(f"{case.name}: meta.json 不是合法 JSON（{exc}）")
                meta = {}
            if meta.get("kind") not in KINDS:
                problems.append(f"{case.name}: meta.kind 必须是 "
                                f"{'/'.join(KINDS)}（现在是 {meta.get('kind')!r}）")
        if (case / "clean.png").is_file():
            with Image.open(case / "clean.png") as im:
                clean_shape = np.asarray(im).shape[:2]
            if clean_shape != shape:
                problems.append(f"{case.name}: clean.png {clean_shape} 与图 "
                                f"{shape} 尺寸不一致")
    print(f"检查 {count} 个用例（其中 {clean} 例掩膜全黑，按干净样本"
          "计入假阳性探测）：", end="")
    if not problems:
        print("全部合规")
        return problems
    print(f"{len(problems)} 处问题")
    for one in problems:
        print(f"  [!] {one}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="真值样本库脚手架（导出初稿 / 校验 / 叠加预览）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("export", help="把真图目录导出成 bench 用例初稿")
    one.add_argument("sources", help="图片目录或单张图")
    one.add_argument("out", help="输出用例库目录")
    one.add_argument("--kind", choices=KINDS, help="强制类别（默认自动判）")
    one.add_argument("--limit", type=int, help="最多导出几张")
    two = sub.add_parser("overlay", help="按 mask.png 重新生成红色叠加预览")
    two.add_argument("cases")
    two.add_argument("--force", action="store_true", help="已存在也重做")
    three = sub.add_parser("check", help="校验用例库格式")
    three.add_argument("cases")
    args = parser.parse_args(argv)
    if args.cmd == "export":
        export(args.sources, args.out, kind=args.kind, limit=args.limit)
        return 0
    if args.cmd == "overlay":
        overlay(args.cases, force=args.force)
        return 0
    return 1 if check(args.cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
