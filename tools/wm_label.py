# -*- coding: utf-8 -*-
"""真值样本库脚手架：把一批真图变成 tools/wm_bench.py 能跑的用例。

去水印这条线此前的样本全是合成的（白纸文档 / 屏摄 / 草地照片），阈值调得
再漂亮也不知道在真图上如何。真实样本按 bench 的目录格式丢进来就能一起跑，
缺的只是“从零标注一批掩膜”的起步成本——一页一页用画图工具描水印笔画，
20 张就要一下午。

这里把起步成本压到“改初稿”：先跑一遍现成检测当草稿，人工只补漏、删多余
像素，然后在图上叠一层红色半透明看得见对没对上。

    python tools/wm_label.py export 真图目录 bench/real          # 生成用例初稿
    python tools/wm_label.py refine bench/real/某例 --boxes 10,20,300,80  # 框标注自动细化
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
    """检测初稿：文档走全局掩膜，照片走网格，平铺另给周期提示。

    返回 (掩膜, 周期或 None, 色系)。色系也给出来，audit 拿它回答“什么水印”，
    不必再把 detect_tint 跑一遍（12 MP 上一次几秒）。
    """
    if kind == "photo":
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for det in wm._grid_light_boxes(rgb):
            if det.mask is not None:
                mask |= det.mask
        return mask, None, ""
    tint = wm.detect_tint(rgb)
    if not tint:
        return np.zeros(rgb.shape[:2], dtype=bool), None, ""
    mask = wm._detect_mask(rgb, tint)
    if mask.any():
        luma, chroma = wm._luma_chroma(rgb)
        mask = wm._recall_mask(rgb, mask, tint, stats=(luma, chroma))
    # 平铺水印整页印满：检测多半只认出一部分，用“按周期折叠”的掩膜当地盘
    # 更完整（重叠部分交给人工删）。先只估周期（自相关，便宜）；估不出就不
    # 做整页重建——那是每张图一次全尺寸折叠，人工标注时没必要付这个钱。
    period = tpl.tiled_period(rgb)
    if period is None:
        return mask, None, tint
    _fixed, info = tpl.remove_tiled_watermark(rgb, period=period)
    if info is not None and "mask" in info:
        mask = mask | info["mask"]
    return mask, (period[0], period[1]), tint


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
        mask, period, tint = _draft_mask(rgb, found_kind)
        Image.fromarray((mask.astype(np.uint8)) * 255).save(case / "mask.png")
        Image.fromarray(_overlay(rgb, mask)).save(case / "_overlay.png")
        note = "初稿：检测掩膜，人工补漏/删多余"
        if period:
            note += f"；平铺周期 {period[1]}×{period[0]}"
        (case / "meta.json").write_text(json.dumps({
            "kind": found_kind, "tiled": bool(period), "tint": tint,
            "note": note, "source": str(path)}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        made.append(case)
        print(f"  [{index}/{len(images)}] {case.name}: {found_kind}"
              f"、掩膜 {int(mask.sum())} 像素")
    print(f"生成 {len(made)} 个用例初稿 -> {out_dir}")
    print("下一步：在画图工具里改 mask.png（白色=水印），"
          "再跑 overlay 看红色叠加确认")
    return made


def _numbers(item, count):
    """把一个标注解析成定长数字串："10,20,30,40" 或 [10, 20, 30, 40]。"""
    if isinstance(item, str):
        parts = [p for p in item.replace(" ", "").split(",") if p]
    else:
        parts = list(item)
    if len(parts) != count:
        raise SystemExit(f"坐标要 {count} 个数，收到 {item!r}")
    return [float(p) for p in parts]


def _marks(boxes, points, marks_json):
    """把命令行标注和 JSON 标注合并成像素坐标的 (框列表, 点列表)。

    JSON 就是 GUI「存标注框」导出的那种：{"boxes": [[x0,y0,x1,y1], ...],
    "points": [[x,y], ...]}，也可以是纯框列表。坐标都是原图像素，因为
    归一化比例离开图片尺寸就没有意义，图尺寸又只有读图时才知道——
    在导出端换算成像素，两边就不会对不上。
    """
    found_boxes = list(boxes or [])
    found_points = list(points or [])
    if marks_json:
        data = json.loads(Path(marks_json).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            found_boxes += list(data.get("boxes") or [])
            found_points += list(data.get("points") or [])
        elif isinstance(data, list):
            found_boxes += list(data)
        else:
            raise SystemExit(f"{marks_json} 不是标注文件")
    return ([_numbers(b, 4) for b in found_boxes],
            [_numbers(p, 2) for p in found_points])


def refine(cases, boxes=None, points=None, marks_json=None, replace=False,
           radius=80):
    """把人工的点/框标注细化成掩膜，合并进 mask.png。

    标注只要落到水印附近就行（拖一个框、或点一下笔画），检测负责把
    框内/点附近的笔画描出来：一页几十个水印也只用手点几下，不用再逐个
    描边。点会被扩成边长为 2*radius 的方框再交给同一个检测器。

    默认与已有掩膜取并集（保留先前人工修过的地方）；--replace 则从全黑
    重来——整页重标时用，免得旧草稿的错误一直传下去。
    """
    boxes, points = _marks(boxes, points, marks_json)
    if not boxes and not points:
        raise SystemExit("没有标注：给 --boxes / --points 或 --json")
    root = Path(cases)
    if (root / "input.png").is_file():
        dirs = [root]
    else:
        dirs = sorted(p for p in root.iterdir()
                      if (p / "input.png").is_file())
    if not dirs:
        raise SystemExit(f"{root} 里没有用例（需要 <case>/input.png）")
    touched = []
    for case in dirs:
        with Image.open(case / "input.png") as im:
            rgb = np.asarray(im.convert("RGB"))
        height, width = rgb.shape[:2]
        mask_path = case / "mask.png"
        mask = np.zeros((height, width), dtype=bool)
        if mask_path.is_file() and not replace:
            with Image.open(mask_path) as im:
                old = np.asarray(im.convert("L")) > 127
            if old.shape != (height, width):
                raise SystemExit(f"{case.name}: 旧 mask 尺寸不一致，先跑 check")
            mask = old
        before = int(mask.sum())

        def add_box(px_box):
            """像素框 -> 比例框（检测器吃 0~1），把检出掩膜并入。"""
            ratio = (px_box[0] / width, px_box[1] / height,
                     px_box[2] / width, px_box[3] / height)
            det = wm.detect_light_in_box(rgb, ratio)
            if det.mask is not None and det.mask.any():
                mask[:] |= det.mask
                return True
            return False

        hit = 0
        for one in boxes:
            hit += add_box(one)
        for x, y in points:
            hit += add_box((x - radius, y - radius, x + radius, y + radius))
        added = int(mask.sum()) - before
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        Image.fromarray(_overlay(rgb, mask)).save(case / "_overlay.png")
        touched.append(case)
        print(f"  {case.name}: 标注 {len(boxes) + len(points)} 处、"
              f"{hit} 处检出，掩膜 +{added} 像素（现 {int(mask.sum())}）")
    print(f"细化 {len(touched)} 个用例。下一步：overlay 看一眼，再 bench 跑分")
    return touched


def audit(sources, kind=None, out_json=None):
    """只检测、不改像素：批量报出哪些图真的值得处理。

    一键抠之前先看一眼：整批里混着干净截图和纹理丰富的照片时，逐张跑
    一遍处理会把干净图也重存一次（JPEG 再压、元数据丢失），而结论可能只是
    “没找到水印”。审计用和处理时同一套判据（照片走网格与 8% 整图护栏、
    文档屏摄走全局掩膜、平铺估周期），但不落盘、不改图，跑完给一张表。
    """
    images = _images(sources)
    rows = []
    for path in images:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        found_kind = _kind_of(rgb, kind)
        mask, period, tint = _draft_mask(rgb, found_kind)
        share = float(mask.mean())
        if not mask.any():
            verdict = "干净（未检出）"
        elif period:
            verdict = f"平铺水印（周期 {period[1]}×{period[0]}）"
        elif found_kind == "photo" and share > wm._PHOTO_MASK_MAX:
            # 与照片路径同一条护栏：九格并集占两成是纹理，不是水印
            verdict = f"疑似纹理（{share:.0%}），建议跳过"
        elif share > 0.30:
            verdict = f"掩膜过大（{share:.0%}），疑似底色误判"
        else:
            verdict = "有水印"
        rows.append({"file": str(path), "kind": found_kind,
                     "tint": tint or "-", "mask_px": int(mask.sum()),
                     "share": round(share, 4), "tiled": bool(period),
                     "verdict": verdict})
        print(f"  {path.name[:30]:<32}{found_kind:<7}{tint or '-':<3}"
              f"{share:>7.1%}  {verdict}")
    hits = sum(1 for r in rows if r["verdict"] == "有水印")
    print(f"\n审计 {len(rows)} 张：{hits} 张有水印、"
          f"{len(rows) - hits} 张跳过/干净")
    if out_json:
        target = Path(out_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"rows": rows}, ensure_ascii=False,
                                     indent=1), encoding="utf-8")
        print(f"报告已写入 {target}")
    return rows


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
    five = sub.add_parser("audit", help="只检测不改图：批量判断哪些图值得处理")
    five.add_argument("sources", help="图片目录或单张图")
    five.add_argument("--kind", choices=KINDS, help="强制类别（默认自动判）")
    five.add_argument("--json", dest="out_json", help="把审计结果写成 JSON")
    four = sub.add_parser("refine", help="点/框标注 → 自动细化成掩膜")
    four.add_argument("cases", help="用例目录（或整个用例库）")
    four.add_argument("--boxes", nargs="*", default=[],
                     help="像素框 x0,y0,x1,y1（可多给）")
    four.add_argument("--points", nargs="*", default=[],
                     help="像素点 x,y（可多给，按 --radius 扩成方框）")
    four.add_argument("--json", dest="marks_json",
                     help="标注文件（GUI 导出，或 {boxes, points}）")
    four.add_argument("--radius", type=int, default=80,
                     help="点标注扩成的方框半径（像素，默认 80）")
    four.add_argument("--replace", action="store_true",
                     help="丢掉旧掩膜，从全黑重来")
    args = parser.parse_args(argv)
    if args.cmd == "export":
        export(args.sources, args.out, kind=args.kind, limit=args.limit)
        return 0
    if args.cmd == "overlay":
        overlay(args.cases, force=args.force)
        return 0
    if args.cmd == "audit":
        audit(args.sources, kind=args.kind, out_json=args.out_json)
        return 0
    if args.cmd == "refine":
        refine(args.cases, boxes=args.boxes, points=args.points,
               marks_json=args.marks_json, replace=args.replace,
               radius=args.radius)
        return 0
    return 1 if check(args.cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
