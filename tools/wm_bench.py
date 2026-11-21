# -*- coding: utf-8 -*-
"""去水印评测基准：一图一真值，一次跑出一张报表。

为什么要有它：这条线此前所有阈值变更都靠临时脚本手测（薄笔画合成图只检出
329 像素、JPEG 蓝天格子 205 像素、屏摄图把黄投成粉，全是现算的），没有可
比较的基线，改完只能说“看起来更好”。这个脚本把“更好”变成几个数：

    err_ratio  框内误差还剩多少（0 = 抹净，1 = 一个像素没动）
    hit        水印掩膜里被真正处理掉的像素占比
    damage     掩膜外（外扩 8 px 之外）被改动的像素占比，越接近 0 越好
    honest     status 说 unchanged 时，输出必须与输入逐像素相同
    fp         干净样本上被改动的像素占比，必须是 0
    sec / MB   单张耗时与峰值内存

用例格式（一个目录一个用例）：

    <case>/input.png    带水印的输入
    <case>/mask.png     水印真值掩膜（255 = 水印像素）
    <case>/clean.png    干净参考（可选；没有就只算 hit/damage/fp）
    <case>/meta.json    {"kind": "doc|photo|screen", "tiled": false}

用法：

    python tools/wm_bench.py --make-fixtures bench_cases   # 生成内置样本
    python tools/wm_bench.py bench_cases                   # 跑并打印报表
    python tools/wm_bench.py bench_cases --save bench/last.json
    python tools/wm_bench.py bench_cases --baseline bench/last.json
        # 有基线时逐项比：err_ratio 变差 >10%、damage 变差、fp 非零、
        # hit 掉 >0.02 任一出现就以退出码 1 结束，供提交前把关

内置样本是合成的（白纸文档 / 屏摄 / 照片三类各几张，含干净样本用于假阳性
探测），它们是“可复现的尺子”，不是真实场景的代表；真实样本按上面的目录格式
丢进来即可一起跑。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

import dedup_watermark as wm  # noqa: E402

KINDS = ("doc", "photo", "screen")


# ---------------------------------------------------------------- 样本生成
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # 老版本 Pillow 没有 size 参数
        return ImageFont.load_default()


def _paper(width, height, seed, base=252):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    tone = base - 4.0 * (x / max(width - 1, 1)) - 2.0 * (y / max(height - 1, 1))
    arr = np.repeat(tone[..., None], 3, axis=2)
    arr += rng.normal(0, 0.8, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _doc(width=900, height=1200, seed=3, lines=18):
    """文档截图：白纸 + 表格线 + 深色正文。"""
    arr = _paper(width, height, seed)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for y in range(90, height - 60, 46):
        d.line([(40, y), (width - 40, y)], fill=(226, 226, 226))
    for x in range(40, width - 40, 150):
        d.line([(x, 90), (x, height - 60)], fill=(232, 232, 232))
    f = _font(17)
    for i in range(lines):
        d.text((50, 100 + i * 46), f"SELECT order_id, amount FROM t{i} WHERE x = {i}",
               fill=(38, 38, 42), font=f)
    return np.asarray(img)


def _photo(width=800, height=600, seed=5, palette="grass"):
    """照片底：渐变 + 噪点（模拟草地/天空这类有纹理的背景）。"""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    t = (x / max(width - 1, 1) + y / max(height - 1, 1)) / 2
    mixes = {"grass": ((70, 96, 52), (128, 154, 86)),
             "sky": ((168, 202, 238), (206, 228, 250)),
             "wall": ((222, 200, 176), (238, 224, 208))}
    a, b = mixes[palette]
    arr = np.stack([a[c] + (b[c] - a[c]) * t for c in range(3)], -1)
    arr = arr + rng.normal(0, 6, arr.shape)
    if palette == "grass":       # 一点草叶纹理，别让背景太光滑
        arr += (8 * np.sin(x / 3.0) * np.sin(y / 5.0))[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def _stamp(img_arr, text, alpha, color, positions, size=64):
    """在图上叠半透明水印，返回 (带水印图, 水印掩膜)。"""
    height, width = img_arr.shape[:2]
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = _font(size)
    for x, y in positions:
        d.text((x, y), text, fill=(color[0], color[1], color[2],
                                  int(round(alpha * 255))), font=f)
    base = Image.fromarray(img_arr).convert("RGBA")
    out = Image.alpha_composite(base, layer)
    mask = np.asarray(layer)[..., 3]
    return np.asarray(out.convert("RGB")), mask > 0


def _as_jpeg(arr, quality=55):
    """存一次 JPEG 再读回来：模拟“转发过一次”的图（8×8 块噪声 + 色度下采样）。"""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def _tiled_positions(width, height, step_x, step_y, phase=(30, 26)):
    xs = list(range(phase[0], width, step_x))
    ys = list(range(phase[1], height, step_y))
    return [(x, y) for y in ys for x in xs]


def _screen_transform(arr, seed):
    """屏摄质感：暗角 + 摩尔纹 + 轻微模糊 + 传感器噪声。"""
    rng = np.random.default_rng(seed)
    height, width = arr.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    vignette = 1.0 - 0.10 * np.clip(r, 0, 1.4) ** 2
    moire = 5.0 * np.sin(2 * math.pi * (x * 0.22 + y * 0.08))
    out = arr.astype(np.float32) * vignette[..., None] + moire[..., None] * 0.4
    out += rng.normal(0, 2.5, out.shape)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(img)


def make_fixtures(out_dir):
    """生成内置样本（确定性，可复现）。"""
    out_dir = Path(out_dir)
    cases = []

    def save(name, arr, mask, clean, kind, tiled=False, note=""):
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(d / "input.png")
        Image.fromarray((mask.astype(np.uint8)) * 255).save(d / "mask.png")
        if clean is not None:
            Image.fromarray(clean).save(d / "clean.png")
        (d / "meta.json").write_text(
            json.dumps({"kind": kind, "tiled": tiled, "note": note},
                       ensure_ascii=False), encoding="utf-8")
        cases.append(name)

    # 1. 文档 + 淡黄水印
    clean = _doc(seed=3)
    arr, mask = _stamp(clean, "2023/12/24", 0.30, (252, 248, 196),
                       [(120, 300), (420, 700), (200, 980)], size=58)
    save("doc_yellow", arr, mask, clean, "doc")

    # 2. 文档 + 淡灰水印（空白区）
    clean = _doc(seed=7, lines=6)
    arr, mask = _stamp(clean, "CONFIDENTIAL", 0.25, (228, 228, 228),
                       [(60, 900), (320, 1000)], size=72)
    save("doc_gray", arr, mask, clean, "doc")

    # 3. 文档 + 平铺水印（现在会被 >70% 拒掉那一类）
    clean = _doc(seed=11, lines=10)
    arr, mask = _stamp(clean, "SAMPLE", 0.22, (252, 248, 196),
                       _tiled_positions(900, 1200, 300, 260), size=52)
    save("doc_tiled", arr, mask, clean, "doc", tiled=True,
         note="period=300x260")

    # 4. 屏摄：文档 + 摩尔纹 + 淡黄水印
    # 参考图必须是“同一次屏摄、但没印水印”的那张。早先写成未渲染的原始页
    # 面，光渲染本身的差异就有 8 级，任何完美去水印都不可能降到 1 以下；
    # 而且水印先印在屏摄图上、再被渲染第二遍（seed 还不一样），等于两次
    # 拍摄叠加。现在只在页面上印一次、渲染一次，参考用同一 seed 的渲染。
    page = _doc(seed=13, lines=12)
    clean = _screen_transform(page, 21)
    arr, mask = _stamp(page, "DY2024", 0.30, (252, 248, 196),
                       [(150, 420), (470, 820)], size=64)
    arr = _screen_transform(arr, 21)
    save("screen_yellow", arr, mask, clean, "screen")

    # 5. 照片：草地 + 淡黄水印（旧用例一直失败的那类）
    clean = _photo(seed=5, palette="grass")
    arr, mask = _stamp(clean, "IMG2023", 0.35, (252, 248, 196),
                       [(180, 220), (360, 390)], size=72)
    save("photo_yellow", arr, mask, clean, "photo")

    # 6. 照片：蓝天 + 淡黄水印
    clean = _photo(seed=6, palette="sky")
    arr, mask = _stamp(clean, "2023.12", 0.35, (252, 248, 196),
                       [(220, 180), (400, 350)], size=70)
    save("photo_blue_sky", arr, mask, clean, "photo")

    # 7. 文档 + 淡黄水印，但被 JPEG 压过（转发一次的图基本都是这样）：
    # 8×8 块噪声会把“相对纸面的偏色”抬成网格，检测阈值跟着顶高
    clean = _doc(seed=23, lines=14)
    arr, mask = _stamp(clean, "2023/12/24", 0.30, (252, 248, 196),
                       [(140, 380), (400, 760)], size=58)
    save("doc_jpeg", _as_jpeg(arr), mask, _as_jpeg(clean), "doc",
         note="JPEG q=55 二次压缩")

    # 8~10. 干净样本（假阳性探测：改动必须是 0）
    clean = _doc(seed=17)
    save("clean_doc", clean, np.zeros(clean.shape[:2], bool), clean, "doc")
    clean = _photo(seed=8, palette="wall")
    save("clean_photo", clean, np.zeros(clean.shape[:2], bool), clean, "photo")
    clean = _screen_transform(_doc(seed=19, lines=8), 23)
    save("clean_screen", clean, np.zeros(clean.shape[:2], bool), clean, "screen")

    print(f"生成 {len(cases)} 个用例 -> {out_dir}")
    for name in cases:
        print(f"  {name}")
    return cases


# ---------------------------------------------------------------- 指标
def _load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _load_mask(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


def _peak_mb():
    try:
        import os
        import psutil  # 可能没装

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def evaluate(case_dir, sens=0.0):
    case_dir = Path(case_dir)
    meta = {}
    if (case_dir / "meta.json").is_file():
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    rgb = _load(case_dir / "input.png")
    mask = (_load_mask(case_dir / "mask.png")
            if (case_dir / "mask.png").is_file()
            else np.zeros(rgb.shape[:2], bool))
    clean = None
    if (case_dir / "clean.png").is_file():
        clean = _load(case_dir / "clean.png")

    before_mb = _peak_mb()
    started = time.perf_counter()
    try:
        out, status = wm.clean_light_watermark(rgb, sens=sens)
    except TypeError:      # 旧版本还没有灵敏度旋钮，用默认行为量
        out, status = wm.clean_light_watermark(rgb)
    seconds = time.perf_counter() - started
    after_mb = _peak_mb()

    delta = np.abs(out.astype(np.int16) - rgb.astype(np.int16)).max(axis=2)
    changed = delta > 2
    grew = wm._grow_mask(mask, 8) if mask.any() else mask
    inside = float(changed[mask].mean()) if mask.any() else 0.0
    # 误伤只看“改得更远离干净图”的像素：屏摄图的座标偏移/陷波本来就要动
    # 全图（那是修复本身），拿“改了多少”当误伤会把它也算进去。
    if clean is not None:
        to_clean_before = np.abs(rgb.astype(np.int16)
                                 - clean.astype(np.int16)).max(axis=2)
        to_clean_after = np.abs(out.astype(np.int16)
                                - clean.astype(np.int16)).max(axis=2)
        hurt = changed & (to_clean_after > to_clean_before + 4)
    else:
        hurt = changed
    outside = float(hurt[~grew].mean()) if (~grew).any() else 0.0
    err_ratio = None
    if clean is not None and mask.any():
        before = float(np.abs(rgb[mask].astype(np.int16)
                              - clean[mask].astype(np.int16)).mean())
        after = float(np.abs(out[mask].astype(np.int16)
                             - clean[mask].astype(np.int16)).mean())
        err_ratio = after / before if before > 1e-6 else None
    honest = True
    if status.status == "unchanged":
        honest = bool(np.array_equal(out, rgb))
    return {
        "case": case_dir.name,
        "kind": meta.get("kind", "?"),
        "tiled": bool(meta.get("tiled", False)),
        "status": status.status,
        "detail": status.detail,
        "err_ratio": None if err_ratio is None else round(err_ratio, 4),
        "hit": round(inside, 4),
        "damage": round(outside, 6),
        "fp": round(float(changed.mean()), 6) if not mask.any() else None,
        "honest": honest,
        "sec": round(seconds, 2),
        "mb": None if after_mb is None else round(after_mb - (before_mb or 0), 1),
    }


def check_truth_manifest(cases_dir):
    """真实样本目录若带 manifest.json，先核对本地素材有没有被动过。

    真值被改过时 hit / damage 与旧基线不可比，所以只提示、不拦下评测：
    合成样本目录本来就没有清单，缺清单不该让评测跑不起来。
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from wm_truth import manifest_drift
    except Exception:
        return []
    drift = manifest_drift(cases_dir)
    if drift:
        print("真值素材与清单不一致，本次 hit / damage 与旧基线不可比：")
        for msg in drift:
            print(f"  [!] {msg}")
        print("  确认是有意改动后运行 python tools/wm_truth.py --update 重写清单")
    return drift


def run(cases_dir, sens=0.0):
    cases_dir = Path(cases_dir)
    rows = []
    for case in sorted(p for p in cases_dir.iterdir() if p.is_dir()):
        if not (case / "input.png").is_file():
            continue
        rows.append(evaluate(case, sens=sens))
    return rows


def summarize(rows):
    out = {"n": len(rows), "fp_total": 0.0, "damage_max": 0.0,
           "hit_mean": 0.0, "err_mean": 0.0, "err_n": 0, "dishonest": 0,
           "sec_sum": 0.0}
    for r in rows:
        if r["fp"]:
            out["fp_total"] += r["fp"]
        out["damage_max"] = max(out["damage_max"], r["damage"])
        out["hit_mean"] += r["hit"]
        if r["err_ratio"] is not None:
            out["err_mean"] += r["err_ratio"]
            out["err_n"] += 1
        out["sec_sum"] += r["sec"]
        if not r["honest"]:
            out["dishonest"] += 1
    if out["err_n"]:
        out["err_mean"] = round(out["err_mean"] / out["err_n"], 4)
    out["hit_mean"] = round(out["hit_mean"] / max(out["n"], 1), 4)
    out["damage_max"] = round(out["damage_max"], 6)
    out["fp_total"] = round(out["fp_total"], 6)
    out["sec_sum"] = round(out["sec_sum"], 1)
    return out


def print_table(rows, summary):
    print(f"{'用例':<18}{'类':<8}{'状态':<10}{'err':>7}{'hit':>7}"
          f"{'damage':>9}{'fp':>9}{'秒':>6}  detail")
    for r in rows:
        err = "-" if r["err_ratio"] is None else f"{r['err_ratio']:.2f}"
        fp = "-" if r["fp"] is None else f"{r['fp']:.4f}"
        flag = "" if r["honest"] else "  <-- 状态与像素不符"
        print(f"{r['case']:<18}{r['kind']:<8}{r['status']:<10}{err:>7}"
              f"{r['hit']:>7.2f}{r['damage']:>9.4f}{fp:>9}{r['sec']:>6.1f}"
              f"  {r['detail'][:34]}{flag}")
    print(f"\n合计 {summary['n']} 例：err 均值 {summary['err_mean']}"
          f"（{summary['err_n']} 例有真值）· hit 均值 {summary['hit_mean']}"
          f" · 最大误伤 {summary['damage_max']:.4f}"
          f" · 干净样本假阳性 {summary['fp_total']:.6f}"
          f" · 状态不实 {summary['dishonest']} · 总耗时 {summary['sec_sum']}s")


def compare(rows, summary, baseline_path):
    base = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    base_rows = {r["case"]: r for r in base.get("cases", [])}
    problems = []
    for r in rows:
        old = base_rows.get(r["case"])
        if not old:
            continue
        if (r["err_ratio"] is not None and old.get("err_ratio") is not None
                and r["err_ratio"] > old["err_ratio"] * 1.10 + 0.02):
            problems.append(f"{r['case']}: err {old['err_ratio']:.2f}"
                            f" -> {r['err_ratio']:.2f}")
        if r["hit"] < old["hit"] - 0.02:
            problems.append(f"{r['case']}: hit {old['hit']:.2f} -> {r['hit']:.2f}")
        if r["damage"] > old["damage"] * 1.5 + 0.002:
            problems.append(f"{r['case']}: damage {old['damage']:.4f}"
                            f" -> {r['damage']:.4f}")
        if old.get("fp") is not None and r.get("fp") is not None:
            if r["fp"] > 0.0 and r["fp"] > old["fp"] + 0.001:
                problems.append(f"{r['case']}: 干净样本出现假阳性 {r['fp']:.4f}")
        if r["honest"] is False:
            problems.append(f"{r['case']}: 报 unchanged 却改了像素")
    if problems:
        print("\n相对基线变差：")
        for p in problems:
            print(f"  [!] {p}")
    else:
        print("\n相对基线：无回归")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description="去水印评测基准")
    parser.add_argument("cases", nargs="?", help="用例目录")
    parser.add_argument("--make-fixtures", metavar="DIR",
                        help="生成内置样本到该目录")
    parser.add_argument("--sens", type=float, default=0.0,
                        help="检出灵敏度 0~1（越大越激进）")
    parser.add_argument("--save", metavar="JSON", help="把这次结果存成基线")
    parser.add_argument("--baseline", metavar="JSON",
                        help="与基线比较，出现回归就以退出码 1 结束")
    args = parser.parse_args(argv)

    if args.make_fixtures:
        make_fixtures(args.make_fixtures)
        return 0
    if not args.cases:
        parser.print_help()
        return 2
    check_truth_manifest(args.cases)
    rows = run(args.cases, sens=args.sens)
    if not rows:
        print(f"{args.cases} 里没有找到用例（需要 <case>/input.png）")
        return 2
    summary = summarize(rows)
    print_table(rows, summary)
    problems = []
    if args.baseline:
        problems = compare(rows, summary, args.baseline)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(
            {"summary": summary, "sens": args.sens, "cases": rows},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {args.save}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
