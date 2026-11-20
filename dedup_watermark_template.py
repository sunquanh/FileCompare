# -*- coding: utf-8 -*-
"""跨图水印模板与平铺水印。

两件事共用一个思路：水印在同一批图里反复出现时，不必每张都从零猜。

1. 跨图模板（learn_template）：抽几张图各自检测，把反复出现在同一位置、同一
   形状的掩膜聚成一类；之后整批按模板反解，检测只做校验。同相机/同导出的
   一批图几乎都是这个样子（实测合成样本 6 张里模板覆盖 6 张，框 IoU > 0.95）。
   好处有三：召回稳（模板比单张投影可靠）、速度快（省掉每张的检测）、能解释
   （可以告诉用户“这批图有 1 个水印模板，覆盖 480/500 张”）。

   注意它的定位：单张质量与逐张主路径同档（实测模板反解 7.80→5.85，主路径
   5.03），没打算更准；模板掩膜就是“检测到的那部分”，检测漏掉的笔画它也不
   会凭空补回来。价值在整批一致与省时。

2. 平铺水印（tiled_period / remove_tiled_watermark）：整页重复的样图水印
   有一定的周期。掩膜能量图做自相关就能估出周期，把图按周期切瓦片、逐像素
   取中位就得到“水印残差模板”（背景在瓦片间变了、水印没变，中位把背景压掉），
   再按相位把模板减回去——比逐块补图自然，也不受“掩膜占比超七成就拒绝”的
   限制（那条门是为框选/网格路径设的）。

两个入口都只吃 numpy 数组，文件/批量封装留给调用方；不在这里碰 IO。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import dedup_watermark as wm

SIG = 32          # 模板签名分辨率（聚类用）
MIN_IOU = 0.55    # 归一化框相似度门槛
MIN_SIG_IOU = 0.45


@dataclass
class Template:
    """一张批图里反复出现的水印模板。"""

    box: tuple                     # 归一化框（x0, y0, x1, y1）
    mask: np.ndarray               # 框内像素尺寸的水印掩膜
    tint: str = ""
    share: float = 0.0             # 出现该模板的样本占比
    score: float = 0.0
    agreement: float = 0.0         # 学过模板的样本与模板的掩膜 IoU 中位
    images: list = field(default_factory=list, repr=False)


def _load(path):
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _resize_nearest(mask, height, width):
    rows = (np.arange(height) * (mask.shape[0] / height)).astype(np.int32)
    cols = (np.arange(width) * (mask.shape[1] / width)).astype(np.int32)
    return mask[np.clip(rows, 0, mask.shape[0] - 1)][
        :, np.clip(cols, 0, mask.shape[1] - 1)]


def _detections(rgb):
    """一张图上的水印候选：照片走网格，文档在全局掩膜上切格再收紧。

    文档不能把整页掩膜当一个候选：字形只占掩膜外接框的几个点，缩到 32×32
    签名后只剩十几个像素（实测），两边根本没法比。按 3×3 重叠网格切、每格
    收紧到自己的外接框，位置信息才留得住。
    """
    if wm.looks_like_photo(rgb):
        return list(wm._grid_light_boxes(rgb))
    mask = wm.detect_light_watermark(rgb)
    if not mask.any():
        return []
    tint = wm.detect_tint(rgb)
    height, width = mask.shape
    dets = []
    step_y, step_x = height / 3.0, width / 3.0
    for row in range(3):
        for col in range(3):
            y0 = max(0, int(row * step_y - step_y * 0.25))
            y1 = min(height, int((row + 1) * step_y + step_y * 0.25))
            x0 = max(0, int(col * step_x - step_x * 0.25))
            x1 = min(width, int((col + 1) * step_x + step_x * 0.25))
            sub = mask[y0:y1, x0:x1]
            if int(sub.sum()) < 200:
                continue
            rows = np.flatnonzero(sub.any(axis=1))
            cols = np.flatnonzero(sub.any(axis=0))
            box = ((x0 + cols[0]) / width, (y0 + rows[0]) / height,
                   (x0 + cols[-1] + 1) / width, (y0 + rows[-1] + 1) / height)
            dets.append(wm.LightDetection(box=box, tint=tint, mask=mask,
                                          score=1.0, source="doc"))
    return dets


def _crop_of(det):
    """检测框内的真实掩膜裁剪（不缩放）。"""
    y0, y1, x0, x1 = wm.box_pixels(det.box, det.mask.shape[1], det.mask.shape[0])
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    return np.ascontiguousarray(det.mask[y0:y1, x0:x1])


def _signature(det):
    crop = _crop_of(det)
    return None if crop is None else _resize_nearest(crop, SIG, SIG)


def _iou(a, b):
    return wm._box_iou(a, b)


def _sig_iou(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def learn_template(paths, limit: int = 8, min_share: float = 0.4):
    """从前若干张图里学一个水印模板；找不到稳定模板就返回 None。"""
    paths = list(paths)[:max(1, int(limit))]
    if not paths:
        return None
    groups = []
    for path in paths:
        rgb = _load(path)
        for det in _detections(rgb):
            sig = _signature(det)
            crop = _crop_of(det)
            if sig is None or crop is None:
                continue
            for group in groups:
                if (_iou(group["box"], det.box) >= MIN_IOU
                        and _sig_iou(group["sig"], sig) >= MIN_SIG_IOU):
                    group["items"].append((det, sig, crop, str(path)))
                    break
            else:
                groups.append({"box": det.box, "sig": sig,
                               "items": [(det, sig, crop, str(path))]})
    if not groups:
        return None
    best = max(groups, key=lambda g: len(g["items"]))
    share = len(best["items"]) / len(paths)
    if share < min_share:
        return None
    items = best["items"]
    boxes = np.array([d.box for d, _s, _c, _p in items], dtype=np.float32)
    box = tuple(float(v) for v in np.median(boxes, axis=0))
    # 模板掩膜用“中位大小”那张的真实掩膜（不拿 32×32 签名放大：那会变成
    # 一个个方块，反解时打不准笔画，实测误差只降 6%）。
    height, width = items[0][0].mask.shape
    y0, y1, x0, x1 = wm.box_pixels(box, width, height)
    median_px = int(np.median([int(crop.sum()) for _d, _s, crop, _p in items]))
    order = sorted(range(len(items)),
                   key=lambda i: abs(int(items[i][2].sum()) - median_px))
    mask = _resize_nearest(items[order[0]][2], max(4, y1 - y0),
                           max(4, x1 - x0)).astype(bool)
    tint_votes = {}
    for det, _sig, _crop, _path in items:
        tint_votes[det.tint] = tint_votes.get(det.tint, 0) + 1
    tint = max(tint_votes, key=tint_votes.get) if tint_votes else ""
    # 贴合度：这几种图各自的检测掩膜与最终模板（同框裁剪）的 IoU 中位。
    # 手里已经有这些掩膜，顺手量一次——它就是“这批图的水印是不是同一个”
    # 的证据，不必为了报告再跑一遍检测。
    agree = []
    for det, _sig, _crop, _path in items:
        crop = det.mask[y0:y1, x0:x1].astype(bool)
        union = int((crop | mask).sum())
        if union:
            agree.append(int((crop & mask).sum()) / union)
    return Template(box=box, mask=mask, tint=tint,
                    share=round(share, 3),
                    score=float(np.median([d.score for d, _s, _c, _p in items])),
                    agreement=round(float(np.median(agree)), 3) if agree else 0.0,
                    images=[p for _d, _s, _c, p in items])


def template_mask(template, shape):
    """把模板铺回整图尺寸的布尔掩膜。"""
    height, width = shape[:2]
    y0, y1, x0, x1 = wm.box_pixels(template.box, width, height)
    full = np.zeros((height, width), dtype=bool)
    mh, mw = y1 - y0, x1 - x0
    if mh < 4 or mw < 4:
        return full
    full[y0:y1, x0:x1] = _resize_nearest(template.mask, mh, mw)
    return full


def apply_template(rgb, template, strength: float = 1.0):
    """按模板反解一张图。返回 (结果, 掩膜)。"""
    mask = template_mask(template, rgb.shape)
    if not mask.any():
        return np.array(rgb, dtype=np.uint8, copy=True), mask
    # 参考色要在这一张上现估（纸色/曝光逐张不一样），和文档主路径同口径：
    # 直接用写死色表会让 α 系统性偏小（实测误差只降 25%）。
    ref = wm._choose_tint_ref(rgb, mask, template.tint)
    fixed = wm.undo_light_watermark(rgb, tint=template.tint or None, ref=ref,
                                    mask=mask, strength=strength)
    return fixed, mask


def remove_with_template(src, template, dst=None, quality: int = 95,
                         strength: float = 1.0, strip_gps: bool = False):
    """文件封装：读图 → 按模板反解 → 写回（沿用主模块的元数据/alpha 处理）。"""
    from pathlib import Path

    src = Path(src)
    arr, alpha, meta = wm._open_for_edit(src)
    fixed, mask = apply_template(arr, template, strength=strength)
    target = Path(dst) if dst else wm.output_path(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    wm._save_result(Image.fromarray(fixed, "RGB"), target, quality, meta,
                    alpha=alpha, strip_gps=strip_gps)
    return target, mask


def _batch_counts(results):
    counts = {"clean": 0, "partial": 0, "unchanged": 0, "rejected": 0}
    for _path, status, _detail in results:
        if status in counts:
            counts[status] += 1
    return counts


def _batch_target(path, out_dir, subdir):
    return wm.output_path(path, wm.target_dir(path, out_dir, subdir),
                          unique=False)


def batch_remove_with_template(paths, out_dir=None, quality: int = 95,
                               learn_limit: int = 8, strip_gps: bool = False,
                               cancel=None, on_progress=None,
                               subdir: bool = False,
                               report=None) -> dict:
    """整批按同一个水印模板反解：先拿前几张学一次，再逐张套用。

    学不出模板（图上没水印、或位置/形状不一致）就退回逐张自动检测，
    批处理仍然跑得完——只是没有省掉检测那一步。返回格式与
    dedup_watermark.batch_remove_light 一致，另加 template_share /
    template_box / learned。
    """
    paths = list(paths)
    template = learn_template(paths[:max(1, learn_limit)], limit=learn_limit)
    done = skipped = 0
    failed = []
    results = []
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if cancel is not None and cancel.is_set():
            break
        try:
            target = _batch_target(path, out_dir, subdir)
            if target.exists():
                skipped += 1
            else:
                if template is not None:
                    _, mask = remove_with_template(
                        path, template, dst=target, quality=quality,
                        strip_gps=strip_gps)
                    px = int(mask.sum())
                    status, detail = (("clean", f"按模板反解 {px} 像素")
                                      if px else ("unchanged", "模板未命中"))
                else:
                    _, light = wm.remove_light_file(
                        path, target, quality, strip_gps=strip_gps)
                    status, detail = light.status, light.detail
                results.append((str(path), status, detail))
                done += 1
        except Exception as exc:
            failed.append((str(path), str(exc)))
        if on_progress is not None:
            on_progress(index, total, str(path))
    result = {"done": done, "failed": failed, "skipped": skipped,
              "total": total, "results": results,
              "learned": template is not None,
              "template_share": float(template.share) if template else 0.0,
              "template_box": tuple(template.box) if template else None,
              "template_agreement": float(template.agreement) if template else 0.0,
              **_batch_counts(results)}
    if report:
        result["report"] = str(wm.write_batch_report(result, report))
    return result


def batch_remove_tiled(paths, out_dir=None, quality: int = 95,
                       strip_gps: bool = False, cancel=None,
                       on_progress=None, subdir: bool = False,
                       report=None) -> dict:
    """整批按周期重建平铺水印（逐张估周期，测不出就该张原样）。

    和 batch_remove_light 的区别：这里不跑检测，只看“有没有周期性重复”，
    所以适合整页印满样图水印、检测被判成“底色误判”的图。返回格式与
    batch_remove_light 一致。
    """
    from PIL import Image

    paths = list(paths)
    done = skipped = 0
    failed = []
    results = []
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if cancel is not None and cancel.is_set():
            break
        try:
            target = _batch_target(path, out_dir, subdir)
            if target.exists():
                skipped += 1
            else:
                arr, alpha, meta = wm._open_for_edit(path)
                fixed, info = remove_tiled_watermark(arr)
                if info is None:
                    results.append((str(path), "unchanged", "未测出平铺周期"))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    wm._save_result(Image.fromarray(fixed, "RGB"), target,
                                    quality, meta, alpha=alpha,
                                    strip_gps=strip_gps)
                    rows, cols = info["period"]
                    results.append((str(path), "clean",
                                    f"平铺水印已重建（{cols}×{rows}，"
                                    f"{info['tiles']} 块）"))
                done += 1
        except Exception as exc:
            failed.append((str(path), str(exc)))
        if on_progress is not None:
            on_progress(index, total, str(path))
    result = {"done": done, "failed": failed, "skipped": skipped,
              "total": total, "results": results, **_batch_counts(results)}
    if report:
        result["report"] = str(wm.write_batch_report(result, report))
    return result


# ---------------------------------------------------------------- 平铺水印
def _energy(rgb):
    """水印能量图：各色系“相对局部纸面的偏色”，取最大。"""
    luma, chroma = wm._luma_chroma(rgb)
    pale = (luma > 140) & (chroma < 45) & (rgb[..., 1] >= 120)
    arr = rgb.astype(np.float32)
    energy = np.zeros(luma.shape, dtype=np.float32)
    for name in ("黄", "蓝", "粉", "绿"):
        score = wm._tint_score(arr, name)
        resid = score - wm._local_trend(score, pale)
        energy = np.maximum(energy, resid)
    return np.clip(energy, 0, None), luma


def tiled_period(rgb, min_period: int = 24, max_frac: float = 0.34):
    """估平铺周期：返回 (周期行, 周期列, 相关值)，无明显周期返回 None。"""
    energy, _luma = _energy(rgb)
    if float(energy.max()) <= 0.5:
        return None
    centred = energy - float(energy.mean())
    spec = np.fft.rfft2(centred)
    auto = np.fft.irfft2(spec * np.conj(spec), s=centred.shape)
    if auto[0, 0] <= 0:
        return None
    auto = auto / float(auto[0, 0])
    height, width = centred.shape
    limit_y = max(min_period + 1, int(height * max_frac))
    limit_x = max(min_period + 1, int(width * max_frac))
    region = np.array(auto[:limit_y, :limit_x], dtype=np.float64)
    # 两轴的滞后都得够大：只要有一个轴接近 0，自相关就退化成“行/列自己的
    # 相似度”，峰全是假的（实测会在 (0, 300) 出一个 0.83 的假峰）。
    region[:min_period, :] = 0.0
    region[:, :min_period] = 0.0
    idx = np.unravel_index(int(np.argmax(region)), region.shape)
    # FFT 的峰落在整数格上，真周期常卡在两格之间；在 ±2 邻域里按自相关再挑
    # 一次，免得周期差 1 像素时瓦片错拍（错拍的表现是模板被抹平、解完仍有影）
    py, px = int(idx[0]), int(idx[1])
    peak = float(region[py, px])
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y, x = py + dy, px + dx
            if y < min_period or x < min_period or y >= limit_y or x >= limit_x:
                continue
            if float(region[y, x]) > peak:
                py, px, peak = y, x, float(region[y, x])
    if peak < 0.12:
        return None
    return int(py), int(px), round(peak, 3)


def remove_tiled_watermark(rgb, period=None, min_strength: float = 3.0):
    """按周期把平铺水印减掉；返回 (结果, 信息字典或 None)。"""
    got = period or tiled_period(rgb)
    if got is None:
        return np.array(rgb, dtype=np.uint8, copy=True), None
    rows, cols = int(got[0]), int(got[1])
    height, width = rgb.shape[:2]
    if rows < 8 or cols < 8:
        return np.array(rgb, dtype=np.uint8, copy=True), None
    arr = rgb.astype(np.float32)
    # 每个通道先减掉低频背景，剩下的就是“相对于周围纸面的凸起”
    radius = max(8, min(rows, cols) // 2)
    resid = np.empty_like(arr)
    for ch in range(3):
        resid[..., ch] = arr[..., ch] - wm._blur_f(arr[..., ch], radius)
    template, tiles = _fold_template(resid, rows, cols)
    if template is None:            # 瓦片不够时 _fold_template 已经返回 None
        return np.array(rgb, dtype=np.uint8, copy=True), None
    strength = np.abs(template).max(axis=2)
    mad = float(np.median(np.abs(template - np.median(template))))
    threshold = max(float(min_strength), 4.0 * 1.4826 * mad)
    mask = strength > threshold
    if not mask.any():
        return np.array(rgb, dtype=np.uint8, copy=True), None
    _energy_map, luma = _energy(rgb)
    reps = (-(-height // rows), -(-width // cols))
    # 只动“浅色水印”的地盘：纸面够亮、且确实有可减的残差
    touched = np.tile(mask, reps)[:height, :width] & (luma > 140)
    if not touched.any():
        return np.array(rgb, dtype=np.uint8, copy=True), None
    # 模板只用来“圈地”，真正抹掉还是交给成熟的 α 反解：直接减中位残差
    # 会把背景差异一起减掉（实测 err 2.63，比现成反解的 1.45 还差）。
    tint = _tint_of(template, mask)
    fixed = wm.undo_light_watermark(rgb, tint=tint or None, mask=touched)
    info = {"period": (rows, cols),
            "correlation": got[2] if len(got) > 2 else None,
            "tiles": int(tiles), "mask_px": int(touched.sum()),
            "tint": tint,
            # 掩膜也带出去：真值库工具拿它当人工标注的初稿（见 tools/wm_label.py）
            "mask": touched}
    return fixed, info


def _fold_template(resid, rows, cols):
    """按周期折叠残差：同一相位上的样本逐像素取中位，就是水印模板。

    早先按整块切瓦片，切不下的余数（图幅不是周期整数倍）整条丢掉——水印
    印到右下边缘时就永远擦不到；相位也钉死在 (0,0) 上。改成按“余数类”
    折叠后，每个像素都归到一个相位，余数也参与中位（只是样本少一点）。
    背景在瓦片之间不一样、水印在每个瓦片的同一处，中位把背景压掉。

    返回 (模板 rows×cols×3, 参与折叠的瓦片数)；瓦片太少返回 (None, 0)。
    """
    height, width = resid.shape[:2]
    if rows < 2 or cols < 2:
        return None, 0
    ny, nx = -(-height // rows), -(-width // cols)
    tiles = int(ny * nx)
    if tiles < 4:
        return None, 0
    pad_h, pad_w = ny * rows - height, nx * cols - width
    padded = resid
    if pad_h or pad_w:
        padded = np.pad(resid, ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="symmetric")
    blocks = padded.reshape(ny, rows, nx, cols, 3)
    blocks = blocks.transpose(1, 3, 0, 2, 4).reshape(rows, cols, tiles, 3)
    return np.median(blocks, axis=2), tiles


def _tint_of(template, mask):
    """看模板残差的平均颜色投一个色系。"""
    if not mask.any():
        return ""
    mean = template[mask].mean(axis=0).astype(np.float32)
    best, best_val = "", 0.0
    for name, ref in wm.TINT_REFS.items():
        val = float(np.dot(mean, np.asarray(ref, dtype=np.float32) - 240.0))
        if val > best_val:
            best, best_val = name, val
    return best


__all__ = [
    "Template",
    "learn_template",
    "template_mask",
    "apply_template",
    "remove_with_template",
    "batch_remove_with_template",
    "tiled_period",
    "remove_tiled_watermark",
    "batch_remove_tiled",
]
