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
    return Template(box=box, mask=mask, tint=tint,
                    share=round(share, 3),
                    score=float(np.median([d.score for d, _s, _c, _p in items])),
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
                         strength: float = 1.0):
    """文件封装：读图 → 按模板反解 → 写回（沿用主模块的元数据/alpha 处理）。"""
    from pathlib import Path

    src = Path(src)
    arr, alpha, meta = wm._open_for_edit(src)
    fixed, mask = apply_template(arr, template, strength=strength)
    target = Path(dst) if dst else wm.output_path(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    wm._save_result(Image.fromarray(fixed, "RGB"), target, quality, meta,
                    alpha=alpha)
    return target, mask


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
    peak = float(region[idx])
    if peak < 0.12:
        return None
    return int(idx[0]), int(idx[1]), round(peak, 3)


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
    tiles = []
    positions = []
    for y in range(0, height - rows + 1, rows):
        for x in range(0, width - cols + 1, cols):
            tiles.append(resid[y:y + rows, x:x + cols])
            positions.append((y, x))
    if len(tiles) < 4:
        return np.array(rgb, dtype=np.uint8, copy=True), None
    stack = np.stack(tiles)
    # 背景在瓦片之间不一样、水印在每个瓦片的同一处，逐像素中位把背景压掉
    template = np.median(stack, axis=0)
    strength = np.abs(template).max(axis=2)
    mad = float(np.median(np.abs(template - np.median(template))))
    threshold = max(float(min_strength), 4.0 * 1.4826 * mad)
    mask = strength > threshold
    if not mask.any():
        return np.array(rgb, dtype=np.uint8, copy=True), None
    _energy_map, luma = _energy(rgb)
    touched = np.zeros((height, width), dtype=bool)
    for (y, x) in positions:
        # 只动“浅色水印”的地盘：纸面够亮、且确实有可减的残差
        ok = mask & (luma[y:y + rows, x:x + cols] > 140)
        touched[y:y + rows, x:x + cols] |= ok
    if not touched.any():
        return np.array(rgb, dtype=np.uint8, copy=True), None
    # 模板只用来“圈地”，真正抹掉还是交给成熟的 α 反解：直接减中位残差
    # 会把背景差异一起减掉（实测 err 2.63，比现成反解的 1.45 还差）。
    tint = _tint_of(template, mask)
    fixed = wm.undo_light_watermark(rgb, tint=tint or None, mask=touched)
    info = {"period": (rows, cols),
            "correlation": got[2] if len(got) > 2 else None,
            "tiles": len(tiles), "mask_px": int(touched.sum()),
            "tint": tint}
    return fixed, info


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
    "tiled_period",
    "remove_tiled_watermark",
]
