# -*- coding: utf-8 -*-
"""照片去水印：把用户框出的区域用四周内容补上（本地、不联网、不覆盖原图）。

设计取舍：
  * 不引入新依赖 —— 只用 numpy + Pillow，和项目其余部分保持一致；
  * 恢复方式用「按最近的边反射回填」：每个待补像素取它最近那条框边的
    镜像内容，四个方向在接缝处都是连续的，所以角落 logo、底部条、相机
    时间戳这类边缘水印补完基本看不到方形痕迹，纹理（草地、墙砖）也接得上；
  * 反射采不到的地方（框贴住图片边、对面也没空间）退回由外向内扩散，
    只会给出平滑过渡，细节会糊——这是无模型方案的硬上限，界面上要提示。

只处理使用者自己有版权的照片。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # 和项目其余部分一样：HEIC 装了才支持
    from dedup_models import enable_heif

    enable_heif()
except Exception:  # pragma: no cover
    pass

# 位置预设：归一化坐标 (x0, y0, x1, y1)，都按画面宽高取比例
PRESETS = {
    "右下角": (0.70, 0.86, 0.985, 0.985),
    "左下角": (0.015, 0.86, 0.30, 0.985),
    "右上角": (0.70, 0.015, 0.985, 0.14),
    "左上角": (0.015, 0.015, 0.30, 0.14),
    "底部整条": (0.0, 0.90, 1.0, 1.0),
    "顶部整条": (0.0, 0.0, 1.0, 0.10),
}

METHOD_LABELS = {
    "auto": "自动（条形走竖列复制，方形走反射）",
    "lama": "LaMa 模型修复（最自然，慢）",
    "mirror": "四方向中位数反射",
    "strip": "竖列复制（条形水印最好）",
    "smooth": "平滑填充（纯色/渐变背景更干净）",
}


def normalize_box(box) -> tuple:
    """把 (x0, y0, x1, y1) 比例框整理成 x0<x1、y0<y1 且夹在 0~1 内。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    def clip(value):
        return min(max(value, 0.0), 1.0)

    return clip(x0), clip(y0), clip(x1), clip(y1)


def box_pixels(box, width, height) -> tuple:
    """比例框 → 像素框 (y0, y1, x0, x1)，至少 1×1。"""
    x0, y0, x1, y1 = normalize_box(box)
    px0 = min(max(int(round(x0 * width)), 0), width - 1)
    py0 = min(max(int(round(y0 * height)), 0), height - 1)
    px1 = min(max(int(round(x1 * width)), px0 + 1), width)
    py1 = min(max(int(round(y1 * height)), py0 + 1), height)
    return py0, py1, px0, px1


def _diffuse(arr: np.ndarray, known: np.ndarray, rounds: int = 4096) -> None:
    """把 known 之外的值由外向内逐圈用邻居均值填上（就地修改）。"""
    height, width = known.shape
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for _ in range(rounds):
        total = np.zeros_like(arr)
        count = np.zeros((height, width), dtype=np.float32)
        for dy, dx in steps:
            shifted = np.roll(arr, (dy, dx), axis=(0, 1))
            shifted_known = np.roll(known, (dy, dx), axis=(0, 1))
            # 卷绕到另一侧的行列不算邻居
            if dy:
                shifted_known[0 if dy == 1 else height - 1, :] = False
            else:
                shifted_known[:, 0 if dx == 1 else width - 1] = False
            total += shifted * shifted_known[..., None]
            count += shifted_known
        ready = (~known) & (count > 0)
        if not ready.any():
            break
        arr[ready] = total[ready] / count[ready][:, None]
        known |= ready


def _smooth_fill(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> None:
    """整个框交给扩散：纯色/渐变背景干净，纹理会被抹平。"""
    known = np.ones(arr.shape[:2], dtype=bool)
    known[y0:y1, x0:x1] = False
    _diffuse(arr, known)


def _reflect_fill(arr: np.ndarray, y0: int, y1: int,
                  x0: int, x1: int) -> np.ndarray:
    """四条边各反射一份候选，逐像素取中位数，返回已填好的像素掩码。

    为什么不是「取最近那条边」：那等于把框分成四块斜向区域，每块复制一条
    不同的邻边，交界处会出现明显的菱形拼缝（实测在脸部/裙子上很跳）。中位数
    保留各方向共同的色调（局部颜色大致连续），同时抑制单一方向的重复纹理，
    过渡比 Voronoi 分区自然；单方向缺失时（框贴边）也能自动用剩下的三个。
    """
    height, width = arr.shape[:2]
    ys = np.arange(y0, y1)[:, None].astype(np.int64)
    xs = np.arange(x0, x1)[None, :].astype(np.int64)
    rows = y1 - y0
    cols = x1 - x0
    ys = np.broadcast_to(ys, (rows, cols))
    xs = np.broadcast_to(xs, (rows, cols))
    # (源 y, 源 x, 距离)
    candidates = [
        (y0 - (ys - y0) - 1, xs, ys - y0 + 1),      # 上
        (y1 + (y1 - 1 - ys), xs, y1 - ys),          # 下
        (ys, x0 - (xs - x0) - 1, xs - x0 + 1),      # 左
        (ys, x1 + (x1 - 1 - xs), x1 - xs),          # 右
    ]
    stack = []
    for cand_y, cand_x, _dist in candidates:
        inside = ((cand_y >= 0) & (cand_y < height)
                  & (cand_x >= 0) & (cand_x < width))
        in_box = ((cand_y >= y0) & (cand_y < y1)
                  & (cand_x >= x0) & (cand_x < x1))
        ok_px = inside & ~in_box
        vals = arr[np.clip(cand_y, 0, height - 1),
                   np.clip(cand_x, 0, width - 1)].astype(np.float32)
        vals[~ok_px] = np.nan
        stack.append(vals)
    with warnings.catch_warnings():
        # 框贴死四条边时全是 NaN，这里只是避开告警，后面用 good 掩码决定填不填
        warnings.simplefilter("ignore", RuntimeWarning)
        filled = np.nanmedian(np.stack(stack), axis=0)
    good = ~np.isnan(filled[..., 0])
    block = arr[y0:y1, x0:x1].copy()
    block[good] = filled[good]
    arr[y0:y1, x0:x1] = block
    return good


def _strip_fill(arr: np.ndarray, y0: int, y1: int,
                x0: int, x1: int) -> np.ndarray:
    """条形水印：整条从上方（或下方）镜像搬下来，并把上下接缝的亮度对齐。

    横向水印字通常只有几十像素高、却横跨半个画面。这时按列整条复制竖向
    邻域，衣服边缘、发丝、手臂轮廓这些「竖结构」会自然接上；四方向取中位数
    反而会把横向的内容差别平均成一条一条的色块（实测在裙子上就是几块补丁）。
    复制完用上下两侧的差值做线性校正，否则接缝处会看到一条亮度差。
    返回已填好的像素掩码（贴边到无竖直空间可用时退回中位数反射）。
    """
    height = arr.shape[0]
    rows = y1 - y0
    if rows <= 0:
        return np.zeros((rows, x1 - x0), dtype=bool)
    if y0 >= rows:
        src = arr[y0 - rows:y0, x0:x1][::-1].astype(np.float32)
        top_gap = arr[y0 - 1, x0:x1].astype(np.float32) - src[0]
        bot_gap = (arr[y1, x0:x1].astype(np.float32) - src[-1]
                   if y1 < height else np.zeros_like(top_gap))
    elif height - y1 >= rows:
        src = arr[y1:y1 + rows, x0:x1][::-1].astype(np.float32)
        bot_gap = (arr[y1, x0:x1].astype(np.float32) - src[-1]
                   if y1 < height else np.zeros_like(src[0]))
        top_gap = (arr[y0 - 1, x0:x1].astype(np.float32) - src[0]
                   if y0 > 0 else np.zeros_like(src[0]))
        top_gap = -top_gap
    else:
        return _reflect_fill(arr, y0, y1, x0, x1)
    ramp = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None, None]
    corr = (top_gap * 0.5) * (1.0 - ramp) + (bot_gap * 0.5) * ramp
    arr[y0:y1, x0:x1] = np.clip(src + corr, 0, 255)
    return np.ones((rows, x1 - x0), dtype=bool)


def as_box_list(box) -> list:
    """把「一个框」或「一批框」统一成 [(x0, y0, x1, y1), ...]。

    一张图上同时有多个水印（比如左右各一个水印字样 + 右下角 AI 标签）时，
    传 list 就行；只补一次时仍可直接传四个数。
    """
    if box is None:
        return []
    first = box[0] if len(box) else None
    if isinstance(first, (list, tuple)):
        return [tuple(float(v) for v in one) for one in box if len(one) >= 4]
    if len(box) >= 4:
        return [tuple(float(v) for v in box[:4])]
    return []


def pick_method(box, width: int, height: int, method: str = "auto") -> str:
    """auto：扁长框（水印字那种横带）走竖列复制，接近方/竖长的框走中位数反射。"""
    if method != "auto":
        return method
    y0, y1, x0, x1 = box_pixels(box, width, height)
    rows = max(1, y1 - y0)
    cols = max(1, x1 - x0)
    return "strip" if cols >= rows * 2 else "mirror"


def inpaint(rgb: np.ndarray, box, method: str = "auto") -> np.ndarray:
    """把框覆盖的区域补掉，返回新的 uint8 RGB 数组。

    box 可以是一个 (x0, y0, x1, y1) 比例框，也可以是多个框的列表。
    框内内容完全丢弃；框外像素一个都不动。method 取 auto / strip / mirror / smooth。
    """
    if method == "lama":
        # 模型级修复单独一层：装不了/没模型时直接报错，让界面提示用户
        import dedup_watermark_lama as lama
        return lama.inpaint(rgb, as_box_list(box))
    out = np.array(rgb, dtype=np.uint8, copy=True)
    height, width = out.shape[:2]
    arr = out.astype(np.float32)
    for one in as_box_list(box):
        y0, y1, x0, x1 = box_pixels(one, width, height)
        chosen = pick_method(one, width, height, method)
        if chosen == "smooth":
            _smooth_fill(arr, y0, y1, x0, x1)
        elif chosen == "strip":
            ok = _strip_fill(arr, y0, y1, x0, x1)
            if not ok.all():
                known = np.ones((height, width), dtype=bool)
                known[y0:y1, x0:x1] = ok
                _diffuse(arr, known)
        else:
            ok = _reflect_fill(arr, y0, y1, x0, x1)
            if not ok.all():
                known = np.ones((height, width), dtype=bool)
                known[y0:y1, x0:x1] = ok
                _diffuse(arr, known)
        out[y0:y1, x0:x1] = np.clip(arr[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    return out


def output_path(src, out_dir=None, suffix: str = "_去水印",
                unique: bool = True) -> Path:
    """输出路径：默认同目录加后缀；unique=False 时不避让（批处理用来判断是否已做过）。"""
    src = Path(src)
    target_dir = Path(out_dir) if out_dir else src.parent
    stem = src.stem + suffix
    candidate = target_dir / (stem + src.suffix)
    if not unique:
        return candidate
    index = 2
    while candidate.exists():
        candidate = target_dir / f"{stem}({index}){src.suffix}"
        index += 1
    return candidate


def remove_watermark_file(src, box, dst=None, method: str = "auto",
                          quality: int = 95) -> Path:
    """处理一张图，写到 dst（默认同目录加后缀，绝不覆盖原图）。"""
    src = Path(src)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"))
        fixed = inpaint(arr, box, method)
    target = Path(dst) if dst else output_path(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(fixed, "RGB")
    if target.suffix.lower() in (".jpg", ".jpeg"):
        image.save(target, quality=quality, subsampling=0)
    elif target.suffix.lower() == ".webp":
        image.save(target, quality=quality)
    else:
        image.save(target)
    return target


def iter_photos(folder):
    from dedup_models import PHOTO_EXT

    for path in sorted(Path(folder).rglob("*")):
        if path.is_file() and path.suffix.lower() in PHOTO_EXT:
            yield path


def batch_remove(paths, box, out_dir=None, method: str = "auto",
                 quality: int = 95, cancel=None, on_progress=None) -> dict:
    """批量处理；返回 {"done", "failed", "skipped", "total"}。

    同名输出已存在就跳过——重复点一次不会生成一堆 `_去水印(2)(3)`。
    """
    paths = list(paths)
    done = skipped = 0
    failed = []
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if cancel is not None and cancel.is_set():
            break
        try:
            target = output_path(path, out_dir, unique=False)
            if target.exists():
                skipped += 1
            else:
                remove_watermark_file(path, box, target, method, quality)
                done += 1
        except Exception as exc:  # 单张失败不影响整批
            failed.append((str(path), str(exc)))
        if on_progress is not None:
            on_progress(index, total, str(path))
    return {"done": done, "failed": failed, "skipped": skipped,
            "total": total}


# ---------------------------------------------------------------- 浅色水印
# 半透明水印的参考色（亮黄），用来反推覆盖率 α
LIGHT_REF = (252.0, 248.0, 196.0)

# 常见浅色水印的参考色：反解时要知道水印是什么颜色
TINT_REFS = {
    "黄": (252.0, 248.0, 196.0),
    "蓝": (196.0, 236.0, 252.0),
    "粉": (252.0, 214.0, 238.0),
    "绿": (214.0, 250.0, 214.0),
}


def _tint_scores(rgb: np.ndarray) -> dict:
    """各色系的“偏向程度”：正数越大表示越像那种淡色。"""
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return {
        "黄": (r + g) * 0.5 - b,
        "蓝": b - (r + g) * 0.5,
        "粉": r - g,
        "绿": g - (r + b) * 0.5,
    }


def _robust_threshold(score: np.ndarray, floor: float = 3.0,
                      cap: float = 8.0) -> float:
    """用中位数绝对偏差估噪声，阈值随图片自动定：干净白底上能抓到 ⩽3 的淡差。"""
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med)))
    return float(np.clip(4.0 * 1.4826 * mad, floor, cap))


def detect_tint(rgb: np.ndarray, gmin: int = 120,
                luma_gate: float = 140.0) -> str:
    """判断这张图上的浅色水印偏哪种颜色（黄/蓝/粉/绿），全没就返回空串。"""
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = arr.max(axis=2) - arr.min(axis=2)
    pale = (luma > luma_gate) & (chroma < 45) & (g >= gmin)
    if not pale.any():
        return ""
    best, best_count = "", 0
    for name, score in _tint_scores(rgb).items():
        limit = _robust_threshold(score[pale])
        count = int(((score > limit) & pale).sum())
        if count > best_count:
            best, best_count = name, count
    # 占比太小就不算水印（至少千分之一）
    return best if best_count >= max(200, luma.size // 1000) else ""


def detect_light_watermark(rgb: np.ndarray, diff=None, gmin: int = 120,
                           luma_gate: float = 140.0,
                           tint=None) -> np.ndarray:
    """找浅色（淡黄/淡蓝/淡粉/淡绿）半透明水印的像素。

    判据：偏那种颜色 + 够亮 + 不饱和。低饱和是为了避开彩色 logo、选中框
    这类真·彩色元素；阈值默认由图自身的噪声自动定，不再写死。
    """
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = arr.max(axis=2) - arr.min(axis=2)
    pale = (luma > luma_gate) & (chroma < 45) & (g >= gmin)
    name = tint if tint is not None else detect_tint(rgb, gmin, luma_gate)
    if not name:
        return np.zeros(luma.shape, dtype=bool)
    score = _tint_scores(rgb)[name]
    limit = float(diff) if diff is not None else _robust_threshold(score[pale])
    return (score > limit) & pale


def _run_lengths(mask: np.ndarray):
    """每个像素所在连续段有多长（横、竖各一张），用来区分“笔画”和“长直线”。"""
    height, width = mask.shape
    rows = np.broadcast_to(np.arange(width), (height, width))
    row_start = np.maximum.accumulate(np.where(mask, 0, rows + 1), axis=1)
    run_row = rows - row_start + 1
    row_end = np.minimum.accumulate(
        np.where(mask, width, rows - 1)[:, ::-1], axis=1)[:, ::-1]
    run_row = np.minimum(run_row, row_end - rows + 1)

    cols = np.broadcast_to(np.arange(height)[:, None], (height, width))
    col_start = np.maximum.accumulate(np.where(mask, 0, cols + 1), axis=0)
    run_col = cols - col_start + 1
    col_end = np.minimum.accumulate(
        np.where(mask, height, cols - 1)[::-1], axis=0)[::-1]
    run_col = np.minimum(run_col, col_end - cols + 1)
    return run_row, run_col


def _neutral_mask(rgb: np.ndarray, max_run: int = 60) -> np.ndarray:
    """淡灰色水印：没有偏色，只是比纸面暗一点点。

    难点：灰色网格线和灰色水印亮度几乎一样，只能看形状——网格线很长、
    笔画很短。所以要求横、竖两个方向的连续段都不长。
    """
    arr = rgb.astype(np.float32)
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    background = float(np.percentile(luma, 90))
    deficit = background - luma
    candidate = (chroma < 6.0) & (deficit > 1.5) & (deficit < 40.0)
    run_row, run_col = _run_lengths(candidate)
    return candidate & (run_row < max_run) & (run_col < max_run)


def _nearest4_fill(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """四方向“最近的已知像素”取中位数。

    只对掩膜里的像素取值：笔画穿过表格线时，上下给的是白底、左右给的是线色，
    中位数落回白底，不会像各向扩散那样把线色刷成一块。
    """
    height, width = mask.shape
    if not mask.any():
        return np.array(arr, dtype=np.uint8, copy=True)
    known = ~mask
    idx_w = np.broadcast_to(np.arange(width), (height, width)).copy()
    idx_h = np.broadcast_to(np.arange(height)[:, None], (height, width)).copy()
    left = np.maximum.accumulate(np.where(known, idx_w, -1), axis=1)
    right = np.minimum.accumulate(np.where(known, idx_w, width)[:, ::-1],
                                 axis=1)[:, ::-1]
    up = np.maximum.accumulate(np.where(known, idx_h, -1), axis=0)
    down = np.minimum.accumulate(np.where(known, idx_h, height)[::-1],
                                 axis=0)[::-1]
    ys, xs = np.nonzero(mask)
    cands = []
    for yy, xx, ok in (
            (ys, left[ys, xs], left[ys, xs] >= 0),
            (ys, right[ys, xs], right[ys, xs] <= width - 1),
            (up[ys, xs], xs, up[ys, xs] >= 0),
            (down[ys, xs], xs, down[ys, xs] <= height - 1)):
        pick = arr[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)]
        cands.append(np.where(ok[:, None], pick, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.stack(cands), axis=0)
    out = arr.astype(np.float32).copy()
    good = ~np.isnan(med[:, 0])
    out[ys[good], xs[good]] = med[good]
    return np.clip(out, 0, 255).astype(np.uint8)


def undo_light_watermark(rgb: np.ndarray, strength: float = 1.0,
                        ref=LIGHT_REF, tint=None) -> np.ndarray:
    """把半透明浅色水印“解”掉，而不是“补”掉。

    I = (1-α)·背景 + α·参考色，背景（纸张、表格底）近似中性，于是
    α 可以由 (R-B)/(Wr-Wb)、(G-B)/(Wg-Wb) 估出来，再反解背景。
    这样文字和表格线的像素压根没被动过，不会留下补图常见的暗边。
    strength 是修正强度（1.0 全解，0.5 只解一半），改小可以少伤原图的暖色调。
    """
    if tint is not None and tint in TINT_REFS:
        ref = TINT_REFS[tint]
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    wr, wg, wb = float(ref[0]), float(ref[1]), float(ref[2])
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    # 灰底（例如表格表头）上反解容易把灰色解出色斑，亮度越低解越少
    scale = np.clip((luma - 200.0) / 40.0, 0.35, 1.0) * float(strength)
    alpha = ((r - b) / (wr - wb) + (g - b) / (wg - wb)) * 0.5 * scale
    alpha = np.clip(alpha, 0.0, 0.6)
    alpha = np.where((luma < 150.0) | (alpha < 0.015), 0.0, alpha)
    denom = np.clip(1.0 - alpha, 0.05, 1.0)[..., None]
    fixed = (arr - alpha[..., None] * np.array([wr, wg, wb], dtype=np.float32)) / denom
    return np.clip(fixed, 0, 255).astype(np.uint8)


def _box_blur_known(arr: np.ndarray, known: np.ndarray, radius: int):
    """只对已知像素做归一化盒滤波（积分图实现），得到“周围背景”。"""
    if arr.ndim == 2:
        arr = arr[..., None]
    height, width = known.shape
    pad = radius + 1
    weighted = np.pad(arr.astype(np.float32) * known[..., None],
                      ((pad, pad), (pad, pad), (0, 0)))
    counts = np.pad(known.astype(np.float32), ((pad, pad), (pad, pad)))
    sum_a = weighted.cumsum(0).cumsum(1)
    sum_k = counts.cumsum(0).cumsum(1)
    ys = np.arange(height) + pad
    xs = np.arange(width) + pad

    def rect(total, y0, y1, x0, x1):
        return (total[y1][:, x1] - total[y0 - 1][:, x1]
                - total[y1][:, x0 - 1] + total[y0 - 1][:, x0 - 1])

    y0, y1 = ys - radius, ys + radius + 1
    x0, x1 = xs - radius, xs + radius + 1
    acc = rect(sum_a, y0, y1, x0, x1)
    cnt = rect(sum_k, y0, y1, x0, x1)
    return acc / np.maximum(cnt, 1e-6)[..., None], cnt


def _local_std(gray: np.ndarray, radius: int = 6) -> np.ndarray:
    known = np.ones(gray.shape, dtype=bool)
    mean = _box_blur_known(gray, known, radius)[0][..., 0]
    mean2 = _box_blur_known(gray * gray, known, radius)[0][..., 0]
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))


def _flatten_background(fixed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把水印所在处的“平坦背景”重新铺平。

    反解能去掉颜色，但会把纸上那层薄薄的阴影（水印遮过的位置）留下一点
    亮度差，在浅灰渐变背景上一眼看得出。这里只对“非常平坦 + 附近完全没
    有深色内容 + 本身很亮”的像素，用周围未受水印影响的像素估出背景填回；
    表头、正文、表格线所在的位置一概不碰，宁可少填不弄花。
    """
    grow = _grow_mask(mask, 2)
    if not grow.any():
        return fixed
    known = ~grow
    background, _ = _box_blur_known(fixed, known, 12)
    luma = fixed.astype(np.float32).mean(axis=2)
    std = _local_std(luma, 6)
    dark = ((luma < 145) & known).astype(np.float32)
    dark_frac = _box_blur_known(dark, known, 10)[0][..., 0]
    flat = (std < 0.8) & (dark_frac <= 0.0) & (luma > 195) & grow
    out = fixed.astype(np.float32).copy()
    out[flat] = background[flat]
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_light_watermark(rgb: np.ndarray) -> np.ndarray:
    """浅色水印一键流程：认出色调 → 反解 → 铺平平坦背景 → 补掉残影。

    彩色淡色水印走反解；完全没偏色的淡灰水印走“平坦区重铺背景”。
    """
    tint = detect_tint(rgb)
    if not tint:
        neutral = _neutral_mask(rgb)
        if int(neutral.sum()) < max(300, neutral.size // 500):
            return np.array(rgb, dtype=np.uint8, copy=True)
        grow = _grow_mask(neutral, 2)
        background, _ = _box_blur_known(rgb, ~grow, 10)
        out = rgb.astype(np.float32).copy()
        out[grow] = background[grow]
        return np.clip(out, 0, 255).astype(np.uint8)

    mask = detect_light_watermark(rgb, tint=tint)
    if not mask.any():
        return np.array(rgb, dtype=np.uint8, copy=True)
    first = undo_light_watermark(rgb, tint=tint)
    first = _flatten_background(first, mask)
    residual = detect_light_watermark(first, diff=2.0, gmin=105,
                                      luma_gate=110.0, tint=tint)
    if residual.any():
        # 文字、表格线旁边不补：那里的“残余黄点”其实是笔画自身的抗锯齿边缘，
        # 补下去会把字吐成毛边（实测表头字母会被吞出锯齿）
        known = ~residual
        luma = first.astype(np.float32).mean(axis=2)
        dark = ((luma < 145) & known).astype(np.float32)
        dark_frac = _box_blur_known(dark, known, 8)[0][..., 0]
        residual &= dark_frac <= 0.0
    if residual.any():
        residual = _grow_mask(residual, 1)
        return _nearest4_fill(first, residual)
    return first


def _grow_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """方形膨胀：把笔画边缘抗锯齿留下的淡黄一起圈进来。"""
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def remove_light_file(src, dst=None, quality: int = 95) -> Path:
    """处理一张图（自动抠浅色水印），不覆盖原图。"""
    src = Path(src)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"))
        fixed = clean_light_watermark(arr)
    target = Path(dst) if dst else output_path(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(fixed, "RGB")
    if target.suffix.lower() in (".jpg", ".jpeg"):
        image.save(target, quality=quality, subsampling=0)
    elif target.suffix.lower() == ".webp":
        image.save(target, quality=quality)
    else:
        image.save(target)
    return target


def batch_remove_light(paths, out_dir=None, quality: int = 95, cancel=None,
                       on_progress=None) -> dict:
    """批量自动抠浅色水印；返回格式与 batch_remove 一致。"""
    paths = list(paths)
    done = skipped = 0
    failed = []
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if cancel is not None and cancel.is_set():
            break
        try:
            target = output_path(path, out_dir, unique=False)
            if target.exists():
                skipped += 1
            else:
                remove_light_file(path, target, quality)
                done += 1
        except Exception as exc:
            failed.append((str(path), str(exc)))
        if on_progress is not None:
            on_progress(index, total, str(path))
    return {"done": done, "failed": failed, "skipped": skipped,
            "total": total}


__all__ = [
    "LIGHT_REF",
    "METHOD_LABELS",
    "PRESETS",
    "as_box_list",
    "batch_remove",
    "batch_remove_light",
    "box_pixels",
    "clean_light_watermark",
    "detect_light_watermark",
    "inpaint",
    "iter_photos",
    "normalize_box",
    "output_path",
    "pick_method",
    "remove_light_file",
    "remove_watermark_file",
    "undo_light_watermark",
]
