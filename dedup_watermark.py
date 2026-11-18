# -*- coding: utf-8 -*-
"""照片去水印：把用户框出的区域用四周内容补上（本地、不联网、不覆盖原图）。

浅色半透明水印另有一条“反解”路：框内按局部背景估透明度反解，底纹不动；
白底文档也可整张自动找，照片则只在找到的水印区里动手。
已知难例：蓝水印压蓝天、绿水印压绿草地（色调与背景太近，色度信号弱）；
整块平铺水印（占比超 70% 拒认，本路不覆盖）；任意偏离标准参考色太远
的色调（投票靠方向投影仍能认，但 α 估计会偏）。

设计取舍：
  * 不引入新依赖 —— 只用 numpy + Pillow，和项目其余部分保持一致；
  * 恢复方式用「按最近的边反射回填」：每个待补像素取它最近那条框边的
    镜像内容，四个方向在接缝处都是连续的，所以角落 logo、底部条、相机
    时间戳这类边缘水印补完基本看不到方形痕迹，纹理（草地、墙砖）也接得上；
  * 反射采不到的地方（框贴住图片边、对面也没空间）退回由外向内扩散，
    只会给出平滑过渡，细节会糊——这是无模型方案的硬上限，界面上要提示。

流程分层（检测 → 重建 → 状态）：检测器（detect_light_in_box）只吐
LightDetection（框 + 色系 + 掩膜 + 分数 + 来源 + 拒认原因），重建决策
（reconstruct_light）是唯一决定“反解还是补图”的地方，框选和一键抠
两条入口都走它；每张图最后给一个 LightStatus（干净 / 残留 / 未找到 /
拒绝），处理完不认怂——解不干净、处理不了的都如实报，不悄悄说成功。

只处理使用者自己有版权的照片。
"""
from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass, field
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
    "auto": "自动（横条走竖列复制、竖条走横行复制、方形走反射）",
    "lama": "LaMa 模型修复（最自然，慢）",
    "mirror": "四方向中位数反射",
    "strip": "竖列复制（横向水印条最好）",
    "strip_v": "横行复制（左右侧边栏式竖条最好）",
    "patchmatch": "纹理合成（重复纹理比反射自然，纯 numpy）",
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


def _resize_bilinear(a: np.ndarray, height: int, width: int) -> np.ndarray:
    """双线性缩放到任意尺寸（纯 numpy，对齐角点；同尺寸时是恒等变换）。

    金字塔上下采样都走这里：不用 PIL 是为了保持 float32 精度，也避免
    每次推理都倒一遍 uint8 丢小数。
    """
    src_h, src_w = a.shape[:2]
    if (src_h, src_w) == (height, width):
        return a
    ys = np.linspace(0.0, src_h - 1, height)
    xs = np.linspace(0.0, src_w - 1, width)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    top = a[y0][:, x0] * (1.0 - wx[..., None]) + a[y0][:, x1] * wx[..., None]
    bot = a[y1][:, x0] * (1.0 - wx[..., None]) + a[y1][:, x1] * wx[..., None]
    return top * (1.0 - wy[..., None]) + bot * wy[..., None]


def _relax_laplace(arr: np.ndarray, fixed: np.ndarray,
                   known: np.ndarray, rounds: int) -> np.ndarray:
    """矩形区域上解 Laplace：unknown 取四邻均值，known 处锁死为 fixed。

    用 edge padding 取邻居，绝不像 np.roll 那样把对边的像素卷绕进来；
    裁剪区最外一圈本来就是 known，边界怎么处理都不影响框内结果。

    这里刻意保持 ω=1 的纯 Jacobi：加权 Jacobi(ω>1) 会把棋盘格高频模态
    （特征值接近 -1）放大到发散，实测会填出 NaN。加速完全交给金字塔层数。
    """
    height, width = known.shape
    padded = np.empty((height + 2, width + 2, arr.shape[2]), dtype=np.float32)
    inner = padded[1:-1, 1:-1]
    cur = arr
    for _ in range(rounds):
        # 自己拼 edge padding 比每轮 np.pad(mode="edge") 少一层 Python 开销
        inner[:] = cur
        padded[0, 1:-1] = cur[0]
        padded[-1, 1:-1] = cur[-1]
        padded[1:-1, 0] = cur[:, 0]
        padded[1:-1, -1] = cur[:, -1]
        padded[0, 0] = cur[0, 0]
        padded[0, -1] = cur[0, -1]
        padded[-1, 0] = cur[-1, 0]
        padded[-1, -1] = cur[-1, -1]
        avg = (padded[:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.25
        avg[known] = fixed[known]
        cur = avg
    return cur


def _diffuse_pyramid(arr: np.ndarray, known: np.ndarray,
                     rounds: int = 4096) -> None:
    """coarse-to-fine 多重网格：先在 1/8 分辨率解，上采样作初值，再逐级细化。

    纯 Jacobi 要把边界信息传进框中心，轮数得跟框的边长同量级（几千轮）；
    在 1/8 图上同样的信息只要一两百轮就能铺满，再把这个「已经大致对」的
    解放大回去做几十轮局部松弛即可。总计算量从「全图 × 4096」降到
    「小区域 × 几百」，视觉上都是同一个 Dirichlet 问题的平滑解。
    """
    height, width = known.shape
    if not known.any():
        return                      # 无边界条件时保持原值（和旧实现一致）
    kf_full = known.astype(np.float32)[..., None]
    weighted_full = arr.astype(np.float32) * kf_full

    sizes = [(height, width)]
    # 最多减半三次（1/8）；太小的区域再降采样就没意义了
    while len(sizes) < 4 and min(sizes[-1]) >= 32:
        sizes.append((max(1, sizes[-1][0] // 2),
                      max(1, sizes[-1][1] // 2)))

    coarse = None
    for level, (h, w) in enumerate(reversed(sizes)):
        if (h, w) == (height, width):
            kc = known              # 最细层用精确掩码，保证框外一个像素不动
            fixed = arr.astype(np.float32)
        else:
            kf = _resize_bilinear(kf_full, h, w)[..., 0]
            kv = _resize_bilinear(weighted_full, h, w)
            kc = kf > 1e-4
            fixed = kv / np.maximum(kf[..., None], 1e-4)
        if coarse is None:
            seed = (fixed[kc].mean(axis=0) if kc.any()
                    else arr.mean(axis=(0, 1)))
            guess = np.where(kc[..., None], fixed,
                             np.broadcast_to(seed, (h, w, arr.shape[2])))
        else:
            guess = np.where(kc[..., None], fixed,
                             _resize_bilinear(coarse, h, w))
        # 粗层多松弛几轮，细层只做局部修正；rounds 仍作为调用方的上限
        budget = min(rounds, (160, 64, 24, 12)[min(level, 3)])
        coarse = _relax_laplace(guess, fixed, kc, budget)
    arr[~known] = coarse[~known]


def _diffuse(arr: np.ndarray, known: np.ndarray, rounds: int = 4096) -> None:
    """把 known 之外的值由外向内扩散填上（就地修改）。

    两处性能关键：只在「待补区域包围盒 + 少量边距」的裁剪块里求解，
    不再对整张 4000×3000 的图反复 np.roll；块内改用金字塔由粗到细。
    签名与旧版一致，调用方（_smooth_fill / inpaint 兜底）无需改动。
    """
    # 两次布尔归约就能框出待补区域，比全图 np.nonzero 便宜一个数量级
    rows = np.flatnonzero(~known.all(axis=1))
    if not rows.size:
        return
    cols = np.flatnonzero(~known.all(axis=0))
    height, width = known.shape
    margin = 16                    # 外圈留一圈已知像素当 Dirichlet 边界
    gy0 = max(0, int(rows[0]) - margin)
    gy1 = min(height, int(rows[-1]) + 1 + margin)
    gx0 = max(0, int(cols[0]) - margin)
    gx1 = min(width, int(cols[-1]) + 1 + margin)
    sub_arr = np.ascontiguousarray(arr[gy0:gy1, gx0:gx1], dtype=np.float32)
    sub_known = known[gy0:gy1, gx0:gx1]
    if not sub_known.any():
        return
    _diffuse_pyramid(sub_arr, sub_known, rounds)
    local_unknown = ~sub_known
    region = arr[gy0:gy1, gx0:gx1]
    region[local_unknown] = sub_arr[local_unknown]


def _smooth_fill(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> None:
    """整个框交给扩散：纯色/渐变背景干净，纹理会被抹平。"""
    known = np.ones(arr.shape[:2], dtype=bool)
    known[y0:y1, x0:x1] = False
    _diffuse(arr, known)


def _smooth_edge(values: np.ndarray, window: int) -> np.ndarray:
    """沿边方向做盒式平滑，只留下低频亮度/颜色渐变。

    边界差里混着纹理的高频起伏（草叶、字边），直接铺进整块会把反射
    出来的纹理抹成色带；平滑后相当于只校正「框外整体比框内亮一点」
    这类渐变台阶。edge padding 保证首尾不会被补零拉低。
    """
    length = values.shape[0]
    if length <= 2 or window <= 1:
        return values
    w = min(window, length)
    w = w if w % 2 else w - 1
    if w <= 1:
        return values
    pad = (w - 1) // 2
    ext = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.full(w, 1.0 / w, dtype=np.float32)
    cols = [np.convolve(ext[:, c], kernel, mode="valid")
            for c in range(values.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def _reflect_seam_correction(arr: np.ndarray, block: np.ndarray,
                             good: np.ndarray, y0: int, y1: int,
                             x0: int, x1: int) -> None:
    """四边接缝校正：把「外侧实际值 − 填充值」按到各边的距离插回框内。

    _strip_fill 只做了上下两道的线性校正；这里推广到四条边，用 Coons
    双线性片（等价于给矩形边界做一次廉价的 Poisson/调和插值）：边界上
    精确等于该边的差值，向框内平滑过渡。天空、逆光这种整体渐变背景上，
    四边反射回来的内容不再在框边出现一圈台阶。
    只作用于确实由反射填好的像素；没填到的留给后面的扩散，避免把水印
    残值也一起抬亮。
    """
    height, width = arr.shape[:2]
    rows, cols = block.shape[:2]
    channels = block.shape[2]
    zeros_x = np.zeros((cols, channels), dtype=np.float32)
    zeros_y = np.zeros((rows, channels), dtype=np.float32)

    # 外侧紧邻一圈的实际像素 − 框内对应边界行的填充值
    top = (arr[y0 - 1, x0:x1].astype(np.float32) - block[0]
           if y0 > 0 else zeros_x.copy())
    bottom = (arr[y1, x0:x1].astype(np.float32) - block[-1]
              if y1 < height else zeros_x.copy())
    left = (arr[y0:y1, x0 - 1].astype(np.float32) - block[:, 0]
            if x0 > 0 else zeros_y.copy())
    right = (arr[y0:y1, x1].astype(np.float32) - block[:, -1]
             if x1 < width else zeros_y.copy())
    if not any(v.any() for v in (top, bottom, left, right)):
        return

    wx = max(3, min(61, cols // 4))
    wy = max(3, min(61, rows // 4))
    top, bottom = _smooth_edge(top, wx), _smooth_edge(bottom, wx)
    left, right = _smooth_edge(left, wy), _smooth_edge(right, wy)
    # 边界本来就贴合（理想线性渐变上反射能精确复原）时不要再动：量到的
    # 0.1 级差只是离散采样噪声，插回去反而会把完美渐变推出一点误差。
    if float(np.mean([np.abs(v).mean() for v in
                      (top, bottom, left, right)])) < 1.0:
        return

    ty = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
    tx = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
    # 四个角点的共识值：相邻两条边各报一个数，取平均保证 Coons 片不自相矛盾
    c_tl = (top[0] + left[0]) * 0.5
    c_tr = (top[-1] + right[0]) * 0.5
    c_bl = (bottom[0] + left[-1]) * 0.5
    c_br = (bottom[-1] + right[-1]) * 0.5

    corr = ((1.0 - ty)[..., None] * top[None, :, :]
            + ty[..., None] * bottom[None, :, :]
            + (1.0 - tx)[..., None] * left[:, None, :]
            + tx[..., None] * right[:, None, :])
    corr -= (((1.0 - ty) * (1.0 - tx))[..., None] * c_tl
             + ((1.0 - ty) * tx)[..., None] * c_tr
             + (ty * (1.0 - tx))[..., None] * c_bl
             + (ty * tx)[..., None] * c_br)
    corr = np.clip(corr, -48.0, 48.0)     # 极端边缘（框边压着强反差）不许带跑整块
    corr[~good] = 0.0
    block += corr


def _reflect_fill(arr: np.ndarray, y0: int, y1: int,
                  x0: int, x1: int, correct: bool = True) -> np.ndarray:
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
    block = arr[y0:y1, x0:x1].astype(np.float32)
    block[good] = filled[good]
    if correct:
        # arr 此刻还是原图，四条边外侧的实际值可以直接拿来当边界条件
        _reflect_seam_correction(arr, block, good, y0, y1, x0, x1)
    arr[y0:y1, x0:x1] = np.clip(block, 0, 255)
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


def _strip_fill_v(arr: np.ndarray, y0: int, y1: int,
                  x0: int, x1: int) -> np.ndarray:
    """竖长条水印（左右侧边栏式）：_strip_fill 转置一下就是横行复制版本。

    transpose 返回的是视图而不是副本，写进去就是写回原图；横纵参数互换后，
    「上方/下方」自然变成「左侧/右侧」，接缝的线性校正也跟着沿 x 方向走。
    返回的掩码同样要转置回 (rows, cols)，供上层判断哪些像素还需要扩散兜底。
    """
    ok = _strip_fill(arr.transpose(1, 0, 2), x0, x1, y0, y1)
    return ok.T


def _patchmatch_fill(arr: np.ndarray, y0: int, y1: int,
                     x0: int, x1: int,
                     iterations: int = 2) -> np.ndarray:
    """简化版 PatchMatch：在框周围找最像的 patch，由外向内一圈圈长进去。

    反射中位数是「每个像素各问一条边」，这里改成「整块一起问周围」：草地、
    布料、砖墙这类重复纹理能保住颗粒走向，不会把四个方向平均成糊块。纯
    numpy、不引新依赖，代价是比反射慢一个数量级，所以只作为显式档位，
    不进 auto 分派。

    步骤：
      1. 框内铺锚点网格，按到框边的距离分环，从最外环往里逐环合成，
         上一环的结果就是下一环的已知上下文；
      2. 每个锚点先扫一组结构化平移（覆盖常见纹理周期的整倍数），再做
         propagation（上/左邻居的最优偏移）+ random search（半径逐轮减半）。
         候选是任意整数坐标、不吸附到网格，周期纹理能找到精确相位；
      3. 匹配只在「已知像素」上算 7×7 采样点的 SSD；源块只要碰到还没填的
         框内像素就一票否决，绝不让水印白块被当成上下文学回去；
      4. 重叠 patch 用抬高的三角窗加权平均，最后复用四边接缝校正。
    返回已填好的像素掩码；没盖到的洞给个中性初值，交给 _diffuse 兜底。
    """
    height, width = arr.shape[:2]
    rows, cols = y1 - y0, x1 - x0
    if rows <= 0 or cols <= 0:
        return np.zeros((max(0, rows), max(0, cols)), dtype=bool)
    # 锚点间距越大，总锚点数越少；patch 取 2*step+1 保证相邻锚点的块有重叠
    step = int(np.clip(np.sqrt(rows * cols) / 40.0, 2, 12))
    ps = min(2 * step + 1, min(height, width))
    max_y, max_x = height - ps, width - ps
    if ps < 3 or max_y < 0 or max_x < 0:
        return _reflect_fill(arr, y0, y1, x0, x1)
    # 框几乎铺满整张图：上下或左右总有一侧找不到够大的已知源块
    if (y0 < ps and y1 > max_y) or (x0 < ps and x1 > max_x):
        return _reflect_fill(arr, y0, y1, x0, x1)

    rng = np.random.default_rng(20240617)  # 固定种子，同样输入给同样结果
    work = np.asarray(arr, dtype=np.float32)  # float32 时是视图，只写框内
    known = np.ones((height, width), dtype=bool)
    known[y0:y1, x0:x1] = False
    block = np.ascontiguousarray(work[y0:y1, x0:x1]).copy()
    filled = np.zeros((rows, cols), dtype=bool)

    # ---- 锚点网格与分环 ----
    ays = np.arange(y0, y1, step, dtype=np.int64)
    axs = np.arange(x0, x1, step, dtype=np.int64)
    na_y, na_x = int(ays.size), int(axs.size)
    gy_idx = np.broadcast_to(np.arange(na_y)[:, None], (na_y, na_x))
    gx_idx = np.broadcast_to(np.arange(na_x)[None, :], (na_y, na_x))
    ring_a = np.minimum(np.minimum(gy_idx, gx_idx),
                        np.minimum(na_y - 1 - gy_idx, na_x - 1 - gx_idx))
    n_rings = int(ring_a.max()) + 1
    ring_p = np.minimum(
        np.minimum(np.arange(rows)[:, None] // step,
                   np.arange(cols)[None, :] // step),
        np.minimum((rows - 1 - np.arange(rows))[:, None] // step,
                   (cols - 1 - np.arange(cols))[None, :] // step))

    field = np.full((na_y, na_x, 2), -1, dtype=np.int64)
    win1 = 0.15 + 0.85 * (1.0 - np.abs(np.linspace(-1.0, 1.0, ps)))
    win = np.outer(win1, win1).astype(np.float32)
    half, center = ps // 2, step // 2
    ii, jj = np.arange(ps)[:, None], np.arange(ps)[None, :]
    si = np.linspace(0, ps - 1, 7).astype(np.int64)
    sy, sx = si[:, None], si[None, :]

    # 结构化平移：近处细扫 + 远处按 1/4 跨度扫，覆盖常见纹理周期的整倍数
    radius = int(max(3 * ps, 32))
    vals = np.unique(np.concatenate([
        np.array([-ps, 0, ps], dtype=np.int64),
        np.linspace(-radius, radius, 5).astype(np.int64)]))
    oy, ox = np.meshgrid(vals, vals, indexing="ij")
    offsets = np.stack([oy.ravel(), ox.ravel()], axis=1)
    chunk = 96

    def _clip_pos(pos: np.ndarray) -> np.ndarray:
        pos[..., 0] = np.clip(pos[..., 0], 0, max_y)
        pos[..., 1] = np.clip(pos[..., 1], 0, max_x)
        return pos

    def _cost(ay, ax, pos, bad_box) -> np.ndarray:
        """(na, m, 2) 个源块左上角 → (na, m) 的 masked SSD。"""
        ty = np.clip(ay[:, None, None] + sy, 0, height - 1)
        tx = np.clip(ax[:, None, None] + sx, 0, width - 1)
        target = work[ty, tx]
        valid = known[ty, tx]
        cnt = np.maximum(valid.sum(axis=(1, 2)).astype(np.float32), 3.0)
        py = pos[..., 0][..., None, None] + sy
        px = pos[..., 1][..., None, None] + sx
        diff = target[:, None] - work[py, px]
        cost = np.sum(diff * diff * valid[:, None, :, :, None],
                      axis=(2, 3, 4)) / cnt[:, None]
        # 源块只要和「还没填的区域」相交就作废：水印白块绝不能进上下文
        by0, by1, bx0, bx1 = bad_box
        sy0, sx0 = pos[..., 0], pos[..., 1]
        cost[(sy0 < by1) & (sy0 + ps > by0)
             & (sx0 < bx1) & (sx0 + ps > bx0)] = np.inf
        return cost

    for ring in range(n_rings):
        bad_rows = np.flatnonzero((~known).any(axis=1))
        bad_cols = np.flatnonzero((~known).any(axis=0))
        if not bad_rows.size:
            break
        bad_box = (int(bad_rows[0]), int(bad_rows[-1]) + 1,
                   int(bad_cols[0]), int(bad_cols[-1]) + 1)
        by0, by1 = bad_box[0], bad_box[1]
        # 所有候选都失效时的安全源：贴着待填区上沿或下沿取一块纯已知内容
        safe_y = (int(np.clip(by0 - ps, 0, max_y)) if y0 >= ps
                  else int(np.clip(by1, 0, max_y)))
        safe = np.array([safe_y, int(np.clip(x0, 0, max_x))], dtype=np.int64)
        sel = np.argwhere(ring_a == ring)
        if not sel.size:
            continue
        na = int(sel.shape[0])
        # patch 以锚点格子的中心为原点，最外环因此有一半落在框外当上下文
        ay = y0 + sel[:, 0] * step + center - half
        ax = x0 + sel[:, 1] * step + center - half
        base = np.stack([ay, ax], axis=1)
        up = field[np.maximum(sel[:, 0] - 1, 0), sel[:, 1]].copy()
        left = field[sel[:, 0], np.maximum(sel[:, 1] - 1, 0)].copy()
        up_ok = (sel[:, 0] > 0) & (up[:, 0] >= 0)
        left_ok = (sel[:, 1] > 0) & (left[:, 0] >= 0)
        idx = np.arange(na)

        def _pick(cost, pos):
            k = np.argmin(cost, axis=1)
            best = pos[idx[:cost.shape[0]], k]
            dead = np.isinf(cost[idx[:cost.shape[0]], k])
            best[dead] = safe
            return best

        # pass 1：结构化平移粗搜，分块控制峰值内存
        cur = np.empty((na, 2), dtype=np.int64)
        for s in range(0, na, chunk):
            e = min(s + chunk, na)
            pos = _clip_pos(base[s:e, None, :] + offsets[None, :, :])
            cur[s:e] = _pick(_cost(ay[s:e], ax[s:e], pos, bad_box), pos)
        # pass 2：propagation + random search
        for it in range(max(1, iterations)):
            r = max(2 * ps, 16) * (0.5 ** it)
            rand = cur[:, None, :] + rng.uniform(-r, r, (na, 3, 2)).astype(np.int64)
            up_c = np.where(up_ok[:, None, None],
                            (up + np.array([step, 0]))[:, None, :],
                            cur[:, None, :])
            left_c = np.where(left_ok[:, None, None],
                              (left + np.array([0, step]))[:, None, :],
                              cur[:, None, :])
            cand = _clip_pos(np.concatenate(
                [cur[:, None, :], up_c, left_c, rand], axis=1))
            cur = _pick(_cost(ay, ax, cand, bad_box), cand)
        field[sel[:, 0], sel[:, 1]] = cur

        # 重叠 patch 加权平均，只写当前环，保证由外向内生长
        ring_mask = ring_p == ring
        acc = np.zeros((rows, cols, 3), dtype=np.float32)
        wsum = np.zeros((rows, cols, 1), dtype=np.float32)
        patches = work[cur[:, 0, None, None] + ii,
                       cur[:, 1, None, None] + jj]
        for k in range(na):
            gy0, gx0 = int(ay[k]) - y0, int(ax[k]) - x0
            py0, px0 = max(0, -gy0), max(0, -gx0)
            ly0, lx0 = max(0, gy0), max(0, gx0)
            ly1, lx1 = min(rows, gy0 + ps), min(cols, gx0 + ps)
            if ly1 <= ly0 or lx1 <= lx0:
                continue
            h, w = ly1 - ly0, lx1 - lx0
            ww = win[py0:py0 + h, px0:px0 + w] * ring_mask[ly0:ly1, lx0:lx1]
            acc[ly0:ly1, lx0:lx1] += patches[k, py0:py0 + h,
                                            px0:px0 + w] * ww[..., None]
            wsum[ly0:ly1, lx0:lx1] += ww[..., None]
        have = (wsum[..., 0] > 1e-6) & ring_mask
        if have.any():
            block[have] = acc[have] / wsum[have]
            filled |= have
            known[y0:y1, x0:x1] = filled
            work[y0:y1, x0:x1] = block

    if not filled.all():
        # 洞里别留着水印本身的颜色，给后面的扩散一个中性初值
        if filled.any():
            neutral = np.median(block[filled], axis=0)
        else:
            # 没填到任何像素时取框外紧邻一圈，别对整张图取中位数（大图几百 MB）
            strips = []
            if y0 > 0:
                strips.append(work[y0 - 1, x0:x1])
            if y1 < height:
                strips.append(work[y1, x0:x1])
            if x0 > 0:
                strips.append(work[y0:y1, x0 - 1])
            if x1 < width:
                strips.append(work[y0:y1, x1])
            neutral = (np.median(np.concatenate(strips), axis=0) if strips
                       else np.full(3, 128.0, dtype=np.float32))
        block[~filled] = neutral
    if filled.any():
        _reflect_seam_correction(arr, block, filled, y0, y1, x0, x1)
    result = np.clip(block, 0, 255)
    if np.issubdtype(arr.dtype, np.integer):
        result = np.rint(result)
    arr[y0:y1, x0:x1] = result
    return filled


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
    """auto：横带走竖列复制，竖条走横行复制，接近方形的框走中位数反射。"""
    if method != "auto":
        return method
    y0, y1, x0, x1 = box_pixels(box, width, height)
    rows = max(1, y1 - y0)
    cols = max(1, x1 - x0)
    if cols >= rows * 2:
        return "strip"
    if rows >= cols * 2:
        return "strip_v"
    return "mirror"


def inpaint(rgb: np.ndarray, box, method: str = "auto") -> np.ndarray:
    """把框覆盖的区域补掉，返回新的 uint8 RGB 数组。

    box 可以是一个 (x0, y0, x1, y1) 比例框，也可以是多个框的列表。
    框内内容完全丢弃；框外像素一个都不动。
    method 取 auto / strip / strip_v / mirror / patchmatch / smooth / lama。
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
        elif chosen in ("strip", "strip_v", "patchmatch"):
            if chosen == "strip":
                ok = _strip_fill(arr, y0, y1, x0, x1)
            elif chosen == "strip_v":
                ok = _strip_fill_v(arr, y0, y1, x0, x1)
            else:
                ok = _patchmatch_fill(arr, y0, y1, x0, x1)
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


def target_dir(path, out_dir=None, subdir: bool = False) -> Path:
    """单张图的输出目录。

    out_dir 为主时统一存那里；否则存原图旁边，subdir=True 再加一层
    `_去水印`。零散选来的若干张照片往往来自不同文件夹，所以必须按张算，
    不能拿第一张的目录当全体。
    """
    if out_dir:
        return Path(out_dir)
    parent = Path(path).parent
    return parent / "_去水印" if subdir else parent


def batch_remove(paths, box, out_dir=None, method: str = "auto",
                 quality: int = 95, cancel=None, on_progress=None,
                 subdir: bool = False) -> dict:
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
            target = output_path(path, target_dir(path, out_dir, subdir),
                                 unique=False)
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


def _tint_score(arr: np.ndarray, name: str) -> np.ndarray:
    """单个色系的“偏向程度”：正数越大表示越像那种淡色。"""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if name == "黄":
        return (r + g) * 0.5 - b
    if name == "蓝":
        return b - (r + g) * 0.5
    if name == "粉":
        return r - g
    if name == "绿":
        return g - (r + b) * 0.5
    return np.zeros(arr.shape[:2], dtype=np.float32)


def _tint_scores(rgb: np.ndarray) -> dict:
    arr = rgb.astype(np.float32)
    return {name: _tint_score(arr, name)
            for name in ("黄", "蓝", "粉", "绿")}


def looks_like_photo(rgb: np.ndarray, limit: int = 25) -> bool:
    """照片护栏：文档截图的中位色度近 0（白纸），人像/风景随便就 30~60。

    只抽 1/16 的像素就行——中位数对抽样不敏感，而全图算一遍要 0.8 秒：
    批量大图时这一步本来就是快进快出的分支，不值得付全价。
    """
    sample = rgb[::4, ::4]
    if not sample.size:
        return False
    r = sample[..., 0].astype(np.int16)
    g = sample[..., 1].astype(np.int16)
    b = sample[..., 2].astype(np.int16)
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    return float(np.median(high - low)) > limit


_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _luma_chroma(rgb: np.ndarray):
    """一次算好亮度与色度，返回 (luma float32, chroma uint8)。

    大图上最大的坑是 ``arr.max(axis=2)``：跨通道归约是跳着读的，12 MP 一次要
    3.5 秒，而一次一键抠会反复调它 8 次（detect_tint / 反解 / 灰水印 / 照片
    护栏各自算一遍）。换成“先转 int32 再两两比较”后同样的图只要 0.2 秒；
    亮度用 77/150/29 的整数权重（≈ Rec.601 × 256），和浮点公式差不到 1/255。
    """
    # tensordot 一次点乘三个通道（就是调 BLAS），12 MP 只要 0.09 秒；
    # 先转 int32 再整数加权要 0.60 秒（uint16 反而更慢，0.97 秒）。
    luma = np.tensordot(rgb, _LUMA_WEIGHTS, axes=([2], [0]))
    r8, g8, b8 = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    high = np.maximum(np.maximum(r8, g8), b8)
    low = np.minimum(np.minimum(r8, g8), b8)
    return luma, high - low


def _robust_threshold(score: np.ndarray, floor: float = 3.0,
                      cap: float = 8.0) -> float:
    """用中位数绝对偏差估噪声，阈值随图片自动定：干净白底上能抓到 ⩽3 的淡差。"""
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med)))
    return float(np.clip(4.0 * 1.4826 * mad, floor, cap))


def _local_trend(score: np.ndarray, valid: np.ndarray,
                 radius: int = 48) -> np.ndarray:
    """局部“纸面基准”：只用 valid 像素算的归一化均值趋势。

    score 减掉它就是相对周围纸面的偏色：整片彩色、渐变色块、整条高亮行
    都是低频，减完归零，不再把全图门限抬上去，极淡的水印（只差 1.5~3）
    也就不会再沉到门限以下；水印笔画是紧凑的高频结构，减完还剩大半。
    和 detect_light_in_box 里“全投影减去自身模糊”是同一个思路，只是这里
    在 1/8 分辨率上算趋势，12 MP 也只要几十毫秒。

    为什么不按 128×128 分块取中位数：块内取常数、或把块中位数插值回去，
    遇到“灰表头压着暖色纸”这种色调边界时整片暖纸相对基准都是正的（实测
    15% 的画面被判成淡黄）；均值趋势下大块区域的残差为零，只剩边界一圈。
    """
    height, width = score.shape
    step = 8 if min(height, width) >= 64 else 1
    small = np.asarray(score[::step, ::step], dtype=np.float32)
    vs = np.asarray(valid[::step, ::step], dtype=bool)
    if not vs.any():
        return np.zeros_like(score, dtype=np.float32)
    r = max(2, int(round(radius / step)))
    acc, cnt = _box_blur_known(small, vs, r)
    trend = acc[..., 0]
    # 窗口里没有 valid 像素（整片彩色/整块高亮）：退回全图中位，不能拿 0 当基准
    trend = np.where(cnt >= 1, trend, float(np.median(small[vs])))
    if step > 1:
        # 趋势本来就是低频量，最近邻放大到原尺寸：8 像素一级台阶只有零点几
        # 级色阶，够不着阈值
        trend = np.repeat(np.repeat(trend, step, axis=0), step, axis=1)
    return np.ascontiguousarray(trend[:height, :width], dtype=np.float32)


def detect_tint(rgb: np.ndarray, gmin: int = 120,
                luma_gate: float = 140.0, stats=None) -> str:
    """判断这张图上的浅色水印偏哪种颜色（黄/蓝/粉/绿），全没就返回空串。"""
    arr = rgb.astype(np.float32)
    g = arr[..., 1]
    luma, chroma = stats if stats is not None else _luma_chroma(rgb)
    pale = (luma > luma_gate) & (chroma < 45) & (g >= gmin)
    if not pale.any():
        return ""
    best, best_count = "", 0
    scores = _tint_scores(arr)
    for name, score in scores.items():
        # 减掉局部纸面趋势：整片彩色抬高不了门限，极淡的水印也能进投票
        resid = score - _local_trend(score, pale)
        limit = _robust_threshold(resid[pale], floor=2.0, cap=8.0)
        # 还要“绝对方向上真的偏这个色”：局部基准只能说明它比周围纸面淡，
        # 而彩色纸上的白字、高光边缘同样“更不像那个底色”（实测蓝灰底
        # 屏幕照片会被投成淡黄），它们自身的偏色其实是 0。
        count = int(((resid > limit) & pale & (score > 0.5)).sum())
        if count > best_count:
            best, best_count = name, count
    # 占比太小就不算水印（至少千分之一）
    return best if best_count >= max(200, luma.size // 1000) else ""


def detect_light_watermark(rgb: np.ndarray, diff=None, gmin: int = 120,
                           luma_gate: float = 140.0,
                           tint=None, stats=None) -> np.ndarray:
    """找浅色（淡黄/淡蓝/淡粉/淡绿）半透明水印的像素。

    判据：相对周围纸面的偏色 + 够亮 + 不饱和。低饱和是为了避开彩色 logo、
    选中框这类真·彩色元素；基准取局部纸面趋势（_local_trend），
    所以整片彩色（图表、高亮行）不再把门限抬高，交替底色的表格上也能抓住
    相对周围只差 1.5~3 的极淡水印；阈值仍由图自身的噪声自动定。
    """
    g = rgb[..., 1]
    luma, chroma = stats if stats is not None else _luma_chroma(rgb)
    pale = (luma > luma_gate) & (chroma < 45) & (g >= gmin)
    name = tint if tint is not None else detect_tint(rgb, gmin, luma_gate,
                                                    stats=(luma, chroma))
    if not name:
        return np.zeros(luma.shape, dtype=bool)
    arr = rgb.astype(np.float32)
    score = _tint_score(arr, name)
    if diff is not None:
        # 复检残影时调用方给的是绝对差门（要问“还剩多少偏色”），照旧直接用
        return (score > float(diff)) & pale
    resid = score - _local_trend(score, pale)
    limit = _robust_threshold(resid[pale], floor=2.0, cap=8.0)
    # 绝对方向门（同 detect_tint）：彩色底上的白字/高光“比周围纸面淡”，
    # 但它自己并不偏这个色，偏色量≈0，不能当成水印
    return (resid > limit) & pale & (score > 0.5)


def estimate_tint_ref(rgb: np.ndarray, mask: np.ndarray,
                      fallback=None, tint=None,
                      top_pct: float = 5.0) -> tuple:
    """从掩膜里最亮的一小撮像素自估水印参考色，替代写死的 TINT_REFS。

    取掩膜内亮度最高 5% 像素的中位色 —— 它们是「最淡的水印压在白底」，
    偏色方向可信、离群少（抗锯齿/黑字边缘被排除），但色度被 α 稀释过：
    直接拿它当 W 会让 (I-B)/(W-B) 系统性偏大，实测蓝色水印被过矫正到
    比原来更蓝。所以先估纸面亮度（浅色水印总有一个通道几乎不衰减，用
    观测色的最大通道近似），再把这一点偏色按固定参考色的色度幅度还原。
    结果是：色相方向跟图走（各家 App 水印色自适应），色度幅度与反解公式
    的标定保持一致（α 不漂移）。

    护栏：偏色太弱、放大倍数离谱、色相方向与固定色系相反、结果过暗时，
    一律退回固定参考色。
    """
    if fallback is None:
        fallback = TINT_REFS.get(tint, LIGHT_REF) if tint else LIGHT_REF
    base = np.asarray(fallback, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return tuple(float(v) for v in base)
    arr = np.asarray(rgb, dtype=np.float32)
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    vals, lum = arr[mask], luma[mask]
    # 12 MP 的扫描页上掩膜能有上百万像素，argpartition 逐个算要 2 秒；
    # 取“最亮的 5% 的中位色”只需要一个抽样代表值，封顶 20 万个像素足够。
    if vals.shape[0] > 200_000:
        pick = np.linspace(0, vals.shape[0] - 1, 200_000).astype(np.int64)
        vals, lum = vals[pick], lum[pick]
    k = min(vals.shape[0], max(1, int(np.ceil(vals.shape[0] * top_pct / 100.0))))
    obs = np.median(vals[np.argpartition(lum, -k)[-k:]], axis=0).astype(np.float32)
    paper = float(obs.max())
    delta = obs - paper                       # 相对纸面的偏色（各通道 ≤0）
    chroma_obs = float(-delta.min())
    chroma_base = float(base.max() - base.min())
    if chroma_obs < 2.0 or chroma_base < 2.0:
        return tuple(float(v) for v in base)
    scale = float(np.clip(chroma_base / chroma_obs, 0.5, 12.0))
    base_delta = base - float(base.max())
    denom = float(np.linalg.norm(delta) * np.linalg.norm(base_delta))
    if denom > 0 and float(np.dot(delta, base_delta) / denom) < 0.3:
        return tuple(float(v) for v in base)   # 偏色方向反了，多半是误检
    ref = np.clip(paper + delta * scale, 0.0, 255.0)
    if float(ref.min()) < 120.0:
        return tuple(float(v) for v in base)
    return tuple(float(v) for v in ref)


def _choose_tint_ref(rgb: np.ndarray, mask: np.ndarray, tint: str,
                     strength: float = 1.0) -> tuple:
    """在「固定参考色 / 自估参考色 / 两者折中」里选反解后残影最小的那个。

    α 与水印色度在单张图上无法分离（观测到的偏色是 α·(W-B)），所以纯自估
    偶尔会跑偏。这里用图自身做闭环：反解完看水印区还剩多少偏色（残影和
    过矫正都会让 |偏色| 变大），挑最小的。固定表因此只在确实更好时才被
    替换，标准四色不会有任何回归。
    """
    base = TINT_REFS.get(tint, LIGHT_REF)
    # 只比“哪个参考色反解后残影更小”，是均值型判定，不必上全分辨率：
    # 12 MP 整幅反解一次要 2 秒，三个候选就是 6 秒；抽 1/3 像素后 0.7 秒，
    # 选出的参考色再拿去全分辨率反解（这一步的精度才真正影响像素）。
    step = 3 if rgb.shape[0] * rgb.shape[1] > 2_000_000 else 1
    probe_rgb = rgb[::step, ::step] if step > 1 else rgb
    probe_mask = mask[::step, ::step] if step > 1 else mask
    if probe_mask.sum() < 200:
        probe_rgb, probe_mask = rgb, mask
    estimated = estimate_tint_ref(probe_rgb, probe_mask, tint=tint)
    mid = tuple((np.asarray(base, dtype=np.float32)
                 + np.asarray(estimated, dtype=np.float32)) * 0.5)
    best, best_cost = base, float("inf")
    for ref in (base, estimated, mid):
        fixed = undo_light_watermark(probe_rgb, strength, ref=ref, tint=tint,
                                     mask=probe_mask)
        cost = float(np.abs(_tint_score(fixed.astype(np.float32),
                                        tint)[probe_mask]).mean())
        if cost < best_cost:
            best, best_cost = ref, cost
    return best


def _run_lengths(mask: np.ndarray):
    """每个像素所在连续段有多长（横、竖各一张），用来区分“笔画”和“长直线”。"""
    height, width = mask.shape
    # int16 而不是默认的 int64：大图上这四道 accumulate 原本占 2.8 秒，窄类型
    # 把内存带宽降下来一半又一半（结果完全一样；超宽图才退回 int32）
    dtype = np.int16 if max(height, width) < 32000 else np.int32
    rows = np.broadcast_to(np.arange(width, dtype=dtype),
                           (height, width))
    row_start = np.maximum.accumulate(np.where(mask, 0, rows + 1), axis=1)
    run_row = rows - row_start + 1
    row_end = np.minimum.accumulate(
        np.where(mask, width, rows - 1)[:, ::-1], axis=1)[:, ::-1]
    run_row = np.minimum(run_row, row_end - rows + 1)

    cols = np.broadcast_to(np.arange(height, dtype=dtype)[:, None],
                           (height, width))
    col_start = np.maximum.accumulate(np.where(mask, 0, cols + 1), axis=0)
    run_col = cols - col_start + 1
    col_end = np.minimum.accumulate(
        np.where(mask, height, cols - 1)[::-1], axis=0)[::-1]
    run_col = np.minimum(run_col, col_end - cols + 1)
    return run_row, run_col


def _neutral_mask(rgb: np.ndarray, stats=None) -> np.ndarray:
    """淡灰色水印：没有偏色，只是比纸面暗一点点。

    难点：灰色网格线和灰色水印亮度几乎一样，只能看形状——网格线很长、
    笔画很短。所以要求横、竖两个方向的连续段都不长。
    """
    luma, chroma = stats if stats is not None else _luma_chroma(rgb)
    background = float(np.percentile(luma, 90))
    deficit = background - luma
    candidate = (chroma < 6.0) & (deficit > 1.5) & (deficit < 40.0)
    run_row, run_col = _run_lengths(candidate)
    # 60 像素以上的连续段算“长直线”（表格线、扫描行），不算笔画
    return candidate & (run_row < 60) & (run_col < 60)


def _near_dark(mask: np.ndarray, luma: np.ndarray, radius: int = 8) -> np.ndarray:
    """掩膜里的像素附近有没有深色内容（正文笔画、表格线、阴影）。

    重铺背景是从 ~mask 估出来的平滑色，铺到“字的抗锯齿边”上就是把字刷掉
    一半：合成文字页实测 10536 个 AA 像素被重铺，改动中位 73 级。AA 边在
    亮度/色度上和淡灰水印一模一样（中性、比纸暗二十几级），唯一的分界是
    位置——水印盖在空白纸上，AA 边紧贴笔画。
    大图降到 1/2 分辨率算（同 _flatten_background 与残影补洞那两处），半径
    同步减半；只看附近的已知像素（掩膜自身不算“深色内容”，否则粗笔画会
    自己把自己挡住）。
    """
    known = ~mask
    step = 2 if luma.size > 4_000_000 else 1
    known_small = known[::step, ::step]
    dark_small = (luma[::step, ::step] < 145) & known_small
    frac = _box_blur_known(dark_small.astype(np.float32), known_small,
                           max(3, radius // step))[0][..., 0]
    if step > 1:
        frac = _upsample2(frac, luma.shape)
    return frac > 0.0


def _nearest4_fill(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """四方向“最近的已知像素”取中位数。

    只对掩膜里的像素取值：笔画穿过表格线时，上下给的是白底、左右给的是线色，
    中位数落回白底，不会像各向扩散那样把线色刷成一块。
    """
    height, width = mask.shape
    if not mask.any():
        return np.array(arr, dtype=np.uint8, copy=True)
    known = ~mask
    ys, xs = np.nonzero(mask)
    # 只算“有水印像素的那些行/列”。全图建索引再 accumulate 要开 4 张 12 MP
    # 的 int64 数组（每张 96 MB），实测 4.4 秒；而一张扫描页上的残影往往
    # 只占几十行，抽出来算不到 0.4 秒，结果逐像素一致。
    rows = np.unique(ys)
    cols = np.unique(xs)
    row_pos = np.searchsorted(rows, ys)
    col_pos = np.searchsorted(cols, xs)
    idx_w = np.arange(width, dtype=np.int32)
    idx_h = np.arange(height, dtype=np.int32)
    sub_rows = known[rows]
    left = np.maximum.accumulate(np.where(sub_rows, idx_w, -1), axis=1)[row_pos, xs]
    right = np.minimum.accumulate(np.where(sub_rows, idx_w, width)[:, ::-1],
                                  axis=1)[:, ::-1][row_pos, xs]
    sub_cols = known[:, cols]
    up = np.maximum.accumulate(np.where(sub_cols, idx_h[:, None], -1),
                               axis=0)[ys, col_pos]
    down = np.minimum.accumulate(np.where(sub_cols, idx_h[:, None], height)[::-1],
                                 axis=0)[::-1][ys, col_pos]
    cands = []
    for yy, xx, ok in ((ys, left, left >= 0),
                       (ys, right, right <= width - 1),
                       (up, xs, up >= 0),
                       (down, xs, down <= height - 1)):
        pick = arr[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)]
        cands.append(np.where(ok[:, None], pick, np.nan).astype(np.float32))
    # 手写 NaN 中位数：np.nanmedian 内部走 masked array，12 MP 上仅这一层
    # 包装就要 1 秒。把 NaN 换成 +inf 排序后取中间两个即可（4 个候选）。
    stack = np.stack(cands)
    finite = ~np.isnan(stack[..., 0])
    count = finite.sum(axis=0)
    stack[~finite] = np.inf
    stack.sort(axis=0)
    lo = np.maximum((count - 1) // 2, 0)
    hi = np.maximum(count // 2, 0)
    med = (np.take_along_axis(stack, lo[None, :, None], axis=0)[0]
           + np.take_along_axis(stack, hi[None, :, None], axis=0)[0]) * 0.5
    out = arr.copy()
    good = count > 0
    out[ys[good], xs[good]] = np.clip(med[good], 0, 255).astype(np.uint8)
    return out


def _undo_exclude(arr: np.ndarray, ref: np.ndarray, luma: np.ndarray,
                  mask) -> np.ndarray:
    """估局部背景时要排除的像素：给了掩膜就用它，没给就先粗找一遍。

    背景里一旦混进水印色，B̂ 被拉向水印，α 系统性偏小、解不干净，所以
    排除区必须比掩膜略大（膨胀 2 像素把抗锯齿边缘一起排除）。
    没传掩膜时用最亮的一成像素当纸面粗估 α，把明显偏色的像素当排除区：
    只是个引导量，不求准，够让第一版 B̂ 落在纸上就行。
    """
    if mask is not None:
        return _grow_mask(np.asarray(mask, dtype=bool), 2)
    paper = np.median(arr[luma >= np.percentile(luma, 90.0)], axis=0)
    w = ref - paper
    den = float(np.dot(w, w))
    if den < 64.0:
        return np.zeros(luma.shape, dtype=bool)
    return (arr - paper) @ w / den > 0.05


def _local_paper(arr: np.ndarray, unknown: np.ndarray, radius: int = 12):
    """只用水印外的已知像素估局部底层色 B̂（纸面、灰表头、暖色底都算进来）。

    _box_blur_known 是归一化盒滤波：窗口里没有已知像素的位置会摊到最近的
    已知色上，所以水印笔画很密（60 px 大字）时也能越过笔画取到两侧纸面。
    大图降一半分辨率算——B̂ 本来就是平滑量，省掉 3/4 的积分图开销。
    返回 (background, count)；没有可用像素时返回 None。
    """
    known = ~unknown
    height, width = arr.shape[:2]
    if int(known.sum()) < 64:
        return None
    if known.size <= 1_000_000:
        return _box_blur_known(arr, known, radius)
    # B̂ 是“这张纸是什么颜色”级别的平滑量：1/2、1/4 分辨率都够用，而三通道
    # 积分图的代价按面积走，12 MP 上 1/4 比分半又快一倍
    step = 2 if known.size <= 4_000_000 else 4
    sub = arr[::step, ::step]
    background, count = _box_blur_known(sub, known[::step, ::step],
                                        max(2, radius // step))
    background = np.repeat(np.repeat(background, step, axis=0), step, axis=1)
    count = np.repeat(np.repeat(count, step, axis=0), step, axis=1)
    return background[:height, :width], count[:height, :width]


def undo_light_watermark(rgb: np.ndarray, strength: float = 1.0,
                        ref=None, tint=None, mask=None) -> np.ndarray:
    """把半透明浅色水印“解”掉，而不是“补”掉。

    I = (1-α)·B + α·W，逐像素用局部背景 B̂ 做投影估 α：
    α = ⟨I-B̂, W-B̂⟩ / ‖W-B̂‖²，再反解 B = (I - αW)/(1-α)。

    为什么必须用局部 B̂：原先把背景当中性色，直接拿 (R-B)/(Wr-Wb)、
    (G-B)/(Wg-Wb) 的均值估 α，等于丢掉 (1-α)(Br-Bb) —— 灰表头、淡蓝底、
    暖色纸上 α 全估偏，只能靠 clip((luma-200)/40, 0.35, 1) 硬压修正强度，
    表头亮度 210~225 时只有 0.35 的力度，黄印解不干净、还留一道浅影。
    换成局部背景后，背景自身的色偏被投影吸收，那个补偿系数整个不需要了。

    中性变暗（灰色表格线、阴影）不能当成水印：它的色度偏离是 0，投影却
    因为 ‖W-B̂‖ 里那个大负分量而偏大，所以再加一道色度同向门——只认色度
    往 W 那边偏的像素，正文和表格线压根不动。
    strength 是修正强度（1.0 全解，0.5 只解一半）。
    ref 显式传入时优先（传自估值）；否则按 tint 查固定表。mask 是水印像素
    （不给就自己粗找，见 _undo_exclude）。
    """
    if ref is None:
        ref = TINT_REFS.get(tint, LIGHT_REF) if tint else LIGHT_REF
    arr = rgb.astype(np.float32)
    wref = np.asarray(ref, dtype=np.float32)
    luma = np.tensordot(arr, _LUMA_WEIGHTS, axes=([2], [0]))
    unknown = _undo_exclude(arr, wref, luma, mask)
    paper = _local_paper(arr, unknown, 12)
    if paper is None:
        return np.array(rgb, dtype=np.uint8, copy=True)   # 全图都是“水印”：无从反解
    background, count = paper
    lost = count <= 0
    if lost.any():
        # 窗口里一个已知像素都没有（整片都被排除）：退回整图中位色，
        # 别拿积分图的 0 当纸面（那会把像素解到溢出）
        rest = arr[~unknown]
        background[lost] = (np.median(rest, axis=0) if rest.size
                            else arr.mean(axis=(0, 1)))
    w = wref - background
    den = np.einsum("ijk,ijk->ij", w, w)
    dev = arr - background
    dot = np.einsum("ijk,ijk->ij", dev, w)
    alpha = np.clip(dot / np.maximum(den, 64.0) * float(strength), 0.0, 0.6)
    alpha[(luma < 150.0) | (alpha < 0.015)] = 0.0
    if mask is None:
        # 没有掩膜时会整幅反解，必须挡掉“中性变暗”（灰色表格线、阴影）：
        # 它们的色度偏离是 0，投影却因为 ‖W-B̂‖ 里那个大负分量而偏大。
        # 色度同向门 = Σ_c(dev_c-d̄)(w_c-w̄) = dev·w - 3·d̄·w̄，不用另开数组。
        # 有掩膜时已经限定在掩膜内，这一趟 12 MP 的求均值可以省掉。
        chroma_dot = dot - 3.0 * (dev.mean(axis=2) * w.mean(axis=2))
        alpha[chroma_dot <= 0.0] = 0.0
    if mask is not None:
        # 只在“水印像素（含抗锯齿外扩一圈）”上反解：局部背景在色调边界上
        # 必然估不准，放开就是边界两侧互相“纠正”、把好像素洗出色偏。
        # 掩膜外的像素本来就该一个都不动。
        alpha[~unknown] = 0.0
    denom = np.clip(1.0 - alpha, 0.05, 1.0)
    # 逐通道就地在 arr 上算：(arr - α·W) / (1-α)。原来走的是
    # (arr - alpha[...,None]*W)/denom[...,None]，要多开两个 12 MP×3 通道的
    # 临时数组（各 144 MB），整次反解 0.7 秒里一半花在分配和填零上。
    for ch, wc in ((arr[..., 0], wref[0]), (arr[..., 1], wref[1]),
                   (arr[..., 2], wref[2])):
        np.subtract(ch, alpha * wc, out=ch)
        np.divide(ch, denom, out=ch)
    np.clip(arr, 0, 255, out=arr)
    # 护栏：反解不该把像素打得比原像素暗太多（参考色估错、蓝灰底被误认成
    # 蓝水印时，(I-αW)/(1-α) 会把背景打成近黑）。暗超 40 的恢复原值。
    fixed_luma = np.tensordot(arr, _LUMA_WEIGHTS, axes=([2], [0]))
    too_dark = (alpha > 0) & (fixed_luma < luma - 40.0)
    arr[too_dark] = rgb[too_dark]
    return arr.astype(np.uint8)


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
    # 窗口是连续的：用切片取四个角，不要用 fancy 索引（
    # total[y1][:, x1] 会先把整行块拷一遍，12 MP 上四个角就是四次全图拷贝，
    # 原实现光这一步要 8 秒）。pad = radius + 1，所以左上从 0 开始、
    # 右下从 2·radius+2 开始，四个切片直接相减即可。
    span = 2 * radius + 2
    def rect(total):
        return (total[span:span + height, span:span + width]
                - total[:height, span:span + width]
                - total[span:span + height, :width]
                + total[:height, :width])

    acc = rect(sum_a)
    cnt = rect(sum_k)
    return acc / np.maximum(cnt, 1e-6)[..., None], cnt


def _upsample2(small: np.ndarray, shape) -> np.ndarray:
    """最近邻放大两倍并裁到原尺寸（只用于“周围背景”这种平滑量）。"""
    big = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    return big[:shape[0], :shape[1]]


def _local_std(gray: np.ndarray, radius: int = 6) -> np.ndarray:
    """局部标准差（判“这块是不是平坦”用）。

    大图上降到 1/2 分辨率、半径同步减半：它只用来看“纹理多不多”，粗一倍
    不影响门限，却能省掉 3/4 的积分图开销（12 MP 上两次累加和约 1 秒）。
    """
    step = 2 if gray.size > 4_000_000 else 1
    small = gray[::step, ::step] if step > 1 else gray
    r = max(2, int(round(radius / step)))
    known = np.ones(small.shape, dtype=bool)
    mean = _box_blur_known(small, known, r)[0][..., 0]
    mean2 = _box_blur_known(small * small, known, r)[0][..., 0]
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    if step == 1:
        return std
    std = np.repeat(np.repeat(std, step, axis=0), step, axis=1)
    return std[:gray.shape[0], :gray.shape[1]]


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
    luma = fixed.mean(axis=2, dtype=np.float32)
    # “周围背景”本来就要平滑，降到 1/2 分辨率算（半径同步减半）再放大回来，
    # 12 MP 上两次三通道积分图从 4.5 秒降到 1.2 秒，肉眼看不出差别。
    step = 2 if luma.size > 4_000_000 else 1
    if step > 1:
        background, count = _box_blur_known(
            fixed[::step, ::step], known[::step, ::step], 6)
        background = _upsample2(background, fixed.shape[:2])
        # 半分辨率上抽样的 known 与全分辨率不是一一对应的：某个位置周围
        # 其实有已知像素，它的抽样格却可能整格都是“未知”。计数为 0 的格子
        # 拿到的是积分图的 0，铺上去就是纯黑——所以计数也要一起放大回来。
        count = _upsample2(count, fixed.shape[:2])
    else:
        background, count = _box_blur_known(fixed, known, 12)
    std = _local_std(luma, 6)
    # count == 0 表示窗口里一个已知像素都没有（周围一整圈都是掩膜），
    # 这时盒滤波拿到的是积分图的 0，铺上去就是一片黑——“打黑出横杠”的
    # 根源之一，直接不填。
    flat = ((std < 0.8) & ~_near_dark(grow, luma, 10) & (luma > 195) & grow
            & (count > 0))
    out = fixed.astype(np.float32)
    out[flat] = background[flat]
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# ---------------------------------------------------------------- 框内反解（照片水印）
# 半透明浅色水印压在照片上时，全图反解有两个死结：一是背景不再是白纸，
# 全局门限（亮度>140、色度<45）抓不到压在深色草地/肤色上的黄字；二是照片
# 护栏为保皮肤、汗光、天空直接放过整张图。所以改成“框内”做：框由用户拖、
# 或由下面的网格检测给出；框内用框外一圈的中位色当地背景，逐像素按“偏离
# 背景朝水印色方向的投影”估 α 再反解，框外一个像素不动。
#
# 投影分两路：全投影 (I-B)·(W-B)/|W-B|，像素 I=(1-α)B+αW 时正好等于
# α·|W-B|（反解用）；色度投影是去掉亮度分量后再算，用来认——亮度渐变
# （比如天空由亮到暗）会让全投影误报，色度投影对它免疫；反过来，全投影
# 的符号还能挡掉“变暗的树叶”这类反向偏离。

# 框内反解的候选色系：标准四色 + 深黄。品牌黄水印比表里的浅黄深得多
# （如 (255,226,130)），只用标准色投票时暖墙上的深黄只能检出几十像素、
# 直接被数量门拒掉；多一个候选，投票自己会选更贴近的那支。
BOX_TINT_REFS = {
    "黄": (252, 248, 196),
    "深黄": (255, 226, 130),
    "蓝": (196, 236, 252),
    "粉": (252, 214, 238),
    "绿": (214, 250, 214),
}


def _box_ring_mask(shape, y0, y1, x0, x1, pad):
    """框外一圈的布尔掩膜（供估计局部背景用）。"""
    height, width = shape
    ring = np.zeros((height, width), dtype=bool)
    ry0 = max(0, y0 - pad)
    ry1 = min(height, y1 + pad)
    rx0 = max(0, x0 - pad)
    rx1 = min(width, x1 + pad)
    ring[ry0:ry1, rx0:rx1] = True
    ring[y0:y1, x0:x1] = False
    return ring


def _local_background(arr, ring):
    """环内像素逐通道中位数（抗离群：环里混进一点文字/树叶不影响中位）。"""
    if not ring.any():
        return None
    vals = arr[ring]
    if vals.shape[0] < 20:
        return None
    return np.median(vals, axis=0).astype(np.float32)


def _blur3(arr):
    """3×3 均值模糊（edge padding）：算投影前先压一遍纹理噪声。

    水印笔画是低频大结构，模糊不伤它；草叶、噪点是高频，模糊后投影的
    噪声能降到三分之一，阈值才立得住。
    """
    if arr.ndim == 2:
        arr = arr[..., None]
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
           + pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
           + pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]) / 9.0
    return out


def _blur_f(arr2d, radius):
    """2D 浮点数组的大半径均值模糊（积分图实现，边缘按有效窗口平均）。

    用来给投影去趋势：亮度渐变是低频，模糊后就是它自己，相减归零；
    水印笔画是紧凑的高频结构，模糊后只剩一小部分，相减后还留大半。
    （PIL 的 BoxBlur 不支持 F 模式，所以这里手写积分图。）
    内存：float64 积分图，12MP 图上单格约 50MB，4 色循环峰值约 100MB
    临时内存；框很大时留意。
    """
    if radius <= 0:
        return np.asarray(arr2d, dtype=np.float32)
    a = np.asarray(arr2d, dtype=np.float64)
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - radius, 0, h - 1)
    y1 = np.clip(ys + radius, 0, h - 1)
    x0 = np.clip(xs - radius, 0, w - 1)
    x1 = np.clip(xs + radius, 0, w - 1)
    s = (ii[y1 + 1][:, x1 + 1] - ii[y0][:, x1 + 1]
         - ii[y1 + 1][:, x0] + ii[y0][:, x0])
    cnt = (y1 - y0 + 1)[:, None] * (x1 - x0 + 1)[None, :]
    return (s / cnt).astype(np.float32)


def _projections(arr, bg, ref):
    """逐像素算两路投影，返回 (全投影, 色度投影)。

    全投影 = (I-B)·(W-B)/|W-B|；色度投影是去掉亮度分量后再算，亮度渐变
    投出来是 0。
    """
    w = np.asarray(ref, dtype=np.float32) - bg
    norm = float(np.sqrt(np.dot(w, w)))
    if norm < 1.0:
        shape = arr.shape[:2]
        return (np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32))
    dev = arr - bg
    full = (dev * w).sum(axis=2) / norm
    wc = w - w.mean()
    norm_c = float(np.sqrt(np.dot(wc, wc)))
    if norm_c < 1.0:
        return full, np.zeros(arr.shape[:2], dtype=np.float32)
    dev_c = dev - dev.mean(axis=2, keepdims=True)
    chroma = (dev_c * wc).sum(axis=2) / norm_c
    return full, chroma


@dataclass
class LightDetection:
    """一处浅色水印的检测结果：检测器只吐这个，重建只认这个。

    box 是归一化框；tint 是赢的色系（"" 表示没认出）；mask 全图尺寸
    （只有框内可能为真）；score 是检出强度——框内检出像素的中位去趋势
    投影除以阈值，>1 才过线，越大越笃定；source 说明框从哪来
    （"box" 用户框选 / "grid" 一键抠网格 / "doc" 文档全局）；
    rejected 是拒认原因（"" 表示没拒；只有“平铺水印占比超 70%”才会拒，
    其余情况没认出就是没认出，不算拒）；bg 是这次检测用的背景色（RGB
    三元组 float，None 表示没检出）——反解直接用它，不再重算环背景。
    """

    box: tuple
    tint: str = ""
    mask: np.ndarray = field(default=None, repr=False)
    score: float = 0.0
    source: str = "box"
    rejected: str = ""
    bg: tuple = None

    @property
    def found(self) -> bool:
        return bool(self.tint) and self.mask is not None and bool(self.mask.any())


@dataclass
class LightStatus:
    """一张图的处理结论：只报结果，不报过程。

    status：clean（水印已清，纹理保留）；partial（反解没解干净，不得不
    整块补图，纹理被重建）；unchanged（没找到水印，什么都没动）；
    rejected（平铺水印或框过大，明确不处理）。detail 是一句人话。
    """

    status: str  # "clean" | "partial" | "unchanged" | "rejected"
    detail: str = ""


def detect_light_in_box(rgb, box, tint=None, source="box"):
    """框内局部检测浅色水印，返回 LightDetection。

    背景取框外一圈中位色，色调在框内投票，阈值按框内投影分布自定——
    压在深色草地/肤色上的黄字也能被认出来，全图投“蓝”的蓝天照片不再
    把黄水印带跑。认不出（占比太小）时 tint 为 ""、mask 全假；
    占比超过七成（多半是整块底色被当成水印，比如一整片黄沙滩）时
    rejected 记“平铺水印”，同样不认。

    两遍：第一遍用框外环背景投票；检出后，用掩膜外一圈（膨胀 10px
    减掉掩膜本身）重估背景，再投一次色系。格子大、背景有渐变时环中
    位色会偏（sky 蓝的格子环中位色比文字处真实背景 R 低 16），偏了的
    偏差近乎无色，色度符号门会把真水印误杀（蓝压蓝天被投成“绿”）。
    掩膜紧贴的邻域才是文字处真正的背景，第二遍的色系和背景都以它
    为准。邻域像素太少（<100）时不重投，留第一遍的结果。
    第二遍没检出时也留第一遍——新背景只影响这一格的判断，不降级。
    """
    height, width = rgb.shape[:2]
    y0, y1, x0, x1 = box_pixels(box, width, height)
    empty = np.zeros((height, width), dtype=bool)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return LightDetection(box, source=source)
    arr = rgb.astype(np.float32)
    pad = max(8, int(min(y1 - y0, x1 - x0) * 0.25))
    ring = _box_ring_mask((height, width), y0, y1, x0, x1, pad)
    bg = _local_background(arr, ring)
    if bg is None:
        # 框贴边、圈外没像素：退回框的最外一圈当背景
        edge = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        bg = np.median(arr[y0:y1, x0:x1][edge], axis=0).astype(np.float32)

    def run(bg):
        """用给定背景投一次色系，返回 LightDetection（bg 记这次用的）。"""
        crop = _blur3(arr[y0:y1, x0:x1])
        radius = max(8, min(crop.shape[:2]) // 8)
        names = (tint,) if tint else tuple(BOX_TINT_REFS)
        best_name, best_mask, best_count, best_score = "", empty, 0, 0.0
        for name in names:
            ref = BOX_TINT_REFS[name]
            full, chroma = _projections(crop, bg, ref)
            # 判据 = 去趋势后的全投影：亮度渐变（天空、墙面）被模糊吃掉，
            # 只剩紧凑的水印笔画；再加色度投影 >0 的符号门，挡掉亮绿草叶
            # 这类“只亮不带黄色调”的偏离（暖墙上的黄水印色度信号本来就弱，
            # 只看色度强度会漏，所以色度只作符号门，不作阈值）。
            score = full - _blur_f(full, radius)
            limit = _robust_threshold(score, floor=5.0, cap=12.0)
            # α 门：只有半透明（α≤0.75）才配反解。白字徽标这类不透明的
            # （α≈1）投影也很大，但底下什么都没剩、反解不出东西，只能整块补；
            # 逐像素筛，混在一个框里的白字和水印能各走各的路。α 用没模糊过的
            # 原图算：模糊会把笔画边的 α 稀释到 0.5 左右，白字轮廓就漏网了。
            local = (score > limit) & (chroma > 0.0)
            if local.any():
                raw, _ = _projections(arr[y0:y1, x0:x1], bg, ref)
                norm = float(np.sqrt(np.dot(
                    np.asarray(ref, dtype=np.float32) - bg,
                    np.asarray(ref, dtype=np.float32) - bg)))
                local &= (raw / max(norm, 1.0) <= 0.75)
            count = int(local.sum())
            # 偏离幅度门（逐色系）：这个色系的掩膜相对背景的中位偏离太小，
            # 说明它只是底色本身的波动，不是真水印——不让它参与赢家投票。
            # 否则一个假阳性色系会靠数量赢过真水印，再在事后把整格判死
            # （第四批实测：小格子里的整掩膜中位偏离——假阳性底色只有
            # 个位数，真水印 47~119；那是小格口径，不是本框的量法）。
            # 门限 15→8（第十一批）：用户真图大框上极淡黄水印的最强信号
            # 像素偏离只有 10.2（α 很低、颜色贴近底色），15 会把它连同底色
            # 一起拦掉；降到 8 后黄（10.2）能过，矩阵 24 格仍 0 不合格、
            # 远角损伤 0.00，未见误伤。黄与绿（10.3）的胜负只差约 1% 像素，
            # 仍偏脆弱，留待后续观察。
            if count:
                # 偏离门改看信号最强的像素：掩膜常混进大量底色渐变像素，
                # 整掩膜中位被底色拉走（实测黄掩膜 27.8 万像素、中位偏离
                # 仅 6.4）；真水印只占其中一小撮笔画。取 score 最高的前
                # 10%（至少 100 像素）算中位色，再判偏离与亮度。
                sub_scores = score[local]
                sub = arr[y0:y1, x0:x1][local]
                k = max(100, count // 10)
                if count > k:
                    idx = np.argpartition(sub_scores, -k)[-k:]
                    sub = sub[idx]
                med = np.median(sub, axis=0)
                dev = float(np.max(np.abs(med - bg)))
                if dev < 8.0:
                    continue
                # 亮度门（逐色系）：掩膜中位亮度比背景暗太多，说明是深色
                # 文字/内容，不是浅色水印。实测：真实裁块的深色 SQL 文字
                # 亮度下降约 45~144，淡黄水印只降 9~17，矩阵 24 格水印全是
                # 更亮（下降为负）；门限 30 介于两者之间。
                bg_luma = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                med_luma = 0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2]
                if bg_luma - med_luma > 30.0:
                    continue
                # 掩膜上限门（逐色系）：单个色系掩膜占框超 30%，说明它是底色
                # 渐变、不是水印——不让它参与赢家投票。否则蓝灰底会靠数量压过
                # 真水印（实测：用户真图框里蓝掩膜占框 34.6%，黄水印只有几万像素）。
                if count > (y1 - y0) * (x1 - x0) * 0.30:
                    continue
            if count > best_count:
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[y0:y1, x0:x1] = local
                best_name, best_mask, best_count = name, full_mask, count
                best_score = (float(np.median(score[local])) / limit
                              if local.any() else 0.0)
        # 占比太小不算水印（框内至少千分之一且 ≥100 像素）；占比超过七成
        # 多半是整块底色被当成水印（比如一整片黄沙滩），也不认、且明确拒。
        # 注：去趋势会把均匀填充吃掉，均匀色块的掩膜到不了七成，这条目前
        # 几乎触发不了，作安全网保留。
        bg_tuple = tuple(float(v) for v in bg)
        total = (y1 - y0) * (x1 - x0)
        if best_count < max(100, total // 1000):
            return LightDetection(box, source=source, bg=bg_tuple)
        if best_count > total * 0.7:
            return LightDetection(box, source=source, bg=bg_tuple,
                                  rejected="平铺水印（占比超 70%），不处理")
        return LightDetection(box, tint=best_name, mask=best_mask,
                              score=best_score, source=source, bg=bg_tuple)

    det1 = run(bg)
    if not det1.found:
        return det1
    # 第二遍：掩膜外一圈重估背景，再投一次色系。_grow_mask 用 np.roll
    # 膨胀，贴图边的掩膜会从对边卷进几个像素，中位数扛得住这点污染。
    nb = _grow_mask(det1.mask, 10) & ~det1.mask
    if int(nb.sum()) < 100:
        return det1
    bg2 = np.median(arr[nb], axis=0).astype(np.float32)
    if float(np.max(np.abs(bg2 - bg))) <= 5.0:
        # 背景几乎没动：环背景本来就准，换背景只会让 α 估计多一点
        # 波动（紧框在 sky 上实测差 1~2 个点），不重投，留第一遍。
        # 格子的大偏差（R 差 16）照修——阈值 5 介于两者之间。
        return det1
    det2 = run(bg2)
    return det2 if det2.found else det1


def _estimate_box_ref(arr, det, fixed_ref):
    """从检出掩膜估计水印参考色：方向取实测色，幅度取写死表。

    写死色是“标准”水印色，真水印往往比它淡得多（拍屏幕照片的黄水印
    中位色 (194,206,182)，写死的“深黄”是 (255,226,130)，R 高出 60，
    反解时 R 会被多减、掩膜区越解越暗）。直接拿中位色当参考色也不行：
    它是水印和背景的混合色，幅度太小，α 被高估、框误差反而变差
    （矩阵实测 3~10 倍）。折中：方向（色相）信实测色，幅度
    （|W-B|）信写死表——α 标定不变，R 不再多减。护栏：中位偏离太小
    或方向与写死色相反时返回 None（调用方退回写死表）。
    方向取法分两档：整掩膜中位偏离 >= 15 时用它（掩膜干净、中位
    就是水印色，最准）；偏离 < 15 时掩膜里混进大量底色像素、中位
    被底色拉走（用户真图黄掩膜 39.5 万像素、中位与背景只差 (7,4,-1)，
    估出的参考色贴近底色、反解残留 38%），改取偏离背景最大的前 10%
    （至少 100 像素）的中位色——那才是水印笔画本身。
    注：这里的 15 只是“中位色是否可信”的干净掩膜线，与检测偏离门
    （现为 8）用途不同，数值相同纯属巧合，改一处时别误改另一处。
    """
    if det.mask is None or not det.mask.any() or det.bg is None:
        return None
    bg = np.asarray(det.bg, dtype=np.float32)
    sub = np.asarray(arr, dtype=np.float32)[det.mask]
    med = np.median(sub, axis=0)
    if float(np.max(np.abs(med - bg))) < 15.0:
        dev = np.max(np.abs(sub - bg), axis=1)
        k = max(100, len(sub) // 10)
        if len(sub) > k:
            sub = sub[np.argpartition(dev, -k)[-k:]]
        med = np.median(sub, axis=0)
    d = med - bg
    n = float(np.sqrt(np.dot(d, d)))
    wv = np.asarray(fixed_ref, dtype=np.float32) - bg
    m = float(np.sqrt(np.dot(wv, wv)))
    if n < 1.0 or m < 1.0:
        return None
    if float(np.dot(d, wv)) / (n * m) < 0.0:
        return None
    est = bg + d / n * m
    return tuple(float(v) for v in np.clip(est, 0, 255))


def undo_light_in_box(rgb, box, tint=None, ref=None, strength=1.0):
    """框内反解浅色水印。

    返回 (fixed, mask, residual)：fixed 是全图 uint8（只有框内动过），
    mask 是框内检出的水印像素，residual 是反解后框内仍检出的水印像素
    （用来决定要不要再补）。框太小、框占图超过七成、或框内认不出水印
    时原样返回，不动任何像素。
    """
    height, width = rgb.shape[:2]
    y0, y1, x0, x1 = box_pixels(box, width, height)
    empty = np.zeros((height, width), dtype=bool)
    unchanged = np.array(rgb, dtype=np.uint8, copy=True)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return unchanged, empty, empty
    if (y1 - y0) * (x1 - x0) > height * width * 0.7:
        return unchanged, empty, empty
    det = detect_light_in_box(rgb, box, tint=tint)
    if not det.found:
        return unchanged, empty, empty
    name, mask = det.tint, det.mask
    arr = rgb.astype(np.float32)
    crop = arr[y0:y1, x0:x1]
    sub = mask[y0:y1, x0:x1]
    if ref is None:
        # 参考色：方向取掩膜中位色（实测水印色相），幅度取写死表
        # （α 标定）。_choose_tint_ref 是给白底文档做的（按“纸面亮度”
        # 自估），用在照片框上同样会估跑偏。估不出时退回写死表。
        fixed = BOX_TINT_REFS.get(name, LIGHT_REF)
        ref = _estimate_box_ref(arr, det, fixed) or fixed
    # 反解用检测那遍的背景：框外环背景在大格子 + 渐变底色上会偏
    # （sky 蓝的格子环中位色比文字处 R 低 16），偏了的偏差近乎无色，
    # α 估计和色度判断都跟着错。检测里第二遍已用掩膜外一圈重估过
    # 背景，直接用它；没给时才退回环背景。
    bg = (np.asarray(det.bg, dtype=np.float32) if det.bg is not None
          else _local_background(
              arr, _box_ring_mask((height, width), y0, y1, x0, x1,
                                  max(8, int(min(y1 - y0, x1 - x0) * 0.25)))))
    if bg is None:
        return unchanged, empty, empty
    w = np.asarray(ref, dtype=np.float32) - bg
    norm = float(np.sqrt(np.dot(w, w)))
    if norm < 1.0:
        return unchanged, empty, empty
    smooth = _blur3(crop)
    full, _ = _projections(smooth, bg, ref)
    alpha = np.clip(full / norm, 0.0, 0.6) * float(strength)
    alpha[~sub] = 0.0
    denom = np.clip(1.0 - alpha, 0.05, 1.0)
    fixed_crop = crop.copy()
    for ch in range(3):
        fixed_crop[..., ch] = (crop[..., ch] - alpha * float(ref[ch])) / denom
    # 护栏：反解不该把像素打得比背景暗太多。参考色/α 一旦估错
    # （比如蓝灰底被误认成蓝水印、亮参考色反解），(I-αW)/(1-α) 会
    # 把背景打成近黑的横杠。反解后的合理结果应贴近背景 bg；比 bg
    # 暗超过 40 的像素视为估错，直接保留原值。
    bg_luma = float(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    fixed_luma = (0.299 * fixed_crop[..., 0] + 0.587 * fixed_crop[..., 1]
                  + 0.114 * fixed_crop[..., 2])
    too_dark = sub & (fixed_luma < bg_luma - 40.0)
    fixed_crop[too_dark] = crop[too_dark]
    out = arr.copy()
    out[y0:y1, x0:x1] = np.clip(fixed_crop, 0, 255)
    fixed = np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    # 反解后框内再检一次：还有多少没解掉（没检出就是空掩膜，不是 None）
    res_det = detect_light_in_box(fixed, box, tint=name)
    residual = res_det.mask if res_det.mask is not None else empty
    return fixed, mask, residual


def reconstruct_light(arr, box, det=None, method="auto", allow_inpaint=True,
                      source="box"):
    """重建决策：检测和重建之间唯一的分叉点。

    输入检测记录（没给就现检），按同一套规则决定反解、补残影还是整块补，
    框选路径和一键抠网格都走这里，不再各自写一套“反解还是补”的判断：
    - 没检出 → ("none", "未找到水印")，原样返回；
    - 检测被拒（平铺水印）→ ("rejected", 拒认原因)，不碰；
    - mask 太小（<200 像素且 < 框面积 2%）→ 不透明徽标，整块补
      （allow_inpaint=False 时原样返回，action "none"）；
    - 反解后无残留 → ("unblend", "反解干净")；
    - 残留 ≤30% → 小补残影 ("fill", "反解后残留 X%，已小补")；
    - 残留 >30% → 整块补 ("inpaint-residual", ...)，纹理被重建；
    - 框占图超 70%（undo 内部拒掉）→ ("rejected", "框过大，不处理")。
    返回 (fixed, action, detail)：action 是 unblend / fill /
    inpaint-opaque / inpaint-residual / none / rejected，detail 是一句人话。
    det 是在原图上探出的检测（remove_light_file 的框路径先在原图探、
    再在当前图上反解）；det 为 None 时在 arr 上现检。
    """
    height, width = arr.shape[:2]
    y0, y1, x0, x1 = box_pixels(box, width, height)
    unchanged = np.array(arr, dtype=np.uint8, copy=True)
    if det is None:
        det = detect_light_in_box(arr, box, source=source)
    if det.rejected:
        return unchanged, "rejected", det.rejected
    if not det.found:
        return unchanged, "none", "未找到水印"
    box_area = (y1 - y0) * (x1 - x0)
    # 框占图超七成先拒掉——不管框里是什么，整块补/反解整张图都是错的；
    # 这个检查必须在“mask 太小就整块补”之前，否则白字徽标配大框会把
    # 整张图都补掉
    if box_area > height * width * 0.7:
        return unchanged, "rejected", "框占图超 70%，不处理"
    # 框只框住字也很常见；向外留一圈上下文，模型才不会把文字边缘当成
    # “要保留的内容”接回去（实测不留余量时白字会残一半）
    pad = max(6, int((y1 - y0) * 0.35))
    grown = (max(0, x0 - pad) / width, max(0, y0 - pad) / height,
             min(width, x1 + pad) / width, min(height, y1 + pad) / height)
    if int(det.mask.sum()) < max(200, box_area * 0.02):
        # 原图框里几乎没有可反解的浅色水印：不透明徽标（白字只剩抗锯齿
        # 轮廓也算），整块补
        if allow_inpaint:
            return (inpaint(arr, grown, method), "inpaint-opaque",
                    "不透明徽标，已整块补图")
        return unchanged, "none", "不透明徽标，整块补已关"
    # 框内有浅色水印：先反解一次，再看残留决定
    fixed, mask2, residual = undo_light_in_box(arr, box, tint=det.tint)
    if not mask2.any():
        # 探时有、解时没了：水印已被自动抠处理掉，不硬补
        return unchanged, "none", "已由自动抠处理"
    if not residual.any():
        return fixed, "unblend", "反解干净"
    frac = float(residual.sum()) / max(float(det.mask.sum()), 1.0)
    if frac <= 0.30:
        return (_nearest4_fill(fixed, _grow_mask(residual, 1)), "fill",
                f"反解后残留 {frac:.0%}，已小补")
    if allow_inpaint:
        return (inpaint(fixed, grown, method), "inpaint-residual",
                f"反解残留 {frac:.0%}，已整块补图")
    return (_nearest4_fill(fixed, _grow_mask(residual, 1)), "fill",
            f"反解残留 {frac:.0%}，已小补（整块补已关）")


def _box_iou(a, b):
    """两个归一化框的 IoU。"""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _grid_light_boxes(rgb, rows=3, cols=3, overlap=0.25):
    """一键抠在照片上的分区检测：把图切成重叠网格，逐格局部投票。

    水印通常只占一角或一条边，整图投票会被主导色（蓝天、黄沙）带跑；
    格子化后每格只看自己的一小片背景。返回 LightDetection 列表
    （重叠的已合并，掩膜取并集，色系取像素多的那格）。
    """
    height, width = rgb.shape[:2]
    cell_h = 1.0 / rows
    cell_w = 1.0 / cols
    found = []
    for r in range(rows):
        for c in range(cols):
            y0 = max(0.0, r * cell_h - cell_h * overlap)
            y1 = min(1.0, (r + 1) * cell_h + cell_h * overlap)
            x0 = max(0.0, c * cell_w - cell_w * overlap)
            x1 = min(1.0, (c + 1) * cell_w + cell_w * overlap)
            box = (x0, y0, x1, y1)
            det = detect_light_in_box(rgb, box, source="grid")
            count = int(det.mask.sum()) if det.found else 0
            py0, py1, px0, px1 = box_pixels(box, width, height)
            total = (py1 - py0) * (px1 - px0)
            # 证据要够强：格内至少 200 像素且占比不超过 25%（太少是噪声，
            # 太多是整块底色）；太少时 detect 里已经返回空了，这里只补上限
            if count >= 200 and count <= total * 0.25:
                found.append(det)
    # 合并重叠的格子（IoU>0.3 合成外接框），避免同一处水印反解两次。
    # 没合并的相邻格（IoU 约 0.14）重叠区会被两个格各反解一次，是自收敛
    # 的（第二次按重新估计的残余 α 再解，不会过冲），不是 bug。
    merged = []
    for det in found:
        placed = False
        for i, old in enumerate(merged):
            if _box_iou(det.box, old.box) > 0.3:
                box = (min(det.box[0], old.box[0]), min(det.box[1], old.box[1]),
                       max(det.box[2], old.box[2]), max(det.box[3], old.box[3]))
                merged[i] = LightDetection(
                    box,
                    tint=(det.tint if int(det.mask.sum()) >= int(old.mask.sum())
                          else old.tint),
                    mask=det.mask | old.mask,
                    score=max(det.score, old.score),
                    source="grid")
                placed = True
                break
        if not placed:
            merged.append(det)
    return merged


def _thin_strokes(mask: np.ndarray) -> np.ndarray:
    """只留“笔画”：长游程占多数的检测直接不认，零星长带像素丢掉。

    半透明水印是细笔画，横竖游程都短；“灰表头压着暖色纸”这种色调边界、
    整条高亮行、整块色块是一整条长带（游程几百上千）。局部基准对色调
    边界天生敏感（暖纸相对灰表头就是“淡黄”），没有这一道，边界那一圈
    会被当成水印，反解时把暖纸洗成冷调（实测 15% 画面被改、最大差 62 级）。
    门限随图幅走：小图 64 像素起，大图放到 1/16 边长的级别，60 px 字号的
    笔画仍然过得去。长带占多数时整格返回空掩膜——宁可不动，也不把底色
    当水印洗一遍。
    """
    if not mask.any():
        return mask
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    # 只在掩膜的包围盒里算游程：边角水印是常见情形，全图算白花两秒
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    crop = mask[y0:y1, x0:x1]
    limit = max(64, min(height, width) // 16)
    run_row, run_col = _run_lengths(crop)
    band = (run_row >= limit) | (run_col >= limit)
    # 分母是掩膜像素本身，不是包围盒：包围盒里大片空白会把比例稀释掉
    if float(band.sum()) > 0.5 * float(crop.sum()):
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = crop & ~band
    return out


def _negligible_change(before: np.ndarray, after: np.ndarray) -> bool:
    """改完几乎没动过（只有零星几个像素差一两级）就别声称清理成功。

    块内基准比全图基准灵敏：白字、高光的抗锯齿边缘在彩色底上也会被读成
    “相对纸面偏色”，这些像素在反解时都被“不许变暗”的护栏拦回去了，真正
    落地的改动只剩个位数色阶。这种情况如实报“未找到”，并且原样返回——
    “没水印的图一个像素都不动”这条不变量比“报个 clean”重要。
    """
    cut_before, cut_after = before[::2, ::2], after[::2, ::2]
    diff = np.abs(np.subtract(np.asarray(cut_after), np.asarray(cut_before),
                              dtype=np.int16))
    changed = int((diff.max(axis=2) > 2).sum())
    # 抽样后阈值同比例缩回去（2×2 抽样，像素数少 4 倍）
    return changed <= max(16, cut_before.shape[0] * cut_before.shape[1] // 2000)


# ------------------------------------------------------- 屏摄水波纹（摩尔纹）
# 拍屏幕出现的摩尔纹是两组周期性栅格（相机像素 × 屏幕像素）拍出来的差频，
# 在频域里就是一对对远离中心的孤立谱峰，画面上是一层斜向彩色细条。它对浅色
# 水印这一路有两个害处：多了层假纹理；而且条纹本身会被检测读成“淡色水印
# 笔画”——实测一张合成屏摄页，流程会改动 2.5 万像素、留下 37 个被压暗到 40 级
# （反解护栏上限）的点。
#
# 处理不必上模型：把谱峰（连同中心对称的那一个）在频谱上压掉即可，纯 numpy，
# 一组谱峰一次 FFT。关键是**只在真的检出谱峰时才动手**——扫描件、纯文档页
# 没有这种孤立斜峰，一个像素都不碰。

MOIRE_STRENGTH = 0.9      # 陷波后保留的幅度比例（0.9 = 压掉九成）
MOIRE_RADIUS = 3          # 陷波半径（全图频谱上的像素）


def _moire_peaks(rgb: np.ndarray) -> list:
    """找摩尔纹谱峰，返回归一化频率 [(fy, fx), ...]（周期/像素），最多 8 个。

    只在居中的 crop×crop 裁片上看频谱：摩尔纹是遍布全图的寄生条纹，一小块
    裁片就够认出频率；而全图 FFT 在 12 MP 上要几秒、还要几百兆内存。归一化
    频率与裁片大小无关，拿到全图上照样能定位。

    五道门，全过了才认：
    - 离中心够远：low_frac=0.02 即周期 50 像素以内才算“条纹”，纸张渐变、
      光照不均这类慢变分量不算；
    - 纵向、横向分量都不为零：表格线、扫描行的谱峰落在 fx=0 / fy=0 的轴上
      （实测合成页的行距 34 px 峰就在 fy=0.029、fx=0），摩尔纹是斜的，
      这一条把两者分开；
    - 峰值比同半径环带的中位数高 ratio 倍：真条纹是尖峰，噪声毛刺不是；
    - **绝对幅度**：把谱峰换算成正弦幅度（幅度 A 的正弦，谱峰 |X| = A·N/2），
      要求 A ≥ min_amp·rms。只比环带的相对门在没有噪声的图上形同虚设：
      干净合成页上文字抗锯齿也能凑出“孤立尖峰”，但 A/rms 只有 0.05；
      真摩尔纹的 A/rms 是 0.35~1.06（实测）。这一条是把它拦下来的关键；
    - **与最强峰同量级**：A ≥ share·A最强。文字被摩尔纹调制出的边带只有
      最强峰的 0.15，压掉它们会在字形旁留下残影，反而被水印检测误读。

    先减掉 5×5 均值（去低频）再变换：谱峰位置不变，但 1/f 的底噪被压平，
    环带中位数才代表得了局部噪声；rms 也从这张高通图上算，量的是“条纹级”
    的起伏，而不是画面本身的明暗。
    """
    # 选峰参数：最多 8 个峰、峰/环带 ≥ 8 倍、正弦幅度 ≥ 0.15·rms、
    # 只用最强峰的 1/4 以上、周期 < 50 像素（low_frac=0.02）才算条纹、
    # 频谱在居中的 1024² 裁片上算
    keep, ratio, min_amp, share = 8, 8.0, 0.15, 0.25
    low_frac, crop, radius = 0.02, 1024, MOIRE_RADIUS
    top = 0.0
    height, width = rgb.shape[:2]
    side = int(min(height, width, crop))
    if side < 64:
        return []
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    gray = np.tensordot(rgb[y0:y0 + side, x0:x0 + side].astype(np.float32),
                        _LUMA_WEIGHTS, axes=([2], [0]))
    gray = gray - _blur_f(gray, 2)
    rms = float(np.sqrt((gray ** 2).mean()))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy, cx = side // 2, side // 2
    ys = (np.arange(side) - cy) / float(side)
    xs = (np.arange(side) - cx) / float(side)
    fy = ys[:, None] + np.zeros((1, side), dtype=np.float32)
    fx = xs[None, :] + np.zeros((side, 1), dtype=np.float32)
    # 低频盘和两条轴带都不参与选峰
    work = np.where((fy ** 2 + fx ** 2 > low_frac ** 2)
                    & (np.abs(fy) > 0.01) & (np.abs(fx) > 0.01), mag, 0.0)
    r_out = 8 * radius
    grid_y = np.arange(2 * r_out + 1)
    dist = np.sqrt((grid_y[:, None] - r_out) ** 2
                   + (grid_y[None, :] - r_out) ** 2)
    ring_sel = (dist >= 4 * radius) & (dist <= r_out)
    peaks = []
    for _ in range(keep):
        idx = int(np.argmax(work))
        if work.flat[idx] <= 0.0:
            break
        py, px = divmod(idx, side)
        if py < r_out or py >= side - r_out or px < r_out or px >= side - r_out:
            break                     # 贴边的峰算不出环带底噪，不认
        win = mag[py - r_out:py + r_out + 1, px - r_out:px + r_out + 1]
        base = float(np.median(win[ring_sel]))
        if float(mag[py, px]) < ratio * max(base, 1e-6):
            break                     # 后面的峰只会更小，不用再看
        amp = 2.0 * float(mag[py, px]) / float(side * side)
        if amp < min_amp * rms:
            break                     # 绝对幅度不够：是纹素/抗锯齿的碎峰
        # 占比门：摩尔纹是几组强干涉，峰与峰同量级；文字被摩尔纹调制出来的
        # 边带只有最强峰的 0.14~0.15（实测 4.3 MP 屏摄页）。压这些边带等于
        # 在字形周围挖掉一条窄频带，留下与水印笔画极像的周期残影——实测
        # 多压这 4 个峰，水印流程就会把整页当成淡黄水印去反解（改 3.7 万
        # 像素、报 clean）；不碰它们，同一张图原地不动。
        if top > 0.0 and amp < share * top:
            break
        top = max(top, amp)
        peaks.append((float(fy[py, px]), float(fx[py, px])))
        work[py - radius:py + radius + 1,
             px - radius:px + radius + 1] = 0.0
    return peaks


def moire_notch(rgb: np.ndarray) -> tuple:
    """屏摄水波纹陷波：只在检出谱峰时动手。

    返回 (image, detail)。detail 为空字符串表示“没检出谱峰，原样返回”——
    这时返回的是副本，像素与原图逐一相同。

    陷波是在频域乘一个掩膜：每个谱峰连同中心对称的共轭峰，在半径
    MOIRE_RADIUS 的圆盘上乘 (1-MOIRE_STRENGTH)。对实图来说谱必然是中心对称
    的，只压一边会算出复数残差，所以两边都要压。

    代价与分辨率成正比：12 MP 上三通道各一次 fft2，约 3~4 秒、峰值内存约
    600 MB，且只在真检出摩尔纹时才走这条路。压掉之后再用实际改动量复核一遍
    （标准差 < 0.5 级就当没动），免得“检出了但压不动”的图被白白重存一次。
    """
    peaks = _moire_peaks(rgb)
    if not peaks:
        return np.array(rgb, dtype=np.uint8, copy=True), ""
    arr = rgb.astype(np.float32)
    height, width = arr.shape[:2]
    cy, cx = height // 2, width // 2
    keep = np.float32(1.0 - MOIRE_STRENGTH)
    band = MOIRE_RADIUS
    mask = np.ones((height, width), dtype=np.float32)
    for peak_y, peak_x in peaks:
        for sign in (1, -1):
            py = cy + int(round(sign * peak_y * height))
            px = cx + int(round(sign * peak_x * width))
            y0, y1 = max(0, py - band), min(height, py + band + 1)
            x0, x1 = max(0, px - band), min(width, px + band + 1)
            gy = np.arange(y0, y1) - py
            gx = np.arange(x0, x1) - px
            cut = (gy[:, None] ** 2 + gx[None, :] ** 2) <= band * band
            sub = mask[y0:y1, x0:x1]
            sub[cut] = np.minimum(sub[cut], keep)
    out = np.empty_like(arr)
    for channel in range(3):
        spec = np.fft.fftshift(np.fft.fft2(arr[..., channel]))
        spec *= mask
        out[..., channel] = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))
    np.clip(out, 0, 255, out=out)
    fixed = np.rint(out).astype(np.uint8)
    removed = fixed.astype(np.float32) - arr
    # 复核两道：压掉的量太小，或者改动只落在画面一小块里，都不算摩尔纹——
    # 摩尔纹是整幅铺开的寄生条纹，改动必然遍布全图（实测干净页覆盖 0%，
    # 合成屏摄页 100%）。这一步挡住“检出了但其实是局部网点/花斑”的误动。
    cover = float((np.abs(removed).max(axis=2) > 1.0).mean())
    if float(removed.std()) < 0.5 or cover < 0.15:
        return np.array(rgb, dtype=np.uint8, copy=True), ""
    return fixed, f"屏摄水波纹已陷波（{len(peaks)} 组频率）"


def clean_light_watermark(rgb: np.ndarray):
    """浅色水印一键流程：屏摄水波纹陷波 → 认出色调 → 反解 → 补残影。

    水波纹是前置的一级，只在检出谱峰时生效（见 moire_notch）。放在反解之前
    是因为条纹比水印笔画强得多，夹在里面会把“相对局部背景的偏离”整片抬高，
    检测阈值跟着水涨船高，淡色水印反而认不出来。
    状态语义不变：unchanged 仍然是“没找到水印”的结论，水波纹压掉了会在
    detail 里写明，免得调用方以为这张图一个像素都没动。
    """
    notched, moire_detail = moire_notch(rgb)
    fixed, status = _clean_light_core(notched)
    if not moire_detail:
        return fixed, status
    if status.status == "unchanged":
        return fixed, LightStatus("unchanged", f"{moire_detail}，未找到水印")
    return fixed, LightStatus(status.status,
                              f"{status.detail}；{moire_detail}".lstrip("；"))


def _clean_light_core(rgb: np.ndarray):
    """浅色水印本体流程（不含水波纹陷波，见 clean_light_watermark）。

    彩色淡色水印走反解；完全没偏色的淡灰水印走“平坦区重铺背景”。
    返回 (image, LightStatus)：照片走网格检测 + 统一重建决策
    （reconstruct_light，一键抠不整块补）；文档走原全局路径。
    """
    # 照片护栏（分区版）：文档截图的中位色度接近 0（白纸），人像/风景随便
    # 就 30~60。在照片上强行全图反解会把皮肤、汗光、天空当水印抹掉（实测
    # 一张人像整张发灰、平均差 5.8），所以不再整张放过、也不再整张反解：
    # 网格找水印区，只在找到的区里框内反解，区外一个像素不动；没找到就
    # 原样返回。
    if looks_like_photo(rgb):
        out = np.array(rgb, dtype=np.uint8, copy=True)
        dets = _grid_light_boxes(rgb)
        if not dets:
            return out, LightStatus("unchanged", "未找到浅色水印")
        changed = 0
        for det in dets:
            fixed, action, _detail = reconstruct_light(
                out, det.box, det=det, allow_inpaint=False)
            if action in ("unblend", "fill"):
                out = fixed
                changed += 1
        if not changed:
            return out, LightStatus("unchanged", "水印区已处理过")
        # 处理后复检：反解没解净的，报 partial 不报 clean，跟文件流的
        # 校验（_verify_light_watermark）对齐；直接调这个函数的调用方
        # 拿到的也是诚实的状态。
        verify = _verify_light_watermark(out)
        if verify is not None:
            return out, verify
        return out, LightStatus("clean", f"{changed} 个水印区已反解")
    luma, chroma = _luma_chroma(rgb)
    stats = (luma, chroma)

    tint = detect_tint(rgb, stats=stats)
    if not tint:
        neutral = _neutral_mask(rgb, stats=stats)
        if int(neutral.sum()) < max(300, neutral.size // 500):
            return (np.array(rgb, dtype=np.uint8, copy=True),
                    LightStatus("unchanged", "未找到浅色水印"))
        # 照片护栏：真的浅灰水印是“在纸上盖一层字”，占比很小（文档样本
        # 0.5%~3%）。风景/人像里的反光、白皮肤、天空随便就能占 7%，照旧
        # 重铺会把整张照片抹成一片，所以超过这个比例就不认为它是文档水印。
        if neutral.mean() > 0.05:
            return (np.array(rgb, dtype=np.uint8, copy=True),
                    LightStatus("unchanged", "疑似照片纹理，未处理"))
        # 邻域深色门：淡灰水印是盖在空白纸上的小字，四下没有深色内容；
        # 文字的抗锯齿边同样“中性、比纸暗”，却紧贴着笔画。合成文字页实测
        # 掩膜 10536 像素全是 AA 边（2 万多的重铺改动、中位 73 级），这一道
        # 把它们全部排掉后该页一个像素不动。
        neutral &= ~_near_dark(neutral, stats[0])
        if int(neutral.sum()) < max(300, neutral.size // 500):
            return (np.array(rgb, dtype=np.uint8, copy=True),
                    LightStatus("unchanged", "未找到浅色水印"))
        grow = _grow_mask(neutral, 2)
        background, _ = _box_blur_known(rgb, ~grow, 10)
        out = rgb.astype(np.float32)
        out[grow] = background[grow]
        return (np.clip(out, 0, 255).astype(np.uint8),
                LightStatus("clean", "淡灰水印已重铺背景"))

    mask = _thin_strokes(detect_light_watermark(rgb, tint=tint, stats=stats))
    if not mask.any():
        return (np.array(rgb, dtype=np.uint8, copy=True),
                LightStatus("unchanged", "未找到浅色水印"))
    # 掩膜过大护栏：水印是盖在图上的小字/小图，占比不会大。掩膜占图超三成
    # 说明检测把底色本身当成了水印（实测：拍屏幕照片的蓝灰底被投成“蓝”，
    # 掩膜 51.8%），再往下反解+铺平会把整片背景重铺，铺平用的模糊背景一旦
    # 沾到附近深色文字就整片打黑出横杠。直接原样返回，不处理。
    if float(mask.mean()) > 0.30:
        return (np.array(rgb, dtype=np.uint8, copy=True),
                LightStatus("unchanged", "水印占比过大（疑似底色误判），未处理"))
    ref = _choose_tint_ref(rgb, mask, tint)
    # 把掩膜一起传下去：局部背景 B̂ 只从水印外的像素估，混进水印色会把
    # B̂ 拉向水印、α 系统性偏小
    first = undo_light_watermark(rgb, tint=tint, ref=ref, mask=mask)
    first = _flatten_background(first, mask)
    # 反解后的亮度/色度算一次就够：残影检测与紧跟的“附近有没有深色内容”
    # 都要用同一张图，不传下去就是同一次归约算两遍（12 MP 约 0.5 秒）。
    stats2 = _luma_chroma(first)
    residual = detect_light_watermark(first, diff=2.0, gmin=105,
                                      luma_gate=110.0, tint=tint,
                                      stats=stats2)
    if residual.any():
        # 文字、表格线旁边不补：那里的“残余黄点”其实是笔画自身的抗锯齿边缘，
        # 补下去会把字吐成毛边（实测表头字母会被吞出锯齿）
        residual &= ~_near_dark(residual, stats2[0])
    if residual.any():
        residual = _grow_mask(residual, 1)
        first = _nearest4_fill(first, residual)
    if _negligible_change(rgb, first):
        return (np.array(rgb, dtype=np.uint8, copy=True),
                LightStatus("unchanged", "未找到浅色水印"))
    return first, LightStatus("clean", f"{tint}色水印已反解")


def _shift_or(mask: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """沿一个轴膨胀 radius（边界外当空，绝不跨边卷绕）。"""
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    src = np.asarray(mask, dtype=bool)
    out = src.copy()
    for s in range(1, radius + 1):
        if axis == 0:
            out[s:] |= src[:-s]
            out[:-s] |= src[s:]
        else:
            out[:, s:] |= src[:, :-s]
            out[:, :-s] |= src[:, s:]
    return out


def _grow_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """方形膨胀：把笔画边缘抗锯齿留下的淡黄一起圈进来。

    可分离实现（先竖后横），半径 r 只要 2r 次数组运算而不是 r² 个位移；
    结果与方形结构元一致，但不再用 np.roll —— 贴边的掩膜不会从对边卷回
    来一份（贴右边缘的水印以前会在左边缘凭空多出一片掩膜，去水印时连带
    把那里也改掉）。
    """
    if radius <= 0:
        return mask
    return _shift_or(_shift_or(mask, radius, 0), radius, 1)


def _verify_light_watermark(arr):
    """处理后校验（只针对照片）：在输出图上用同一套网格检测再检一遍，
    确认水印真的没了。返回 None 表示通过，否则返回 partial。

    文档不走这里：文档路径内部已经用同一套残影检测复检并补过，靠近
    文字笔画的残留是按设计留下的（那是笔画抗锯齿，不是水印），这里
    再报就是误报。平铺水印网格本身不认（格内上限 25%），校验也检不
    出——那是检测的已知局限，不在这里硬报。
    """
    if not looks_like_photo(arr):
        return None
    dets = _grid_light_boxes(arr)
    if not dets:
        return None
    top = max(det.score for det in dets)
    return LightStatus(
        "partial",
        f"处理后仍有 {len(dets)} 个水印区残留（检出分数 {top:.2f}，"
        "越接近 1 越可能是轻微残留）")


def remove_light_file(src, dst=None, quality: int = 95, boxes=None,
                     method: str = "auto"):
    """处理一张图（自动抠浅色水印），不覆盖原图。

    返回 (目标路径, LightStatus)：先跑一键自动抠，再按用户框走统一重建
    决策（reconstruct_light：先反解、解不净再补、不透明徽标整块补、
    平铺水印明确拒绝）。boxes 给了就先反解淡色水印，再把这些框（比如
    记下来的徽标位置）整块补掉：徽标上的白字是不透明的，反解救不回来，
    只能靠补。
    """
    src = Path(src)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"))
        fixed, auto = clean_light_watermark(arr)
        box_notes = []
        rejected = None
        partial = None
        cleaned = auto.status in ("clean", "partial")
        for one in as_box_list(boxes) if boxes else []:
            # 探：先在原图上检测（框里有什么），再在当前图上重建；重建决策
            # 集中在 reconstruct_light，这里只汇总每框的结论
            det = detect_light_in_box(arr, one, source="box")
            fixed, action, detail = reconstruct_light(
                fixed, one, det=det, method=method, allow_inpaint=True)
            if action == "rejected":
                rejected = detail
                box_notes.append(f"框已拒绝：{detail}")
            elif action == "inpaint-residual":
                partial = detail
                box_notes.append(f"框：{detail}")
            elif action == "inpaint-opaque":
                cleaned = True
                box_notes.append(f"框：{detail}")
            elif action in ("unblend", "fill"):
                cleaned = True
                box_notes.append(f"框：{detail}")
            # action "none"：框里没水印，不记
    target = Path(dst) if dst else output_path(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 没改动（照片护栏直接放过、或者这张图本来就没水印）就直接拷原文件：
    # 重存一遍既费时（12 MP 约 0.4 秒）又白白损一次 JPEG 质量。
    if not boxes and np.array_equal(fixed, arr):
        shutil.copyfile(src, target)
        return target, LightStatus("unchanged", "未找到浅色水印")
    image = Image.fromarray(fixed, "RGB")
    if target.suffix.lower() in (".jpg", ".jpeg"):
        image.save(target, quality=quality, subsampling=0)
    elif target.suffix.lower() == ".webp":
        image.save(target, quality=quality)
    else:
        image.save(target)
    if rejected is not None and not cleaned and partial is None:
        # 唯一结果就是被拒：什么都没动，如实报 rejected
        notes = ([auto.detail]
                 if auto.status in ("clean", "partial") and auto.detail
                 else []) + box_notes
        detail = "；".join(notes) if notes else rejected
        base = LightStatus("rejected", detail)
    elif partial is not None:
        # 反解没解干净，不得不整块补图——纹理被重建，报 partial
        notes = ([auto.detail]
                 if auto.status in ("clean", "partial") and auto.detail
                 else []) + box_notes
        detail = "；".join(notes) if notes else partial
        base = LightStatus("partial", detail)
    elif auto.status == "partial":
        # 自动抠已报 partial（处理后仍有残留）：如实报 partial，不降级成
        # unchanged，也不升成 clean；文件流的校验随后会再补一句。
        notes = ([auto.detail] if auto.detail else []) + box_notes
        detail = "；".join(notes) if notes else "水印未除净"
        base = LightStatus("partial", detail)
    elif cleaned:
        # 框被拒但水印已被自动抠清掉：结论仍是 clean，拒的框只记一句
        notes = ([auto.detail]
                 if auto.status in ("clean", "partial") and auto.detail
                 else []) + box_notes
        detail = "；".join(notes)
        base = LightStatus("clean", detail or "水印已清")
    else:
        base = LightStatus("unchanged", "未找到浅色水印")
    # 处理后校验：自称干净的，输出图上再检一遍；校验发现残留就改口
    # partial，不悄悄说成功。已经是 partial/rejected 的只补一句。
    verify = (_verify_light_watermark(fixed)
              if base.status in ("clean", "partial") else None)
    if verify is not None:
        if base.status == "clean":
            base = verify
        elif verify.detail not in base.detail:
            # auto 已报 partial 时里面已含同一份校验结论（同一张图、
            # 没过框），不再重复追加；框改过图、结论变了时才追加。
            base = LightStatus(base.status,
                               base.detail + "；" + verify.detail)
    return target, base


def batch_remove_light(paths, out_dir=None, quality: int = 95, cancel=None,
                       on_progress=None, subdir: bool = False, boxes=None,
                       method: str = "auto") -> dict:
    """批量自动抠浅色水印；返回格式与 batch_remove 一致，再加逐张状态。

    results 是 [(原路径, 状态, 一句人话)]，供界面逐张报“干净 / 残留 /
    未找到 / 拒绝”；clean/partial/unchanged/rejected 是各状态的张数。
    """
    paths = list(paths)
    done = skipped = 0
    failed = []
    results = []
    counts = {"clean": 0, "partial": 0, "unchanged": 0, "rejected": 0}
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if cancel is not None and cancel.is_set():
            break
        try:
            target = output_path(path, target_dir(path, out_dir, subdir),
                                 unique=False)
            if target.exists():
                skipped += 1
            else:
                _, status = remove_light_file(path, target, quality,
                                              boxes=boxes, method=method)
                results.append((str(path), status.status, status.detail))
                counts[status.status] += 1
                done += 1
        except Exception as exc:
            failed.append((str(path), str(exc)))
        if on_progress is not None:
            on_progress(index, total, str(path))
    return {"done": done, "failed": failed, "skipped": skipped,
            "total": total, "results": results, **counts}


def batch_remove_light_boxes(paths, boxes, out_dir=None, quality: int = 95,
                            cancel=None, on_progress=None,
                            subdir: bool = False) -> dict:
    """批量框选去浅色水印：只在用户框里做浅色反解，不跑自动检测，不整块补图。

    每个框走 reconstruct_light（allow_inpaint=False）：先反解，残留小补；
    框里没水印就不动；框占图超 70% 或平铺水印会被拒。返回格式与
    batch_remove_light 一致（results 是 [(原路径, 状态, 一句人话)]）。
    """
    paths = list(paths)
    boxes = as_box_list(boxes) if boxes else []
    done = skipped = 0
    failed = []
    results = []
    counts = {"clean": 0, "partial": 0, "unchanged": 0, "rejected": 0}
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if cancel is not None and cancel.is_set():
            break
        try:
            target = output_path(path, target_dir(path, out_dir, subdir),
                                 unique=False)
            if target.exists():
                skipped += 1
            else:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im)
                    arr = np.asarray(im.convert("RGB"))
                fixed = np.array(arr, dtype=np.uint8, copy=True)
                notes = []
                cleaned = False
                rejected = None
                for one in boxes:
                    det = detect_light_in_box(arr, one, source="box")
                    fixed, action, detail = reconstruct_light(
                        fixed, one, det=det, allow_inpaint=False)
                    if action == "rejected":
                        rejected = detail
                        notes.append(f"框已拒绝：{detail}")
                    elif action in ("unblend", "fill"):
                        cleaned = True
                        notes.append(f"框：{detail}")
                    # action "none"：框里没水印，不记
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cleaned and np.array_equal(fixed, arr):
                    shutil.copyfile(path, target)
                    if rejected is not None:
                        status = LightStatus("rejected", rejected)
                    else:
                        status = LightStatus("unchanged", "框里未找到浅色水印")
                else:
                    image = Image.fromarray(fixed, "RGB")
                    if target.suffix.lower() in (".jpg", ".jpeg"):
                        image.save(target, quality=quality, subsampling=0)
                    elif target.suffix.lower() == ".webp":
                        image.save(target, quality=quality)
                    else:
                        image.save(target)
                    if rejected is not None and not cleaned:
                        status = LightStatus("rejected", rejected)
                    else:
                        status = LightStatus(
                            "clean", "；".join(notes) or "框内水印已反解")
                results.append((str(path), status.status, status.detail))
                counts[status.status] += 1
                done += 1
        except Exception as exc:
            failed.append((str(path), str(exc)))
        if on_progress is not None:
            on_progress(index, total, str(path))
    return {"done": done, "failed": failed, "skipped": skipped,
            "total": total, "results": results, **counts}


__all__ = [
    "LIGHT_REF",
    "METHOD_LABELS",
    "PRESETS",
    "LightDetection",
    "LightStatus",
    "as_box_list",
    "batch_remove",
    "batch_remove_light",
    "batch_remove_light_boxes",
    "box_pixels",
    "clean_light_watermark",
    "detect_light_in_box",
    "detect_light_watermark",
    "estimate_tint_ref",
    "inpaint",
    "iter_photos",
    "moire_notch",
    "normalize_box",
    "output_path",
    "pick_method",
    "reconstruct_light",
    "remove_light_file",
    "remove_watermark_file",
    "target_dir",
    "undo_light_in_box",
    "undo_light_watermark",
]
