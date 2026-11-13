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
    estimated = estimate_tint_ref(rgb, mask, tint=tint)
    mid = tuple((np.asarray(base, dtype=np.float32)
                 + np.asarray(estimated, dtype=np.float32)) * 0.5)
    best, best_cost = base, float("inf")
    for ref in (base, estimated, mid):
        fixed = undo_light_watermark(rgb, strength, ref=ref, tint=tint)
        cost = float(np.abs(_tint_score(fixed.astype(np.float32), tint)[mask]).mean())
        if cost < best_cost:
            best, best_cost = ref, cost
    return best


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


def _safe_den(value: float) -> float:
    """反解分母的下限保护：绿色参考色的 Wr-Wb 正好是 0，直接除会出 inf。"""
    if abs(value) >= 4.0:
        return value
    return 4.0 if value >= 0 else -4.0


def undo_light_watermark(rgb: np.ndarray, strength: float = 1.0,
                        ref=None, tint=None) -> np.ndarray:
    """把半透明浅色水印“解”掉，而不是“补”掉。

    I = (1-α)·背景 + α·参考色，背景（纸张、表格底）近似中性，于是
    α 可以由 (R-B)/(Wr-Wb)、(G-B)/(Wg-Wb) 估出来，再反解背景。
    这样文字和表格线的像素压根没被动过，不会留下补图常见的暗边。
    strength 是修正强度（1.0 全解，0.5 只解一半），改小可以少伤原图的暖色调。
    ref 显式传入时优先（传自估值）；否则按 tint 查固定表。
    """
    if ref is None:
        ref = TINT_REFS.get(tint, LIGHT_REF) if tint else LIGHT_REF
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    wr, wg, wb = float(ref[0]), float(ref[1]), float(ref[2])
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    # 灰底（例如表格表头）上反解容易把灰色解出色斑，亮度越低解越少
    scale = np.clip((luma - 200.0) / 40.0, 0.35, 1.0) * float(strength)
    alpha = ((r - b) / _safe_den(wr - wb)
             + (g - b) / _safe_den(wg - wb)) * 0.5 * scale
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
    ref = _choose_tint_ref(rgb, mask, tint)
    first = undo_light_watermark(rgb, tint=tint, ref=ref)
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
    "estimate_tint_ref",
    "inpaint",
    "iter_photos",
    "normalize_box",
    "output_path",
    "pick_method",
    "remove_light_file",
    "remove_watermark_file",
    "undo_light_watermark",
]
