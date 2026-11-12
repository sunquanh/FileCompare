"""相似照片：dHash 粗筛 + pHash 复核 + 颜色直方图 + 完全链接聚类。

与旧实现相比：
  * 8 个方向（4 旋转 × 2 镜像）都参与匹配，翻拍/镜像图也能归组；
  * 粗筛用「8 个方向的 dHash 最小值」作为旋转不变签名，单次 popcount 即可过滤；
  * 命中候选再用 16×16 pHash（256 位，DCT）逐方向复核，并用颜色直方图排除
    「结构像但内容完全不同」的图；
  * 最后两道与颜色无关的复核：局部高频的归一化互相关（不看整体明暗，专看
    边缘细节是否对得上）+ 8×8 空间颜色布局（不看全局配色，专看颜色在画面
    里的位置），挡住「衣服和背景像、人和姿态不同」的照片；
  * 聚类改成完全链接，避免 A~B、B~C 却把 A、C 硬并成一组。
"""
from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

try:
    import numpy as np
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    np = None
    Image = None
    ImageOps = None

from dedup_models import check_cancel, enable_heif, ignore

enable_heif()  # 可选：装了 pillow-heif 就能读 HEIC/HEIF

ORIENTS = 8
HASH_BITS = 64
PHASH_BITS = 256
DCT_SIZE = 32
LOW_SIZE = 16
# 只存整幅描述子；裁过边的副本改用下面的「缩放搜索」现场对齐。原先预切
# 1.0/0.88/0.76 三个固定裁边层，实测只能盖住四边各裁 6%/12% 这两档：
# 4%/8%/10%/14% 正好落在层间，结构相关掉到 0.42~0.84，整类漏判。
CROP_LEVELS = (1.0,)
LEVELS = len(CROP_LEVELS)
HIST_CHROMA = 8                     # 色度 8×8
HIST_LUMA = 32                      # 相对亮度 32 档
HIST_SIZE = HIST_CHROMA * HIST_CHROMA + HIST_LUMA
# 缩放对齐之后的结构相关下限：重采样有损失，比同尺度低一档（实测真重复
# 在 0.94 以上，「同背景换人」在 0.51 附近，两者不重叠）
CROP_NCC_TOL = 0.89
# 整幅粗筛在 dist 之上留的余量：裁过边的副本整幅哈希会偏移，留少了会漏
PREFILTER_SLACK = 14
# 裁边候选单独放宽一档：实测裁 14%~20% 的整幅哈希差到了 23~28 位，按 14 的
# 余量根本进不了候选。这一档只用来「进缩放搜索」，不参与同尺度判定。
PREFILTER_ZOOM_SLACK = 30
# 放宽那一档的额外条件：色分布也必须几乎一致。裁边副本的直方图几乎不变
# （实测 ≤0.02），而不相干的两张图即便亮度相近也很难到这个程度。
# 缩放路径的布局阈值：差异集中在被裁掉的外圈，中心区仍会到 13 上下，而
# 「换个人」是 26.6，两者之间有足够空隙
ZOOM_LAYOUT_TOL = 16.0
# 缩放搜索档位：中心保留 1/z。z=1.43 ≈ 四边各裁 15%，1.67 ≈ 20%。
# 先按 6% 粗扫，再在最好的两档附近按 1% 细扫：粗扫单档最坏错位 3%（相关
# 值只剩 0.76~0.86），细扫一轮就能回到 0.95 以上。
ZOOM_LADDER = (1.00, 1.06, 1.13, 1.20, 1.28, 1.37, 1.46, 1.56, 1.67)
THUMB_SIZE = 64            # 缓存里存的中间图边长，缩放搜索的原料
ZOOM_MAX_PAIRS = 128       # 每块最多对多少对做缩放搜索（按色分布由近到远取）
# 同机型 + 拍摄时间相差 ≤2 秒：EXIF 本身就对得上，结构阈值放宽一档
EXIF_TIME_TOL = 2
EXIF_NCC_TOL = 0.80
_DCT_CACHE: dict[int, np.ndarray] = {}


# ---------------------------------------------------------------- 基础
def _dct_matrix(n=32) -> np.ndarray:
    cached = _DCT_CACHE.get(n)
    if cached is None:
        k = np.arange(n)[:, None]
        x = np.arange(n)[None, :]
        cached = np.cos(np.pi * (2 * x + 1) * k / (2.0 * n)).astype(np.float32)
        _DCT_CACHE[n] = cached
    return cached


def popcount(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = np.asarray(x, dtype=np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    packed = np.packbits(bits.astype(np.uint8))
    pad = (-packed.size) % 8
    if pad:
        packed = np.concatenate([packed, np.zeros(pad, dtype=np.uint8)])
    return np.frombuffer(packed.tobytes(), dtype=">u8").astype(np.uint64)


def dhash64(gray: Image.Image) -> int:
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).ravel()
    return int(_pack_bits(bits)[0])


def phash256(gray: Image.Image, size=32, low=16) -> np.ndarray:
    """DCT 感知哈希：32×32 → 低频 16×16（256 位）。"""
    small = gray.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    mat = _dct_matrix(size)
    coeff = mat @ arr @ mat.T
    block = coeff[:low, :low].ravel()
    median = np.median(block[1:])
    return _pack_bits(block > median)


def histogram(rgb: Image.Image, edge=64) -> np.ndarray:
    """曝光/白平衡无关的颜色分布（96 桶），用于 Bhattacharyya 距离。

    旧版是 4×4×4 的绝对 RGB：同一张照片把亮度调高 18%，直方图距离就会飙到
    0.39~0.72，被当成「颜色完全不同」而漏掉（实测 21/216 对）。这里把每个
    像素拆成两半互不干扰的信息：
      * 色度 (r, g)/(r+g+b)，8×8 量化 —— 与曝光、白平衡无关；
      * 相对亮度 L/mean(L)，按 log2 分 32 档 —— 只管明暗结构，也被归一化掉了。
    两项按 0.75 / 0.25 加权：既保留「颜色在画面里的占比」，也保留明暗层次。
    """
    small = rgb.resize((edge, edge), Image.Resampling.BOX)
    arr = np.asarray(small, dtype=np.float32) + 1.0
    total = arr.sum(axis=2)
    chroma = arr[..., :2] / total[..., None]
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    luma = luma / max(float(luma.mean()), 1e-3)
    ci = np.clip((chroma * HIST_CHROMA).astype(np.int32), 0, HIST_CHROMA - 1)
    li = np.clip((np.log2(np.maximum(luma, 1e-3)) + 2.0) * (HIST_LUMA / 4.0),
                 0, HIST_LUMA - 1).astype(np.int32)
    idx = ci[..., 0] * HIST_CHROMA + ci[..., 1]
    hist = np.zeros(HIST_SIZE, dtype=np.float32)
    hist[:HIST_CHROMA * HIST_CHROMA] = np.bincount(
        idx.ravel(), minlength=HIST_CHROMA * HIST_CHROMA) * 0.75
    hist[HIST_CHROMA * HIST_CHROMA:] = np.bincount(
        li.ravel(), minlength=HIST_LUMA) * 0.25
    mass = float(hist.sum())
    if mass > 0:
        hist /= mass
    return hist


def _box_blur(arr, radius=2):
    """环绕边界（与原 np.roll 版本等价）的盒式均值。

    原来是 25 次 np.roll：单个 32×32 数组要 0.5 毫秒，缩放搜索每比对一对图
    要做十来次，全表扫描几乎全耗在这里。改成两轴各一次累积和相减。
    """
    a = np.asarray(arr, dtype=np.float32)
    if radius <= 0:
        return a.copy()
    k = radius * 2 + 1
    out = a
    for axis in (0, 1):
        n = out.shape[axis]
        head = [slice(None)] * out.ndim
        tail = [slice(None)] * out.ndim
        head[axis] = slice(n - radius, n)
        tail[axis] = slice(0, radius)
        padded = np.concatenate([out[tuple(head)], out, out[tuple(tail)]],
                                axis=axis)
        cs = np.cumsum(padded, axis=axis)
        zero = [slice(None)] * out.ndim
        zero[axis] = slice(0, 1)
        cs = np.concatenate([np.zeros_like(cs[tuple(zero)]), cs], axis=axis)
        hi = [slice(None)] * out.ndim
        lo = [slice(None)] * out.ndim
        hi[axis] = slice(k, k + n)
        lo[axis] = slice(0, n)
        out = (cs[tuple(hi)] - cs[tuple(lo)]) / k
    return out


def _orients(arr):
    """数组版的 4 旋转 × 2 镜像（与 _orientations 的顺序一致）。"""
    out = []
    current = arr
    for _ in range(4):
        out.append(current)
        out.append(np.fliplr(current))
        current = np.rot90(current)
    return out


def _highpass_arr(arr):
    """对已经缩到 32×32 的灰度数组做高通（缩放搜索直接复用，省一次重采样）。"""
    g = np.asarray(arr, dtype=np.float32)
    hp = g - _box_blur(g, 2)
    hp -= hp.mean()
    energy = float(np.abs(hp).mean())
    norm = float(np.sqrt(float((hp * hp).sum())))
    if norm < 1e-6:
        return np.zeros((32, 32), dtype=np.float32), energy
    return (hp / norm).astype(np.float32), energy


def _highpass(gray):
    """返回 (单位范数高频分量, 细节能量)。

    能量 = 高频的绝对均值：纯色天空、雾面墙这类画面能量很低，哈希和相关的
    结果基本由压缩噪声决定，配对判据必须比有纹理的照片更严。
    """
    g = np.asarray(gray.resize((32, 32), Image.Resampling.BOX),
                   dtype=np.float32)
    hp = g - _box_blur(g, 2)
    hp -= hp.mean()
    energy = float(np.abs(hp).mean())
    norm = float(np.sqrt(float((hp * hp).sum())))
    if norm < 1e-6:
        return np.zeros((32, 32), dtype=np.float32), energy
    return (hp / norm).astype(np.float32), energy


def local_structure(gray) -> np.ndarray:
    """32×32 高频分量，去均值后归一化。

    减去低频（盒滤波）后再归一化，等于只看「边缘和纹理」，整体明暗、色调、
    白平衡都不参与；两张图做点积就是归一化互相关。
    """
    return _highpass(gray)[0]


def color_grid(rgb) -> np.ndarray:
    """8×8 网格的平均颜色（0..255）：颜色在画面中的位置，而非全局配色。"""
    return np.asarray(rgb.resize((8, 8), Image.Resampling.BOX), dtype=np.uint8)


def structure_ncc(a, b) -> float:
    """局部结构归一化互相关（-1..1），取 8 个方向里最好的一个。"""
    if a is None or b is None:
        return 1.0
    return max(float((a * o).sum()) for o in _orients(b))


def _layout_norm(grid) -> np.ndarray:
    """网格先按通道去均值（消整体色偏），再按整体幅度归一（消曝光/对比度）。

    只去均值不够：调亮是乘性的，偏差会一起放大，同一张图调亮 25% 就能让
    距离从 8 涨到 13、越过阈值。归一后纯曝光差异≈0，位置差异照旧保留。
    """
    arr = np.asarray(grid, dtype=np.float32)
    arr = arr - arr.mean(axis=(0, 1), keepdims=True)
    scale = float(np.abs(arr).mean())
    return arr if scale < 1e-6 else arr * (64.0 / scale)


def layout_distance_inner(a, b, keep=6) -> float:
    """只比中心 keep×keep 块的颜色布局，用均值。

    缩放对齐后，差异几乎都集中在被裁掉的最外圈：整幅网格的 75 分位会被这几
    块顶到 12 以上，而中心区的均值能把「真裁边」（8.5~10.8）和「主体挪了」
    （14.3~15.9）分开。
    """
    if a is None or b is None:
        return 0.0
    n = a.shape[0]
    keep = max(2, min(keep, n))
    off = (n - keep) // 2
    best = 1e9
    af = _layout_norm(a)[off:off + keep, off:off + keep]
    for o in _orients(b):
        bf = _layout_norm(o)[off:off + keep, off:off + keep]
        best = min(best, float(np.abs(af - bf).mean(axis=2).mean()))
    return best


def layout_distance(a, b) -> float:
    """8×8 网格颜色差的 75 分位（0..128），取 8 个方向里最小的一个。

    用分位数而不是均值：主体只占画面一部分时，均值会被大片相同背景稀释，
    分位数能反映「有一块地方明显对不上」。
    """
    if a is None or b is None:
        return 0.0
    af = _layout_norm(a)
    best = 1e9
    for o in _orients(b):
        diff = np.abs(af - _layout_norm(o)).mean(axis=2)
        best = min(best, float(np.percentile(diff, 75)))
    return best


def zone_shape(image, target=96, low=8, high=16):
    """按图片尺寸挑网格：每块尽量是 target 像素的正方形。

    两个方向用同一个块尺寸算，极端宽高比（比如 4:1 全景）才不会把方块拉成
    扁条。块太细则重压缩噪点冒充差异（实测真实截图从 20 格起，真重复也会
    冒出 3~4 个「细节差异」块），所以长边格数封在 16。
    """
    if image is None:
        return (low, low)
    width, height = image.size
    long_side = max(width, height, 1)
    block = max(float(target), long_side / float(high))
    block = min(block, long_side / float(low))
    cols = max(1, int(round(width / block)))
    rows = max(1, int(round(height / block)))
    return (min(high, rows), min(high, cols))


def block_diff_maps(left, right, cells=8):
    """逐块比较两张已对齐的 RGB 图，返回 (颜色差, 细节相关)。

    cells 可以是整数（正方形网格）或 (行数, 列数)，返回的 color/corr 都是
    (行数, 列数)。color 是去掉整体色偏后每块的平均颜色差（0..255），corr 是
    每块高频结构的余弦相似度（-1..1，1 = 完全一致）。
    与 match_photos 的判据同源：界面标出的「哪里不一样」就是算法眼里的差异。

    工作分辨率跟着格数走（颜色每块 8 像素、细节每块 4 像素），所以块变细时
    每块看到的仍是同等大小的一小片画面，阈值不用跟着调。
    """
    if left is None or right is None or np is None or Image is None:
        return None, None
    if isinstance(cells, (tuple, list)):
        rows, cols = int(cells[0]), int(cells[1])
    else:
        rows = cols = int(cells)
    rows, cols = max(1, rows), max(1, cols)
    cw, ch = cols * 8, rows * 8
    a = np.asarray(left.resize((cw, ch), Image.Resampling.BOX),
                   dtype=np.float32)
    b = np.asarray(right.resize((cw, ch), Image.Resampling.BOX),
                   dtype=np.float32)
    a = a - a.mean(axis=(0, 1), keepdims=True)
    b = b - b.mean(axis=(0, 1), keepdims=True)

    def blocks(arr):
        return arr[:ch, :cw].reshape(rows, 8, cols, 8, -1).mean(axis=(1, 3))

    color = np.abs(blocks(a) - blocks(b)).mean(axis=2)

    gw, gh = max(32, cols * 4), max(32, rows * 4)
    radius = max(2, int(round(min(gw, gh) / 16.0)))
    ga = np.asarray(left.convert("L").resize((gw, gh),
                                             Image.Resampling.BOX),
                    dtype=np.float32)
    gb = np.asarray(right.convert("L").resize((gw, gh),
                                              Image.Resampling.BOX),
                    dtype=np.float32)
    ha = ga - _box_blur(ga, radius)
    hb = gb - _box_blur(gb, radius)
    bw, bh = max(1, gw // cols), max(1, gh // rows)

    def flat(arr):
        return (arr[:bh * rows, :bw * cols].reshape(rows, bh, cols, bw)
                .transpose(0, 2, 1, 3).reshape(rows, cols, -1))

    fa, fb = flat(ha), flat(hb)
    na = np.sqrt((fa * fa).sum(axis=2))
    nb = np.sqrt((fb * fb).sum(axis=2))
    dot = (fa * fb).sum(axis=2)
    corr = np.where((na > 1e-6) & (nb > 1e-6),
                    dot / np.maximum(na * nb, 1e-6), 1.0)
    return color.astype(np.float32), corr.astype(np.float32)


def pair_similarity(fine, ncc=1.0) -> int:
    """把「细节位差」和「结构相关」折成一个 0..100 的相似度。

    只用 pHash 位差会把 42/256（84%）当成很高的相似度；结构相关能把
    「结构大致像但细节对不上」的情况压到 95% 以下，交给用户人工确认。
    """
    detail = 1.0 - min(1.0, max(0.0, fine / 256.0))
    corr = min(1.0, max(0.0, float(ncc)))
    return max(0, min(100, round(100 * (0.45 * detail + 0.55 * corr))))


def _orientations(gray: Image.Image):
    out = []
    current = gray
    for _ in range(4):
        out.append(current)
        current = current.transpose(Image.Transpose.ROTATE_90)
    current = ImageOps.mirror(gray)
    for _ in range(4):
        out.append(current)
        current = current.transpose(Image.Transpose.ROTATE_90)
    return out


def _orientation_grids(gray: Image.Image, big=DCT_SIZE, small=9):
    """每个方向只准备两张小图：32×32（pHash）和 9×9（dHash）。

    旧实现把 1024px 整图旋转 8 次、再做 16 次大图缩放，实测占单张耗时 60%
    以上。这里整图只缩两次，方向在全方阵小数组上用 np.rot90 / fliplr 得到，
    对正方形目标与「先转整图再缩放」严格等价（散射可分离且对称）。
    """
    big_arr = np.asarray(gray.resize((big, big), Image.Resampling.LANCZOS),
                         dtype=np.float32)
    small_arr = np.asarray(gray.resize((small, small),
                                       Image.Resampling.LANCZOS),
                           dtype=np.int16)
    out = []
    for turn in range(4):
        b = np.rot90(big_arr, turn)
        s = np.rot90(small_arr, turn)
        out.append((b, s))
        out.append((np.fliplr(b), np.fliplr(s)))
    return out


def _phash_array(arr: np.ndarray, mat: np.ndarray, low=LOW_SIZE):
    coeff = mat @ arr @ mat.T
    block = coeff[:low, :low].ravel()
    med = np.median(block[1:])
    return _pack_bits(block > med)


def _dhash_grid(grid: np.ndarray) -> int:
    """9×9 亮度格的横向差分 → 64 位 dHash（取前 8 行× 8 列）。"""
    return int(_pack_bits((grid[:, 1:] > grid[:, :-1]).ravel())[0])


# ---------------------------------------------------------------- 指纹
_HP_SCALE = 512.0


def _pack_hp(arr) -> str:
    """32×32 高频分量压成 int8（量化误差 ≈ 0.002，对相关值影响在小数点后第三位）。

    直接存 float 列表每张要 7 KB 以上，一张 32×32 的 int8 + base64 只要 1.4 KB；
    层级从 1 提到 3 之后，缓存体积反而是原来的 1/3 左右。
    """
    q = np.clip(np.rint(np.asarray(arr, dtype=np.float32) * _HP_SCALE),
                -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _unpack_hp(text: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(text.encode("ascii")), dtype=np.int8)
    out = (raw.astype(np.float32) / _HP_SCALE).reshape(32, 32)
    # 量化会把单位范数弄偏一点点，复原后重新归一：相关值必须回到 0..1
    norm = float(np.sqrt(float((out * out).sum())))
    if norm > 1e-6:
        out /= norm
    return out


def _pack_thumb(arr) -> str:
    """64×64 RGB 中间图压成 PNG 再 base64（自然照片约 4~8 KB）。"""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGB").save(
        buf, "PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _unpack_thumb(text: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(str(text).encode("ascii"))
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _zoom_window(n, z):
    """中心取景窗 (边长, y0, x0)：保留中间 1/z 的区域。"""
    side = max(8, min(n, int(round(n / float(z)))))
    off = (n - side) // 2
    return side, off, off


def _zoom_view(thumb, z, size):
    """按 1/z 中心取景，再缩到 size×size（缩放搜索共用）。"""
    n = thumb.shape[0]
    side, y0, x0 = _zoom_window(n, z)
    piece = thumb[y0:y0 + side, x0:x0 + side]
    return Image.fromarray(np.ascontiguousarray(piece), "RGB").resize(
        (size, size), Image.Resampling.LANCZOS)


def zoom_hp(thumb, z) -> np.ndarray:
    """缩放 z 倍后的 32×32 高频单位向量。

    从 64×64 中间图直接做面积平均（积分图，纯 numpy）：缩放搜索里这一步要
    被调用上千次，走 Pillow 重采样每次 0.9 毫秒，全表扫描时就是几秒的差别。
    """
    return zoom_hp_gray(_thumb_gray(thumb), z)


def _thumb_gray(thumb) -> np.ndarray:
    return np.asarray(thumb, dtype=np.float32) @ np.array(
        [0.299, 0.587, 0.114], dtype=np.float32)


def zoom_hp_gray(gray, z) -> np.ndarray:
    """同 zoom_hp，但直接吃已经算好的 64×64 灰度（缩放搜索复用）。"""
    n = gray.shape[0]
    side, y0, x0 = _zoom_window(n, z)
    piece = gray[y0:y0 + side, x0:x0 + side]
    cs = np.cumsum(np.cumsum(piece, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    edges = np.round(np.arange(33) * side / 32.0).astype(int)
    lo = edges[:-1]
    hi = np.maximum(edges[1:], lo + 1)
    sums = (cs[np.ix_(hi, hi)] - cs[np.ix_(lo, hi)]
            - cs[np.ix_(hi, lo)] + cs[np.ix_(lo, lo)])
    counts = np.outer(hi - lo, hi - lo).astype(np.float32)
    return _highpass_arr(sums / counts)[0]


def zoom_grid(thumb, z) -> np.ndarray:
    return np.asarray(_zoom_view(thumb, z, 8), dtype=np.uint8)


def zoom_desc(thumb, z, mat=None):
    """缩放 z 倍后的 (dHash, pHash)，用来给出真实的粗差/细差。"""
    gray = _zoom_view(thumb, z, 32).convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    mat = _dct_matrix(DCT_SIZE) if mat is None else mat
    small = np.asarray(_zoom_view(thumb, z, 9).convert("L"),
                       dtype=np.int16)
    return _dhash_grid(small), _phash_array(arr, mat)


def exif_shot(path):
    """(拍摄时间 epoch 秒, 机型)；读不到就是 (0, "")。"""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get(36867) or exif.get(306) or ""
            model = str(exif.get(272) or "").strip()
    except Exception:
        return 0, ""
    text = str(raw).strip()
    if not text:
        return 0, ""
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(text, fmt).timestamp()), model
        except Exception:
            continue
    return 0, ""


@dataclass
class PhotoFP:
    dh: np.ndarray          # (LEVELS, 8) uint64，每个裁边层级 × 8 个方向的 dHash
    ph: np.ndarray          # (LEVELS, 8, 4) uint64，同上方向的 16×16 pHash
    hist: np.ndarray        # (96,) float32，曝光无关颜色分布
    width: int
    height: int
    canon: int              # 旋转不变签名（整幅 8 个方向取最小）
    hp: np.ndarray | None = None    # (LEVELS, 32, 32) 零均值单位范数高频分量
    grid: np.ndarray | None = None  # (LEVELS, 8, 8, 3) uint8 空间颜色布局
    energy: np.ndarray | None = None  # (LEVELS,) float32 各层细节能量
    thumb: np.ndarray | None = None   # (64, 64, 3) uint8，缩放搜索的原料
    exif_time: int = 0                # 拍摄时间（epoch 秒），0 = 没有
    exif_model: str = ""              # 机型，空 = 没有

    def to_cache(self):
        return {
            "v": 3,
            "dh": [[int(x) for x in row] for row in self.dh],
            "ph": [[[int(x) for x in row] for row in lv] for lv in self.ph],
            "hist": [round(float(x), 5) for x in self.hist],
            "w": int(self.width), "h": int(self.height),
            "hp": None if self.hp is None else [_pack_hp(lv) for lv in self.hp],
            "grid": None if self.grid is None else self.grid.tolist(),
            "energy": (None if self.energy is None
                       else [round(float(x), 4) for x in self.energy]),
            "thumb": (None if self.thumb is None
                      else _pack_thumb(self.thumb)),
            "exif_time": int(self.exif_time or 0),
            "exif_model": str(self.exif_model or ""),
        }

    @classmethod
    def from_cache(cls, data):
        if int(data.get("v", 1)) < 3:
            # v1 用绝对 RGB 直方图、v2 是固定裁边层；语义都变了，留着会静默算错
            raise ValueError("照片指纹缓存是旧格式")
        dh = np.array(data["dh"], dtype=np.uint64).reshape(LEVELS, ORIENTS)
        ph = np.array(data["ph"], dtype=np.uint64).reshape(LEVELS, ORIENTS, 4)
        hist = np.array(data["hist"], dtype=np.float32).reshape(HIST_SIZE)
        hp = data.get("hp")
        grid = data.get("grid")
        hp = (None if hp is None
              else np.stack([_unpack_hp(x) for x in hp]))
        grid = (None if grid is None
                else np.array(grid, dtype=np.uint8).reshape(LEVELS, 8, 8, 3))
        energy = data.get("energy")
        energy = (None if energy is None
                  else np.array(energy, dtype=np.float32).reshape(LEVELS))
        thumb = data.get("thumb")
        thumb = (None if not thumb else _unpack_thumb(thumb))
        return cls(dh, ph, hist, int(data["w"]), int(data["h"]),
                   int(dh[0].min()), hp, grid, energy, thumb,
                   int(data.get("exif_time") or 0),
                   str(data.get("exif_model") or ""))


def open_small(path, max_edge=1024):
    """按 EXIF 转正、缩到长边 max_edge 的 RGB 图（JPEG 会走 draft 加速）。

    没有方向标记就跳过 exif_transpose（省一次全图拷贝），RGB 图直接 copy。
    """
    with Image.open(path) as im:
        try:
            im.draft("RGB", (max_edge, max_edge))
        except Exception as exc:
            ignore(__name__, exc)
        try:
            orientation = im.getexif().get(274, 1)
        except Exception:
            orientation = 1
        if orientation not in (0, 1):
            im = ImageOps.exif_transpose(im)
        im = im.copy() if im.mode == "RGB" else im.convert("RGB")
        if im.width > max_edge or im.height > max_edge:
            im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return im


def probe_size(path):
    """只读图片头，返回 (宽, 高)；读不到返回 None（不解码像素）。"""
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def fingerprint(path, max_edge=1024) -> PhotoFP | None:
    try:
        rgb = open_small(path, max_edge)
        width, height = rgb.size
        if width < 16 or height < 16:
            return None
        # 裁边层级从 256×256 中间图里切：直接在 64×64 上切会让「裁边版」和
        # 「整幅版」的视频域重采样差太多（实测结构相关只有 0.84~0.94，余量不够
        # 区分误判），从 256 切再缩到 32 就接近真正对齐。
        mid = rgb.resize((256, 256), Image.Resampling.BOX)
        small = rgb.resize((64, 64), Image.Resampling.BOX)
        mat = _dct_matrix(DCT_SIZE)
        dh: list[list[int]] = []
        ph: list[list[np.ndarray]] = []
        hps: list[np.ndarray] = []
        grids: list[np.ndarray] = []
        energies: list[float] = []
        for keep in CROP_LEVELS:
            side = max(48, int(round(256 * keep)))
            off = (256 - side) // 2
            piece = mid.crop((off, off, off + side, off + side))
            gray = piece.convert("L")
            hp_unit, energy = _highpass(gray)
            hps.append(hp_unit)
            energies.append(energy)
            grids.append(color_grid(piece))
            row_dh, row_ph = [], []
            for big_arr, small_arr in _orientation_grids(gray, DCT_SIZE):
                row_dh.append(_dhash_grid(small_arr))
                row_ph.append(_phash_array(big_arr, mat))
            dh.append(row_dh)
            ph.append(row_ph)
        dh_arr = np.array(dh, dtype=np.uint64).reshape(LEVELS, ORIENTS)
        ph_arr = np.array(ph, dtype=np.uint64).reshape(LEVELS, ORIENTS, 4)
        thumb = np.asarray(rgb.resize((THUMB_SIZE, THUMB_SIZE),
                                      Image.Resampling.BOX), dtype=np.uint8)
        shot_time, shot_model = exif_shot(path)
        return PhotoFP(dh_arr, ph_arr, histogram(small), width, height,
                       int(dh_arr[0].min()), np.stack(hps), np.stack(grids),
                       np.array(energies, dtype=np.float32), thumb,
                       shot_time, shot_model)
    except Exception:
        return None


# ---------------------------------------------------------------- 匹配

def _shot_ratio(fp) -> float:
    return fp.width / fp.height if fp.height else 1.0


def _unique_fingerprints(fps):
    """把指纹完全相同的条目折叠成一个代表元。

    dh / ph / hist / 宽高比 都相同的两张图，比对时每一项差值只由指纹决定，
    因此它们在原算法里的结果必然一致：先在唯一集合上跑 O(m^2) 比对，
    再把每条边展开成原来的所有下标对，结果与逐对比对完全相同。
    """
    groups = {}
    for idx, fp in enumerate(fps):
        key = (fp.dh.tobytes(), fp.ph.tobytes(), fp.hist.tobytes(),
               _shot_ratio(fp))
        groups.setdefault(key, []).append(idx)
    members = list(groups.values())
    reps = [group[0] for group in members]
    return members, reps


def _corr_rows(ha, hb) -> np.ndarray:
    """(kk,32,32) 两组单位向量取 8 个方向里最大的相关（structure_ncc 的批量版）。"""
    best = np.full(ha.shape[0], -1.0, dtype=np.float32)
    flat = ha.reshape(ha.shape[0], -1)
    for turn in range(4):
        turned = np.rot90(hb, turn, axes=(1, 2))
        for flipped in (False, True):
            cur = turned[:, :, ::-1] if flipped else turned
            np.maximum(best, np.einsum("ij,ij->i", flat,
                                       np.ascontiguousarray(cur).reshape(
                                           cur.shape[0], -1)), out=best)
    return best


def match_photos(fps, dist=8, fine=42, hist_tol=0.26, ratio_tol=0.12,
                 ncc_tol=0.95, layout_tol=12.0, crop_ncc_tol=None,
                 cancel=None, on_progress=None):
    """返回 [(i, j, 粗差, 细差, 直方图距离, 结构相关, 布局差)]，i < j。

    三道路径，代价从低到高：
      1) 同尺度：dHash 粗差 + pHash 细差 + 结构相关 + 色彩布局；
      2) EXIF：同机型且拍摄时间相差 ≤2 秒，结构阈值放宽一档——裁切、加字、
         换色调的版本几何对不上，但拍摄信息还在；
      3) 缩放搜索：把一方的 64×64 中间图按 1/z 重新对齐后再比。原先预切的
         固定裁边层只能覆盖四边各裁 6%/12% 两档，实测 4%/8%/10%/14% 整类漏判。
    """
    n = len(fps)
    if n < 2:
        return []
    # 缩放对齐要经过一次重采样，相关值天然比同尺度低：实测真重复在 0.94
    # 以上，「同背景换人」在 0.51 附近，所以同尺度用 0.95、对齐后用 0.89。
    if crop_ncc_tol is None:
        crop_ncc_tol = min(float(ncc_tol), CROP_NCC_TOL)
    members, reps = _unique_fingerprints(fps)
    uniq = [fps[i] for i in reps]
    m = len(uniq)

    dh = np.array([f.dh for f in uniq], dtype=np.uint64).reshape(
        m, LEVELS, ORIENTS)
    ph = np.array([f.ph for f in uniq], dtype=np.uint64).reshape(
        m, LEVELS, ORIENTS, 4)
    hist = np.array([f.hist for f in uniq], dtype=np.float32).reshape(
        m, HIST_SIZE)
    ratios = np.array([_shot_ratio(f) for f in uniq], dtype=np.float64)
    has_hp = np.array([f.hp is not None for f in uniq])
    hp0 = np.zeros((m, 32, 32), dtype=np.float32)
    for i, f in enumerate(uniq):
        if f.hp is not None:
            hp0[i] = f.hp[0]
    exif_t = np.array([int(f.exif_time or 0) for f in uniq], dtype=np.int64)
    model_ids = {name: i + 1 for i, name in
                 enumerate(sorted({f.exif_model for f in uniq if f.exif_model}))}
    exif_m = np.array([model_ids.get(f.exif_model, 0) for f in uniq],
                      dtype=np.int32)
    mat = _dct_matrix(DCT_SIZE)

    out = []
    # 同指纹组内部的边：粗差、细差恒为 0，直方图距离按原公式（同一数组）算一次
    for group in members:
        if len(group) < 2:
            continue
        h0 = fps[group[0]].hist
        hist_d = float(1.0 - np.sqrt(h0 * h0).sum())
        if hist_d > hist_tol:
            continue
        # 指纹量化后撞到一起的两张图（哈希相同、内容不同）也要过结构复核
        base = fps[group[0]]
        keep = {group[0]}
        corr_v = {group[0]: 1.0}
        layout_v = {group[0]: 0.0}
        for i in group[1:]:
            corr_v[i] = (1.0 if base.hp is None or fps[i].hp is None
                         else structure_ncc(base.hp[0], fps[i].hp[0]))
            if corr_v[i] < ncc_tol:
                continue
            layout_v[i] = (0.0 if base.grid is None or fps[i].grid is None
                           else layout_distance(base.grid[0],
                                                fps[i].grid[0]))
            if layout_v[i] > layout_tol:
                continue
            keep.add(i)
        if len(keep) < 2:
            continue
        for pos, i in enumerate(group):
            if i not in keep:
                continue
            for j in group[pos + 1:]:
                if j not in keep:
                    continue
                out.append((i, j, 0, 0, hist_d, corr_v[i],
                            layout_v.get(i, 0.0)))
    if m < 2:
        out.sort(key=lambda edge: (edge[0], edge[1]))
        return out

    chunk = max(8, min(512, int(400_000 / max(m, 1))))
    prefilter = int(dist) + PREFILTER_SLACK
    prefilter_zoom = int(dist) + PREFILTER_ZOOM_SLACK
    all_idx = np.arange(m)
    total_blocks = (m + chunk - 1) // chunk
    dh0 = dh[:, 0, :]
    # b 的 8 个循环移位：min over r/o 等价于 min over 全部 (o, o') 方向对
    shifts = np.stack([np.roll(dh0, -r, axis=1) for r in range(ORIENTS)])

    ladder_cache = {}
    orient_cache = {}
    gray_cache = {}

    def _gray(pos):
        g = gray_cache.get(pos)
        if g is None:
            fp = uniq[pos]
            g = () if fp.thumb is None else _thumb_gray(fp.thumb)
            gray_cache[pos] = g
        return g
    def _dst_rows(pos):
        """目标图的 8 个方向预展开成 (8, 1024)：缩放搜索里同一张目标图会被
        几十个配对反复比对，每次重建方向既慢又白费。"""
        rows = orient_cache.get(pos)
        if rows is None:
            fp = uniq[pos]
            if fp.hp is None:
                rows = ()
            else:
                rows = np.stack([o.ravel() for o in _orients(fp.hp[0])])
                rows = rows.astype(np.float32)
            orient_cache[pos] = rows
        return rows

    def _ladder(pos):
        """一次算好这张图在 9 个档位上的高频向量：同一张图会出现在很多配对
        里，不缓存的话每次都要重采样一遍，全表扫描时占掉大半时间。"""
        rows = ladder_cache.get(pos)
        if rows is None:
            fp = uniq[pos]
            if fp.thumb is None:
                rows = ()
            else:
                g = _gray(pos)
                rows = np.stack([zoom_hp_gray(g, z) for z in ZOOM_LADDER])
            ladder_cache[pos] = rows
        return rows

    def _zoom_hit(src_pos, dst_pos, tol):
        """把 src 的中间图按 1/z 对齐到 dst，返回 (相关, 布局差, z)。

        z = 0 表示「粗档差得太远」。
        """
        src, dst = uniq[src_pos], uniq[dst_pos]
        if src.thumb is None or src.hp is None or dst.hp is None:
            return -2.0, None, 0.0
        rows = _ladder(src_pos)
        targets = _dst_rows(dst_pos)
        if not len(rows) or not len(targets):
            return -2.0, None, 0.0
        ladder = np.stack(rows).reshape(len(rows), -1) @ targets.T
        scores = [(float(v), float(z))
                  for v, z in zip(ladder.max(axis=1), ZOOM_LADDER)]
        scores.sort(reverse=True)
        best_v, best_z = scores[0]
        if best_v < tol - 0.15:
            return best_v, None, 0.0
        # 粗档有时会被相邻档抢走（真峰在 1.40，最高一档却在 1.46），所以前两
        # 档各细扫一遍
        for _, center in scores[:2]:
            for z in np.arange(center - 0.03, center + 0.0301, 0.01):
                v = float((zoom_hp(src.thumb, float(z)).ravel()
                           @ targets.T).max())
                if v > best_v:
                    best_v, best_z = v, float(z)

        if best_v < tol:
            return best_v, None, best_z
        lay = 0.0
        if dst.grid is not None:
            lay = layout_distance_inner(zoom_grid(src.thumb, best_z),
                                        dst.grid[0])
            if lay > ZOOM_LAYOUT_TOL:
                return best_v, None, best_z
        return best_v, lay, best_z

    def run_block(start, end):
        found = []
        blk = all_idx[start:end]
        ratio = ratios[blk, None]
        ratio_ok = (np.abs(ratio / ratios[None, :] - 1.0) <= ratio_tol) | (
            np.abs(ratio * ratios[None, :] - 1.0) <= ratio_tol)
        cols = np.nonzero(ratio_ok.any(axis=0))[0]
        # 只需要 j > i >= blk[0] 的列：右上半三角以外本来就会被 mask 丢掉
        cols = cols[cols > blk[0]]
        if not cols.size:
            return found
        # 严格等价于全方向最小汉明距离（旧版用「最小哈希值」当签名，一旦取
        # 最小的方向变了，距离就完全失真，会静默漏掉真匹配）
        best = np.full((blk.size, cols.size), 255, dtype=np.uint8)
        rows = dh0[blk]
        for r in range(ORIENTS):
            shifted = shifts[r][cols]
            for orient in range(ORIENTS):
                diff = popcount(rows[:, None, orient]
                                ^ shifted[None, :, orient]).astype(np.uint8)
                np.minimum(best, diff, out=best)
        base = ratio_ok[:, cols] & (cols[None, :] > blk[:, None])
        # EXIF 一致的配对不受粗筛限制：裁切/加字/换色调会把整幅哈希推得很远
        blk_t = exif_t[blk][:, None]
        blk_m = exif_m[blk][:, None]
        col_t = exif_t[cols][None, :]
        col_m = exif_m[cols][None, :]
        same_shot = ((blk_t > 0) & (blk_m > 0) & (blk_m == col_m)
                     & (np.abs(blk_t - col_t) <= EXIF_TIME_TOL))
        ii, jj = np.nonzero(((best <= prefilter_zoom) | same_shot) & base)
        a = blk[ii]
        b = cols[jj]
        pref = best[ii, jj]
        if not a.size:
            return found

        k = a.size
        step = 4_000
        for s in range(0, k, step):
            sl = slice(s, min(s + step, k))
            hist_dist = 1.0 - np.sqrt(hist[a[sl]] * hist[b[sl]]).sum(axis=1)
            keep = np.nonzero(hist_dist <= hist_tol)[0]
            if not keep.size:
                continue
            ia_all, ib_all = a[sl][keep], b[sl][keep]
            hist_ok = hist_dist[keep]
            allow_same = pref[sl][keep] <= prefilter
            kk = keep.size
            corr = np.ones(kk, dtype=np.float32)
            hp_ok = has_hp[ia_all] & has_hp[ib_all]
            if hp_ok.any():
                picks = np.nonzero(hp_ok)[0]
                corr[picks] = _corr_rows(hp0[ia_all[picks]],
                                         hp0[ib_all[picks]])

            # ---- 1) 同尺度：细差与结构都达标，再看色彩布局
            coarse = popcount(dh[ia_all][:, 0, :, None]
                              ^ dh[ib_all][:, 0, None, :])
            fine_all = popcount(ph[ia_all][:, 0, :, None, :]
                                ^ ph[ib_all][:, 0, None, :, :]).sum(axis=3)
            valid = ((coarse <= dist) & (fine_all <= fine)
                     & allow_same[:, None, None])
            flat = np.where(valid, fine_all, 10_000).reshape(kk, -1)
            arg = flat.argmin(axis=1)
            cur = flat[np.arange(kk), arg]
            coarse_flat = coarse.reshape(kk, -1)
            accepted = np.zeros(kk, dtype=bool)
            for pos in np.nonzero((cur < 10_000) & (corr >= ncc_tol)
                                  & allow_same)[0]:
                ia, ib = int(ia_all[pos]), int(ib_all[pos])
                fa, fb = uniq[ia], uniq[ib]
                lay = (0.0 if fa.grid is None or fb.grid is None
                       else layout_distance(fa.grid[0], fb.grid[0]))
                if lay > layout_tol:
                    continue
                accepted[pos] = True
                found.append((ia, ib, int(coarse_flat[pos, arg[pos]]),
                              int(cur[pos]), float(hist_ok[pos]),
                              float(corr[pos]), lay))

            # ---- 2) EXIF：几何对不上、拍摄信息对得上
            shot = np.nonzero(
                ~accepted
                & (exif_t[ia_all] > 0) & (exif_m[ia_all] > 0)
                & (exif_m[ia_all] == exif_m[ib_all])
                & (np.abs(exif_t[ia_all] - exif_t[ib_all]) <= EXIF_TIME_TOL)
                & (corr >= EXIF_NCC_TOL))[0]
            for pos in shot:
                ia, ib = int(ia_all[pos]), int(ib_all[pos])
                fa, fb = uniq[ia], uniq[ib]
                lay = (0.0 if fa.grid is None or fb.grid is None
                       else layout_distance(fa.grid[0], fb.grid[0]))
                if lay > layout_tol:
                    continue
                f2 = fine_all[pos]
                accepted[pos] = True
                found.append((ia, ib, int(coarse_flat[pos, f2.argmin()]),
                              int(f2.min()), float(hist_ok[pos]),
                              float(corr[pos]), lay))

            # ---- 3) 缩放搜索：任意裁边量。按色分布从近到远排队，限量做
            rest = np.nonzero(~accepted)[0]
            if rest.size:
                order = rest[np.lexsort((-corr[rest], hist_ok[rest]))]
                for pos in order[:ZOOM_MAX_PAIRS]:
                    ia, ib = int(ia_all[pos]), int(ib_all[pos])
                    hit = _zoom_hit(ia, ib, crop_ncc_tol)
                    src, dst = uniq[ia], uniq[ib]
                    # 方向不对称：镜像缩放窗口只能把「被裁的那张」当 src，
                    # 所以失败后必须换方向再试一次（不能因为 z 非 0 就跳过）
                    if hit[1] is None:
                        hit = _zoom_hit(ib, ia, crop_ncc_tol)
                        src, dst = uniq[ib], uniq[ia]
                    if hit[1] is None:
                        continue
                    corr_v, lay, z = hit
                    g_dh, g_ph = zoom_desc(src.thumb, z, mat)
                    c_v = int(popcount(np.uint64(g_dh) ^ dst.dh[0]).min())
                    f_v = int(popcount(np.uint64(g_ph)[None, :]
                                       ^ dst.ph[0]).sum(axis=1).min())
                    found.append((ia, ib, c_v, f_v, float(hist_ok[pos]),
                                  float(corr_v), lay))
        return found

    workers = max(1, min(8, os.cpu_count() or 4))
    raw = []
    if workers == 1 or total_blocks < 2:
        for block_no, start in enumerate(range(0, m, chunk)):
            check_cancel(cancel)
            raw.extend(run_block(start, min(start + chunk, m)))
            if on_progress is not None:
                on_progress(block_no + 1, total_blocks, "照片比对")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_block, start, min(start + chunk, m))
                       for start in range(0, m, chunk)]
            for done, future in enumerate(as_completed(futures), 1):
                check_cancel(cancel)
                raw.extend(future.result())
                if on_progress is not None:
                    on_progress(done, total_blocks, "照片比对")

    # 唯一集合上的边展开回原始下标（同一指纹组内的序号可能交错，故取 min/max）
    for la, lb, coarse_v, fine_v, hist_v, corr_v, lay in raw:
        for i in members[la]:
            for j in members[lb]:
                if i < j:
                    out.append((i, j, coarse_v, fine_v, hist_v, corr_v, lay))
                else:
                    out.append((j, i, coarse_v, fine_v, hist_v, corr_v, lay))
    out.sort(key=lambda edge: (edge[0], edge[1]))
    return out
