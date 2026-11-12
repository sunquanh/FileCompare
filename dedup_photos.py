"""相似照片：dHash 粗筛 + pHash 复核 + 颜色直方图 + 完全链接聚类。

与旧实现相比：
  * 8 个方向（4 旋转 × 2 镜像）都参与匹配，翻拍/镜像图也能归组；
  * 粗筛用「8 个方向的 dHash 最小值」作为旋转不变签名，单次 popcount 即可过滤；
  * 命中候选再用 16×16 pHash（256 位，DCT）逐方向复核，并用颜色直方图排除
    「结构像但内容完全不同」的图；
  * 聚类改成完全链接，避免 A~B、B~C 却把 A、C 硬并成一组。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

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
    """4×4×4 RGB 直方图（64 桶），用于 Bhattacharyya 距离。

    用 BOX（面积平均）降采样，比 BILINEAR 抗频闪，也更便宜。
    """
    small = rgb.resize((edge, edge), Image.Resampling.BOX)
    arr = np.asarray(small, dtype=np.uint16) >> 6
    idx = (arr[..., 0] * 16 + arr[..., 1] * 4 + arr[..., 2]).ravel()
    hist = np.bincount(idx, minlength=64).astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


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
@dataclass
class PhotoFP:
    dh: np.ndarray          # (8,) uint64，8 个方向的 8x8 dHash
    ph: np.ndarray          # (8, 4) uint64，8 个方向的 16x16 pHash
    hist: np.ndarray        # (64,) float32
    width: int
    height: int
    canon: int              # 旋转不变签名（8 个方向取最小）

    def to_cache(self):
        return {
            "dh": [int(x) for x in self.dh],
            "ph": [[int(x) for x in row] for row in self.ph],
            "hist": [round(float(x), 5) for x in self.hist],
            "w": int(self.width), "h": int(self.height),
        }

    @classmethod
    def from_cache(cls, data):
        dh = np.array(data["dh"], dtype=np.uint64).reshape(ORIENTS)
        ph = np.array(data["ph"], dtype=np.uint64).reshape(ORIENTS, 4)
        hist = np.array(data["hist"], dtype=np.float32).reshape(64)
        return cls(dh, ph, hist, int(data["w"]), int(data["h"]), int(dh.min()))


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
        # 先在 RGB 上降到 64×64，再转灰度，省掉对整图的灰度转换
        small = rgb.resize((64, 64), Image.Resampling.BOX)
        gray = small.convert("L")
        mat = _dct_matrix(DCT_SIZE)
        dh: list[int] = []
        ph: list[np.ndarray] = []
        for big_arr, small_arr in _orientation_grids(gray, DCT_SIZE):
            dh.append(_dhash_grid(small_arr))
            ph.append(_phash_array(big_arr, mat))
        dh_arr = np.array(dh, dtype=np.uint64)
        ph_arr = np.array(ph, dtype=np.uint64).reshape(ORIENTS, 4)
        return PhotoFP(dh_arr, ph_arr, histogram(small), width, height,
                       int(dh_arr.min()))
    except Exception:
        return None


# ---------------------------------------------------------------- 匹配
def match_photos(fps, dist=8, fine=42, hist_tol=0.26, ratio_tol=0.12,
                 cancel=None, on_progress=None):
    """返回 [(i, j, 粗差, 细差, 直方图距离)]，i < j。"""
    n = len(fps)
    if n < 2:
        return []
    dh = np.array([f.dh for f in fps], dtype=np.uint64).reshape(n, ORIENTS)
    ph = np.array([f.ph for f in fps], dtype=np.uint64).reshape(n, ORIENTS, 4)
    hist = np.array([f.hist for f in fps], dtype=np.float32).reshape(n, 64)
    ratios = np.array([f.width / f.height if f.height else 1.0 for f in fps],
                      dtype=np.float64)

    chunk = max(8, min(512, int(400_000 / max(n, 1))))
    prefilter = int(dist) + 6
    all_idx = np.arange(n)
    total_blocks = (n + chunk - 1) // chunk
    # b 的 8 个循环移位：min over r/o 就等价于 min over 全部 (o, o') 方向对
    shifts = np.stack([np.roll(dh, -r, axis=1) for r in range(ORIENTS)])

    def run_block(start, end):
        found = []
        blk = all_idx[start:end]
        ratio = ratios[blk, None]
        ratio_ok = (np.abs(ratio / ratios[None, :] - 1.0) <= ratio_tol) | \
                   (np.abs(ratio * ratios[None, :] - 1.0) <= ratio_tol)
        cols = np.nonzero(ratio_ok.any(axis=0))[0]
        if not cols.size:
            return found
        # 严格等价于全方向最小汉明距离（旧版用“最小哈希值”当签名，
        # 一旦取最小的方向变了，距离就完全失真，会静默漏掉真匹配）
        best = np.full((blk.size, cols.size), 255, dtype=np.uint8)
        rows = dh[blk]
        for r in range(ORIENTS):
            shifted = shifts[r][cols]
            for orient in range(ORIENTS):
                diff = popcount(rows[:, None, orient]
                                ^ shifted[None, :, orient]).astype(np.uint8)
                np.minimum(best, diff, out=best)
        mask = (best <= prefilter) & ratio_ok[:, cols] & \
            (cols[None, :] > blk[:, None])
        ii, jj = np.nonzero(mask)
        b = cols[jj]
        if ii.size:
            a = blk[ii]
            k = a.size
            # 分块复核，控制内存
            step = 20_000
            for s in range(0, k, step):
                sl = slice(s, min(s + step, k))
                coarse = popcount(dh[a[sl]][:, :, None] ^ dh[b[sl]][:, None, :])
                fine_all = popcount(
                    ph[a[sl]][:, :, None, :] ^ ph[b[sl]][:, None, :, :]
                ).sum(axis=3)
                hist_dist = 1.0 - np.sqrt(
                    hist[a[sl]] * hist[b[sl]]).sum(axis=1)
                valid = (coarse <= dist) & (fine_all <= fine) & \
                    (hist_dist <= hist_tol)[:, None, None]
                any_valid = valid.reshape(valid.shape[0], -1).any(axis=1)
                if not any_valid.any():
                    continue
                scores = np.where(valid, fine_all, 10_000).reshape(
                    valid.shape[0], -1)
                best = scores.argmin(axis=1)
                flat_idx = np.nonzero(any_valid)[0]
                for pos in flat_idx:
                    flat = int(best[pos])
                    ra, rb = divmod(flat, ORIENTS)
                    la, lb = int(a[sl][pos]), int(b[sl][pos])
                    found.append((min(la, lb), max(la, lb),
                                    int(coarse[pos, ra, rb]),
                                    int(fine_all[pos, ra, rb]),
                                    float(hist_dist[pos])))
        return found

    workers = max(1, min(8, os.cpu_count() or 4))
    out = []
    if workers == 1 or total_blocks < 2:
        for block_no, start in enumerate(range(0, n, chunk)):
            check_cancel(cancel)
            out.extend(run_block(start, min(start + chunk, n)))
            if on_progress is not None:
                on_progress(block_no + 1, total_blocks, "照片比对")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_block, start, min(start + chunk, n))
                       for start in range(0, n, chunk)]
            for done, future in enumerate(as_completed(futures), 1):
                check_cancel(cancel)
                out.extend(future.result())
                if on_progress is not None:
                    on_progress(done, total_blocks, "照片比对")
    out.sort(key=lambda edge: (edge[0], edge[1]))
    return out

# ---------------------------------------------------------------- 缩放 / 裁边匹配
# 副本可能是原图的裁切（对称或偏心）、加边框、或只裁一边，整图指纹对不上时，
# 用直方图预筛出候选对，再对候选对做多档位、多位置、多形状的窗口哈希比对。
# 加边框是裁切的反向：原图 = 加了边框的图的一个窗口，所以双向窗口比对
# 就覆盖了 z < 1 的情况，不需要单独的分支。
# 窗口哈希只在候选对上现算（重开图片），不进缓存，指纹格式不变；但同一张
# 图的窗口哈希在它参与的每个候选对里都一样，所以按图缓存一次，多对复用。
# 每个窗口先算便宜的 dHash，不过门槛就不再算 pHash；两道门槛都比常规
# 放宽（dHash +4、pHash +20），因为窗口下采样会推高哈希距离。
# 候选先按直方图距离取每图最近 16 个；第一遍没轮上的、但直方图很近
# （<= 0.26，真副本的典型范围）的对再试第二遍。
ZOOM_SCALES = (1.11, 1.25, 1.43, 1.67, 2.0)  # 方形窗口边长 = 原图 / scale，对应裁 10%~50%
ZOOM_RECT_FRACS = (0.75, 0.67, 0.6)  # 矩形窗口保留的边长比例（裁 25%/33%/40%）
ZOOM_POSITIONS = ((0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                  (1.0, 1.0), (0.5, 0.0), (0.5, 1.0), (0.0, 0.5),
                  (1.0, 0.5))  # 中心优先，其次四角，再四边中点
ZOOM_HIST_TOL = 0.40
ZOOM_HIST_TOL2 = 0.26
ZOOM_MAX_PER_IMAGE = 16
ZOOM_PASS2_BUDGET = 6000
ZOOM_DIST_SLACK = 4
ZOOM_FINE_SLACK = 20


def _zoom_windows():
    """窗口清单：(宽占比, 高占比, 左上角 x 比例, 左上角 y 比例)。
    方形 5 档 x 9 位；矩形分横条（整宽、裁上下）和竖条（整高、裁左右）。
    """
    wins = []
    for s in ZOOM_SCALES:
        f = 1.0 / s
        for fx, fy in ZOOM_POSITIONS:
            wins.append((f, f, fx, fy))
    for r in ZOOM_RECT_FRACS:
        for p in (0.0, 0.5, 1.0):
            wins.append((1.0, r, 0.0, p))
            wins.append((r, 1.0, p, 0.0))
    return wins


ZOOM_WINDOWS = _zoom_windows()


def _gray64(path):
    """打开图片并按指纹同一流程缩到 64x64 灰度图；失败返回 None。"""
    try:
        rgb = open_small(path, 1024)
        return rgb.resize((64, 64), Image.Resampling.BOX).convert("L")
    except Exception:
        return None


def _window_box(wfrac, hfrac, fx, fy):
    w = 64.0 * wfrac
    h = 64.0 * hfrac
    x0 = int(round((64.0 - w) * fx))
    y0 = int(round((64.0 - h) * fy))
    return (x0, y0, int(round(x0 + w)), int(round(y0 + h)))


def _window_dhash(gray, wfrac, hfrac, fx, fy):
    win = gray.crop(_window_box(wfrac, hfrac, fx, fy))
    small = np.asarray(win.resize((9, 9), Image.Resampling.LANCZOS),
                       dtype=np.int16)
    return _dhash_grid(small)


def _window_phash(gray, wfrac, hfrac, fx, fy):
    win = gray.crop(_window_box(wfrac, hfrac, fx, fy))
    big = np.asarray(win.resize((DCT_SIZE, DCT_SIZE),
                                Image.Resampling.LANCZOS), dtype=np.float32)
    return _phash_array(big, _dct_matrix(DCT_SIZE))


def _zoom_pair(fps, win_dh, win_ph, i, j, hist_dist, dist, fine):
    """一对候选做双向窗口比对；命中返回 (i, j, 粗差, 细差, 直方图距离)。
    窗口按 ZOOM_WINDOWS 顺序试，第一个同时过两道门槛的即命中。
    """
    dist_lim = dist + ZOOM_DIST_SLACK
    fine_lim = fine + ZOOM_FINE_SLACK
    for a, b in ((i, j), (j, i)):
        dh_all = win_dh(a)
        if dh_all is None:
            continue
        dh_b = fps[b].dh
        ph_b = fps[b].ph
        coarse_all = popcount(dh_all[:, None] ^ dh_b[None, :]).min(axis=1)
        for w, coarse in enumerate(coarse_all):
            if coarse > dist_lim:
                continue
            ph_w = win_ph(a, w)
            if ph_w is None:
                continue
            fine_d = int(popcount(ph_w[None, :] ^ ph_b).sum(axis=1).min())
            if fine_d <= fine_lim:
                return (min(i, j), max(i, j), int(coarse), fine_d,
                        float(hist_dist))
    return None


def match_zoom(fps, paths, matched=None, dist=8, fine=42,
               hist_tol=ZOOM_HIST_TOL, max_per_image=ZOOM_MAX_PER_IMAGE,
               cancel=None):
    """缩放 / 裁边匹配，返回 [(i, j, 粗差, 细差, 直方图距离)]，i < j。
    只处理常规匹配没命中的对：先用直方图距离预筛（裁切基本不改变
    颜色分布），每图取最近的若干候选，再对候选对做窗口比对。
    已知缺口：直方图距离 > 0.26 又排在每图 16 名开外的真副本
    可能漏（例如裁掉了画面里的主导色块）；取舍见 ZOOM_HIST_TOL2。
    paths 与 fps 一一对应；paths 为空时直接返回 []（老调用方不受影响）。
    """
    n = len(fps)
    if n < 2 or not paths or len(paths) != n:
        return []
    matched_set = {(min(a, b), max(a, b)) for a, b in (matched or ())}
    hist = np.array([f.hist for f in fps], dtype=np.float32).reshape(n, 64)
    sq = np.sqrt(hist)
    all_cand = {}
    chunk = 256
    for start in range(0, n, chunk):
        check_cancel(cancel)
        sim = sq[start:start + chunk] @ sq.T
        d = 1.0 - sim
        for r in range(d.shape[0]):
            i = start + r
            row = d[r]
            for j in np.nonzero(row <= hist_tol)[0]:
                j = int(j)
                if j <= i or (i, j) in matched_set:
                    continue
                all_cand.setdefault(i, []).append((float(row[j]), j))
    pairs = []
    tried = set()
    gray_cache = {}
    dh_cache = {}
    ph_cache = {}

    def gray_of(idx):
        if idx not in gray_cache:
            gray_cache[idx] = _gray64(paths[idx])
        return gray_cache[idx]

    def win_dh(idx):
        if idx not in dh_cache:
            gray = gray_of(idx)
            if gray is None:
                dh_cache[idx] = None
            else:
                dh_cache[idx] = np.array(
                    [_window_dhash(gray, wf, hf, fx, fy)
                     for wf, hf, fx, fy in ZOOM_WINDOWS], dtype=np.uint64)
        return dh_cache[idx]

    def win_ph(idx, w):
        key = (idx, w)
        if key not in ph_cache:
            gray = gray_of(idx)
            if gray is None:
                ph_cache[key] = None
            else:
                wf, hf, fx, fy = ZOOM_WINDOWS[w]
                ph_cache[key] = _window_phash(gray, wf, hf, fx, fy)
        return ph_cache[key]

    def try_pair(i, j, hd):
        key = (i, j)
        if key in tried:
            return
        tried.add(key)
        check_cancel(cancel)
        hit = _zoom_pair(fps, win_dh, win_ph, i, j, hd, dist, fine)
        if hit is not None:
            pairs.append(hit)

    for i, lst in sorted(all_cand.items()):
        lst.sort()
        for hd, j in lst[:max_per_image]:
            try_pair(i, j, hd)
    rest = []
    for i, lst in all_cand.items():
        for hd, j in lst[max_per_image:]:
            if hd <= ZOOM_HIST_TOL2 and (i, j) not in tried:
                rest.append((hd, i, j))
    rest.sort()
    for hd, i, j in rest[:ZOOM_PASS2_BUDGET]:
        try_pair(i, j, hd)
    pairs.sort(key=lambda e: (e[0], e[1]))
    return pairs
