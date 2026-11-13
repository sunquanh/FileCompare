"""LaMa 修复模型后端（可选）：语义级补图，专门对付压在人物/结构上的水印。

为什么单独一层：LaMa 需要 onnxruntime 和 198 MiB 的模型文件，不该拖累
默认安装。没装 onnxruntime 或没放模型时，available() 返回 False，上层
自动退回 dedup_watermark 里的 numpy 镜像填充，行为不变。

模型：lama_fp32.onnx（Apache-2.0，Carve/LaMa-ONNX 导出）
输入：image float32 [1,3,512,512]（**0~1**，RGB），mask float32 [1,1,512,512]（0/1）
输出：float32 [1,3,512,512]，**已是 0~255**（导出时内部做过反归一化）
两个缩放都踩过坑：输入按 0~255 传会输出泛白，输出再乘 255 就整张白屏。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

MODEL_NAME = "lama_fp32.onnx"
SIZE = 512
_session = None
_session_path = None
_lock = threading.Lock()


def _roots():
    """模型搜索目录：程序目录、程序目录下的 models、数据目录下的 models。"""
    roots = []
    try:
        import dedup_models as models
        roots.append(models.program_dir())
        roots.append(models.program_dir() / "models")
        roots.append(models.app_dir() / "models")
    except Exception:
        pass
    roots.append(Path(__file__).resolve().parent)
    roots.append(Path(__file__).resolve().parent / "models")
    return roots


def model_path():
    """找到可用的模型文件；找不到返回 None。环境变量 LAMA_MODEL 优先。"""
    env = os.environ.get("LAMA_MODEL")
    if env and Path(env).is_file():
        return Path(env)
    for root in _roots():
        candidate = Path(root) / MODEL_NAME
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate
    return None


def available() -> bool:
    """能不能用 LaMa：onnxruntime 装了、模型也在。"""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return model_path() is not None


def status_text() -> str:
    """给界面显示的一行状态说明。"""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return "未安装 onnxruntime，高质量模式不可用"
    path = model_path()
    if path is None:
        return f"缺少模型文件 {MODEL_NAME}"
    return f"就绪：{path.name}（{path.stat().st_size / 1048576:.0f} MB）"


def _session_for(path):
    global _session, _session_path
    with _lock:
        if _session is not None and _session_path == str(path):
            return _session
        import onnxruntime as ort
        options = ort.SessionOptions()
        # 实测 8 逻辑核：4 线程 7.1 s，默认/8 线程 11.6 s，2 线程 11.6 s
        options.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        options.log_severity_level = 3
        _session = ort.InferenceSession(str(path), sess_options=options,
                                        providers=["CPUExecutionProvider"])
        _session_path = str(path)
        return _session


def warmup() -> bool:
    """把会话和首次推理做掉：拿来在界面里提前预热，避免用户点"开始"后干等。"""
    path = model_path()
    if path is None:
        return False
    session = _session_for(path)
    image = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
    mask = np.zeros((1, 1, SIZE, SIZE), dtype=np.float32)
    mask[:, :, SIZE // 2 - 16:SIZE // 2 + 16,
         SIZE // 2 - 16:SIZE // 2 + 16] = 1.0
    session.run(["output"], {"image": image, "mask": mask})
    return True


def _box_pixels(box, width, height):
    x0, y0, x1, y1 = box
    px0 = int(round(min(x0, x1) * width))
    py0 = int(round(min(y0, y1) * height))
    px1 = int(round(max(x0, x1) * width))
    py1 = int(round(max(y0, y1) * height))
    px0 = min(max(px0, 0), width - 1)
    px1 = min(max(px1, px0 + 1), width)
    py0 = min(max(py0, 0), height - 1)
    py1 = min(max(py1, py0 + 1), height)
    return px0, py0, px1, py1


def _crop_window(px0, py0, px1, py1, width, height, pad_ratio):
    """把框扩成一个带上下文的正方形窗口（尽量居中），并夹在画面内。"""
    box_w = px1 - px0
    box_h = py1 - py0
    side = int(round(max(box_w, box_h) * pad_ratio))
    side = max(side, 64)
    side = min(side, max(width, height))
    cx = (px0 + px1) // 2
    cy = (py0 + py1) // 2
    left = cx - side // 2
    top = cy - side // 2
    left = min(max(left, 0), max(0, width - side))
    top = min(max(top, 0), max(0, height - side))
    side = min(side, width, height)
    return left, top, left + side, top + side


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """方形结构元的膨胀（纯 numpy，用于把掩膜向外放一点再羽化）。"""
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(0, dy), mask.shape[0] + min(0, dy))
            xs = slice(max(0, dx), mask.shape[1] + min(0, dx))
            ys_src = slice(max(0, -dy), mask.shape[0] + min(0, -dy))
            xs_src = slice(max(0, -dx), mask.shape[1] + min(0, -dx))
            shifted[ys, xs] = mask[ys_src, xs_src]
            out |= shifted
    return out


def _reflect_index(length: int, before: int, after: int) -> np.ndarray:
    """反射填充用的索引序列，支持补边比原尺寸还长（np.pad 也能，但这样更可控）。"""
    pos = np.arange(-before, length + after, dtype=np.int64)
    if length == 1:
        return np.zeros_like(pos)
    period = 2 * length - 2
    idx = np.mod(pos, period)
    return np.where(idx >= length, period - idx, idx)


def _reflect_pad_to(rgb: np.ndarray, height: int, width: int,
                    off_y: int, off_x: int) -> np.ndarray:
    """把 (h,w,3) 原样贴进 height×width 画布的 (off_y,off_x)，四周反射补边。

    关键是不做 resize：窗口小于 512 时原来是「先放大 → 推理 → 再缩回」，
    两次重采样把框内细节糊掉；现在只是补上下文，输出仍按原像素取回。
    """
    h, w = rgb.shape[:2]
    iy = _reflect_index(h, off_y, height - h - off_y)
    ix = _reflect_index(w, off_x, width - w - off_x)
    return rgb[iy][:, ix]


def _neutralize_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """反射补边前把掩膜内换成左右最近的框外像素均值。

    掩膜内本来就要被模型替换，输入值无所谓；但如果不处理，反射平铺会把
    白块水印复制到画布上好几个位置，模型会把白色当成「周围纹理」学过去
    （实测小图上水印反而被补得更白）。中和后镜像出去的仍是连续纹理。
    """
    if not mask.any():
        return rgb
    height, width = mask.shape
    if not (~mask).any():
        return np.broadcast_to(rgb.mean(axis=(0, 1)), rgb.shape).copy()
    xx = np.broadcast_to(np.arange(width), (height, width))
    left = np.maximum.accumulate(np.where(~mask, xx, -1), axis=1)
    right = np.minimum.accumulate(np.where(~mask, xx, width)[:, ::-1],
                                  axis=1)[:, ::-1]
    left = np.clip(left, 0, width - 1)
    right = np.clip(right, 0, width - 1)
    rows = np.arange(height)[:, None]
    out = rgb.copy()
    out[mask] = ((rgb[rows, left] + rgb[rows, right]) * 0.5)[mask]
    return out


def _run_tile(session, rgb: np.ndarray, mask: np.ndarray,
              dilate: int = 4) -> np.ndarray:
    """单个 512 推理单元，返回与输入窗口同尺寸、按原像素对齐的预测（0~255）。

    crop 居中贴进 512 画布、掩膜同步平移；推理完只裁回 crop 那一块。
    全程没有一次缩放，窗口 ≤512 时框内细节保持原分辨率。
    """
    h, w = mask.shape
    if h > SIZE or w > SIZE:
        raise ValueError("窗口超过 512，请先滑窗分块")
    off_y, off_x = (SIZE - h) // 2, (SIZE - w) // 2
    source = np.asarray(rgb, dtype=np.float32)
    if h < SIZE or w < SIZE:
        source = _neutralize_mask(source, mask)   # 只有需要补边时才镜像
    canvas = _reflect_pad_to(source, SIZE, SIZE, off_y, off_x)
    image = (canvas / 255.0)[None].transpose(0, 3, 1, 2)   # NCHW, RGB, 0~1
    m512 = np.zeros((SIZE, SIZE), dtype=bool)
    m512[off_y:off_y + h, off_x:off_x + w] = mask
    if dilate > 0:
        m512 = _dilate(m512, dilate)
    pred = session.run(["output"],
                       {"image": image,
                        "mask": m512.astype(np.float32)[None, None]})[0]
    # 输出已是 0~255，不要再乘 255
    pred = np.clip(pred[0].transpose(1, 2, 0), 0, 255)
    return pred[off_y:off_y + h, off_x:off_x + w].astype(np.float32)


def _feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """0/1 掩膜 → 羽化 alpha（框内 1、框外 0、边界渐变），避免硬贴方框。"""
    from PIL import Image, ImageFilter
    if radius <= 0:
        return mask.astype(np.float32)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    blurred = img.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _tiled_inpaint(session, rgb_roi: np.ndarray, mask_roi: np.ndarray,
                   stride: int = 256, dilate: int = 4,
                   feather: float = 3.0) -> np.ndarray:
    """ROI 超过 512 时带重叠地滑 512 窗口，重叠区按羽化权重平均。

    不把大框整体缩小成 512（那样大框必糊）；每个窗口都在原生分辨率上
    推理，只把掩膜内的结果贴回去。跳过完全不含掩膜的窗口省推理时间。
    """
    h, w = mask_roi.shape
    th, tw = min(SIZE, h), min(SIZE, w)
    step = max(32, min(stride, th, tw))
    ys = list(range(0, max(1, h - th + 1), step))
    if ys[-1] != h - th:
        ys.append(h - th)
    xs = list(range(0, max(1, w - tw + 1), step))
    if xs[-1] != w - tw:
        xs.append(w - tw)
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + th, x0 + tw
            sub = mask_roi[y0:y1, x0:x1]
            if not sub.any():
                continue
            filled = _run_tile(session, rgb_roi[y0:y1, x0:x1], sub, dilate)
            a = _feather(sub, feather)[..., None]
            acc[y0:y1, x0:x1] += filled * a
            weight[y0:y1, x0:x1] += a
    return np.where(weight > 0, acc / np.maximum(weight, 1e-6), rgb_roi)


def inpaint(rgb: np.ndarray, boxes, pad_ratio: float = 1.8,
            blur: int = 6, tile_stride: int = 256,
            dilate: int = 4) -> np.ndarray:
    """用 LaMa 补掉所有框，返回新的 uint8 RGB 数组。

    小窗口（含上下文后 ≤512）直接贴进 512 画布做推理，不缩放；大框改用
    带重叠的滑窗，在原生分辨率上逐块修，不再整体缩小成 512。
    pad_ratio 决定给模型多少上下文，blur 是贴回时边缘的羽化像素数，
    tile_stride 是滑窗步长（越小重叠越多、越慢但接缝越干净）。
    """
    path = model_path()
    if path is None:
        raise RuntimeError("没有找到 LaMa 模型文件，无法使用高质量模式")
    if isinstance(boxes, list) is False:
        boxes = [boxes]
    session = _session_for(path)
    height, width = rgb.shape[:2]
    out = np.array(rgb, dtype=np.float32)
    for box in boxes:
        px0, py0, px1, py1 = _box_pixels(box, width, height)
        box_mask = np.zeros((height, width), dtype=bool)
        box_mask[py0:py1, px0:px1] = True
        # 上下文只随框大小适度增长；封顶避免大框滑出一堆没必要的窗口
        context = max(24, min(192,
                              int(max(px1 - px0, py1 - py0)
                                  * (pad_ratio - 1.0) * 0.5)))
        wy0, wy1 = max(0, py0 - context), min(height, py1 + context)
        wx0, wx1 = max(0, px0 - context), min(width, px1 + context)
        # 图够大且框装得下时，直接取原图的 512×512 原生窗口：比把小 crop
        # 镜像平铺到 512 更真实（不会把水印复制成周围纹理），也不用缩放
        cx, cy = (px0 + px1) // 2, (py0 + py1) // 2
        if (width >= SIZE and height >= SIZE
                and px1 - px0 <= SIZE - 128 and py1 - py0 <= SIZE - 128):
            wx0 = min(max(cx - SIZE // 2, 0), width - SIZE)
            wy0 = min(max(cy - SIZE // 2, 0), height - SIZE)
            wx1, wy1 = wx0 + SIZE, wy0 + SIZE
        roi = out[wy0:wy1, wx0:wx1]
        local_mask = box_mask[wy0:wy1, wx0:wx1]
        if roi.shape[0] <= SIZE and roi.shape[1] <= SIZE:
            filled = _run_tile(session, roi, local_mask, dilate)
        else:
            filled = _tiled_inpaint(session, roi, local_mask,
                                    tile_stride, dilate)
        # 羽化会渗到框边外几像素；框外一律换回原图，保证框外一个像素不动，
        # 框内边缘仍有 alpha 渐变（不会出现硬方框）
        a = _feather(local_mask, blur)[..., None]
        filled = np.where(local_mask[..., None], filled, roi)
        out[wy0:wy1, wx0:wx1] = roi * (1.0 - a) + filled * a
    # rint 而不是截断：alpha=0 处的浮点往返误差不该把 128 变成 127
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


def inpaint_mask(rgb: np.ndarray, mask: np.ndarray, stride: int = 256,
                 feather: int = 3) -> np.ndarray:
    """按像素掩膜修（不是框）：适合逐字抠出来的水印笔画。

    掩膜外接框扩一圈后，用 512 窗口带重叠地滑过去；每个窗口同样不缩放，
    重叠区按羽化权重真正做累加平均（旧版是后一个窗口直接覆盖）。
    """
    path = model_path()
    if path is None:
        raise RuntimeError("没有找到 LaMa 模型文件，无法使用高质量模式")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.array(rgb, dtype=np.uint8, copy=True)
    session = _session_for(path)
    height, width = rgb.shape[:2]
    ys, xs = np.nonzero(mask)
    pad = SIZE // 3
    top, left = max(0, ys.min() - pad), max(0, xs.min() - pad)
    bottom = min(height, ys.max() + pad + 1)
    right = min(width, xs.max() + pad + 1)
    roi = np.asarray(rgb[top:bottom, left:right], dtype=np.float32)
    sub_mask = mask[top:bottom, left:right]
    filled = _tiled_inpaint(session, roi, sub_mask, stride,
                            dilate=2, feather=feather)
    a = _feather(sub_mask, feather)[..., None]
    filled = np.where(sub_mask[..., None], filled, roi)
    out = np.array(rgb, dtype=np.float32)
    out[top:bottom, left:right] = roi * (1.0 - a) + filled * a
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


def remove_watermark_file(src, box, dst=None, pad_ratio: float = 1.8):
    """按文件处理一张照片，输出路径规则与 dedup_watermark 保持一致。"""
    from PIL import Image, ImageOps
    import dedup_watermark as wm
    src = Path(src)
    with Image.open(src) as im:
        transposed = ImageOps.exif_transpose(im)
        rgb = np.asarray(transposed.convert("RGB"))
    fixed = inpaint(rgb, box, pad_ratio)
    target = Path(dst) if dst else wm.output_path(src)
    image = Image.fromarray(fixed)
    if target.suffix.lower() in (".jpg", ".jpeg"):
        image.save(target, quality=95, subsampling=0)
    else:
        image.save(target)
    return target
