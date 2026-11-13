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


def inpaint(rgb: np.ndarray, boxes, pad_ratio: float = 1.8,
            blur: int = 6) -> np.ndarray:
    """用 LaMa 补掉所有框，返回新的 uint8 RGB 数组。

    pad_ratio 决定给模型多少上下文（框越大需要的周围信息越多），blur 是
    贴回时边缘的羽化像素数——模型输出和原图在框外会有细微色差，硬贴会
    看到方框。
    """
    path = model_path()
    if path is None:
        raise RuntimeError("没有找到 LaMa 模型文件，无法使用高质量模式")
    if isinstance(boxes, list) is False:
        boxes = [boxes]
    session = _session_for(path)
    height, width = rgb.shape[:2]
    out = np.array(rgb, dtype=np.uint8, copy=True)
    from PIL import Image, ImageFilter
    for box in boxes:
        px0, py0, px1, py1 = _box_pixels(box, width, height)
        win = _crop_window(px0, py0, px1, py1, width, height, pad_ratio)
        wx0, wy0, wx1, wy1 = win
        crop = Image.fromarray(out[wy0:wy1, wx0:wx1])
        crop512 = crop.resize((SIZE, SIZE), Image.BILINEAR)
        image = np.asarray(crop512, dtype=np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)  # NCHW, RGB, 0~1

        mask = np.zeros((wy1 - wy0, wx1 - wx0), dtype=np.uint8)
        mask[py0 - wy0:py1 - wy0, px0 - wx0:px1 - wx0] = 1
        mask_img = Image.fromarray(mask * 255).resize((SIZE, SIZE),
                                                      Image.NEAREST)
        m512 = (np.asarray(mask_img) > 127).astype(np.float32)
        m512 = _dilate(m512 > 0, 4).astype(np.float32)
        mask_in = m512[None, None]

        pred = session.run(["output"], {"image": image, "mask": mask_in})[0]
        # 输出已是 0~255，不要再乘 255
        pred = np.clip(pred[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        filled = Image.fromarray(pred).resize((wx1 - wx0, wy1 - wy0),
                                              Image.BILINEAR)

        # 只贴回框内（边缘羽化），框外像素保持原样
        alpha = np.zeros(mask.shape, dtype=np.float32)
        alpha[py0 - wy0:py1 - wy0, px0 - wx0:px1 - wx0] = 1.0
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
        if blur > 0:
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(blur))
        a = (np.asarray(alpha_img, dtype=np.float32) / 255.0)[..., None]
        region = out[wy0:wy1, wx0:wx1].astype(np.float32)
        blended = region * (1.0 - a) + np.asarray(filled, dtype=np.float32) * a
        out[wy0:wy1, wx0:wx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def inpaint_mask(rgb: np.ndarray, mask: np.ndarray, stride: int = 256,
                 feather: int = 3) -> np.ndarray:
    """按像素掩膜修（不是框）：适合逐字抠出来的水印笔画。

    做法：把掩膜的外接框扩一圈，用 512×512 的窗口带重叠地滑过去，每个
    窗口只把模型生成的“掩膜内”像素贴回去——窗口重叠区靠重叠平均，不会
    出现拼缝。窗口比整张小就是为了保住分辨率，不要把整页缩成一团。
    """
    path = model_path()
    if path is None:
        raise RuntimeError("没有找到 LaMa 模型文件，无法使用高质量模式")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.array(rgb, dtype=np.uint8, copy=True)
    session = _session_for(path)
    from PIL import Image, ImageFilter
    height, width = rgb.shape[:2]
    out = np.array(rgb, dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    ys, xs = np.nonzero(mask)
    pad = SIZE // 3
    top = max(0, ys.min() - pad)
    left = max(0, xs.min() - pad)
    bottom = min(height, ys.max() + pad)
    right = min(width, xs.max() + pad)
    step = max(32, min(stride, SIZE))
    for y0 in range(top, bottom, step):
        for x0 in range(left, right, step):
            y1 = min(y0 + SIZE, height)
            x1 = min(x0 + SIZE, width)
            if y1 - y0 < 32 or x1 - x0 < 32:
                continue
            sub_mask = mask[y0:y1, x0:x1]
            if not sub_mask.any():
                continue
            crop = Image.fromarray(np.asarray(rgb[y0:y1, x0:x1]))
            img512 = crop.resize((SIZE, SIZE), Image.BILINEAR)
            image = (np.asarray(img512, dtype=np.float32) / 255.0)[None]
            image = image.transpose(0, 3, 1, 2)
            m = Image.fromarray((sub_mask * 255).astype(np.uint8))
            m512 = np.asarray(m.resize((SIZE, SIZE), Image.NEAREST)) > 127
            m512 = _dilate(m512, 2).astype(np.float32)
            pred = session.run(["output"],
                               {"image": image,
                                "mask": m512[None, None]})[0]
            pred = np.clip(pred[0].transpose(1, 2, 0), 0, 255)
            filled = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(
                (x1 - x0, y1 - y0), Image.BILINEAR), dtype=np.float32)
            alpha = Image.fromarray((sub_mask * 255).astype(np.uint8))
            if feather > 0:
                alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
            a = (np.asarray(alpha, dtype=np.float32) / 255.0)[..., None]
            region = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = region * (1 - a) + filled * a
            weight[y0:y1, x0:x1] += a
    out = np.where(weight > 0, out, rgb.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


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
