"""照片 embedding 后端（可选，全程离线）。

与 LaMa 同一套分层模式：需要 ``onnxruntime`` + 本地 ``photo_embed.onnx``。
缺任一个就 ``available() == False``，``embed_rgb`` 返回 ``(None, "")``，
照片去重自动退回纯手工特征，行为与没有本模块时完全一致。

**没有任何联网逻辑**：不下载模型、不检查更新、不调用外部服务。
模型是 ``tools/train_photo_embed.py`` 在本机离线生成的，随包提供。
运行时依赖只有项目已有的 numpy / Pillow / 可选 onnxruntime。

模型约定（导出时已自检）：
    输入 image  float32 [1,3,64,64]，0~1，RGB 顺序
    输出 embedding float32 [1,128]，已 L2 归一化
换模型会改变 ``model_id``，指纹缓存里的旧向量自动失效。
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np

MODEL_NAME = "photo_embed.onnx"
DIM = 128
SIZE = 64
_session = None
_session_path = None
_model_id_cache = None
_path_cache = None          # None=未探测；False=探测过但没找到；Path=命中
_lock = threading.Lock()


def _roots():
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
    """找到可用模型；找不到返回 None。环境变量 PHOTO_EMBED_MODEL 优先。

    ``_roots()`` 里的 app_dir() 会做一次可写性探测（创建临时文件），
    每张图都重扫会让指纹计算慢一个数量级，所以结果缓存一次。
    """
    global _path_cache
    if _path_cache is not None:
        return _path_cache if _path_cache is not False else None
    env = os.environ.get("PHOTO_EMBED_MODEL")
    found = None
    if env and Path(env).is_file():
        found = Path(env)
    else:
        for root in _roots():
            candidate = Path(root) / MODEL_NAME
            if candidate.is_file() and candidate.stat().st_size > 10_000:
                found = candidate
                break
    with _lock:
        _path_cache = found if found is not None else False
    return found


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return model_path() is not None


def status_text() -> str:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return "未安装 onnxruntime，深度特征不可用（不影响手工去重）"
    path = model_path()
    if path is None:
        return f"缺少模型文件 {MODEL_NAME}（不影响手工去重）"
    return f"就绪：{path.name}（{path.stat().st_size / 1024:.0f} KiB）"


def model_id() -> str:
    """模型内容指纹，写进每张图的缓存；换模型时旧向量自动失效。"""
    global _model_id_cache
    path = model_path()
    if path is None:
        return ""
    with _lock:
        if _model_id_cache is not None and _model_id_cache[0] == str(path):
            return _model_id_cache[1]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    ident = f"pemb-{h.hexdigest()[:12]}-{DIM}"
    with _lock:
        _model_id_cache = (str(path), ident)
    return ident


def _session_for(path):
    global _session, _session_path
    with _lock:
        if _session is not None and _session_path == str(path):
            return _session
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
        options.log_severity_level = 3
        _session = ort.InferenceSession(str(path), sess_options=options,
                                        providers=["CPUExecutionProvider"])
        _session_path = str(path)
        return _session


def warmup() -> bool:
    """提前加载会话 + 跑一次空推理，避免用户点「开始」后干等。"""
    path = model_path()
    if path is None:
        return False
    try:
        session = _session_for(path)
        session.run(["embedding"],
                    {"image": np.zeros((1, 3, SIZE, SIZE), np.float32)})
        model_id()
        return True
    except Exception:
        return False


def embed_rgb(rgb):
    """PIL RGB → (L2 归一化的 128 维向量, model_id)。

    模型缺失 / 没装 onnxruntime / 推理失败一律返回 ``(None, "")``，
    调用方据此跳过整层，不抛异常、不联网、不改变其它指纹字段。
    """
    path = model_path()
    if path is None:
        return None, ""
    try:
        from PIL import Image
        if not isinstance(rgb, Image.Image):
            return None, ""
        small = rgb.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BOX)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        tensor = arr.transpose(2, 0, 1)[None].copy()
        session = _session_for(path)
        out = session.run(["embedding"], {"image": tensor})[0][0]
        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        if vec.size != DIM or not np.isfinite(vec).all():
            return None, ""
        norm = float(np.sqrt(float((vec * vec).sum())))
        if norm > 1e-6:
            vec /= norm
        return vec, model_id()
    except Exception:
        return None, ""
