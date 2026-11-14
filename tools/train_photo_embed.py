# -*- coding: utf-8 -*-
"""离线生成照片 embedding 模型（一次性工具，主程序不会 import 本文件）。

用途
----
产出 ``models/photo_embed.onnx``：一个约 0.6 MB 的轻量 CNN，输入
``[1,3,64,64]`` 的 0~1 RGB，输出 ``[1,128]`` 的 L2 归一化向量。
运行时由 ``dedup_photo_embed.py`` 用项目已有的 onnxruntime 推理；
没装 onnxruntime 或删掉模型，照片去重自动退回纯手工特征。

约束（重要）
------------
* **全程离线**：不下载任何预训练权重、不联网、不依赖 torchvision。
* 训练数据是本机程序化生成的合成场景 + 仓库自带两张样例照片的随机裁块。
* 只有本脚本需要 torch/onnx；它们是开发期依赖，不进运行时。
* 用 InfoNCE 对比学习：同一张图的两个随机增强视图互为正样本，让向量
  对「重调色 / 裁切 / 贴纸 / 强压缩 / 白边 / 镜像」保持不变。

定位与边界
----------
这不是 ImageNet/CLIP 级别的语义模型（离线拿不到预训练权重），而是一个
「变换不变的浅层纹理 + 配色描述子」。因此上层只把它当**辅助判据**：
低余弦用来否决容易误并的路径，高余弦必须和手工特征门联合才放行，
绝不单独决定两张图是否重复。

用法
----
    C:/Python314/python.exe tools/train_photo_embed.py
可选环境变量 ``PHOTO_EMBED_OUT`` 指定输出路径。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
SIDE = 64
DIM = 128
STEPS = 300
BATCH = 24
SEED = 20240617


# ---------------------------------------------------------------- 数据
def base_scene(seed) -> np.ndarray:
    """一张 96×96 的合成场景：独立配色 + 结构块 + 高频纹理。"""
    r = np.random.default_rng(seed)
    h = w = 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pal = r.uniform(20, 230, 3).astype(np.float32)
    slope = r.uniform(-90, 90, (3, 2)).astype(np.float32)
    a = (pal[None, None, :]
         + slope[:, 0][None, None, :] * (xx / w)[..., None]
         + slope[:, 1][None, None, :] * (yy / h)[..., None])
    a = a + (np.sin(yy / (9 + seed % 13) + seed) * 22
             + np.cos(xx / (11 + seed % 7)) * 15)[..., None]
    a = np.clip(a + r.normal(0, 10, (h, w, 1)), 0, 255).astype(np.uint8)
    im = Image.fromarray(a, "RGB")
    d = ImageDraw.Draw(im)
    for k in range(int(r.integers(4, 11))):
        x = int(r.integers(0, w - 12))
        y = int(r.integers(0, h - 12))
        ww = int(r.integers(8, 45))
        hh = int(r.integers(8, 40))
        c = tuple(int(v) for v in r.integers(20, 240, 3))
        box = [x, y, min(w - 1, x + ww), min(h - 1, y + hh)]
        if k % 3 == 0:
            d.ellipse(box, fill=c, outline=(15, 15, 15))
        else:
            d.rectangle(box, fill=c, outline=(15, 15, 15))
    return np.asarray(im, dtype=np.float32)


def augment(a: np.ndarray, r) -> np.ndarray:
    """随机几何 + 光度 + 局部编辑增强，纯 numpy（训练循环里不调 PIL）。"""
    h, w = a.shape[:2]
    frac = r.uniform(0.58, 1.0)
    cw, ch = max(16, int(w * frac)), max(16, int(h * frac))
    y0 = int(r.integers(0, h - ch + 1))
    x0 = int(r.integers(0, w - cw + 1))
    crop = a[y0:y0 + ch, x0:x0 + cw]
    iy = np.linspace(0, ch - 1, SIDE).astype(int)
    ix = np.linspace(0, cw - 1, SIDE).astype(int)
    out = crop[iy[:, None], ix[None, :]]
    # 曝光 / gamma / 色温 / 逐通道增益
    out = (np.clip(out, 0, 255) / 255.0) ** r.uniform(0.7, 1.4)
    out = out * r.uniform(0.72, 1.3)
    out = out * np.array([r.uniform(0.85, 1.15), r.uniform(0.90, 1.10),
                          r.uniform(0.85, 1.20)], np.float32)
    # 强压缩：粗量化 + 横向模糊
    q = r.uniform(6, 26)
    out = np.round(out * 255.0 / q) * q / 255.0
    if r.random() < 0.35:
        out = 0.65 * out + 0.35 * np.roll(out, 1, axis=1)
    # 贴纸 / 大字
    if r.random() < 0.40:
        sy = int(r.integers(0, SIDE - 12))
        sx = int(r.integers(0, SIDE - 20))
        sh = int(r.integers(6, 20))
        sw = int(r.integers(14, min(40, SIDE - sx)))
        out[sy:sy + sh, sx:sx + sw] = r.uniform(0.88, 1.0)
    # 白边
    if r.random() < 0.20:
        b = int(r.integers(2, 6))
        out[:b] = 1.0
        out[-b:] = 1.0
        out[:, :b] = 1.0
        out[:, -b:] = 1.0
    if r.random() < 0.50:
        out = out[:, ::-1]
    if r.random() < 0.30:
        out = out[::-1]
    return np.clip(out, 0, 1).transpose(2, 0, 1).copy()


def build_bases(rng):
    bases = [base_scene(s) for s in range(1000, 1400)]
    # 仓库根目录里现成的照片：补一点自然纹理，避免只见过合成矩形。
    # 不写死文件名（样例照片不入库），有多少用多少，最多取 8 张。
    samples = sorted(p for p in ROOT.glob("*")
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    for path in samples[:8]:
        name = path.name
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                rgb.thumbnail((420, 420))
                a = np.asarray(rgb, dtype=np.float32)
            for _ in range(60):
                y = int(rng.integers(0, max(1, a.shape[0] - 96)))
                x = int(rng.integers(0, max(1, a.shape[1] - 96)))
                bases.append(a[y:y + 96, x:x + 96].copy())
        except Exception as exc:
            print(f"  跳过 {name}: {exc}", flush=True)
    return bases


# ---------------------------------------------------------------- 模型
class TinyEmbed(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return F.normalize(self.features(x), dim=1)


def train():
    torch.manual_seed(7)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    print("生成训练基图 ...", flush=True)
    bases = build_bases(rng)
    print(f"  {len(bases)} 张基图，{time.perf_counter() - t0:.1f}s", flush=True)

    net = TinyEmbed()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    net.train()
    eye = torch.eye(2 * BATCH, dtype=torch.bool)
    pos = torch.arange(BATCH)
    labels = torch.cat([pos + BATCH, pos])
    for step in range(STEPS):
        idx = rng.integers(0, len(bases), BATCH)
        v1 = np.stack([augment(bases[i], rng) for i in idx])
        v2 = np.stack([augment(bases[i], rng) for i in idx])
        x = torch.from_numpy(np.concatenate([v1, v2]).astype(np.float32))
        z = net(x)
        sim = (z @ z.T).masked_fill(eye, -9.0)
        loss = F.cross_entropy(sim / 0.18, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 60 == 0:
            print(f"  step {step:3d}  loss {loss.item():.3f}", flush=True)
    print(f"训练完成 {time.perf_counter() - t0:.1f}s，最终 loss {loss.item():.3f}",
          flush=True)
    return net.eval()


def export(net, out_path: Path):
    import io
    dummy = torch.zeros(1, 3, SIDE, SIDE, dtype=torch.float32)
    buf = io.BytesIO()
    torch.onnx.export(
        net, dummy, buf,
        input_names=["image"], output_names=["embedding"],
        opset_version=13, dynamo=False,
        dynamic_axes=None,
    )
    data = buf.getvalue()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"已导出 {out_path}  {len(data) / 1024:.0f} KiB", flush=True)

    # 导出后立刻用 onnxruntime 自检，避免把跑不通的模型交出去
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path),
                                providers=["CPUExecutionProvider"])
    got = sess.run(["embedding"],
                   {"image": np.zeros((1, 3, SIDE, SIDE), np.float32)})[0]
    assert got.shape == (1, DIM), got.shape
    assert np.isfinite(got).all(), "输出含非有限值"
    print(f"  onnxruntime 自检通过：{got.shape}，输入 0~1 RGB / 输出已 L2 归一",
          flush=True)


def main():
    out = Path(os.environ.get("PHOTO_EMBED_OUT")
               or (ROOT / "models" / "photo_embed.onnx"))
    net = train()
    export(net, out)
    print("\n提示：本模型是可选的。删掉它或没装 onnxruntime 时，照片去重"
          "会自动退回纯手工特征，功能不受影响。", flush=True)


if __name__ == "__main__":
    sys.exit(main())
