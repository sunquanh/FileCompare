# -*- coding: utf-8 -*-
"""R20 性能回归门槛：小规模基准，召回 + 计时双断言。

生成 12 张图（3 底图 × 4 变体：原图/中心裁 20%/旋转 5°/加 10% 白边，
变体构造与完整基准 fc_eval_user_r18.py 一致：裁切与加边不缩回原尺寸），
跑 fingerprint + match_photos，断言：
  - 3 组真对全找到（每组底图 vs 3 个变体，共 9 对），且无跨组误报；
  - 总耗时 ≤ 10 秒（宽松，防共享 CPU 波动误报；真值约 1~2 秒，
    只有数量级回归才会触发）。
这是门槛测试，不是精确计时；300 张完整基准仍以
~/workspace/filecompare-verify/fc_eval_user_r18.py 为准。
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from dedup_photos import fingerprint, match_photos  # noqa: E402

TIME_BUDGET_S = 10.0
SIZE = 512


def _base(seed):
    rng = np.random.default_rng(seed)
    top = rng.integers(0, 255, 3).astype(np.float32)
    bot = rng.integers(0, 255, 3).astype(np.float32)
    t = np.linspace(0, 1, SIZE, dtype=np.float32)[:, None, None]
    arr = (top[None, None, :] * (1 - t)
           + bot[None, None, :] * t).astype(np.uint8)
    arr = np.repeat(arr, SIZE, axis=1)
    im = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(im)
    horizon = SIZE // 2 + int(rng.integers(-60, 60))
    ground = tuple(int(x) for x in rng.integers(0, 120, 3))
    d.rectangle([0, horizon, SIZE, SIZE], fill=ground)
    for _ in range(5):
        x0 = int(rng.integers(0, SIZE - 80))
        bw = int(rng.integers(40, 120))
        bh = int(rng.integers(60, 200))
        col = tuple(int(x) for x in rng.integers(0, 255, 3))
        d.rectangle([x0, horizon - bh, x0 + bw, horizon], fill=col)
    for _ in range(8):
        col = tuple(int(x) for x in rng.integers(0, 255, 3))
        x0, y0 = int(rng.integers(0, SIZE - 100)), int(rng.integers(0, SIZE - 100))
        x1, y1 = x0 + int(rng.integers(40, 200)), y0 + int(rng.integers(40, 200))
        if rng.random() < 0.5:
            d.rectangle([x0, y0, x1, y1], fill=col)
        else:
            d.ellipse([x0, y0, x1, y1], fill=col)
    return im


def _variants(im):
    w, h = im.size
    out = [im.copy()]
    # 中心裁 20%（每边 10%），不缩回
    out.append(im.crop((int(w * 0.10), int(h * 0.10),
                        int(w * 0.90), int(h * 0.90))))
    # 旋转 5°
    out.append(im.rotate(5, resample=Image.Resampling.BICUBIC,
                         expand=False, fillcolor=(255, 255, 255)))
    # 加 10% 白边，不缩回
    b = int(w * 0.10)
    bordered = Image.new("RGB", (w + 2 * b, h + 2 * b), (255, 255, 255))
    bordered.paste(im, (b, b))
    out.append(bordered)
    return out


class TestPerfGate(unittest.TestCase):
    def test_small_benchmark_recall_and_time(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for bi in range(3):
                base = _base(100 + bi)
                for vi, var in enumerate(_variants(base)):
                    p = os.path.join(td, "g%d_v%d.png" % (bi, vi))
                    var.save(p)
                    paths.append(p)
            t0 = time.perf_counter()
            fps = [fingerprint(p) for p in paths]
            self.assertEqual(len([f for f in fps if f is not None]), 12)
            edges = match_photos(fps)
            el = time.perf_counter() - t0
            # 真对：只断言“底图 vs 各变体”（单次变换），跨变体组合
            # （旋转+裁切、旋转+加边）是双重变换，不在算法设计覆盖内，
            # 完整基准里也不计这类对子。
            got = {(e[0], e[1]) for e in edges}
            want = set()
            for bi in range(3):
                base_idx = bi * 4
                for vi in (1, 2, 3):
                    want.add((base_idx, base_idx + vi))
            missing = want - got
            self.assertEqual(missing, set(), "漏真对: %s" % sorted(missing))
            # 误报：跨组边
            fp = {(a, b) for a, b in got if a // 4 != b // 4}
            self.assertEqual(fp, set(), "误报: %s" % sorted(fp))
            self.assertLessEqual(
                el, TIME_BUDGET_S,
                "小基准耗时 %.1fs 超过门槛 %.0fs" % (el, TIME_BUDGET_S))


if __name__ == "__main__":
    unittest.main()
