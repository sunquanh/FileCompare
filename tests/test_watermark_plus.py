# -*- coding: utf-8 -*-
"""去水印新能力的回归：连贯度门、灵敏度旋钮、跨图模板、平铺水印。

样本由 tools/wm_bench.py 的生成器现做（确定性），不依赖仓库里的图片文件。
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import dedup_watermark as W  # noqa: E402
import dedup_watermark_template as T  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "wm_bench", ROOT / "tools" / "wm_bench.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["wm_bench"] = bench
_spec.loader.exec_module(bench)


def _err(a, clean, mask):
    return float(np.abs(a[mask].astype(np.int16)
                        - clean[mask].astype(np.int16)).mean())


class TestStrokeDensityGate(unittest.TestCase):
    """色系投票要挡掉“白纸上的彩色噪点”，这是干净屏摄图被改 56% 的根因。"""

    def test_clean_screen_photo_is_left_alone(self):
        # 屏摄质感（暗角 + 摩尔纹 + 噪声）的干净文档页
        clean = bench._screen_transform(bench._doc(seed=19, lines=8), 23)
        out, status = W.clean_light_watermark(clean)
        self.assertEqual(status.status, "unchanged", status.detail)
        self.assertTrue(np.array_equal(out, clean),
                        "干净图上一个像素都不许动")

    def test_real_watermark_still_detected(self):
        clean = bench._doc(seed=3)
        arr, mask = bench._stamp(clean, "2023/12/24", 0.30, (252, 248, 196),
                                 [(120, 300), (420, 700)], size=58)
        detected = W.detect_light_watermark(arr)
        self.assertGreater(int(detected.sum()), 2000)
        # 连贯度门不该把真水印挡掉：覆盖真值掩膜的一半以上
        self.assertGreater(float(detected[mask].mean()), 0.5)

    def test_density_separates_noise_from_strokes(self):
        rng = np.random.default_rng(0)
        noise = rng.random((200, 200)) > 0.98
        strokes = np.zeros((200, 200), dtype=bool)
        strokes[40:60, 40:160] = True
        self.assertLess(W._stroke_density(noise), 0.5)
        self.assertGreater(W._stroke_density(strokes), 0.9)


class TestSensitivity(unittest.TestCase):
    """灵敏度旋钮：往下推门限，检出只增不减。"""

    def test_higher_sensitivity_detects_more(self):
        clean = bench._photo(seed=5, palette="grass")
        arr, _mask = bench._stamp(clean, "IMG2023", 0.35, (252, 248, 196),
                                  [(180, 220), (360, 390)], size=72)
        box = (0.1, 0.2, 0.7, 0.8)
        low = W.detect_light_in_box(arr, box, sens=0.0)
        high = W.detect_light_in_box(arr, box, sens=0.6)
        low_px = int(low.mask.sum()) if low.found else 0
        high_px = int(high.mask.sum()) if high.found else 0
        self.assertGreaterEqual(high_px, low_px)
        self.assertGreater(high_px, 0)

    def test_default_is_unchanged(self):
        # 默认 sens=0 时行为与不传一样
        clean = bench._doc(seed=7, lines=6)
        arr, _ = bench._stamp(clean, "CONFIDENTIAL", 0.25, (228, 228, 228),
                              [(60, 900)], size=72)
        a, sa = W.clean_light_watermark(arr)
        b, sb = W.clean_light_watermark(arr, sens=0.0)
        self.assertEqual(sa.status, sb.status)
        self.assertTrue(np.array_equal(a, b))


class TestCrossImageTemplate(unittest.TestCase):
    """同批同款水印：学一次模板，整批按模板反解。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix=".tmp_tpl_"))

    def tearDown(self):
        for item in sorted(self.tmp.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        self.tmp.rmdir()

    def _batch(self, count=5):
        paths = []
        for i in range(count):
            base = bench._doc(seed=3 + i * 4, lines=14)
            arr, mask = bench._stamp(base, "2023/12/24", 0.30,
                                     (252, 248, 196), [(150, 320)], size=58)
            path = self.tmp / f"page{i}.png"
            Image.fromarray(arr).save(path)
            paths.append(path)
            if i == 0:
                self.clean, self.mask = base, mask
        return paths

    def test_learns_and_applies(self):
        paths = self._batch()
        template = T.learn_template(paths)
        self.assertIsNotNone(template, "同款同位置的水印应该能学出模板")
        self.assertGreaterEqual(template.share, 0.8)
        self.assertEqual(template.tint, "黄")
        rgb = np.asarray(Image.open(paths[0]).convert("RGB"))
        before = _err(rgb, self.clean, self.mask)
        fixed, full = T.apply_template(rgb, template)
        after = _err(fixed, self.clean, self.mask)
        # 模板反解的质量与逐张主路径同档（实测 7.80 -> 5.85，主路径 5.03）：
        # 它的价值在整批一致性（一次学到、每张都套），不是单张更准。
        self.assertLess(after, before * 0.8,
                        f"模板反解：{before:.2f} -> {after:.2f}")
        # 模板掩膜要落在真值掩膜附近，不能把整页都算进来
        self.assertGreater(float(full[self.mask].mean()), 0.4)
        outside = ~W._grow_mask(self.mask, 8)
        drift = float(np.abs(fixed[outside].astype(np.int16)
                             - rgb[outside].astype(np.int16)).mean())
        self.assertLess(drift, 1.0, f"水印外被动了 {drift:.2f}")

    def test_no_template_when_nothing_found(self):
        base = bench._photo(seed=8, palette="wall")
        path = self.tmp / "plain.png"
        Image.fromarray(base).save(path)
        self.assertIsNone(T.learn_template([path]))


class TestTiledWatermark(unittest.TestCase):
    """平铺水印：周期可测、能减掉、且不碰无关区域。"""

    def _case(self):
        clean = bench._doc(seed=11, lines=10)
        arr, mask = bench._stamp(clean, "SAMPLE", 0.22, (252, 248, 196),
                                 bench._tiled_positions(900, 1200, 300, 260),
                                 size=52)
        return arr, clean, mask

    def test_period_is_exact(self):
        arr, _clean, _mask = self._case()
        period = T.tiled_period(arr)
        self.assertIsNotNone(period)
        self.assertEqual((period[0], period[1]), (260, 300))

    def test_removes_and_leaves_rest(self):
        arr, clean, mask = self._case()
        before = _err(arr, clean, mask)
        fixed, info = T.remove_tiled_watermark(arr)
        self.assertIsNotNone(info)
        self.assertEqual(info["period"], (260, 300))
        after = _err(fixed, clean, mask)
        self.assertLess(after, before * 0.6,
                        f"平铺重建：{before:.2f} -> {after:.2f}")
        outside = ~W._grow_mask(mask, 8)
        drift = float(np.abs(fixed[outside].astype(np.int16)
                             - arr[outside].astype(np.int16)).mean())
        self.assertLess(drift, 1.0, f"水印外被动了 {drift:.2f}")

    def test_clean_page_has_no_period(self):
        clean = bench._doc(seed=17)
        self.assertIsNone(T.tiled_period(clean),
                          "没有平铺水印的页面不该报周期")


if __name__ == "__main__":
    unittest.main()
