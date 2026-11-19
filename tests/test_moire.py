# -*- coding: utf-8 -*-
"""屏摄水波纹（摩尔纹）与淡灰水印的回归用例。

图都是合成的，但每一条都对应一个真踩过的坑：

- **屏摄页必须报 unchanged、且不许把像素压暗**：水波纹的条纹会被浅色水印
  检测读成“淡色水印笔画”，曾经在一张合成屏摄页上改动 2.5 万像素、留下 37
  个被压暗到护栏上限的点；4.3 MP 上还会因为多压了字形被调制出的边带，把整页
  判成淡黄水印（反解 3.7 万像素、报 clean）。
- **干净文字页必须一个像素不动**：文字的抗锯齿边与淡灰水印在亮度、色度上
  完全一样，缺邻域深色门时会被“重铺背景”改掉 3.4 万像素。
- **真的淡灰水印仍然要清掉**：上面那道门不能把该洗的也放过。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import dedup_watermark as wm  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
# (高, 宽)：小图是慢速用例，大图 4.3 MP 专门盯“大图上多压边带”那个坑
SIZES = {"小图": (900, 1200), "大图": (1800, 2400)}


def text_page(height=900, width=1200, text_height=None):
    """白底 + 每 34 像素一行黑字和一条浅灰表格线（仿文档截图）。"""
    im = Image.new("RGB", (width, height), (246, 246, 246))
    d = ImageDraw.Draw(im)
    stop = height - 40 if text_height is None else min(height - 40, text_height)
    for y in range(60, stop, 34):
        d.text((40, y), "2024/22.24  IP  1102  SWR1024/A9", fill=(35, 35, 35),
               font=ImageFont.load_default(size=15))
        d.line([(30, y + 22), (width - 30, y + 22)], fill=(222, 222, 222))
    return np.asarray(im)


def add_moire(arr, scale=1.0):
    """叠加两组斜向彩色条纹，模拟拍屏幕时的摩尔纹。"""
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    stripe = (0.055 * np.sin(2 * np.pi * (xx * 0.21 + yy * 0.13))
              + 0.04 * np.sin(2 * np.pi * (xx * 0.083 - yy * 0.19))) * scale
    out = arr.astype(np.float32) * (1.0 + stripe)[..., None]
    out += np.stack([28 * stripe, 6 * stripe, -18 * stripe], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_yellow_watermark(arr, alpha=0.2):
    """盖一层半透明淡黄水印，模拟压在文档上的浅色水印。"""
    height, width = arr.shape[:2]
    layer = Image.new("L", (width, height), 0)
    ImageDraw.Draw(layer).text((width // 5, height // 3), "INTERNAL", fill=255,
                               font=ImageFont.load_default(
                                   size=max(60, height // 12)))
    mask = np.asarray(layer).astype(np.float32)[..., None] / 255.0 * alpha
    out = (arr.astype(np.float32) * (1 - mask)
           + np.array([252, 248, 196], dtype=np.float32) * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def luma(arr):
    return np.tensordot(arr.astype(np.float32), LUMA, axes=([2], [0]))


def changed(before, after, tol=3):
    """改动超过 tol 级的像素数。"""
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return int((diff.max(axis=2) > tol).sum())


def darkened(before, after, limit=40):
    """被压暗超过 limit 级的像素数（反解护栏就是按这个口径定的）。"""
    return int(((luma(before) - luma(after)) > limit).sum())


class ScreenPhotoCase(unittest.TestCase):
    """拍屏幕的照片：压掉水波纹，但不许被当成有水印。"""

    @classmethod
    def setUpClass(cls):
        cls.pages = {}
        for name, (height, width) in SIZES.items():
            clean = text_page(height, width)
            cls.pages[name] = {
                "clean": clean,
                "moire": add_moire(clean),
                "weak": add_moire(clean, 0.5),      # 幅度减半的弱水波纹
                "wm": add_moire(add_yellow_watermark(clean)),
            }

    def test_screen_photo_reports_unchanged_and_stays_bright(self):
        for name, page in self.pages.items():
            with self.subTest(size=name):
                out, status = wm.clean_light_watermark(page["moire"])
                self.assertEqual(status.status, "unchanged")
                self.assertIn("水波纹", status.detail)
                self.assertEqual(darkened(page["moire"], out), 0,
                                 "水波纹处理不许把像素压暗")
                # 水印流程本身一个像素都不该动：输出应等于“只做陷波”的结果
                notched, detail = wm.moire_notch(page["moire"])
                self.assertTrue(detail, "屏摄页应该检出水波纹谱峰")
                np.testing.assert_array_equal(out, notched)

    def test_weak_moire_is_still_notched(self):
        for name in ("小图",):
            page = self.pages[name]
            with self.subTest(size=name):
                out, status = wm.clean_light_watermark(page["weak"])
                self.assertEqual(status.status, "unchanged")
                self.assertIn("水波纹", status.detail)
                self.assertEqual(darkened(page["weak"], out), 0)

    def test_screen_photo_with_watermark_is_still_cleaned(self):
        for name in ("小图",):
            page = self.pages[name]
            with self.subTest(size=name):
                out, status = wm.clean_light_watermark(page["wm"])
                self.assertEqual(status.status, "clean")
                self.assertIn("水波纹", status.detail)
                self.assertEqual(darkened(page["wm"], out), 0)

    def test_only_injected_frequencies_are_picked(self):
        """只认注入的两组条纹（两对共轭峰），不认字形被调制出的边带。

        边带只有最强峰的 0.15；压它们会在字形旁留下残影，反而被水印检测
        读成淡色笔画——4.3 MP 上就是这么把整页判成有水印的。
        """
        for name, page in self.pages.items():
            with self.subTest(size=name):
                peaks = wm._moire_peaks(page["moire"])
                self.assertEqual(len(peaks), 4)
                got = {(round(abs(fy), 2), round(abs(fx), 3)) for fy, fx in peaks}
                self.assertEqual(got, {(0.13, 0.21), (0.19, 0.083)})


class CleanPageCase(unittest.TestCase):
    """本来就没水印的文档：一个像素都不许动。"""

    def test_clean_text_page_is_not_touched(self):
        for name, (height, width) in SIZES.items():
            with self.subTest(size=name):
                page = text_page(height, width)
                out, status = wm.clean_light_watermark(page)
                self.assertEqual(status.status, "unchanged")
                np.testing.assert_array_equal(out, page)

    def test_clean_text_page_with_noise_is_not_touched(self):
        rng = np.random.default_rng(0)
        page = text_page()
        noisy = np.clip(page.astype(np.float32)
                        + rng.normal(0, 1.6, page.shape), 0, 255).astype(np.uint8)
        out, status = wm.clean_light_watermark(noisy)
        self.assertEqual(status.status, "unchanged")
        self.assertEqual(changed(noisy, out), 0)


class GrayWatermarkCase(unittest.TestCase):
    """淡灰水印：没有偏色，靠“平坦区重铺背景”清。"""

    def page_with_gray_watermark(self):
        """上半是正文，下半空白处盖一枚淡灰水印。"""
        page = text_page(1000, 1300, text_height=520)
        im = Image.fromarray(page)
        ImageDraw.Draw(im).text((300, 700), "DRAFT", fill=(232, 232, 232),
                                font=ImageFont.load_default(size=90))
        return np.asarray(im)

    def test_gray_watermark_is_cleaned(self):
        page = self.page_with_gray_watermark()
        out, status = wm.clean_light_watermark(page)
        self.assertEqual(status.status, "clean")
        paper = float(luma(page)[200:400, 900:1200].mean())
        box = (slice(700, 800), slice(300, 560))
        self.assertLess(float(luma(page)[box].mean()), paper - 1.5,
                        "水印区本来就该比纸面暗")
        self.assertAlmostEqual(float(luma(out)[box].mean()), paper, delta=2.0)
        self.assertEqual(darkened(page, out), 0)

    def test_body_text_and_rules_are_not_touched(self):
        """水印洗掉，但正文笔画和浅灰表格线一个像素都不许动。

        表格线（222）只比纸暗 24 级，既不是“深色内容”（邻域深色门拦不住），
        也不是水印。早先它会漏掉两重守卫：一是重钺的是掩膜外扩 2 像素的整个
        圈，二是 _neutral_mask 只用“到最近断点的距离”判长直线——一条 1240
        像素的长线，两↑端头各有 60 像素被当成短笔画。一张 1300×1000 的页面
        上就是 845 个线色像素被刷成纸色（表格线断成一截一截）。
        """
        page = self.page_with_gray_watermark()
        out, _status = wm.clean_light_watermark(page)
        self.assertEqual(changed(page[:520], out[:520]), 0,
                         "正文笔画和表格线都不许动")
        diff = np.abs(out.astype(np.int16) - page.astype(np.int16)).max(axis=2) > 3
        rules = np.all(page == 222, axis=2) & diff
        self.assertEqual(int(rules.sum()), 0, "表格线不许被刷成纸色")

    def test_gray_watermark_on_blank_paper_is_cleaned(self):
        page = np.full((600, 800, 3), 248, np.uint8)
        im = Image.fromarray(page)
        ImageDraw.Draw(im).text((200, 250), "DRAFT", fill=(228, 228, 228),
                                font=ImageFont.load_default(size=80))
        watermark = np.asarray(im)
        out, status = wm.clean_light_watermark(watermark)
        self.assertEqual(status.status, "clean")
        self.assertEqual(darkened(watermark, out), 0)


if __name__ == "__main__":
    unittest.main()
