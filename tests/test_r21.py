# -*- coding: utf-8 -*-
"""R21 格式与序列：多帧 GIF、连拍序列、RAW 内嵌预览。

构造与真实使用一致的最小样本（512x512 以下小图，跑得快）：
  多帧：3 帧 GIF（红/绿/蓝）与同动画循环移位版（绿/蓝/红）应全帧配上；
        只共用第一帧的（红/黄/紫）只配 1 对。
  连拍：burst_key / detect_bursts 的分组与断点。
  RAW：合成 CR2（随机头 + 真 JPEG 预览 + 随机尾）经预览出指纹，
       与原 JPEG 指纹一致；无预览 RAW 返回 None；DNG 式 TIFF 直接可开。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dedup_photos as photos
from dedup_models import burst_key, detect_bursts


def _base(seed=7):
    rng = np.random.default_rng(seed)
    return (rng.random((128, 128, 3)) * 255).astype(np.uint8)


def _frame(arr, color):
    px = np.array(arr)
    px[:32, :32] = color
    return Image.fromarray(px)


class TestMultiFrame(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        base = _base()
        red, green, blue = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        yellow, purple = (255, 255, 0), (128, 0, 128)
        for name, colors in [("a", [red, green, blue]),
                             ("b", [green, blue, red]),
                             ("c", [red, yellow, purple])]:
            frames = [_frame(base, c) for c in colors]
            frames[0].save(os.path.join(self.d, name + ".gif"),
                           save_all=True, append_images=frames[1:],
                           duration=100, loop=0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_frame_count(self):
        fps = photos.frame_fingerprints(os.path.join(self.d, "a.gif"))
        self.assertEqual(len(fps), 3)

    def test_frame0_matches_single(self):
        p = os.path.join(self.d, "a.gif")
        single = photos.fingerprint(p)
        frames = photos.frame_fingerprints(p)
        self.assertTrue((frames[0].dh == single.dh).all())

    def test_same_animation_all_frames_match(self):
        fa = photos.frame_fingerprints(os.path.join(self.d, "a.gif"))
        fb = photos.frame_fingerprints(os.path.join(self.d, "b.gif"))
        pairs = photos.match_photos(fa + fb)
        self.assertEqual(len(pairs), 3)

    def test_only_first_frame_shared(self):
        fa = photos.frame_fingerprints(os.path.join(self.d, "a.gif"))
        fc = photos.frame_fingerprints(os.path.join(self.d, "c.gif"))
        pairs = photos.match_photos(fa + fc)
        self.assertEqual(len(pairs), 1)


class TestBurst(unittest.TestCase):
    def test_burst_key(self):
        self.assertEqual(burst_key("IMG_1001.jpg"), ("IMG_", 1001))
        self.assertEqual(burst_key("DSC_0001.png"), ("DSC_", 1))
        # N4：纯数字/日期型文件名 stem 为空，不算连拍
        self.assertIsNone(burst_key("123.jpg"))
        self.assertIsNone(burst_key("20261112.jpg"))
        self.assertIsNone(burst_key("photo.jpg"))
        self.assertIsNone(burst_key("IMG_1001A.jpg"))

    def test_detect_bursts(self):
        paths = ["/x/IMG_1001.jpg", "/x/IMG_1002.jpg", "/x/IMG_1003.jpg",
                 "/x/IMG_1005.jpg", "/x/IMG_1006.jpg", "/x/photo.jpg",
                 "/x/DSC_0001.png", "/x/DSC_0002.png"]
        bursts = detect_bursts(paths)
        self.assertEqual(len(bursts), 3)
        sizes = sorted(len(b) for b in bursts)
        self.assertEqual(sizes, [2, 2, 3])
        # 1004 断开：1001-1003 与 1005-1006 是两组
        flat = [p for b in bursts for p in b]
        self.assertIn("/x/IMG_1001.jpg", flat)
        self.assertIn("/x/IMG_1005.jpg", flat)
        self.assertNotIn("/x/photo.jpg", flat)

    def test_detect_bursts_cross_dir(self):
        # N4：不同目录的同名序列不并进同一 burst
        paths = ["/a/IMG_1001.jpg", "/b/IMG_1002.jpg", "/a/IMG_1003.jpg"]
        self.assertEqual(detect_bursts(paths), [])
        paths2 = ["/a/IMG_1001.jpg", "/a/IMG_1002.jpg", "/a/IMG_1003.jpg"]
        self.assertEqual(len(detect_bursts(paths2)), 1)


class TestRaw(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        im = Image.fromarray(_base(11))
        im.save(os.path.join(self.d, "preview.jpg"), quality=90)
        self.jpeg = Path(os.path.join(self.d, "preview.jpg")).read_bytes()
        # N7：确定性填充（不用 os.urandom，测试可复现）；
        # 填充字节特意避开 FFD8/FFD9，不干扰预览提取
        filler = b"\x00\x01\x02\x03" * 1024
        raw = filler + self.jpeg + filler
        Path(os.path.join(self.d, "shot.cr2")).write_bytes(raw)
        Path(os.path.join(self.d, "noprev.nef")).write_bytes(filler * 2)
        im.save(os.path.join(self.d, "shot.dng"), format="TIFF")

    def tearDown(self):
        self.tmp.cleanup()

    def test_preview_extracted_byte_identical(self):
        prev = photos.extract_jpeg_preview(os.path.join(self.d, "shot.cr2"))
        self.assertEqual(prev, self.jpeg)

    def test_preview_none_when_absent(self):
        self.assertIsNone(
            photos.extract_jpeg_preview(os.path.join(self.d, "noprev.nef")))

    def test_raw_fingerprint_via_preview(self):
        fp = photos.fingerprint(os.path.join(self.d, "shot.cr2"))
        fp_jpg = photos.fingerprint(os.path.join(self.d, "preview.jpg"))
        self.assertIsNotNone(fp)
        self.assertTrue((fp.dh == fp_jpg.dh).all())

    def test_raw_without_preview_skipped(self):
        self.assertIsNone(photos.fingerprint(os.path.join(self.d, "noprev.nef")))

    def test_dng_tiff_opens_directly(self):
        self.assertIsNotNone(photos.fingerprint(os.path.join(self.d, "shot.dng")))


if __name__ == "__main__":
    unittest.main()
