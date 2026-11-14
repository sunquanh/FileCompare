"""去水印框内反解 tests：浅色水印压在照片上
（黄/蓝/粉/绿 × 天空渐变/草地/肤色墙 × α 0.35/0.5）。

合成图是唯一能“有干净对照”的办法：同一张底图，一份压水印、
一份干净，误差按水印框量；框外远角量附带损伤。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dedup_watermark as W

WIDTH, HEIGHT = 600, 400


def _font(size=44):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _background(kind, seed=7):
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, HEIGHT)[:, None]
    xx = np.linspace(0, 1, WIDTH)[None, :]
    if kind == "sky":
        base = np.stack([(90 + 120 * yy) * np.ones_like(xx),
                         (150 + 60 * yy) * np.ones_like(xx),
                         (200 - 40 * yy) * np.ones_like(xx)], axis=2)
    elif kind == "grass":
        base = np.zeros((HEIGHT, WIDTH, 3))
        base[..., 1] = 120 + 30 * xx
        base[..., 0] = 60 + 20 * yy
        base[..., 2] = 40
    elif kind == "wall":
        # 室内暖色墙 + 一张人脸肤色椭圆（水印字会压过脸，测皮肤不被伤）
        base = np.zeros((HEIGHT, WIDTH, 3))
        base[..., 0] = 210
        base[..., 1] = 180
        base[..., 2] = 150
        gy, gx = np.mgrid[0:HEIGHT, 0:WIDTH]
        skin = (((gx - WIDTH * 0.6) / 90) ** 2
                + ((gy - HEIGHT * 0.5) / 110) ** 2) < 1
        base[skin] = [225, 180, 150]
    else:
        raise ValueError(kind)
    base = base + rng.normal(0, 6, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def _watermark(clean, tint_name, alpha, text="WATERMARK", color=None):
    """压半透明水印字，返回 (watermarked, text_box_px)。

    color 不给就用该色系的标准参考色；传 (255, 255, 255) 就是白字徽标。
    """
    ref = (np.array(color, dtype=np.float32) if color is not None
           else np.array(W.TINT_REFS[tint_name], dtype=np.float32))
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font()
    pos = (40, int(HEIGHT * 0.55))
    bbox = draw.textbbox(pos, text, font=font)
    a = int(round(alpha * 255))
    draw.text(pos, text, font=font,
              fill=(int(ref[0]), int(ref[1]), int(ref[2]), a))
    wm = np.asarray(overlay).astype(np.float32)
    al = wm[..., 3:4] / 255.0
    out = (1 - al) * clean.astype(np.float32) + al * ref
    return np.clip(out, 0, 255).astype(np.uint8), bbox


def _box_err(a, b, y0, y1, x0, x1):
    return float(np.abs(a[y0:y1, x0:x1].astype(np.float32)
                        - b[y0:y1, x0:x1].astype(np.float32)).mean())


def _text_box_norm(bbox, margin=24):
    x0, y0, x1, y1 = bbox
    return (max(0.0, (x0 - margin) / WIDTH),
            max(0.0, (y0 - margin) / HEIGHT),
            min(1.0, (x1 + margin) / WIDTH),
            min(1.0, (y1 + margin) / HEIGHT))


class TestBoxUnblend(unittest.TestCase):
    """框内反解：水印框误差大幅下降，远角几乎不动。"""

    def test_yellow_on_three_backgrounds(self):
        for kind in ("sky", "grass", "wall"):
            clean = _background(kind)
            wm, bbox = _watermark(clean, "黄", 0.35)
            x0, y0, x1, y1 = bbox
            before = _box_err(wm, clean, y0, y1, x0, x1)
            fixed, mask, residual = W.undo_light_in_box(
                wm, _text_box_norm(bbox))
            after = _box_err(fixed, clean, y0, y1, x0, x1)
            self.assertGreater(mask.sum(), 200,
                               f"{kind}: mask too small")
            self.assertLess(after, before * 0.6,
                            f"{kind}: box err {before:.2f} -> {after:.2f}")
            # 远角（右上，无水印）几乎不动
            far = _box_err(fixed, wm, 10, 70, WIDTH - 110, WIDTH - 10)
            self.assertLess(far, 1.0, f"{kind}: far corner moved {far:.2f}")

    def test_all_tints_on_grass(self):
        # 绿水印压绿草地本色就弱，门槛放低；其余三色降 30% 以上
        bars = {"黄": 0.7, "蓝": 0.7, "粉": 0.7, "绿": 0.85}
        clean = _background("grass")
        for tint_name, bar in bars.items():
            wm, bbox = _watermark(clean, tint_name, 0.35)
            x0, y0, x1, y1 = bbox
            before = _box_err(wm, clean, y0, y1, x0, x1)
            fixed, mask, _ = W.undo_light_in_box(wm, _text_box_norm(bbox))
            after = _box_err(fixed, clean, y0, y1, x0, x1)
            self.assertGreater(mask.sum(), 100, f"{tint_name}: no mask")
            self.assertLess(after, before * bar,
                            f"{tint_name}: box err {before:.2f} -> {after:.2f}")

    def test_nonstandard_yellow(self):
        # 品牌黄 (255,226,130) 比标准参考色 (252,248,196) 深得多：只用
        # 标准色投票时暖墙上只能检出几十像素、直接被数量门拒掉（实测 0%），
        # “深黄”候选进投票后三底色都能解。墙面是最弱一格（暖墙 + 框内
        # 有肤色边缘，单色背景估计在交界处会过冲），门槛按实测值定，不虚报。
        bars = {"sky": 0.5, "grass": 0.5, "wall": 0.8}
        for kind in ("sky", "grass", "wall"):
            clean = _background(kind)
            wm, bbox = _watermark(clean, "黄", 0.35, color=(255, 226, 130))
            x0, y0, x1, y1 = bbox
            before = _box_err(wm, clean, y0, y1, x0, x1)
            fixed, mask, _ = W.undo_light_in_box(wm, _text_box_norm(bbox))
            after = _box_err(fixed, clean, y0, y1, x0, x1)
            self.assertGreater(mask.sum(), 100,
                               f"{kind}: nonstandard yellow not detected")
            self.assertLess(after, before * bars[kind],
                            f"{kind}: nonstandard yellow "
                            f"{before:.2f} -> {after:.2f}")

    def test_strong_alpha(self):
        clean = _background("sky")
        wm, bbox = _watermark(clean, "黄", 0.5)
        x0, y0, x1, y1 = bbox
        before = _box_err(wm, clean, y0, y1, x0, x1)
        fixed, mask, _ = W.undo_light_in_box(wm, _text_box_norm(bbox))
        after = _box_err(fixed, clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.6,
                        f"alpha 0.5: box err {before:.2f} -> {after:.2f}")

    def test_opaque_white_logo_not_unblended(self):
        # 白字不透明徽标：反解只能认出笔画边的抗锯齿轮廓（不到框面积 2%），
        # 字身动不了（反解前后框内误差几乎不变）；真正去掉它靠整块补，
        # 见 test_white_logo_box_still_inpainted。
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 1.0, color=(255, 255, 255))
        box = _text_box_norm(bbox)
        fixed, mask, _ = W.undo_light_in_box(wm, box)
        py0, py1, px0, px1 = W.box_pixels(box, WIDTH, HEIGHT)
        area = (py1 - py0) * (px1 - px0)
        self.assertLess(int(mask.sum()), 0.02 * area,
                        "white logo mask should be outline-only")
        x0b, y0b, x1b, y1b = bbox
        before = _box_err(wm, clean, y0b, y1b, x0b, x1b)
        after = _box_err(fixed, clean, y0b, y1b, x0b, x1b)
        self.assertGreater(after, before * 0.8,
                           "un-blend must not pretend to remove the logo")

    def test_clean_photo_box_returns_empty(self):
        clean = _background("sky")
        name, mask = W.detect_light_in_box(
            clean, (0.1, 0.4, 0.7, 0.7))
        self.assertEqual(name, "")
        self.assertEqual(int(mask.sum()), 0)


class TestCleanPhoto(unittest.TestCase):
    """一键抠在照片上：网格分区找水印区反解，没水印原样返回。"""

    def test_grid_path_removes_yellow(self):
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 0.35)
        x0, y0, x1, y1 = bbox
        before = _box_err(wm, clean, y0, y1, x0, x1)
        fixed = W.clean_light_watermark(wm)
        after = _box_err(fixed, clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.6,
                        f"grid path: box err {before:.2f} -> {after:.2f}")
        far = _box_err(fixed, wm, 10, 70, WIDTH - 110, WIDTH - 10)
        self.assertLess(far, 1.0, f"grid path far corner moved {far:.2f}")

    def test_no_watermark_unchanged(self):
        clean = _background("wall")
        fixed = W.clean_light_watermark(clean)
        self.assertTrue(np.array_equal(fixed, clean),
                        "photo without watermark must be untouched")


class TestCleanDoc(unittest.TestCase):
    """白底文档回归：全局反解路径不受照片分区改动影响。"""

    def test_doc_still_works(self):
        doc = np.full((HEIGHT, WIDTH, 3), 250, dtype=np.uint8)
        img = Image.fromarray(doc)
        draw = ImageDraw.Draw(img)
        for y in range(40, HEIGHT, 60):
            draw.rectangle([30, y, WIDTH - 30, y + 30], outline=(180, 180, 180))
        doc = np.asarray(img)
        wm, bbox = _watermark(doc, "黄", 0.35)
        x0, y0, x1, y1 = bbox
        before = _box_err(wm, doc, y0, y1, x0, x1)
        fixed = W.clean_light_watermark(wm)
        after = _box_err(fixed, doc, y0, y1, x0, x1)
        self.assertLess(after, before * 0.7,
                        f"doc: box err {before:.2f} -> {after:.2f}")


class TestRemoveLightFile(unittest.TestCase):
    """端到端：框内先反解，解不净/不透明才整块补。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_yellow_box_unblended_not_inpainted(self):
        clean = _background("sky")
        wm, bbox = _watermark(clean, "黄", 0.35)
        src = self.tmp / "photo.jpg"
        Image.fromarray(wm).save(src, quality=95)
        # 门槛按“管线实际看到的 JPEG”算：4:2:0 色度抽样把水印色晕到周围
        # 背景像素上——干净图本身 roundtrip 框内误差约 4.3，带水印的 JPEG
        # 框内非水印像素误差约 7.6，这部分反解救不回来；内存内同款测试
        # 已证算法本身能降 68%，这里只验端到端接线。
        loaded = np.asarray(Image.open(src).convert("RGB"))
        dst = W.remove_light_file(src, boxes=[_text_box_norm(bbox)])
        out = np.asarray(Image.open(dst).convert("RGB"))
        x0, y0, x1, y1 = bbox
        before = _box_err(loaded, clean, y0, y1, x0, x1)
        after = _box_err(out, clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.65,
                        f"file: box err {before:.2f} -> {after:.2f}")
        # 框外远角几乎不动（整块补会把草地/天空抹掉，这里不该发生）
        far = _box_err(out, loaded, 10, 70, WIDTH - 110, WIDTH - 10)
        self.assertLess(far, 1.0, f"file far corner moved {far:.2f}")

    def test_white_logo_box_still_inpainted(self):
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 1.0, color=(255, 255, 255))
        src = self.tmp / "logo.jpg"
        Image.fromarray(wm).save(src, quality=95)
        dst = W.remove_light_file(src, boxes=[_text_box_norm(bbox)])
        out = np.asarray(Image.open(dst).convert("RGB"))
        x0, y0, x1, y1 = bbox
        moved = _box_err(out, wm, y0, y1, x0, x1)
        self.assertGreater(moved, 5.0,
                           "opaque logo box must be inpainted, not left")


if __name__ == "__main__":
    unittest.main()
