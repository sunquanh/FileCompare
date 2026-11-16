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
        det = W.detect_light_in_box(clean, (0.1, 0.4, 0.7, 0.7))
        self.assertEqual(det.tint, "")
        self.assertFalse(det.found)
        self.assertEqual(det.rejected, "")
        self.assertEqual(det.score, 0.0)


class TestCleanPhoto(unittest.TestCase):
    """一键抠在照片上：网格分区找水印区反解，没水印原样返回。"""

    def test_grid_path_removes_yellow(self):
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 0.35)
        x0, y0, x1, y1 = bbox
        before = _box_err(wm, clean, y0, y1, x0, x1)
        fixed, status = W.clean_light_watermark(wm)
        self.assertEqual(status.status, "clean")
        after = _box_err(fixed, clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.6,
                        f"grid path: box err {before:.2f} -> {after:.2f}")
        far = _box_err(fixed, wm, 10, 70, WIDTH - 110, WIDTH - 10)
        self.assertLess(far, 1.0, f"grid path far corner moved {far:.2f}")

    def test_no_watermark_unchanged(self):
        clean = _background("wall")
        fixed, status = W.clean_light_watermark(clean)
        self.assertEqual(status.status, "unchanged")
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
        fixed, status = W.clean_light_watermark(wm)
        self.assertEqual(status.status, "clean")
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
        dst, status = W.remove_light_file(src, boxes=[_text_box_norm(bbox)])
        self.assertEqual(status.status, "clean")
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
        dst, status = W.remove_light_file(src, boxes=[_text_box_norm(bbox)])
        self.assertEqual(status.status, "clean")
        self.assertIn("不透明", status.detail)
        out = np.asarray(Image.open(dst).convert("RGB"))
        x0, y0, x1, y1 = bbox
        moved = _box_err(out, wm, y0, y1, x0, x1)
        self.assertGreater(moved, 5.0,
                           "opaque logo box must be inpainted, not left")


class TestDetection(unittest.TestCase):
    """检测只吐 LightDetection：框 + 色系 + 掩膜 + 分数 + 来源 + 拒认原因。"""

    def test_box_detection_fields(self):
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 0.35)
        det = W.detect_light_in_box(wm, _text_box_norm(bbox))
        self.assertTrue(det.found)
        self.assertEqual(det.tint, "黄")
        self.assertGreater(det.score, 1.0)
        self.assertEqual(det.source, "box")
        self.assertEqual(det.rejected, "")
        self.assertGreater(int(det.mask.sum()), 100)
        self.assertEqual(det.box, _text_box_norm(bbox))

    def test_grid_detections_carry_source(self):
        clean = _background("grass")
        wm, _bbox = _watermark(clean, "黄", 0.35)
        dets = W._grid_light_boxes(wm)
        self.assertGreater(len(dets), 0)
        for det in dets:
            self.assertEqual(det.source, "grid")
            self.assertTrue(det.found)


class TestReconstruct(unittest.TestCase):
    """重建决策是唯一的分叉点：同一套规则，不再两条路各写一套。"""

    def test_yellow_box_unblend_or_fill(self):
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 0.35)
        x0, y0, x1, y1 = bbox
        before = _box_err(wm, clean, y0, y1, x0, x1)
        fixed, action, _detail = W.reconstruct_light(
            wm, _text_box_norm(bbox))
        self.assertIn(action, ("unblend", "fill"))
        after = _box_err(fixed, clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.6)

    def test_strong_yellow_box_inpaint_residual(self):
        # 草地 α=0.5：反解留 47% 残留，不得不整块补（partial 的决策层）
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 0.5)
        _fixed, action, detail = W.reconstruct_light(
            wm, _text_box_norm(bbox))
        self.assertEqual(action, "inpaint-residual")
        self.assertIn("残留", detail)

    def test_white_logo_box_inpaint_opaque(self):
        clean = _background("grass")
        wm, bbox = _watermark(clean, "黄", 1.0, color=(255, 255, 255))
        _fixed, action, detail = W.reconstruct_light(
            wm, _text_box_norm(bbox))
        self.assertEqual(action, "inpaint-opaque")
        self.assertIn("不透明", detail)

    def test_clean_box_none(self):
        clean = _background("sky")
        _fixed, action, _detail = W.reconstruct_light(
            clean, (0.1, 0.4, 0.7, 0.7))
        self.assertEqual(action, "none")

    def test_whole_image_box_rejected(self):
        clean = _background("sky")
        wm, _bbox = _watermark(clean, "黄", 0.35)
        _fixed, action, detail = W.reconstruct_light(
            wm, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(action, "rejected")
        self.assertIn("70%", detail)


class TestStatus(unittest.TestCase):
    """处理结论如实报：干净 / 残留 / 未找到 / 拒绝，不悄悄说成功。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, arr):
        src = self.tmp / name
        Image.fromarray(arr).save(src, quality=95)
        return src

    def test_clean_photo_unchanged(self):
        src = self._write("clean.jpg", _background("wall"))
        _dst, status = W.remove_light_file(src)
        self.assertEqual(status.status, "unchanged")

    def test_yellow_photo_clean(self):
        clean = _background("sky")
        wm, bbox = _watermark(clean, "黄", 0.35)
        src = self._write("photo.jpg", wm)
        _dst, status = W.remove_light_file(
            src, boxes=[_text_box_norm(bbox)])
        self.assertEqual(status.status, "clean")

    def test_whole_image_box_rejected(self):
        # 白字徽标 + 整图大框：自动抠不管白字，框又占图超 70% → 明确拒绝
        clean = _background("grass")
        wm, _bbox = _watermark(clean, "黄", 1.0, color=(255, 255, 255))
        src = self._write("logo.jpg", wm)
        _dst, status = W.remove_light_file(src, boxes=[(0.0, 0.0, 1.0, 1.0)])
        self.assertEqual(status.status, "rejected")
        self.assertIn("70%", status.detail)

    def test_whole_image_box_clean_with_note(self):
        # 黄字照片 + 整图大框：水印是自动抠清的，结论仍是 clean，
        # 但被拒的框必须记一句，不能悄悄吞掉
        clean = _background("sky")
        wm, _bbox = _watermark(clean, "黄", 0.35)
        src = self._write("photo.jpg", wm)
        _dst, status = W.remove_light_file(src, boxes=[(0.0, 0.0, 1.0, 1.0)])
        self.assertEqual(status.status, "clean")
        self.assertIn("拒绝", status.detail)

    def test_batch_counts_and_results(self):
        clean = _background("wall")
        wm_y, _b = _watermark(_background("sky"), "黄", 0.35)
        wm_l, _b2 = _watermark(_background("grass"), "黄", 1.0,
                               color=(255, 255, 255))
        paths = [self._write("clean.jpg", clean),
                 self._write("yellow.jpg", wm_y),
                 self._write("logo.jpg", wm_l)]
        result = W.batch_remove_light(paths)
        self.assertEqual(result["done"], 3)
        self.assertEqual(result["clean"], 1)
        self.assertEqual(result["unchanged"], 2)
        self.assertEqual(result["partial"], 0)
        self.assertEqual(result["rejected"], 0)
        by_name = {Path(p).name: s for p, s, _d in result["results"]}
        self.assertEqual(by_name["yellow.jpg"], "clean")
        self.assertEqual(by_name["clean.jpg"], "unchanged")
        self.assertEqual(by_name["logo.jpg"], "unchanged")

    def test_batch_rejected_count(self):
        clean = _background("grass")
        wm_l, _b = _watermark(clean, "黄", 1.0, color=(255, 255, 255))
        src = self._write("logo.jpg", wm_l)
        result = W.batch_remove_light([src], boxes=[(0.0, 0.0, 1.0, 1.0)])
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["results"][0][1], "rejected")

class TestSkyBlueFix(unittest.TestCase):
    """sky 蓝文件流弱项的回归：格子大、背景有渐变时环中位色会偏，
    偏了的偏差近乎无色，色度门把真蓝水印误杀、色系投成“绿”。
    修法：检测到掩膜后，用掩膜外一圈重估背景再投一次色系。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, arr):
        src = self.tmp / name
        Image.fromarray(arr).save(src, quality=95)
        return src

    def _jpeg(self, tint, alpha):
        clean = _background("sky")
        wm, bbox = _watermark(clean, tint, alpha)
        return clean, wm, bbox, self._write("photo.jpg", wm)

    def test_grid_cell_votes_blue_on_jpeg_sky(self):
        # 顶格（y 100–300）以前投成“绿”、钉死“蓝” mask=0；
        # 第二遍后必须是“蓝”，掩膜不小于单框检出量的八成
        _clean, _wm, _bbox, src = self._jpeg("蓝", 0.35)
        loaded = np.asarray(Image.open(src).convert("RGB"))
        cell = (0.0, 0.25, 0.41666666666666663, 0.75)
        det = W.detect_light_in_box(loaded, cell)
        self.assertEqual(det.tint, "蓝")
        self.assertGreaterEqual(int(det.mask.sum()), 2000)

    def test_one_click_blue_sky_drops_to_floor(self):
        # 一键抠（JPEG 图）以前只降 16%（11.41→9.57），现在要降到
        # 单框反解的水平（约 6.7），即降幅不低于 35%
        clean, _wm, bbox, src = self._jpeg("蓝", 0.35)
        loaded = np.asarray(Image.open(src).convert("RGB"))
        x0, y0, x1, y1 = bbox
        before = _box_err(loaded, clean, y0, y1, x0, x1)
        fixed, _status = W.clean_light_watermark(loaded)
        after = _box_err(fixed, clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.65)

    def test_file_stream_blue_sky_drops_to_floor(self):
        # 文件流以前只降 15%（11.41→9.68，partial），现在降幅
        # 不低于 35%（约 6.4）。状态仍是 partial 也可以——校验在
        # 输出图上仍检出轻微残留（分数近线，detail 里写明），
        # 误差已到 JPEG 底噪水平，不是没修干净。
        clean, _wm, bbox, src = self._jpeg("蓝", 0.35)
        x0, y0, x1, y1 = bbox
        before = _box_err(np.asarray(Image.open(src).convert("RGB")),
                          clean, y0, y1, x0, x1)
        dst, status = W.remove_light_file(src, boxes=[_text_box_norm(bbox)])
        after = _box_err(np.asarray(Image.open(dst).convert("RGB")),
                         clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.65)
        self.assertIn(status.status, ("clean", "partial"))


class TestRealPhoto(unittest.TestCase):
    """真实照片（拍屏幕）回归：蓝灰底色不再误认成“蓝”水印，
    深色 SQL 文字不再误认成水印，淡黄斜字水印要能去掉；
    状态要诚实（没去净报 partial）。

    图是合成的“屏幕照片”（蓝灰渐变底 + 淡色 SQL 文本行 + 蓝色关键字 +
    淡黄斜字水印），不用用户原图——原图不进公开仓库。
    文字要淡、小，水印要大：锐利的深色文字会在照片路径里压过水印
    （真实照片里文字边缘是糊的），所以图里先做了轻微失焦 + 噪声。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _screen_photo(self, alpha=0.5):
        """返回 (clean, wm, text_bbox)：clean 是无水印的屏幕照片。"""
        rng = np.random.default_rng(11)
        yy = np.linspace(0, 1, HEIGHT)[:, None]
        xx = np.linspace(0, 1, WIDTH)[None, :]
        base = np.stack([(196 - 12 * yy) * np.ones_like(xx),
                         (213 - 8 * yy) * np.ones_like(xx),
                         (237 - 6 * xx) * np.ones_like(yy)], axis=2)
        base = base + rng.normal(0, 5, base.shape)
        img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        font = _font(20)
        lines = [("SELECT diff = 0, value_diff = 0,", (70, 70, 70)),
                 ("  COALESCE(SUM(n_cnt), 0) AS new_rows,", (40, 40, 180)),
                 ("  COALESCE(SUM(o_cnt), 0) AS only_old,", (40, 40, 180)),
                 ("WHERE a = 1 AND NOT (o_used <=> n_used", (70, 70, 70))]
        y = 150
        for text, color in lines:
            draw.text((15, y), text, font=font, fill=color)
            y += 36
        clean = np.asarray(img)
        # 淡黄斜字水印：颜料色 (180,195,130)，α=0.5 时观测色约 (188,204,183)，
        # 接近真实裁块的观测中位色 (194,206,182)，比写死的“深黄”淡得多
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        wm_font = _font(60)
        a = int(round(alpha * 255))
        od.text((60, 20), "1102", font=wm_font,
                fill=(180, 195, 130, a))
        wm_arr = np.asarray(overlay).astype(np.float32)
        al = wm_arr[..., 3:4] / 255.0
        ref = np.array([180.0, 195.0, 130.0], dtype=np.float32)
        out = (1 - al) * clean.astype(np.float32) + al * ref
        bbox = od.textbbox((60, 20), "1102", font=wm_font)
        wm = np.clip(out, 0, 255).astype(np.uint8)
        # 拍屏幕照片：轻微失焦 + 传感器噪声。否则锐利的深色文字边缘会在
        # 照片路径里被当成水印（真实照片里文字边缘是糊的）；噪声对 clean
        # 和 wm 用同一份，框误差里噪声抵消，只量水印处理本身。
        from PIL import ImageFilter
        clean_b = np.asarray(Image.fromarray(clean).filter(
            ImageFilter.GaussianBlur(1))).astype(np.float32)
        wm_b = np.asarray(Image.fromarray(wm).filter(
            ImageFilter.GaussianBlur(1))).astype(np.float32)
        noise = rng.normal(0, 8, (HEIGHT, WIDTH, 3))
        clean2 = np.clip(clean_b + noise, 0, 255).astype(np.uint8)
        wm2 = np.clip(wm_b + noise, 0, 255).astype(np.uint8)
        return clean2, wm2, bbox

    def test_screen_photo_no_blue_false_positive(self):
        # 蓝灰底色以前在 3 格里投出 2 格“蓝”假阳性；两道门之后，
        # 网格里不该再有“蓝”，淡黄水印那格要保留
        _clean, wm, _bbox = self._screen_photo()
        dets = W._grid_light_boxes(wm)
        self.assertTrue(any(d.tint in ("黄", "深黄") for d in dets),
                        f"黄水印格丢了：{[(d.tint, round(d.score, 2)) for d in dets]}")
        self.assertFalse(any(d.tint == "蓝" for d in dets),
                         f"仍有蓝假阳性：{[(d.tint, round(d.score, 2)) for d in dets]}")

    def test_screen_photo_yellow_removed(self):
        # 一键抠后水印框误差至少降一半，状态是 clean 或轻微残留的 partial
        clean, wm, bbox = self._screen_photo()
        x0, y0, x1, y1 = bbox
        before = _box_err(wm, clean, y0, y1, x0, x1)
        fixed, status = W.clean_light_watermark(wm)
        after = _box_err(fixed, clean, y0, y1, x0, x1)
        self.assertLess(after, before * 0.5,
                        f"水印没去净：{before:.2f} -> {after:.2f}")
        self.assertIn(status.status, ("clean", "partial"))

    def test_big_watermark_reports_partial(self):
        # 大字水印（跨多个网格格）一遍反解+小补解不净时，
        # clean_light_watermark 要报 partial 而不是 clean；
        # 文件流同样报 partial，绝不能降级成 unchanged
        clean, _wm, _bbox = self._screen_photo(alpha=0.5)
        ref = np.array(W.TINT_REFS["黄"], dtype=np.float32)
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.text((20, 130), "WATERMARK", font=_font(110),
                fill=(int(ref[0]), int(ref[1]), int(ref[2]), int(0.5 * 255)))
        arr = np.asarray(overlay).astype(np.float32)
        al = arr[..., 3:4] / 255.0
        wm = np.clip((1 - al) * clean.astype(np.float32) + al * ref,
                     0, 255).astype(np.uint8)
        x0, y0, x1, y1 = od.textbbox((20, 130), "WATERMARK", font=_font(110))
        before = _box_err(wm, clean, y0, y1, x0, x1)
        fixed, status = W.clean_light_watermark(wm)
        after = _box_err(fixed, clean, y0, y1, x0, x1)
        self.assertLess(after, before)
        self.assertEqual(status.status, "partial")
        src = self.tmp / "screen.jpg"
        Image.fromarray(wm).save(src, quality=95)
        _dst, fstatus = W.remove_light_file(src)
        self.assertEqual(fstatus.status, "partial")


class TestLightBoxes(unittest.TestCase):
    """第八批：框选去浅色水印（batch_remove_light_boxes）。"""

    def _screen_photo(self):
        """蓝灰底 + 淡黄斜字水印 + 深色 SQL 文字（与 TestRealPhoto 同款底）。"""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        rng = np.random.default_rng(11)
        H, Wd = 768, 1024
        yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
        t = (xx / Wd + yy / H) / 2
        base = np.zeros((H, Wd, 3), dtype=np.float32)
        for c, (a, b) in enumerate([(177, 208), (186, 221), (207, 241)]):
            base[..., c] = a + (b - a) * t
        img = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
            wmf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except Exception:
            font = ImageFont.load_default(); wmf = font
        for i in range(12):
            d.text((40, 50 + i * 52), f"SELECT a{i} FROM t WHERE x = {i}", font=font, fill=(52, 52, 58, 255))
        wm = Image.new("RGBA", (Wd, H), (0, 0, 0, 0))
        dw = ImageDraw.Draw(wm)
        dw.text((200, 150), "1102", font=wmf, fill=(180, 195, 130, 128))
        wm = wm.rotate(-18, resample=Image.BICUBIC, center=(Wd // 2, H // 2))
        img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(1))
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
        return arr

    def test_box_light_removes_watermark_only_inside_box(self):
        """框住水印字：框内水印去掉，框外一个像素不动。"""
        import numpy as np
        from PIL import Image
        import tempfile
        arr = self._screen_photo()
        # 水印字“1102”约在 (200,150)-(500,300) 一带，框给大一点
        box = (0.15, 0.10, 0.55, 0.45)
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "a.png"
            Image.fromarray(arr).save(src)
            out_dir = Path(td) / "out"
            res = W.batch_remove_light_boxes([src], [box], out_dir=str(out_dir))
            self.assertEqual(res["clean"], 1)
            self.assertEqual(res["done"], 1)
            fixed = np.asarray(Image.open(W.output_path(src, str(out_dir), unique=False)).convert("RGB"))
            # 框内：必须有改动（水印被反解掉），且改动后框内像素离背景更近。
            # 背景约 (195,205,225)：用框内像素与背景中位色的平均距离衡量。
            h, w = arr.shape[:2]
            x0, y0 = int(box[0] * w), int(box[1] * h)
            x1, y1 = int(box[2] * w), int(box[3] * h)
            bg = np.median(arr[y0:y1, x0:x1].reshape(-1, 3), axis=0)
            before_d = np.abs(arr[y0:y1, x0:x1].astype(np.float32) - bg).mean()
            after_d = np.abs(fixed[y0:y1, x0:x1].astype(np.float32) - bg).mean()
            self.assertLess(after_d, before_d,
                            "框内反解后应离背景更近（水印去掉）")
            # 框外：逐像素相等
            mask = np.ones(arr.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = False
            self.assertTrue(np.array_equal(fixed[mask], arr[mask]),
                            "框选去浅色不得动框外像素")

    def test_oversized_box_reports_rejected(self):
        """框占图超 70% 且框里有水印：应报 rejected（不是 unchanged），输出原样。"""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import tempfile
        # 大字标准黄水印：框再大检测也能找到它，70% 门才会触发
        H, Wd = 768, 1024
        arr = np.zeros((H, Wd, 3), dtype=np.uint8) + 210
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img, "RGBA")
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 110)
        except Exception:
            f = ImageFont.load_default()
        d.text((100, 300), "WATERMARK", font=f, fill=(252, 248, 196, 255))
        arr = np.asarray(img.convert("RGB"))
        box = (0.05, 0.05, 0.95, 0.95)  # 占图 81%，超 70% 门
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "a.png"
            Image.fromarray(arr).save(src)
            out_dir = Path(td) / "out"
            res = W.batch_remove_light_boxes([src], [box], out_dir=str(out_dir))
            self.assertEqual(res["rejected"], 1)
            fixed = np.asarray(Image.open(W.output_path(src, str(out_dir), unique=False)).convert("RGB"))
            self.assertTrue(np.array_equal(fixed, arr))

    def test_box_without_watermark_unchanged(self):
        """框里没水印：原样返回，状态 unchanged。"""
        import numpy as np
        from PIL import Image
        import tempfile
        arr = self._screen_photo()
        box = (0.75, 0.75, 0.95, 0.95)  # 右下角空白底色，无水印
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "a.png"
            Image.fromarray(arr).save(src)
            out_dir = Path(td) / "out"
            res = W.batch_remove_light_boxes([src], [box], out_dir=str(out_dir))
            self.assertEqual(res["unchanged"], 1)
            fixed = np.asarray(Image.open(W.output_path(src, str(out_dir), unique=False)).convert("RGB"))
            self.assertTrue(np.array_equal(fixed, arr))


class TestMaskCap(unittest.TestCase):
    """第九批：框内单色系掩膜上限（占框超 30% 就跳过该色系）。

    注：本测试是烟雾测试——合成图上蓝掩膜最高只到约 28%，到不了 30%
    门，触发不了上限。真正的验收以用户真图探针为准：用户框里蓝掩膜
    占框 34.6%（>30%），修后应被跳过、黄水印获胜。"""

    def _wave_photo(self, period=200, amp=60):
        """蓝灰底 + 长周期蓝波 + 淡黄水印字 + 深色 SQL 文字。"""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        rng = np.random.default_rng(11)
        H, Wd = 768, 1024
        yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
        t = (xx / Wd + yy / H) / 2
        base = np.zeros((H, Wd, 3), dtype=np.float32)
        for c, (a, b) in enumerate([(177, 208), (186, 221), (207, 241)]):
            base[..., c] = a + (b - a) * t
        wave = amp * np.sin(xx / period * 2 * np.pi)
        base[..., 2] += wave
        base[..., 0] -= wave * 0.4
        base = np.clip(base, 0, 255)
        img = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
            wmf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except Exception:
            font = ImageFont.load_default(); wmf = font
        for i in range(12):
            d.text((40, 50 + i * 52), f"SELECT a{i} FROM t WHERE x = {i}", font=font, fill=(52, 52, 58, 255))
        wm = Image.new("RGBA", (Wd, H), (0, 0, 0, 0))
        dw = ImageDraw.Draw(wm)
        dw.text((200, 150), "1102", font=wmf, fill=(180, 195, 130, 128))
        wm = wm.rotate(-18, resample=Image.BICUBIC, center=(Wd // 2, H // 2))
        img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(1))
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
        return arr

    def test_mask_cap_does_not_fire_below_threshold(self):
        """蓝掩膜约 28%（<30%）：上限不触发，蓝仍获胜——门限确为 30%，不是更低。"""
        import numpy as np
        arr = self._wave_photo()
        box = (0.15, 0.10, 0.55, 0.45)
        det = W.detect_light_in_box(arr, box, source="box")
        self.assertTrue(det.found)
        self.assertEqual(det.tint, "蓝")
        h, w = arr.shape[:2]
        x0, y0 = int(box[0] * w), int(box[1] * h)
        x1, y1 = int(box[2] * w), int(box[3] * h)
        area = (x1 - x0) * (y1 - y0)
        ratio = det.mask.sum() / area
        self.assertLess(ratio, 0.30)
        self.assertGreater(ratio, 0.20)

    def test_normal_yellow_still_wins(self):
        """无蓝波的正常图：黄水印仍获胜，上限没误伤正常检测。"""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        rng = np.random.default_rng(11)
        H, Wd = 768, 1024
        yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
        t = (xx / Wd + yy / H) / 2
        base = np.zeros((H, Wd, 3), dtype=np.float32)
        for c, (a, b) in enumerate([(177, 208), (186, 221), (207, 241)]):
            base[..., c] = a + (b - a) * t
        img = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
            wmf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except Exception:
            font = ImageFont.load_default(); wmf = font
        for i in range(12):
            d.text((40, 50 + i * 52), f"SELECT a{i} FROM t WHERE x = {i}", font=font, fill=(52, 52, 58, 255))
        wm = Image.new("RGBA", (Wd, H), (0, 0, 0, 0))
        dw = ImageDraw.Draw(wm)
        dw.text((200, 150), "1102", font=wmf, fill=(180, 195, 130, 128))
        wm = wm.rotate(-18, resample=Image.BICUBIC, center=(Wd // 2, H // 2))
        img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(1))
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
        box = (0.15, 0.10, 0.55, 0.45)
        det = W.detect_light_in_box(arr, box, source="box")
        self.assertTrue(det.found)
        self.assertIn(det.tint, ("黄", "深黄"))


class TestDarkGuard(unittest.TestCase):
    """第五批：反解不该把背景打黑（横杠噪点护栏回归）。"""

    def _dark_frac(self, a):
        luma = 0.299 * a[..., 0].astype(np.float32) + 0.587 * a[..., 1].astype(np.float32) + 0.114 * a[..., 2].astype(np.float32)
        return float((luma < 80).mean() * 100)

    def test_global_undo_never_darkens(self):
        """第六批：全局反解 undo_light_watermark 也不得把像素打黑。

        用户真图走的是文档路径（looks_like_photo=False），调的是全局
        undo_light_watermark，不是 undo_light_in_box——探针证实后者一次
        都没被调到。所以护栏也要加在全局反解上：反解后比原像素暗超 40
        的恢复原值。本测试只验“不变暗”这条不变量（烟雾测试，见提交说明）。
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        rng = np.random.default_rng(7)
        H, Wd = 600, 800
        yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
        t = (xx / Wd + yy / H) / 2
        base = np.zeros((H, Wd, 3), dtype=np.float32)
        for c, (a, b) in enumerate([(196, 208), (213, 221), (237, 241)]):
            base[..., c] = a + (b - a) * t
        img = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        for i in range(15):
            d.text((30, 30 + i * 34), f"SELECT col{i} FROM t WHERE x={i}",
                   font=font, fill=(60, 60, 60))
        arr = np.asarray(img).astype(np.uint8)
        orig_luma = (0.299 * arr[..., 0].astype(np.float32)
                     + 0.587 * arr[..., 1].astype(np.float32)
                     + 0.114 * arr[..., 2].astype(np.float32))
        for tint in ("蓝", "粉", "黄", "深黄", "绿"):
            fixed = W.undo_light_watermark(arr, tint=tint)
            fixed_luma = (0.299 * fixed[..., 0].astype(np.float32)
                          + 0.587 * fixed[..., 1].astype(np.float32)
                          + 0.114 * fixed[..., 2].astype(np.float32))
            worst = float((orig_luma - fixed_luma).max())
            self.assertLessEqual(
                worst, 40.0 + 1e-6,
                f"全局反解 tint={tint} 把像素打暗 {worst:.1f}（>40），护栏失效")

    def test_oversized_mask_returns_unchanged(self):
        """第七批：掩膜过大护栏——底色被误判成水印时不得处理。

        用户真图探针：detect_tint 投“蓝”，detect_light_watermark 掩膜
        51.80%（半张图都被当成水印），_flatten_background 把整片背景重铺，
        暗像素 2.74%→11.38%。水印占比不会这么大，掩膜超 30% 就原样返回。
        为触发文档路径，这里把 looks_like_photo 临时打桩成 False。
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        H, Wd = 600, 800
        yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
        t = (xx / Wd + yy / H) / 2
        base = np.zeros((H, Wd, 3), dtype=np.float32)
        for c, (a, b) in enumerate([(196, 208), (213, 221), (237, 241)]):
            base[..., c] = a + (b - a) * t
        img = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        for i in range(15):
            d.text((30, 30 + i * 34), f"SELECT col{i} FROM t WHERE x={i}",
                   font=font, fill=(60, 60, 60))
        arr = np.asarray(img).astype(np.uint8)
        orig = W.looks_like_photo
        W.looks_like_photo = lambda rgb, limit=25: False
        try:
            out, st = W.clean_light_watermark(arr)
        finally:
            W.looks_like_photo = orig
        self.assertEqual(st.status, "unchanged")
        self.assertTrue(np.array_equal(out, arr),
                        "掩膜过大时应原样返回，一个像素都不该动")

    def test_one_click_does_not_create_dark_bars(self):
        """蓝灰底照片一键抠后，暗像素占比不得明显上涨（杠子就是暗像素暴涨）。"""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        rng = np.random.default_rng(11)
        H, Wd = 768, 1024
        yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
        t = (xx / Wd + yy / H) / 2
        base = np.zeros((H, Wd, 3), dtype=np.float32)
        for c, (a, b) in enumerate([(177, 208), (186, 221), (207, 241)]):
            base[..., c] = a + (b - a) * t
        img = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
            wmf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except Exception:
            font = ImageFont.load_default(); wmf = font
        for i in range(12):
            d.text((40, 50 + i * 52), f"SELECT a{i} FROM t WHERE x = {i}", font=font, fill=(52, 52, 58, 255))
        wm = Image.new("RGBA", (Wd, H), (0, 0, 0, 0))
        dw = ImageDraw.Draw(wm)
        dw.text((200, 150), "1102", font=wmf, fill=(180, 195, 130, 128))
        wm = wm.rotate(-18, resample=Image.BICUBIC, center=(Wd // 2, H // 2))
        img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(1))
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
        before = self._dark_frac(arr)
        fixed, st = W.clean_light_watermark(arr)
        after = self._dark_frac(fixed)
        self.assertLessEqual(after, before + 1.0,
                             f"一键抠后暗像素 {before:.2f}% -> {after:.2f}%，疑似打黑出杠子")


if __name__ == "__main__":
    unittest.main()
