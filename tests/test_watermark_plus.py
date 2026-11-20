# -*- coding: utf-8 -*-
"""去水印新能力的回归：连贯度门、灵敏度旋钮、跨图模板、平铺水印。

样本由 tools/wm_bench.py 的生成器现做（确定性），不依赖仓库里的图片文件。
"""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class TestBatchWiring(unittest.TestCase):
    """模板与平铺已经接进批量流程，不只是能单独调用。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix=".tmp_batch_"))

    def tearDown(self):
        for item in sorted(self.tmp.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        self.tmp.rmdir()

    def _pages(self, count=5):
        paths = []
        for i in range(count):
            base = bench._doc(seed=3 + i * 4, lines=14)
            arr, mask = bench._stamp(base, "2023/12/24", 0.30,
                                     (252, 248, 196), [(150, 320)], size=58)
            path = self.tmp / f"p{i}.png"
            Image.fromarray(arr).save(path)
            paths.append(path)
            if i == 0:
                self.clean, self.mask = base, mask
                self.raw = arr
        return paths

    def test_light_batch_learns_template(self):
        paths = self._pages()
        out = self.tmp / "out"
        result = W.batch_remove_light(paths, out_dir=out, learn=True)
        self.assertEqual(result["done"], 5)
        self.assertGreaterEqual(result["template_share"], 0.8,
                                "同款同位置的水印应该学出模板")
        self.assertEqual(result["clean"], 5)
        self.assertTrue(all("模板" in d for _p, _s, d in result["results"]))
        target = W.output_path(paths[0], out, unique=False)
        self.assertTrue(target.exists(), "批量该真的写出文件")
        fixed = np.asarray(Image.open(target).convert("RGB"))
        self.assertLess(_err(fixed, self.clean, self.mask),
                        0.8 * _err(self.raw, self.clean, self.mask))

    def test_template_batch_without_learning_falls_back(self):
        # 没有水印的图学不出模板：整批要退回逐张检测，不能一片错误
        plain = self.tmp / "plain.png"
        Image.fromarray(bench._photo(seed=8, palette="wall")).save(plain)
        result = T.batch_remove_with_template([plain], out_dir=self.tmp / "o2")
        self.assertFalse(result["learned"])
        self.assertEqual(result["done"], 1)
        self.assertEqual(result["failed"], [])

    def test_tiled_batch_uses_period(self):
        clean = bench._doc(seed=11, lines=10)
        arr, mask = bench._stamp(clean, "SAMPLE", 0.22, (252, 248, 196),
                                 bench._tiled_positions(900, 1200, 300, 260),
                                 size=52)
        src = self.tmp / "tiled.png"
        Image.fromarray(arr).save(src)
        result = T.batch_remove_tiled([src], out_dir=self.tmp / "t")
        self.assertEqual(result["clean"], 1)
        detail = result["results"][0][2]
        self.assertIn("平铺水印已重建", detail)
        out = np.asarray(Image.open(
            W.output_path(src, self.tmp / "t", unique=False)).convert("RGB"))
        self.assertLess(_err(out, clean, mask), 0.7 * _err(arr, clean, mask))

    def test_tiled_fallback_guard_on_clean_page(self):
        # 兜底接在“占比过大”那条拒认上：没有周期时必须是原样返回
        clean = bench._doc(seed=17)
        out, info = W._tiled_rebuild(clean)
        self.assertIsNone(info)
        self.assertTrue(np.array_equal(out, clean))


class TestMaskRecall(unittest.TestCase):
    """掩膜召回：阈值切掉的半截笔画要补回来，干净页一个像素不长。"""

    def test_recall_grows_mask_and_keeps_clean_pages_alone(self):
        clean = bench._doc(seed=3)
        arr, mask = bench._stamp(clean, "2023/12/24", 0.30, (252, 248, 196),
                                 [(120, 300), (420, 700)], size=58)
        base = W._thin_strokes(W.detect_light_watermark(arr))
        grown = W._recall_mask(arr, base, "黄")
        self.assertGreater(int(grown.sum()), int(base.sum()))
        self.assertGreaterEqual(float(grown[mask].mean()), 0.85)
        # 干净页种子为空：软门再宽也不许凭空长出水印
        page = bench._doc(seed=17)
        empty = np.zeros(page.shape[:2], dtype=bool)
        self.assertFalse(W._recall_mask(page, empty, "黄").any())
        # 召回之后反解更干净（真值误差比原来低四分之一以上）
        before = _err(arr, clean, mask)
        out, status = W.clean_light_watermark(arr)
        self.assertEqual(status.status, "clean")
        self.assertLess(_err(out, clean, mask), 0.75 * before)


class TestSelfCheckGuard(unittest.TestCase):
    """像素级自检：解完比输入还差就整张退回，正常的解不许被拦。"""

    def test_reverts_when_unblend_makes_it_worse(self):
        clean = bench._doc(seed=3)
        arr, mask = bench._stamp(clean, "2023/12/24", 0.30, (252, 248, 196),
                                 [(120, 300), (420, 700)], size=58)
        # 模拟“参考色估错”：把掩膜里的像素再往黄上推一层，比输入更偏
        worse = arr.astype(np.float32)
        worse[mask] = worse[mask] * 0.7 + np.array([252, 248, 196]) * 0.3
        worse = np.clip(worse, 0, 255).astype(np.uint8)
        kept, passed, note = W._self_check(arr, worse, mask, "黄")
        self.assertFalse(passed)
        self.assertTrue(np.array_equal(kept, arr), "更差的结果必须退回原像素")
        self.assertIn("反解没有变好", note)

    def test_normal_unblend_passes(self):
        clean = bench._doc(seed=3)
        arr, mask = bench._stamp(clean, "2023/12/24", 0.30, (252, 248, 196),
                                 [(120, 300), (420, 700)], size=58)
        ref = W._choose_tint_ref(arr, mask, "黄")
        fixed = W.undo_light_watermark(arr, tint="黄", ref=ref, mask=mask)
        fixed = W._flatten_background(fixed, mask)
        kept, passed, note = W._self_check(arr, fixed, mask, "黄")
        self.assertTrue(passed)
        self.assertTrue(np.array_equal(kept, fixed))
        self.assertEqual(note, "")


class TestRecompressDiagnostic(unittest.TestCase):
    """重编码：只做诊断（去块替换实测让人均覆盖降 4 个点，已撤回）。"""

    def _jpeg(self, arr, quality=45):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as im:
            return np.asarray(im.convert("RGB"))

    def test_blockiness_separates_recompressed_content(self):
        # 有纹理的照片最能显块噪声：无损 <1.35，压过之后明显超过
        clean = bench._photo(seed=5, palette="grass")
        self.assertLess(W.blockiness(clean), 1.35)
        self.assertGreater(W.blockiness(self._jpeg(clean, 70)), 1.35)
        # 平滑纸面本身没有块噪声，压过也不该被误判成“二次压缩”
        page = bench._doc(seed=3)
        self.assertLess(W.blockiness(self._jpeg(page)), 1.35)


class TestLargePageFastDetect(unittest.TestCase):
    """文档路径提速：大图降采样找掩膜，位置与全分辨率一致。"""

    def test_fast_matches_full_resolution(self):
        clean = bench._doc(width=2000, height=2600, seed=3, lines=42)
        arr, mask = bench._stamp(clean, "2023/12/24", 0.30, (252, 248, 196),
                                 [(300, 600), (900, 1500)], size=96)
        full = W._thin_strokes(W.detect_light_watermark(arr, tint="黄"))
        fast = W._detect_mask(arr, "黄")
        self.assertGreater(int(full.sum()), 2000)
        iou = (full & fast).sum() / max((full | fast).sum(), 1)
        self.assertGreater(iou, 0.7, f"降采样掩膜与全分辨率 IoU {iou:.2f}")
        self.assertGreater(float(fast[mask].mean()), 0.6)


class TestTiledPhaseFolding(unittest.TestCase):
    """平铺水印按相位折叠：图幅不是周期整数倍时余数区也要处理。"""

    def test_leftover_rows_are_rebuilt(self):
        clean = bench._doc(seed=11, lines=10)
        arr, mask = bench._stamp(clean, "SAMPLE", 0.22, (252, 248, 196),
                                 bench._tiled_positions(900, 1200, 300, 260),
                                 size=52)
        period = T.tiled_period(arr)
        self.assertIsNotNone(period)
        self.assertEqual((period[0], period[1]), (260, 300))
        fixed, info = T.remove_tiled_watermark(arr)
        self.assertIsNotNone(info)
        self.assertEqual(info["period"], (260, 300))
        # 1200 不是 260 的整数倍：底部余数行以前整条丢掉，现在必须处理
        strip = mask.copy()
        strip[:1040] = False
        self.assertTrue(strip.any())
        changed = (np.abs(fixed.astype(np.int16)
                          - arr.astype(np.int16)).max(axis=2) > 2)
        self.assertGreater(float(changed[strip].mean()), 0.3,
                           "右下余数区的水印没被处理")
        self.assertLess(_err(fixed, clean, mask), 0.75 * _err(arr, clean, mask))


class TestTemplateAgreement(unittest.TestCase):
    """模板贴合度：学模板时顺手量一次，不用为报告再跑一遍检测。"""

    def test_agreement_is_reported_by_batch(self):
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_tplrep_"))
        try:
            paths = []
            for i in range(5):
                base = bench._doc(seed=3 + i * 4, lines=14)
                arr, _mask = bench._stamp(base, "2023/12/24", 0.30,
                                          (252, 248, 196), [(150, 320)],
                                          size=58)
                p = tmp / f"p{i}.png"
                Image.fromarray(arr).save(p)
                paths.append(p)
            template = T.learn_template(paths)
            self.assertIsNotNone(template)
            # 同位置、同形状的 5 张：贴合度应当高（不是“模板没用上”）
            self.assertGreaterEqual(template.agreement, 0.5)
            out = T.batch_remove_with_template(
                paths, out_dir=tmp / "out", report=tmp / "r.csv")
            self.assertGreaterEqual(out["template_agreement"], 0.5)
            self.assertTrue(Path(out["report"]).is_file(),
                            "模板批处理也要写报告")
            self.assertEqual(out["clean"], 5)
        finally:
            for item in sorted(tmp.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            tmp.rmdir()


class TestTemplatePersistence(unittest.TestCase):
    """模板存盘：同一款水印学一次，下一批直接读回来用。"""

    def test_save_load_and_batch_uses_file(self):
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_tplfile_"))
        try:
            paths = []
            for i in range(5):
                base = bench._doc(seed=3 + i * 4, lines=14)
                arr, _mask = bench._stamp(base, "2023/12/24", 0.30,
                                          (252, 248, 196), [(150, 320)],
                                          size=58)
                p = tmp / f"p{i}.png"
                Image.fromarray(arr).save(p)
                paths.append(p)
            template = T.learn_template(paths)
            self.assertIsNotNone(template)
            store = T.save_template(template, tmp / "tpl.npz")
            self.assertTrue(store.is_file())
            back = T.load_template(store)
            self.assertIsNotNone(back)
            self.assertEqual(back.box, template.box)
            self.assertEqual(back.tint, template.tint)
            self.assertTrue(np.array_equal(back.mask, template.mask))
            self.assertAlmostEqual(back.share, template.share)
            # 跨批复用：模板从文件读进来，batch 里不再现学
            out = W.batch_remove_light(paths, out_dir=tmp / "out",
                                       template_path=tmp / "tpl.npz")
            self.assertEqual(out["clean"], 5)
            self.assertIsNone(T.load_template(tmp / "missing.npz"))
        finally:
            for item in sorted(tmp.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            tmp.rmdir()


class TestBatchReport(unittest.TestCase):
    """批处理报告：一张 CSV，Excel 双击不乱码。"""

    def test_csv_report_is_written(self):
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_rep_"))
        try:
            paths = []
            for i in range(5):
                base = bench._doc(seed=3 + i * 4, lines=14)
                arr, _mask = bench._stamp(base, "2023/12/24", 0.30,
                                          (252, 248, 196), [(150, 320)],
                                          size=58)
                p = tmp / f"p{i}.png"
                Image.fromarray(arr).save(p)
                paths.append(p)
            result = W.batch_remove_light(paths, out_dir=tmp / "out",
                                          report=tmp / "r.csv")
            self.assertEqual(result.get("report"), str(tmp / "r.csv"))
            text = (tmp / "r.csv").read_text(encoding="utf-8-sig")
            self.assertTrue(text.startswith("文件,状态,说明"))
            self.assertIn("合计", text)
            self.assertEqual(text.count("\n"), len(paths) + 2)
        finally:
            for item in sorted(tmp.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            tmp.rmdir()


class TestPhotoMaskShareGuard(unittest.TestCase):
    """真实照片上几乎每一格都能“检出”一点，九格拼起来占两成画面。

    逐格阈值分不开“水印”与“纹理”（树影、花束、皮肤反光都比局部背景亮），
    整图并集占比能：实测合成水印照片 1.5%~4.7%，真实干净照片 15%~18%。
    """

    def setUp(self):
        self.rgb = bench._photo(seed=5, palette="grass", width=600, height=400)

    def _det(self, share):
        mask = np.zeros(self.rgb.shape[:2], dtype=bool)
        mask[:int(self.rgb.shape[0] * share), :] = True
        return W.LightDetection((0.0, 0.0, 1.0, 0.4), tint="黄", mask=mask,
                                score=3.0, source="grid")

    def test_scattered_pale_texture_is_left_alone(self):
        with mock.patch.object(W, "_grid_light_boxes",
                               return_value=[self._det(0.18)]):
            out, status = W._clean_light_core(self.rgb)
        self.assertEqual(status.status, "unchanged", status.detail)
        self.assertIn("纹理", status.detail)
        self.assertTrue(np.array_equal(out, self.rgb),
                        "判成纹理就一个像素都不许动")

    def test_watermark_sized_mask_still_processed(self):
        # 水印尺度的并集（5%）不能进护栏，否则真水印会被一起挡掉
        with mock.patch.object(W, "_grid_light_boxes",
                               return_value=[self._det(0.05)]):
            _out, status = W._clean_light_core(self.rgb)
        self.assertNotIn("纹理", status.detail)


class TestLabelTool(unittest.TestCase):
    """真值样本库脚手架：导出初稿 → 校验 → 叠加预览，bench 能直接吃。"""

    def _lab(self):
        spec = importlib.util.spec_from_file_location(
            "wm_label", ROOT / "tools" / "wm_label.py")
        lab = importlib.util.module_from_spec(spec)
        sys.modules["wm_label"] = lab
        spec.loader.exec_module(lab)
        return lab

    def test_export_check_overlay(self):
        lab = self._lab()
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_lbl_"))
        try:
            src = tmp / "src"
            src.mkdir()
            arr, _mask = bench._stamp(bench._doc(seed=3), "2023/12/24", 0.30,
                                      (252, 248, 196), [(120, 300)], size=58)
            Image.fromarray(arr).save(src / "page1.png")
            Image.fromarray(bench._photo(seed=5)).save(src / "photo1.jpg")
            cases = lab.export(src, tmp / "cases")
            self.assertEqual(len(cases), 2)
            for case in cases:
                self.assertTrue((case / "input.png").is_file())
                self.assertTrue((case / "mask.png").is_file())
                self.assertTrue((case / "_overlay.png").is_file())
            meta = json.loads((cases[0] / "meta.json").read_text(encoding="utf-8"))
            self.assertIn(meta["kind"], lab.KINDS)
            self.assertEqual(lab.check(tmp / "cases"), [])
            self.assertEqual(len(bench.run(tmp / "cases")), 2)
            self.assertEqual(len(lab.overlay(tmp / "cases", force=True)), 2)
        finally:
            for item in sorted(tmp.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            tmp.rmdir()

    def test_audit_only_reports(self):
        """审计只给结论不落盘：整批里哪些图值得处理，跑完就能看到。"""
        lab = self._lab()
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_aud_"))
        try:
            src = tmp / "src"
            src.mkdir()
            arr, _mask = bench._stamp(bench._doc(seed=3), "2023/12/24", 0.30,
                                      (252, 248, 196), [(120, 300)], size=58)
            Image.fromarray(arr).save(src / "wm.png")
            Image.fromarray(bench._doc(seed=4)).save(src / "plain.png")
            report = tmp / "audit.json"
            rows = lab.audit(src, out_json=report)
            self.assertEqual(len(rows), 2)
            self.assertTrue(report.is_file())
            by_name = {Path(r["file"]).name: r for r in rows}
            self.assertEqual(by_name["wm.png"]["verdict"], "有水印")
            self.assertEqual(by_name["plain.png"]["verdict"], "干净（未检出）")
            # 只读：审计不改任何图（掩膜只在内存里）
            self.assertEqual(
                np.asarray(Image.open(src / "plain.png")).shape,
                bench._doc(seed=4).shape)
        finally:
            for item in sorted(tmp.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            tmp.rmdir()

    def test_refine_from_boxes_and_points(self):
        """点/框标注只给位置，掩膜由检测细化——不必手工描水印笔画。"""
        lab = self._lab()
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_ref_"))
        try:
            src = tmp / "src"
            src.mkdir()
            arr, _mask = bench._stamp(bench._doc(seed=3), "2023/12/24", 0.30,
                                      (252, 248, 196), [(120, 300)], size=58)
            Image.fromarray(arr).save(src / "page1.png")
            case = lab.export(src, tmp / "cases")[0]
            # 从全黑重来：模拟“只拖了框、还没描边”的标注
            Image.fromarray(np.zeros(arr.shape[:2], dtype=np.uint8)).save(
                case / "mask.png")
            lab.refine(case, boxes=["80,260,600,380"])
            mask = np.asarray(Image.open(case / "mask.png").convert("L")) > 127
            self.assertGreater(int(mask.sum()), 500)
            self.assertEqual(lab.check(tmp / "cases"), [])
            # 点标注单独用：扩成方框后同样能细化，--replace 丢掉旧掩膜
            lab.refine(case, points=["200,320"], replace=True)
            again = np.asarray(Image.open(case / "mask.png").convert("L")) > 127
            self.assertGreater(int(again.sum()), 500)
            self.assertEqual(lab.check(tmp / "cases"), [])
        finally:
            for item in sorted(tmp.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            tmp.rmdir()


if __name__ == "__main__":
    unittest.main()
