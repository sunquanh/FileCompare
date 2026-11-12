"""核心逻辑回归测试。

跑法：python -m unittest discover -s tests -v
真实回收站往返默认跳过，需要时设 DEDUP_TEST_RECYCLE=1（会短暂往回收站丢一个
测试文件，测试结束会还原并删除）。
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dedup_audio as audio          # noqa: E402
import dedup_core as core            # noqa: E402
import dedup_files as files          # noqa: E402
import dedup_models as models        # noqa: E402
import dedup_tools as tools          # noqa: E402


def make_item(name, size=1024, duration=0.0):
    return models.FileItem(path=Path(name), size=size, mtime=0.0,
                           duration=duration)


class SkipRules(unittest.TestCase):
    def test_dev_dirs_skipped_at_any_depth(self):
        for name in (".git", "node_modules", "__pycache__", "$RECYCLE.BIN",
                     "System Volume Information"):
            self.assertTrue(models.should_skip_dir(name, 1), name)
            self.assertTrue(models.should_skip_dir(name, 4), name)

    def test_system_dirs_only_at_scan_root(self):
        self.assertTrue(models.should_skip_dir("AppData", 1))
        self.assertTrue(models.should_skip_dir("Windows", 1))
        self.assertTrue(models.should_skip_dir("windows", 1))   # 大小写
        self.assertFalse(models.should_skip_dir("appdata", 2))
        self.assertFalse(models.should_skip_dir("Windows", 3))

    def test_normal_dir_kept(self):
        self.assertFalse(models.should_skip_dir("照片", 1))
        self.assertFalse(models.should_skip_dir("", 1))


class CollectFiles(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp_scan_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_skips_lowercase_system_dir(self):
        # 旧实现按原样比较，小写 windows/build 会被整个扫进来
        deep = self.root / "windows" / "build"
        deep.mkdir(parents=True)
        (deep / "a.bin").write_bytes(b"x" * 8)
        (self.root / "photo.jpg").write_bytes(b"y" * 8)
        found = files.collect_files(models.ScanOptions(root=self.root))[0]
        names = {item.path.name for item in found}
        self.assertEqual(names, {"photo.jpg"})

    def test_exclude_is_case_insensitive(self):
        sub = self.root / "Cache"
        sub.mkdir()
        (sub / "a.bin").write_bytes(b"x" * 8)
        found = files.collect_files(
            models.ScanOptions(root=self.root, exclude=["cache"]))[0]
        self.assertEqual(found, [])


class WastedSpace(unittest.TestCase):
    def test_wasted_follows_kept_item(self):
        small, big = make_item("small.bin", 1 << 20), make_item("big.bin", 10 << 20)
        group = models.DupGroup(kind=core.KIND_EXACT, items=[big, small],
                                similarity=100, keep=0)
        self.assertEqual(group.wasted, 1 << 20)      # 保留大文件
        group.keep = 1
        self.assertEqual(group.wasted, 10 << 20)     # 保留小文件


class AudioClusters(unittest.TestCase):
    def test_no_chained_drift(self):
        # 100/103/106 首尾差 6 秒 > 4 秒容差，不能并成一组
        groups = audio.cluster_audio([
            make_item("a.mp3", duration=100.0),
            make_item("b.mp3", duration=103.0),
            make_item("c.mp3", duration=106.0),
        ])
        self.assertTrue(all(len(g) < 3 for g in groups), [len(g) for g in groups])

    def test_near_duplicates_grouped(self):
        groups = audio.cluster_audio([
            make_item("a.mp3", duration=200.0),
            make_item("b.mp3", duration=202.0),
        ])
        self.assertEqual([len(g) for g in groups], [2])


class RenameUndo(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp_undo_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _manifest(self, records):
        return tools.save_manifest("rename", self.root, records)

    def test_swap_is_restored(self):
        a, b = self.root / "a.txt", self.root / "b.txt"
        # 已经互换过的状态：a 里是 B 的内容，b 里是 A 的内容
        a.write_text("B", encoding="utf-8")
        b.write_text("A", encoding="utf-8")
        path = self._manifest([
            {"act": "rename", "src": str(a), "dst": str(b)},
            {"act": "rename", "src": str(b), "dst": str(a)},
        ])
        done, failed, skipped = tools.undo_manifest(path)
        self.assertEqual((done, failed, skipped), (2, 0, 0))
        self.assertEqual(a.read_text(encoding="utf-8"), "A")
        self.assertEqual(b.read_text(encoding="utf-8"), "B")

    def test_case_only_rename_is_restored(self):
        target = self.root / "photo.JPG"
        target.write_text("pic", encoding="utf-8")
        path = self._manifest([
            {"act": "rename", "src": str(self.root / "Photo.JPG"),
             "dst": str(target)},
        ])
        done, failed, skipped = tools.undo_manifest(path)
        self.assertEqual((done, failed, skipped), (1, 0, 0))
        restored = self.root / "Photo.JPG"
        self.assertTrue(restored.exists())
        self.assertEqual(restored.read_text(encoding="utf-8"), "pic")


class CopyUndo(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp_copy_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_untouched_copy_removed(self):
        source = self.root / "src.bin"
        source.write_bytes(b"1234")
        target = self.root / "dst.bin"
        shutil.copy2(source, target)
        st = target.stat()
        path = tools.save_manifest("organize", self.root, [
            {"act": "copy", "src": str(source), "dst": str(target),
             "size": int(st.st_size), "mtime": float(st.st_mtime)},
        ])
        done, _failed, _skipped = tools.undo_manifest(path)
        self.assertEqual(done, 1)
        self.assertFalse(target.exists())

    def test_modified_copy_is_kept(self):
        source = self.root / "src.bin"
        source.write_bytes(b"1234")
        target = self.root / "dst.bin"
        shutil.copy2(source, target)
        st = target.stat()
        path = tools.save_manifest("organize", self.root, [
            {"act": "copy", "src": str(source), "dst": str(target),
             "size": int(st.st_size), "mtime": float(st.st_mtime)},
        ])
        target.write_bytes(b"12345678")          # 用户后来编辑过
        done, _failed, skipped = tools.undo_manifest(path)
        self.assertEqual((done, skipped), (0, 1))
        self.assertTrue(target.exists())


class ReportExport(unittest.TestCase):
    def test_export_json_is_reexported(self):
        # GUI 的“导出 JSON”菜单直接调 core.export_report_json，缺了会必崩
        self.assertTrue(callable(getattr(core, "export_report_json", None)))
        self.assertTrue(callable(getattr(core, "norm_path", None)))


@unittest.skipUnless(os.name == "nt", "回收站是 Windows 专有")
class RecycleRoundtrip(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp_recycle_"))
        self.path = self.root / "dedup_probe.txt"

    def tearDown(self):
        # 用例中途失败时探针可能还留在回收站里：先还原再删，不留给用户
        try:
            files.restore_from_recycle([str(self.path)])
        except Exception:
            pass
        if self.path.exists():
            self.path.unlink()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_capability_check(self):
        self.path.write_text("x", encoding="utf-8")
        ok, blocked = files.recycle_capable([str(self.path)])
        self.assertEqual(blocked, [])
        self.assertEqual(ok, [str(self.path)])

    def test_missing_path_is_not_blocked(self):
        ok, blocked = files.recycle_capable([str(self.path)])
        self.assertEqual(blocked, [])
        self.assertEqual(ok, [str(self.path)])

    @unittest.skipUnless(os.environ.get("DEDUP_TEST_RECYCLE") == "1",
                         "设 DEDUP_TEST_RECYCLE=1 才真删（会先投入回收站）")
    def test_recycle_and_restore(self):
        self.path.write_text("probe", encoding="utf-8")
        deleted, failed = files.recycle_paths([str(self.path)])
        self.assertEqual(failed, [])
        self.assertEqual(deleted, [str(self.path)])
        recycled, gone = files.verify_recycled(deleted)
        self.assertEqual(gone, set())
        self.assertEqual(recycled, {str(self.path)})
        restored, bad = files.restore_from_recycle([str(self.path)])
        self.assertEqual(bad, [])
        self.assertEqual(restored, [str(self.path)])
        self.assertEqual(self.path.read_text(encoding="utf-8"), "probe")

    @unittest.skipUnless(os.environ.get("DEDUP_TEST_RECYCLE") == "1",
                         "设 DEDUP_TEST_RECYCLE=1 才真删（会先投入回收站）")
    def test_conflicting_destination_is_skipped_without_losing_file(self):
        """原位置已有同名文件：跳过而不是覆盖，且条目必须还在回收站里。"""
        self.path.write_text("original", encoding="utf-8")
        deleted, failed = files.recycle_paths([str(self.path)])
        self.assertEqual((deleted, failed), ([str(self.path)], []))

        self.path.write_text("newer", encoding="utf-8")   # 原位置又出现同名文件
        restored, bad = files.restore_from_recycle([str(self.path)])
        self.assertEqual((restored, bad), ([], [str(self.path)]))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "newer")

        # 跳过后若把 $R/$I 也删了，用户腾开位置就再也拿不回来 —— 必须是可重试的
        recycled, _gone = files.verify_recycled([str(self.path)])
        self.assertIn(str(self.path), recycled)

        self.path.unlink()                                # 腾开位置后重试
        restored, bad = files.restore_from_recycle([str(self.path)])
        self.assertEqual((restored, bad), ([str(self.path)], []))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "original")

    @unittest.skipUnless(os.environ.get("DEDUP_TEST_RECYCLE") == "1",
                         "设 DEDUP_TEST_RECYCLE=1 才真删（会先投入回收站）")
    def test_file_emptied_from_recycle_bin_is_reported(self):
        """回收站被清空：报失败而不是抛异常，也不会凭空造出文件。"""
        self.path.write_text("vanished", encoding="utf-8")
        files.recycle_paths([str(self.path)])

        # 真的把它从回收站取出来再删掉，等价于用户清空了回收站
        restored, bad = files.restore_from_recycle([str(self.path)])
        self.assertEqual((restored, bad), ([str(self.path)], []))
        self.path.unlink()
        self.assertFalse(self.path.exists())

        recycled, _gone = files.verify_recycled([str(self.path)])
        self.assertNotIn(str(self.path), recycled)

        restored, bad = files.restore_from_recycle([str(self.path)])
        self.assertEqual((restored, bad), ([], [str(self.path)]))
        self.assertFalse(self.path.exists())

class RestoreRetry(unittest.TestCase):
    """刚删除的文件可能被扫描器短暂占用：还原要重试，不能一次失败就报错。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp_restore_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_transient_lock_is_retried(self):
        data_file = self.root / "$Rprobe.bin"
        index_file = self.root / "$Iprobe.bin"
        data_file.write_text("payload", encoding="utf-8")
        index_file.write_bytes(bytes(24))
        target = self.root / "nested" / "restored.txt"
        real_move = shutil.move
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("另一个程序正在使用此文件")
            return real_move(src, dst)

        with mock.patch.object(files.shutil, "move", side_effect=flaky):
            ok = files._restore_entry(str(target),
                                      (data_file, index_file, str(target)))
        self.assertTrue(ok, "瞬时占用后第三次尝试应当成功")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(target.read_text(encoding="utf-8"), "payload")
        self.assertFalse(index_file.exists(), "还原后要清掉 $I 元数据")


class PhotoFingerprintDedup(unittest.TestCase):
    """指纹去重 + 只算右上半三角，结果必须与旧定义的逐对暴力实现完全一致。"""

    @staticmethod
    def _photos():
        import dedup_photos as photos
        return photos

    @classmethod
    def _naive(cls, fps, dist=8, fine=42, hist_tol=0.26, ratio_tol=0.12):
        photos = cls._photos()
        np = photos.np
        out = []
        n = len(fps)
        for i in range(n):
            a = fps[i]
            ra = a.width / a.height if a.height else 1.0
            for j in range(i + 1, n):
                b = fps[j]
                rb = b.width / b.height if b.height else 1.0
                if not (abs(ra / rb - 1.0) <= ratio_tol or
                        abs(ra * rb - 1.0) <= ratio_tol):
                    continue
                hist_d = float(1.0 - np.sqrt(a.hist * b.hist).sum())
                if hist_d > hist_tol:
                    continue
                best = None
                for la in range(photos.LEVELS):
                    for lb in range(photos.LEVELS):
                        for oa in range(photos.ORIENTS):
                            for ob in range(photos.ORIENTS):
                                coarse = int(photos.popcount(
                                    np.uint64(int(a.dh[la][oa])
                                              ^ int(b.dh[lb][ob]))))
                                if coarse > dist:
                                    continue
                                fine_c = sum(int(photos.popcount(
                                    np.uint64(int(a.ph[la][oa][w])
                                              ^ int(b.ph[lb][ob][w]))))
                                    for w in range(4))
                                if (fine_c <= fine
                                        and (best is None or fine_c < best[0])):
                                    best = (fine_c, coarse)
                if best is not None:
                    out.append((i, j, best[1], best[0], hist_d, 1.0, 0.0))
        return out

    def test_matches_brute_force(self):
        photos = self._photos()
        np = photos.np
        rng = np.random.default_rng(20240607)

        def make():
            shape = (photos.LEVELS, photos.ORIENTS)
            dh = rng.integers(0, 1 << 64, shape, dtype=np.uint64)
            ph = rng.integers(0, 1 << 64, (photos.LEVELS, photos.ORIENTS, 4),
                              dtype=np.uint64)
            hist = rng.random(photos.HIST_SIZE).astype(np.float32)
            hist = hist / hist.sum()
            return photos.PhotoFP(dh, ph, hist, 640, 480, int(dh[0].min()))

        a, b, c = make(), make(), make()
        near_dh = a.dh.copy()
        near_dh[0, 0] ^= np.uint64(0b1011)
        near_ph = a.ph.copy()
        near_ph[0, 3, 2] ^= np.uint64(0b111)
        near = photos.PhotoFP(near_dh, near_ph, a.hist, 640, 480,
                              int(near_dh[0].min()))
        # 竖版副本：宽高比是倒数，比例预筛按旋转 90° 语义应当接受
        port_dh = a.dh.copy()
        port_dh[0, 1] ^= np.uint64(0b1)
        port = photos.PhotoFP(port_dh, a.ph, a.hist, 480, 640,
                              int(port_dh[0].min()))
        fps = [a, b, near, c, a, port]

        edges = photos.match_photos(fps)
        self.assertTrue(edges)
        self.assertEqual(edges, self._naive(fps))
        self.assertEqual(edges, sorted(edges, key=lambda e: (e[0], e[1])))
        self.assertEqual(len({(e[0], e[1]) for e in edges}), len(edges))
        direct = [e for e in edges if e[0] == 0 and e[1] == 4]
        self.assertEqual(len(direct), 1)
        self.assertEqual((direct[0][2], direct[0][3]), (0, 0))
        portrait = [e for e in edges if e[0] == 0 and e[1] == 5]
        self.assertEqual(len(portrait), 1)
        self.assertEqual((portrait[0][2], portrait[0][3]), (0, 0))

    def test_all_identical(self):
        photos = self._photos()
        np = photos.np
        nx = photos.LEVELS * photos.ORIENTS
        dh = np.arange(nx, dtype=np.uint64).reshape(photos.LEVELS,
                                                    photos.ORIENTS)
        ph = np.arange(nx * 4, dtype=np.uint64).reshape(photos.LEVELS,
                                                        photos.ORIENTS, 4)
        hist = np.full(photos.HIST_SIZE, 1.0 / photos.HIST_SIZE,
                       dtype=np.float32)
        one = photos.PhotoFP(dh, ph, hist, 640, 480, 0)
        edges = photos.match_photos([one] * 5)
        self.assertEqual(len(edges), 10)
        self.assertEqual({(e[0], e[1]) for e in edges},
                         {(i, j) for i in range(5) for j in range(i + 1, 5)})
        self.assertEqual({(e[2], e[3]) for e in edges}, {(0, 0)})

class PhotoStructureGates(unittest.TestCase):
    """衣服背景相似、主体不同不能被并成一组；同一张图的不同版本仍要命中。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp_photo_gate_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _photos():
        import dedup_photos as photos
        return photos

    @staticmethod
    def _scene(dx=0, head=1.0, arms=0.0):
        """同款衣服、同款背景的合成人像；dx/head/arms 模拟不同的人与姿态。"""
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (360, 280), (174, 182, 188))
        d = ImageDraw.Draw(img)
        for y in range(0, 280, 5):
            d.rectangle([0, y, 360, y + 5], fill=(172 + y // 24, 182, 188))
        d.rectangle([18, 18, 118, 148], fill=(206, 216, 222))
        d.rectangle([120 + dx, 30, 300 + dx, 62], fill=(58, 84, 120))
        cx = 176 + dx
        hw = int(40 * head)
        d.ellipse([cx - hw, 112 - int(10 * head), cx + hw, 196],
                  fill=(226, 190, 168))
        d.pieslice([cx - hw - 2, 102 - int(10 * head), cx + hw + 2, 174],
                   180, 360, fill=(62, 46, 38))
        d.ellipse([cx - hw // 2, 142, cx - hw // 2 + 9, 152], fill=(54, 44, 40))
        d.ellipse([cx + hw // 2 - 9, 142, cx + hw // 2, 152], fill=(54, 44, 40))
        d.arc([cx - 13, 158, cx + 13, 184], 20, 160, fill=(150, 90, 88), width=3)
        for side in (-1, 1):
            d.line([(cx + side * (hw + 16), 92),
                    (cx + side * (hw + 16 + 12 * arms), 214)],
                   fill=(58, 84, 120), width=20)
        for x in range(228, 300, 12):
            d.line([(x + dx, 120), (x + dx, 210)], fill=(40, 60, 90), width=3)
        return img

    @staticmethod
    def _fabric_detail_changed():
        """低频几乎不变、细节全变：在人物区域叠一层细棋盘纹理。"""
        import numpy as np
        from PIL import Image
        arr = np.asarray(PhotoStructureGates._scene(), dtype=np.int16).copy()
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((xx // 3 + yy // 3) % 2) * 2 - 1
        mask = np.zeros((h, w), dtype=bool)
        mask[20:230, 90:330] = True
        arr[mask] += (checker[mask] * 24)[:, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _write(self, name, img, **kw):
        path = self.root / name
        img.save(path, **kw)
        return path

    def _pair(self, first, second, **match_kw):
        photos = self._photos()
        fps = [photos.fingerprint(self._write("a.png", first)),
               photos.fingerprint(self._write("b.png", second))]
        return photos, fps, photos.match_photos(fps, **match_kw)

    def test_version_of_same_photo_still_matches(self):
        from PIL import Image
        photos = self._photos()
        base = self._write("base.png", self._scene())
        small = self._write(
            "small.jpg",
            self._scene().resize((252, 196), Image.Resampling.LANCZOS),
            quality=70)
        fps = [photos.fingerprint(base), photos.fingerprint(small)]
        edges = photos.match_photos(fps)
        self.assertEqual([(e[0], e[1]) for e in edges], [(0, 1)])
        self.assertGreaterEqual(edges[0][5], 0.95)
        sim = photos.pair_similarity(edges[0][3], edges[0][5])
        self.assertGreaterEqual(sim, 95)

    def test_same_clothes_different_person_is_rejected(self):
        """同款衣服、同款背景，不同的人与姿态：只放宽旧判据会误判，结构闸门挡住。"""
        photos, fps, edges = self._pair(
            self._scene(), self._scene(dx=16, head=1.15, arms=1.6),
            dist=64, fine=255, hist_tol=1.0)
        self.assertEqual(edges, [])
        ungated = photos.match_photos(fps, dist=64, fine=255, hist_tol=1.0,
                                      ncc_tol=0.0, layout_tol=1e9)
        self.assertEqual([(e[0], e[1]) for e in ungated], [(0, 1)])
        self.assertLess(ungated[0][5], 0.88)

    def test_new_gates_are_brightness_invariant(self):
        """两道新判据不看整体明暗：调亮 + 加对比度后仍然高度吻合。"""
        from PIL import ImageEnhance
        base = self._scene()
        brighter = ImageEnhance.Contrast(
            ImageEnhance.Brightness(base).enhance(1.18)).enhance(1.10)
        photos, fps, _edges = self._pair(base, brighter)
        self.assertGreaterEqual(
            photos.structure_ncc(fps[0].hp[0], fps[1].hp[0]), 0.99)
        self.assertLessEqual(
            photos.layout_distance(fps[0].grid[0], fps[1].grid[0]), 10.0)

    def test_similarity_below_threshold_is_not_auto_selected(self):
        photos = self._photos()
        self.assertGreaterEqual(photos.pair_similarity(2, 0.99), 95)
        self.assertLess(photos.pair_similarity(30, 0.70), 95)
        self.assertLess(photos.pair_similarity(42, 0.60), 85)

    def test_cache_roundtrip_keeps_structure(self):
        photos = self._photos()
        fp = photos.fingerprint(self._write("base.png", self._scene()))
        back = photos.PhotoFP.from_cache(fp.to_cache())
        self.assertAlmostEqual(photos.structure_ncc(fp.hp[0], back.hp[0]),
                               1.0, places=3)
        self.assertLess(photos.layout_distance(fp.grid[0], back.grid[0]),
                        0.6)
        # 缩放搜索靠这两项，缓存里必须一起存下来，否则读缓存就退化成旧行为
        import numpy as np
        self.assertTrue(np.array_equal(fp.thumb, back.thumb))
        self.assertEqual(fp.thumb.shape, (photos.THUMB_SIZE, photos.THUMB_SIZE, 3))
        self.assertEqual(fp.exif_time, back.exif_time)
        self.assertEqual(fp.exif_model, back.exif_model)

    def test_brightened_copy_still_matches(self):
        """整体调亮 1.25 倍（亮部还没被截断）：不该被颜色分布或布局判据挡住。"""
        from PIL import ImageEnhance
        base = self._scene()
        bright = ImageEnhance.Brightness(base).enhance(1.25)
        photos, fps, edges = self._pair(base, bright)
        self.assertEqual([(e[0], e[1]) for e in edges], [(0, 1)])
        self.assertLessEqual(edges[0][4], 0.26)

    def test_cropped_copy_matches_at_any_amount(self):
        """裁掉一圈边框的副本：整幅对不上，靠缩放对齐命中；裁多少都要能认。"""
        from PIL import Image
        base = self._scene()
        w, h = base.size
        for pct in (0.04, 0.10, 0.14, 0.20):
            dx, dy = int(round(w * pct)), int(round(h * pct))
            cropped = base.crop((dx, dy, w - dx, h - dy)).resize(
                (w, h), Image.Resampling.LANCZOS)
            photos, fps, edges = self._pair(base, cropped)
            self.assertEqual([(e[0], e[1]) for e in edges], [(0, 1)], pct)
            same = photos.structure_ncc(fps[0].hp[0], fps[1].hp[0])
            self.assertLess(same, 0.95, pct)      # 只比整幅会漏掉
            # 命中的相关值就是缩放对齐后的值，必须过跨尺度阈值
            self.assertGreaterEqual(edges[0][5], photos.CROP_NCC_TOL, pct)

    def test_same_shot_exif_rescues_heavy_edit(self):
        """同机型 + 拍摄时间一致：几何对不上也应当归到一组。"""
        import numpy as np
        from PIL import Image
        photos = self._photos()
        base = self._scene()
        stamps = {272: "TestCam X1", 36867: "2024:05:06 07:08:09"}

        def save(name, img):
            path = self.root / name
            exif = Image.Exif()
            for key, value in stamps.items():
                exif[key] = value
            img.save(path, exif=exif)
            return path

        # 轻微重新取景：结构相关掉到 0.89~0.95 这一带，同尺度过不了
        edited = self._scene(dx=4)
        fps = [photos.fingerprint(save("exif_a.jpg", base)),
               photos.fingerprint(save("exif_b.jpg", edited))]
        self.assertEqual(fps[0].exif_model, "TestCam X1")
        self.assertEqual(fps[0].exif_time, fps[1].exif_time)
        corr = photos.structure_ncc(fps[0].hp[0], fps[1].hp[0])
        self.assertLess(corr, 0.95)
        self.assertGreaterEqual(corr, photos.EXIF_NCC_TOL)
        # 同尺度与缩放两道阈值都抬高到 1.0：只剩 EXIF 这条路能命中
        strict = photos.match_photos(fps, ncc_tol=1.0, layout_tol=1e9,
                                     crop_ncc_tol=1.0)
        self.assertEqual([(e[0], e[1]) for e in strict], [(0, 1)])
        # 反向保险：拍摄信息一样但根本不是同一张（换了人），EXIF 不能放行
        other = photos.fingerprint(save("exif_c.jpg",
                                        self._scene(dx=16, head=1.15,
                                                    arms=1.6)))
        self.assertEqual(photos.match_photos([fps[0], other]), [])

    def test_flat_scenes_are_not_paired(self):
        """大片纯色 + 少量色块：粗筛过得去，结构闸门必须挡住。"""
        from PIL import Image, ImageDraw

        def flat(blocks):
            img = Image.new("RGB", (320, 240), (208, 208, 204))
            draw = ImageDraw.Draw(img)
            for x, y, bw, bh, col in blocks:
                draw.rectangle([x, y, x + bw, y + bh], fill=col)
            return img

        one = flat([(40, 40, 90, 70, (180, 172, 160)),
                    (200, 120, 80, 90, (96, 104, 112))])
        two = flat([(30, 120, 120, 60, (176, 168, 156)),
                    (190, 30, 90, 80, (100, 108, 116))])
        photos, fps, edges = self._pair(one, two, dist=64, fine=255,
                                       hist_tol=1.0)
        self.assertEqual(edges, [])
        ungated = photos.match_photos(fps, dist=64, fine=255, hist_tol=1.0,
                                      ncc_tol=0.0, layout_tol=1e9)
        self.assertEqual([(e[0], e[1]) for e in ungated], [(0, 1)])
        self.assertLess(ungated[0][5], 0.95)

    def test_cache_from_old_version_is_rejected(self):
        """旧格式缓存（绝对直方图、无层级）必须抛错重算，不能静默沿用。"""
        photos = self._photos()
        for version in (1, 2):
            data = {"v": version,
                    "dh": [[0] * photos.ORIENTS] * photos.LEVELS,
                    "ph": [[[0] * 4] * photos.ORIENTS] * photos.LEVELS,
                    "hist": [0.0] * photos.HIST_SIZE, "w": 10, "h": 10}
            with self.assertRaises(ValueError):
                photos.PhotoFP.from_cache(data)


    def test_zone_grid_scales_with_image_size(self):
        """每块约 96 像素：小图保底 8 格，大图封顶 16 格。"""
        from PIL import Image

        photos = self._photos()
        sizes = [(640, 480, (6, 8)), (1200, 900, (9, 12)),
                 (1600, 1200, (12, 16)), (4000, 3000, (12, 16)),
                 (200, 150, (6, 8))]
        for width, height, want in sizes:
            image = Image.new("RGB", (width, height), (10, 20, 30))
            self.assertEqual(photos.zone_shape(image), want, (width, height))

    def test_zone_grid_keeps_blocks_square(self):
        """极端宽高比也不许把方块拉成扁条，长边仍受 16 格约束。"""
        from PIL import Image

        photos = self._photos()
        for width, height in ((1200, 900), (4000, 1000), (800, 2400),
                              (640, 480), (3000, 3000)):
            rows, cols = photos.zone_shape(Image.new("RGB", (width, height)))
            self.assertLessEqual(rows, 16)
            self.assertLessEqual(cols, 16)
            block_w, block_h = width / cols, height / rows
            self.assertLessEqual(abs(block_w / block_h - 1.0), 0.08,
                                 (width, height, rows, cols))


class ReferenceDirs(unittest.TestCase):
    """参考目录在扫描根之外也要生效：参与比对，且永远不进清理表。"""

    def setUp(self):
        import tempfile
        here = Path(__file__).resolve().parent.parent
        self.base = Path(tempfile.mkdtemp(prefix=".tmp_ref_", dir=here))
        self.root = self.base / "lib"
        self.ref = self.base / "archive"
        self.root.mkdir()
        self.ref.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.base, ignore_errors=True)

    def _opts(self, **kw):
        return models.ScanOptions(root=self.root,
                                  reference_dirs=(self.ref,), **kw)

    def test_external_reference_is_collected_and_protected(self):
        (self.root / "a.txt").write_bytes(b"x" * 64)
        (self.root / "b.txt").write_bytes(b"y" * 64)
        (self.ref / "a.txt").write_bytes(b"x" * 64)
        (self.ref / "nested").mkdir()
        (self.ref / "nested" / "c.txt").write_bytes(b"z" * 64)

        found = files.collect_files(self._opts())[0]
        outside = [f for f in found if self.ref in f.path.parents]
        inside = [f for f in found if self.root in f.path.parents]
        self.assertEqual(len(outside), 2)
        self.assertEqual(len(inside), 2)
        self.assertTrue(all(f.protected for f in outside))
        self.assertFalse(any(f.protected for f in inside))

    def test_reference_inside_root_not_double_counted(self):
        (self.root / "a.txt").write_bytes(b"x" * 32)
        (self.root / "ref").mkdir()
        (self.root / "ref" / "a.txt").write_bytes(b"x" * 32)

        opts = models.ScanOptions(root=self.root,
                                  reference_dirs=(self.root / "ref",))
        found = files.collect_files(opts)[0]
        self.assertEqual(len(found), 2)
        prot = [f for f in found if f.protected]
        self.assertEqual([f.path for f in prot], [self.root / "ref" / "a.txt"])

    def test_missing_reference_is_ignored(self):
        (self.root / "a.txt").write_bytes(b"x" * 32)
        opts = models.ScanOptions(root=self.root,
                                  reference_dirs=(self.base / "nope",))
        found = files.collect_files(opts)[0]
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0].protected)

    def test_scan_keeps_reference_copy(self):
        (self.root / "song.txt").write_bytes(b"same content here")
        (self.ref / "song.txt").write_bytes(b"same content here")
        opts = self._opts(quick_scan=True, photos=False, videos=False,
                          audio=False, names=False, cache=False)
        result = core.scan_folder(opts)
        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertTrue(group.items[group.keep].protected)
        self.assertEqual(group.default_selection(),
                         {str(self.root / "song.txt")})

    def test_materialize_keeps_redundant_copy(self):
        # 参考副本被摘掉后只剩主库那一个：它仍是可清理的冗余副本，不能当低置信丢掉
        import dedup_finder as finder
        (self.root / "song.txt").write_bytes(b"same content here")
        (self.ref / "song.txt").write_bytes(b"same content here")
        opts = self._opts(quick_scan=True, photos=False, videos=False,
                          audio=False, names=False, cache=False)
        result = core.scan_folder(opts)
        out = self.base / "out"
        rows, skipped = finder.materialize(result.groups, out,
                                           core.KIND_EXACT, "copy")
        self.assertEqual(skipped["protected"], 1)
        self.assertEqual(skipped["low_confidence"], 0)
        self.assertEqual([r["文件名"] for r in rows], ["song.txt"])
        origin = Path(rows[0]["原始路径"])
        self.assertEqual(origin.name, "song.txt")
        self.assertIn(self.root, origin.parents)
        self.assertTrue((out / core.KIND_EXACT / "团01" / "01_song.txt").exists())


if __name__ == "__main__":
    unittest.main()
