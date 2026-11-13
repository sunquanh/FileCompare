"""R23 tests: save_manifest mtime cleanup, CLI, ignore list, progress
filename, scan history, threshold overrides."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

import dedup_core
import dedup_models
import dedup_tools
from dedup_core import scan_folder
from dedup_models import ScanOptions


def _make_image(path, seed=0):
    im = Image.new("RGB", (128, 96), (200, 30, 30))
    dr = ImageDraw.Draw(im)
    dr.rectangle([20, 20, 60, 60], fill=(30, 120, 200))
    dr.ellipse([70, 30, 110, 70], fill=(240, 220, 40))
    dr.rectangle([5 + seed * 3, 5, 10 + seed * 3, 10],
                 fill=(210 + seed * 5, 40, 40))
    im.save(path)


class TestSaveManifestMtime(unittest.TestCase):
    """save_manifest 修 bug：清理按 mtime 排序，不再按文件名字典序。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig_dir = dedup_tools.MANIFEST_DIR
        dedup_tools.MANIFEST_DIR = self.tmp / "_操作记录"
        dedup_tools.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        dedup_tools.MANIFEST_DIR = self.orig_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_file_not_deleted_when_name_sorts_before_old(self):
        # 模拟混时区场景：先写 12 个“北京时区”名字（21:xx）的文件，
        # 再写一个字典序更小（13:xx）的新文件。按旧逻辑新文件会被当“最旧”删掉；
        # 按 mtime 排序，新文件是最新的，不该被删。
        kind = "cleanup"
        for i in range(12):
            p = dedup_tools.MANIFEST_DIR / f"{kind}-20261113-21{i:02d}00-000000.json"
            p.write_text(json.dumps({"kind": kind, "items": []}),
                         encoding="utf-8")
            # 旧文件 mtime 设成过去
            os.utime(p, (1000000000 + i, 1000000000 + i))
        # 新文件：名字字典序最小（13:xx），但 mtime 最新
        new_path = dedup_tools.save_manifest(kind, self.tmp, [])
        self.assertTrue(new_path.exists(),
                        "新写的文件刚写完就被清理删掉了（mtime 排序未生效）")
        # 旧文件里最旧的（按 mtime）应该被删掉一个，保持 12 个
        remaining = sorted(dedup_tools.MANIFEST_DIR.glob(f"{kind}-*.json"))
        self.assertEqual(len(remaining), 12)
        self.assertIn(new_path, remaining)


class TestIgnoreList(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.txt").write_text("same content")
        (self.tmp / "b.txt").write_text("same content")
        (self.tmp / "c.txt").write_text("same content")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _scan(self, ignore_paths=()):
        return scan_folder(ScanOptions(
            root=self.tmp, photos=False, videos=False, audio=False,
            names=False, ignore_paths=list(ignore_paths)))

    def test_ignore_one_file(self):
        r = self._scan([str(self.tmp / "b.txt")])
        self.assertEqual(len(r.groups), 1)
        names = {i.path.name for g in r.groups for i in g.items}
        self.assertEqual(names, {"a.txt", "c.txt"})

    def test_ignore_two_files_no_group(self):
        r = self._scan([str(self.tmp / "a.txt"), str(self.tmp / "b.txt")])
        self.assertEqual(len(r.groups), 0)

    def test_ignore_whole_dir(self):
        sub = self.tmp / "sub"
        sub.mkdir()
        (sub / "d.txt").write_text("same content")
        (sub / "e.txt").write_text("same content")
        # 忽略整个 sub 目录：只剩 a/b/c 一组
        r = self._scan([str(sub)])
        self.assertEqual(len(r.groups), 1)
        names = {i.path.name for g in r.groups for i in g.items}
        self.assertEqual(names, {"a.txt", "b.txt", "c.txt"})


class TestProgressFilename(unittest.TestCase):
    def test_photo_progress_includes_filename(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for i, name in enumerate(["a.png", "b.png", "c.png"]):
                _make_image(tmp / name, seed=i)
            seen = []

            def prog(stage, done, total, detail):
                if stage == "photo" and "·" in detail:
                    seen.append(detail)

            r = scan_folder(ScanOptions(root=tmp, photos=True, videos=False,
                                        audio=False, names=False),
                            on_progress=prog)
            self.assertEqual(len(r.groups), 1)
            self.assertTrue(any(".png" in s for s in seen),
                            f"照片进度里没有出现文件名：{seen}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestScanHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig_dir = dedup_tools.SCAN_HISTORY_DIR
        dedup_tools.SCAN_HISTORY_DIR = self.tmp / "_扫描历史"

    def tearDown(self):
        dedup_tools.SCAN_HISTORY_DIR = self.orig_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _scan_dir(self, files):
        d = self.tmp / "data"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for name, content in files.items():
            (d / name).write_text(content)
        return scan_folder(ScanOptions(root=d, photos=False, videos=False,
                                       audio=False, names=False))

    def test_save_list_compare(self):
        r1 = self._scan_dir({"a.txt": "same", "b.txt": "same"})
        p1 = dedup_tools.save_scan_history(r1.root, r1.groups,
                                           r1.file_count, r1.elapsed)
        self.assertTrue(p1.exists())
        r2 = self._scan_dir({"c.txt": "same", "d.txt": "same"})
        p2 = dedup_tools.save_scan_history(r2.root, r2.groups,
                                           r2.file_count, r2.elapsed)
        hist = dedup_tools.list_scan_history()
        self.assertEqual(len(hist), 2)
        added, removed, kept = dedup_tools.compare_scans(
            dedup_tools.load_scan_history(p1),
            dedup_tools.load_scan_history(p2))
        self.assertEqual(len(added), 1)
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(kept), 0)

    def test_history_keeps_only_latest_20(self):
        r = self._scan_dir({"a.txt": "same", "b.txt": "same"})
        for _ in range(22):
            dedup_tools.save_scan_history(r.root, r.groups,
                                          r.file_count, r.elapsed)
            time.sleep(0.01)
        hist = dedup_tools.list_scan_history()
        self.assertEqual(len(hist), dedup_tools.SCAN_HISTORY_KEEP)


class TestThresholdOverrides(unittest.TestCase):
    def test_scan_options_accepts_ncc_layout(self):
        opts = ScanOptions(root=Path("."), img_ncc=0.80, img_layout=18.0)
        self.assertEqual(opts.img_ncc, 0.80)
        self.assertEqual(opts.img_layout, 18.0)

    def test_presets_unchanged(self):
        self.assertEqual(dedup_models.PRESETS["standard"]["img_ncc"], 0.95)
        self.assertEqual(dedup_models.PRESETS["relaxed"]["img_ncc"], 0.72)


class TestCLI(unittest.TestCase):
    def test_cli_scans_and_exports(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "a.txt").write_text("same content")
            (tmp / "b.txt").write_text("same content")
            out_json = tmp / "out.json"
            out_csv = tmp / "out.csv"
            repo = Path(__file__).resolve().parent.parent
            proc = subprocess.run(
                [sys.executable, str(repo / "dedup_cli.py"), str(tmp),
                 "--json", str(out_json), "--csv", str(out_csv)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("重复组：1 组", proc.stdout)
            self.assertTrue(out_json.exists())
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["total_groups"], 1)
            self.assertTrue(out_csv.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cli_bad_dir_returns_1(self):
        repo = Path(__file__).resolve().parent.parent
        proc = subprocess.run(
            [sys.executable, str(repo / "dedup_cli.py"), "/no/such/dir/xyz"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
