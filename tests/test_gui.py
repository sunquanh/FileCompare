"""Tk 交互回归：对比窗口左右切换、差异热图、取消永久删除。

这些用例会真的建窗口（不显示），直接驱动界面的方法，而不是绕过界面去测引擎。
没有图形环境时整体跳过，不让 CI 因为无桌面而红。
"""
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import tkinter as tk  # noqa: E402

from PIL import Image  # noqa: E402

import dedup_core as core  # noqa: E402
import dedup_gui as gui  # noqa: E402
import dedup_models as models  # noqa: E402
import ui_kit  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent

class FakeApp(tk.Tk):
    """App 的最小替身：只提供界面层真正会碰到的属性。"""

    def __init__(self):
        super().__init__()
        # 放到屏幕外而不是 withdraw：对比窗口的 _draw 依赖 winfo_ismapped，
        # 不 map 就会每 80ms 重试一次，销毁后留下一堆无效 after 回调。
        self.geometry("320x200+4000+4000")
        self.theme_name = "dark"
        ui_kit.apply_palette("dark")
        self.theme = ui_kit.Theme(self)
        self.settings = {}
        self.scanning = False
        self.result = None
        self.queue = None
        self.cancel_event = None
        self.logs = []
        self.toasts = []
        self.status = []
        self.posts = []
        self.opened = []

    def log(self, message, level="info"):
        self.logs.append((level, message))

    def toast(self, message, level="info", duration=None):
        self.toasts.append((level, message))

    def set_status(self, text, color=None, detail=None):
        self.status.append(text)

    def post(self, message):
        self.posts.append(message)

    def open_file(self, path):
        self.opened.append(str(path))

    def reveal_file(self, path):
        self.opened.append(str(path))

    def open_folder(self, path):
        self.opened.append(str(path))

    def update_counts(self, *a, **kw):
        pass

    def set_result_count(self, *a, **kw):
        pass

    def update_progress(self, *a, **kw):
        pass

    def show_page(self, *a, **kw):
        pass


class TkCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe = tk.Tk()
        except tk.TclError as exc:      # pragma: no cover - 无桌面环境
            raise unittest.SkipTest(f"没有可用的图形环境：{exc}")
        probe.destroy()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix=".tmp_gui_", dir=HERE))
        # setUp 里失败时 tearDown 不会执行，用 cleanup 保证临时目录一定被删
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._titlebar = gui.enable_dark_titlebar
        gui.enable_dark_titlebar = lambda *a, **kw: None
        self.app = FakeApp()

    def tearDown(self):
        gui.enable_dark_titlebar = self._titlebar
        try:
            self.app.destroy()
        except Exception:
            pass

    # ---------------------------------------------------------- 工具
    def png(self, name, color, size=(48, 40), patch=None):
        """生成一张纯色图；patch=(x, y, w, h, color) 额外涂一块。"""
        image = Image.new("RGB", size, color)
        if patch:
            x, y, w, h, fill = patch
            for px in range(x, min(x + w, size[0])):
                for py in range(y, min(y + h, size[1])):
                    image.putpixel((px, py), fill)
        path = self.tmp / name
        image.save(path)
        return path

    def group(self, paths):
        items = []
        for path in paths:
            stat = path.stat()
            items.append(models.FileItem(path=path, size=stat.st_size,
                                         mtime=stat.st_mtime))
        group = models.DupGroup(models.KIND_EXACT, items, 100, "字节完全一致")
        group.keep = 0
        return group

    def pump(self, dialog, seconds=6.0):
        """跑真正的事件循环直到两张图都解码完。

        载图在后台线程里做，结果由主线程轮询取回。这里分片跑 mainloop：
        单次 mainloop 可能因为队列里遗留的 quit 立刻返回，分片重进才能保证
        回调和轮询都真的被处理。
        """
        deadline = time.time() + seconds
        while time.time() < deadline:
            if dialog._raw[0] is not None and dialog._raw[1] is not None:
                return True
            self.app.after(40, self.app.quit)
            self.app.mainloop()
        return dialog._raw[0] is not None and dialog._raw[1] is not None


class CompareZoneMarks(TkCase):
    """「差异区域」按 8×8 网格标出不一致的块，并给出计数。"""

    def setUp(self):
        super().setUp()
        self.root = Path(tempfile.mkdtemp(dir=ROOT, prefix=".tmp_zone_ui_"))
        self.left = self.root / "left.png"
        self.right = self.root / "right.png"
        self._split(self.left, flip=False)
        self._split(self.right, flip=True)
        self.dialog = gui.CompareDialog(self.app, self.group([self.left,
                                                              self.right]))
        try:
            self.assertTrue(self.pump(self.dialog), "对比窗口载入超时")
        except Exception:
            self.dialog.destroy()
            raise

    @staticmethod
    def _split(path, flip):
        """左红右蓝 / 左蓝右红：整体配色一样，颜色的位置不同。"""
        warm, cool = (200, 30, 30), (30, 30, 200)
        first, second = (cool, warm) if flip else (warm, cool)
        img = Image.new("RGB", (64, 48), first)
        img.paste(Image.new("RGB", (32, 48), second), (32, 0))
        img.save(path)

    def tearDown(self):
        try:
            self.dialog.destroy()
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)
        super().tearDown()

    def _paint(self):
        """离屏窗口没有完成布局，直接驱动右画布重绘后数框。"""
        self.dialog._draw_side(1, 1.0)
        canvas = self.dialog.canvases[1]
        return [item for item in canvas.find_all()
                if canvas.type(item) == "rectangle"]

    def test_marks_every_differing_block(self):
        self.dialog._zones.set(True)
        self.assertEqual(len(self._paint()), 48)
        text = self.dialog.zone_hint.cget("text")
        self.assertIn("颜色差 48", text)

    def test_identical_pair_reports_no_difference(self):
        same = Image.open(self.left).convert("RGB")
        self.dialog._raw = [same, same.copy()]
        self.dialog._zone_stats = None
        self.dialog._zones.set(True)
        self.assertEqual(self._paint(), [])
        self.assertIn("没有明显差异", self.dialog.zone_hint.cget("text"))

    def test_turning_off_clears_marks_and_hint(self):
        self.dialog._zones.set(True)
        self._paint()
        self.dialog._zones.set(False)
        self.dialog._toggle_zones()
        self.assertEqual(self._paint(), [])
        self.assertEqual(self.dialog.zone_hint.cget("text"), "")


    def test_heatmap_and_zones_are_exclusive(self):
        self.dialog._zones.set(True)
        self.dialog._toggle_zones()
        self.assertFalse(self.dialog._diff.get(), "打开方框应自动收起热图")
        self.dialog._diff.set(True)
        self.dialog._toggle_heat()
        self.assertFalse(self.dialog._zones.get(), "打开热图应自动收起方框")
        self.assertEqual(self.dialog.zone_hint.cget("text"), "")

    def test_thresholds_follow_scan_options(self):
        strict = models.PRESETS["strict"]
        opts = models.ScanOptions(root=self.root, img_layout=strict["img_layout"],
                                  img_ncc=strict["img_ncc"])
        dialog = gui.CompareDialog(self.app, self.group([self.left, self.right]),
                                   options=opts)
        try:
            self.assertTrue(self.pump(dialog), "对比窗口载入超时")
            self.assertEqual(dialog._zone_limits(),
                             (strict["img_layout"], strict["img_ncc"]))
            dialog._zones.set(True)
            # 阈值高到没有块能超线：如果还在画框，说明画框没读 _zone_limits
            dialog._zone_limits = lambda: (1e9, -1.0)
            dialog._zone_stats = None
            dialog._draw_side(1, 1.0)
            canvas = dialog.canvases[1]
            rects = [i for i in canvas.find_all()
                     if canvas.type(i) == "rectangle"]
            self.assertEqual(rects, [], "阈值放宽后不应再标出任何块")
            self.assertIn("没有明显差异", dialog.zone_hint.cget("text"))
        finally:
            dialog.destroy()


    def test_grid_follows_image_size(self):
        """大图切得更细：每块大约 96 像素，而不是固定 8×8。"""
        big_left, big_right = self.root / "big_left.png", self.root / "big_right.png"
        for path, shift in ((big_left, 0), (big_right, 28)):
            img = Image.new("RGB", (1200, 900), (30, 40, 60))
            for col, fill in ((0, (200, 60, 60)), (600, (60, 60, 200))):
                box = [col + shift, 0, col + shift + 600, 900]
                img.paste(Image.new("RGB", (600, 900), fill), (box[0], 0))
            img.save(path)
        dialog = gui.CompareDialog(self.app, self.group([big_left, big_right]))
        try:
            self.assertTrue(self.pump(dialog), "对比窗口载入超时")
            dialog._zones.set(True)
            dialog._draw_side(1, 1.0)
            left = dialog._raw[0]
            self.assertEqual(dialog._zone_maps()[0].shape, (9, 12))
            self.assertAlmostEqual(left.width / 12.0, 100.0, delta=10)
            self.assertAlmostEqual(left.height / 9.0, 100.0, delta=10)
        finally:
            dialog.destroy()


    def test_hover_shows_block_numbers(self):
        """悬停在框上给出这一块的颜色差与结构相关，移开就收起。"""
        self.dialog._zones.set(True)
        self._paint()
        canvas = self.dialog.canvases[1]
        item = [i for i in canvas.find_all()
                if canvas.type(i) == "rectangle"][0]
        left, top, right, bottom = canvas.coords(item)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        hit = self.dialog._zone_at(cx, cy)
        self.assertIsNotNone(hit)
        row, col, color, corr = hit
        maps = self.dialog._zone_maps()
        self.assertAlmostEqual(color, float(maps[0][row, col]), places=3)
        self.assertAlmostEqual(corr, float(maps[1][row, col]), places=3)
        self.assertIsNone(self.dialog._zone_at(-999, -999))

        event = mock.Mock(x=cx, y=cy, x_root=200, y_root=200)
        self.dialog._hover_zone(event)
        self.assertEqual(self.dialog._tip.state(), "normal")
        text = self.dialog._tip_label.cget("text")
        self.assertIn("颜色差", text)
        self.assertIn("细节相关", text)
        self.assertIn(f"{color:.1f}", text)

        self.dialog._zone_data = []
        self.dialog._hide_tip()
        self.assertEqual(self.dialog._tip.state(), "withdrawn")


    def test_tip_flips_at_screen_edges(self):
        """贴边时气泡翻到光标另一侧，并且始终完整留在屏幕内。"""
        self.dialog._zones.set(True)
        self._paint()
        canvas = self.dialog.canvases[1]
        item = [i for i in canvas.find_all()
                if canvas.type(i) == "rectangle"][0]
        x0, y0, x1, y1 = canvas.coords(item)
        hit = self.dialog._zone_at((x0 + x1) / 2, (y0 + y1) / 2)
        self.assertIsNotNone(hit)
        screen_w = self.app.winfo_screenwidth()
        screen_h = self.app.winfo_screenheight()

        self.dialog._show_tip(200, 200, hit)
        size, px, py = self.dialog._tip.geometry().split("+")
        self.assertEqual((int(px), int(py)), (200 + 14, 200 + 18))

        self.dialog._show_tip(screen_w - 3, screen_h - 3, hit)
        size, px, py = self.dialog._tip.geometry().split("+")
        tip_w, tip_h = (int(v) for v in size.split("x"))
        px, py = int(px), int(py)
        self.assertLessEqual(px + tip_w, screen_w)
        self.assertLessEqual(py + tip_h, screen_h)
        self.assertLessEqual(px, screen_w - 3 - 14)   # 翻到了左边
        self.assertLessEqual(py, screen_h - 3 - 18)   # 也翻到了上面


    def test_tip_uses_monitor_bounds(self):
        """多屏：按光标所在显示器的工作区翻转，而不是主屏尺寸。"""
        self.dialog._zones.set(True)
        self._paint()
        canvas = self.dialog.canvases[1]
        item = [i for i in canvas.find_all()
                if canvas.type(i) == "rectangle"][0]
        x0, y0, x1, y1 = canvas.coords(item)
        hit = self.dialog._zone_at((x0 + x1) / 2, (y0 + y1) / 2)
        area = (-1920, 0, 0, 1080)      # 副屏在主屏左侧，右边界正好是 0
        self.dialog._tip_bounds = lambda x, y: area
        self.dialog._show_tip(-30, 40, hit)
        size, px, py = self.dialog._tip.geometry().split("+")
        tip_w, tip_h = (int(v) for v in size.split("x"))
        px, py = int(px), int(py)
        self.assertGreaterEqual(px, area[0])
        self.assertLessEqual(px + tip_w, area[2])
        self.assertGreaterEqual(py, area[1])
        self.assertLessEqual(py + tip_h, area[3])
        self.assertLessEqual(px, -30 - 14)          # 翻到了光标左侧
        self.assertLessEqual(py, area[3])           # 仍在这块屏内


    def test_export_writes_comparison_with_boxes(self):
        """导出的是左右并排的原图，右图带差异框。"""
        self.dialog._zones.set(True)
        out = self.root / "shot.png"
        self.assertEqual(str(self.dialog.export_snapshot(out)), str(out))
        shot = Image.open(out).convert("RGB")
        self.assertEqual(shot.size, (64 + 8 + 64, 48))
        self.assertEqual(shot.getpixel((10, 10)),
                         self.dialog._raw[0].getpixel((10, 10)))

        right_x = 64 + 8
        changed = 0
        for x in range(0, 64, 2):
            for y in range(0, 48, 2):
                if shot.getpixel((right_x + x, y)) != \
                        self.dialog._raw[1].getpixel((x, y)):
                    changed += 1
        self.assertGreater(changed, 0, "导出的右图应带差异框")

    def test_export_cancelled_writes_nothing(self):
        with mock.patch.object(gui.filedialog, "asksaveasfilename",
                               return_value=""):
            self.assertIsNone(self.dialog.export_snapshot())


    def test_only_diff_dims_untouched_blocks(self):
        """只看差异：框外的块被压暗，框内保持原样；关掉差异区域就失能。"""
        base = Image.new("RGB", (96, 96), (150, 150, 150))
        for x0 in range(0, 96, 12):
            for y0 in range(0, 96, 12):
                if (x0 // 12 + y0 // 12) % 2 == 0:
                    base.paste((90, 120, 150), (x0, y0, x0 + 12, y0 + 12))
        left = self.root / "only_left.png"
        right = self.root / "only_right.png"
        base.save(left)
        patched = base.copy()
        patched.paste((250, 240, 60), (40, 40, 60, 60))
        patched.save(right)
        dialog = gui.CompareDialog(self.app, self.group([left, right]))
        try:
            self.assertTrue(self.pump(dialog), "对比窗口载入超时")
            self.assertEqual(dialog.only_check.cget("state"), "disabled")

            dialog._zones.set(True)
            dialog._toggle_zones()
            self.assertEqual(dialog.only_check.cget("state"), "normal")

            maps = dialog._zone_maps()
            color, corr = maps[0], maps[1]
            rows, cols = color.shape
            color_tol, ncc_tol = dialog._zone_limits()
            hot, cool = [], []
            for row in range(rows):
                for col in range(cols):
                    item = (row, col)
                    if color[row, col] > color_tol or corr[row, col] < ncc_tol:
                        hot.append(item)
                    else:
                        cool.append(item)
            self.assertTrue(hot and cool, "构造的图应同时有框内和框外")

            aligned = dialog._aligned_right()[1]
            dim = dialog._dim_outside_zones(aligned)
            ch, cw = aligned.height / rows, aligned.width / cols

            def center(cell):
                return (int((cell[1] + 0.5) * cw), int((cell[0] + 0.5) * ch))

            self.assertEqual(dim.getpixel(center(hot[0])),
                             aligned.getpixel(center(hot[0])))
            px = center(cool[0])
            self.assertLess(sum(dim.getpixel(px)), sum(aligned.getpixel(px)))

            dialog._zones.set(False)
            dialog._toggle_zones()
            self.assertEqual(dialog.only_check.cget("state"), "disabled")
            self.assertFalse(dialog._only.get())
        finally:
            dialog.destroy()


    def test_dim_strength_presets(self):
        """强度预设：三档各按自己的比例压暗，框内始终不受影响。"""
        base = Image.new("RGB", (96, 96), (160, 160, 160))
        for x0 in range(0, 96, 12):
            for y0 in range(0, 96, 12):
                if (x0 // 12 + y0 // 12) % 2 == 0:
                    base.paste((80, 120, 160), (x0, y0, x0 + 12, y0 + 12))
        left = self.root / "dim_left.png"
        right = self.root / "dim_right.png"
        base.save(left)
        patched = base.copy()
        patched.paste((240, 220, 80), (48, 48, 60, 60))
        patched.save(right)
        dialog = gui.CompareDialog(self.app, self.group([left, right]))
        try:
            self.assertTrue(self.pump(dialog), "对比窗口载入超时")
            self.assertEqual(dialog.dim_seg.winfo_manager(), "")

            dialog._zones.set(True)
            dialog._toggle_zones()
            dialog._only.set(True)
            dialog._toggle_only()
            self.assertEqual(dialog.dim_seg.winfo_manager(), "pack")

            color, corr = dialog._zone_maps()[0], dialog._zone_maps()[1]
            rows, cols = color.shape
            color_tol, ncc_tol = dialog._zone_limits()
            hot = cool = None
            for row in range(rows):
                for col in range(cols):
                    if color[row, col] > color_tol or corr[row, col] < ncc_tol:
                        hot = hot or (row, col)
                    else:
                        cool = cool or (row, col)
            self.assertTrue(hot and cool)

            aligned = dialog._aligned_right()[1]
            ch, cw = aligned.height / rows, aligned.width / cols

            def center(cell):
                return (int((cell[1] + 0.5) * cw), int((cell[0] + 0.5) * ch))

            for factor in (0.45, 0.62, 0.80):
                dialog.dim_seg.set_value(factor)
                dim = dialog._dim_outside_zones(aligned)
                px = center(cool)
                src = aligned.getpixel(px)
                self.assertEqual(dim.getpixel(px),
                                 tuple(int(v * factor) for v in src))
                hot_px = center(hot)
                self.assertEqual(dim.getpixel(hot_px),
                                 aligned.getpixel(hot_px))

            dialog._only.set(False)
            dialog._toggle_only()
            self.assertEqual(dialog.dim_seg.winfo_manager(), "")
        finally:
            dialog.destroy()


class CompareNavigation(TkCase):
    """左右键在两个非保留版本之间循环，且过期加载结果不会串图。"""

    def setUp(self):
        super().setUp()
        self.paths = [self.png("a.png", (200, 30, 30)),
                      self.png("b.png", (30, 200, 30)),
                      self.png("c.png", (30, 30, 200))]
        self.group = self.group(self.paths)
        self.dialog = gui.CompareDialog(self.app, self.group)
        try:
            self.assertTrue(self.pump(self.dialog), "对比窗口载入超时")
        except Exception:
            self.dialog.destroy()
            raise

    def tearDown(self):
        try:
            self.dialog.destroy()
        except Exception:
            pass
        super().tearDown()

    def test_starts_on_first_non_kept_item(self):
        self.assertEqual(self.dialog.right_index, 1)
        self.assertEqual(self.dialog.title_right.cget("text"), "b.png")
        self.assertEqual(self.dialog.title_left.cget("text"), "a.png")
        self.assertTrue(self.dialog.bind("<Left>"))
        self.assertTrue(self.dialog.bind("<Right>"))

    def test_cycle_walks_versions_and_wraps(self):
        second = self.dialog._load_gen
        self.dialog._cycle(1)
        self.assertTrue(self.pump(self.dialog))
        self.assertEqual(self.dialog.right_index, 2)
        self.assertEqual(self.dialog.title_right.cget("text"), "c.png")
        self.assertGreater(self.dialog._load_gen, second)

        self.dialog._cycle(1)          # 回到第一个非保留项
        self.assertTrue(self.pump(self.dialog))
        self.assertEqual(self.dialog.right_index, 1)
        self.assertEqual(self.dialog.title_right.cget("text"), "b.png")

        self.dialog._cycle(-1)         # 反向回到最后一个
        self.assertTrue(self.pump(self.dialog))
        self.assertEqual(self.dialog.right_index, 2)
        self.assertEqual(self.dialog.title_right.cget("text"), "c.png")

    def test_kept_item_never_shown_on_right(self):
        for _ in range(6):
            self.dialog._cycle(1)
            self.assertTrue(self.pump(self.dialog))
            self.assertNotEqual(self.dialog.right_index, self.group.keep)

    def test_stale_load_is_discarded(self):
        self.dialog._cycle(1)
        self.assertTrue(self.pump(self.dialog))
        stale = self.dialog._load_gen
        self.dialog._cycle(1)
        self.assertTrue(self.pump(self.dialog))
        self.assertGreater(self.dialog._load_gen, stale)
        self.assertIsNotNone(self.dialog._raw[0])
        # 上一轮的读取结果迟到：token 已过期，不能覆盖当前画面
        self.dialog._apply_loaded(stale, [None, None])
        self.assertIsNotNone(self.dialog._raw[0])
        self.assertIsNotNone(self.dialog._raw[1])


class DiffHeatmap(TkCase):
    """差异热图：只影响右侧，无差异处为黑，尺寸不同也不变形。"""

    def setUp(self):
        super().setUp()
        self.left = self.png("left.png", (180, 40, 40))
        self.right = self.png("right.png", (180, 40, 40),
                              patch=(6, 6, 10, 10, (255, 255, 255)))
        self.group = self.group([self.left, self.right])
        self.dialog = gui.CompareDialog(self.app, self.group)
        try:
            self.assertTrue(self.pump(self.dialog), "对比窗口载入超时")
        except Exception:
            self.dialog.destroy()
            raise

    def tearDown(self):
        try:
            self.dialog.destroy()
        except Exception:
            pass
        super().tearDown()

    def test_heatmap_highlights_changed_area(self):
        self.assertFalse(self.dialog._diff.get())
        before = self.dialog._difference()
        self.assertIsNotNone(before)
        self.assertEqual(before.size, self.dialog._raw[0].size)

        self.dialog._diff.set(True)
        self.dialog._draw_side(1, 1.0)
        self.assertIsNotNone(self.dialog._photos[1])

        heat = self.dialog._difference()
        inside = heat.getpixel((10, 10))
        outside = heat.getpixel((40, 35))
        self.assertGreater(inside[0], 0)          # 差异处发红
        self.assertGreater(inside[1], 0)          # 且偏黄（差异很大）
        self.assertEqual(outside, (0, 0, 0))      # 无差异处为黑

    def test_left_side_ignores_heatmap(self):
        self.dialog._diff.set(True)
        self.dialog._draw_side(0, 1.0)
        self.assertIsNotNone(self.dialog._photos[0])
        left = self.dialog._raw[0]
        self.assertEqual(left.getpixel((10, 10)), (180, 40, 40))

    def test_different_sizes_are_letterboxed(self):
        tall = Image.new("RGB", (24, 80), (10, 200, 10))
        self.dialog._raw = [self.dialog._raw[0], tall]
        heat = self.dialog._difference()
        self.assertIsNotNone(heat)
        self.assertEqual(heat.size, self.dialog._raw[0].size)


class CancelPermanentDelete(TkCase):
    """永久删除的每一道确认都能取消，取消后不写记录、不删文件。"""

    def setUp(self):
        super().setUp()
        self.paths = [self.png("keep.png", (20, 20, 20)),
                      self.png("dup.png", (90, 90, 90))]
        self.kept, self.dup = self.paths
        self.group = self.group(self.paths)
        self.page = gui.ResultsPage(self.app, self.app)
        self.result = models.ScanResult(root=self.tmp, groups=[self.group],
                                        file_count=2)
        self.page.result = self.result
        self.page._size_map = {str(p): p.stat().st_size for p in self.paths}
        self.page.selection = {str(self.dup)}

    def tearDown(self):
        try:
            self.page.destroy()
        except Exception:
            pass
        super().tearDown()

    def test_cancel_at_permanent_delete_prompt(self):
        titles = []

        def answer(parent, theme, title, message, **kw):
            titles.append(title)
            return "extra" if len(titles) == 1 else None

        with mock.patch.object(gui, "confirm", side_effect=answer), \
                mock.patch.object(core, "permanent_delete") as perm, \
                mock.patch.object(core, "recycle_paths") as rec, \
                mock.patch.object(gui.tools, "save_manifest") as manifest:
            self.page.ask_delete()
            self.assertFalse(self.page.busy)
            perm.assert_not_called()
            rec.assert_not_called()
            manifest.assert_not_called()
        self.assertEqual(len(titles), 2)
        self.assertIn("永久删除", titles[1])
        self.assertTrue(self.dup.exists())
        self.assertFalse(any("永久删除：" in m
                             for _lvl, m in self.app.logs))

    def test_cancel_when_recycle_bin_unavailable(self):
        blocked = [(str(self.dup), "FAT32 卷没有回收站")]
        with mock.patch.object(gui, "confirm", return_value=None) as conf, \
                mock.patch.object(core, "recycle_capable",
                                  return_value=([], blocked)), \
                mock.patch.object(core, "recycle_paths") as rec, \
                mock.patch.object(core, "permanent_delete") as perm, \
                mock.patch.object(gui.tools, "save_manifest") as manifest:
            self.page._delete([str(self.dup)], recycle=True)
            self.assertFalse(self.page.busy)
            rec.assert_not_called()
            perm.assert_not_called()
            manifest.assert_not_called()
            self.assertEqual(conf.call_count, 1)
            self.assertIn("无法进入回收站", conf.call_args[0][2])
        self.assertTrue(self.dup.exists())

    def test_kept_file_is_never_deleted(self):
        with mock.patch.object(gui, "confirm", return_value="ok") as conf, \
                mock.patch.object(core, "recycle_paths") as rec:
            self.page._delete([str(self.kept)], recycle=True)
            rec.assert_not_called()
            conf.assert_not_called()
        self.assertTrue(self.kept.exists())
        self.assertIn("warn", [lvl for lvl, _m in self.app.toasts])

class CompareEvents(TkCase):
    """真实鼠标/键盘事件驱动对比窗口：走绑定而非直接调方法。"""

    def setUp(self):
        super().setUp()
        paths = [self.png("a.png", (200, 30, 30)),
                 self.png("b.png", (30, 200, 30)),
                 self.png("c.png", (30, 30, 200))]
        self.group = self.group(paths)
        self.dialog = gui.CompareDialog(self.app, self.group)
        self.addCleanup(self._close_dialog)
        self.assertTrue(self.pump(self.dialog), "对比窗口载入超时")
        self.dialog.lift()
        self.dialog.focus_force()
        self.app.update()

    def _close_dialog(self):
        try:
            self.dialog.destroy()
        except Exception:
            pass

    def test_wheel_zooms_and_clamps(self):
        canvas = self.dialog.canvases[0]
        self.assertEqual(self.dialog.zoom, 0.0)          # 初始为「适应窗口」
        canvas.event_generate("<MouseWheel>", delta=120, x=60, y=60, when="now")
        zoomed = self.dialog.zoom
        self.assertGreater(zoomed, 0.0)

        canvas.event_generate("<MouseWheel>", delta=-120, x=60, y=60, when="now")
        self.assertLess(self.dialog.zoom, zoomed)

        for _ in range(40):
            canvas.event_generate("<MouseWheel>", delta=120, when="now")
        self.assertLessEqual(self.dialog.zoom, 12.0)
        for _ in range(80):
            canvas.event_generate("<MouseWheel>", delta=-120, when="now")
        self.assertGreaterEqual(self.dialog.zoom, 0.05)

    def test_wheel_on_second_pane_redraws_it(self):
        canvas = self.dialog.canvases[1]
        self.dialog.zoom = 0.0
        canvas.event_generate("<MouseWheel>", delta=120, x=40, y=40, when="now")
        self.assertGreater(self.dialog.zoom, 0.0)
        self.assertIsNotNone(self.dialog._photos[1])

    def test_drag_pans_both_panes(self):
        canvas = self.dialog.canvases[0]
        self.dialog._set_zoom(1.0)
        canvas.event_generate("<ButtonPress-1>", x=100, y=100, when="now")
        self.assertIsNotNone(self.dialog._drag)

        canvas.event_generate("<B1-Motion>", x=140, y=120, when="now")
        self.assertEqual([round(v) for v in self.dialog.pan[0]], [40, 20])
        self.assertEqual([round(v) for v in self.dialog.pan[1]], [40, 20])

        canvas.event_generate("<ButtonRelease-1>", x=140, y=120, when="now")
        self.assertIsNone(self.dialog._drag)
        canvas.event_generate("<B1-Motion>", x=200, y=200, when="now")
        self.assertEqual([round(v) for v in self.dialog.pan[0]], [40, 20])

    def test_arrow_keys_cycle_versions(self):
        self.assertEqual(self.dialog.right_index, 1)
        self.dialog.event_generate("<Right>", when="now")
        self.app.update()
        self.assertTrue(self.pump(self.dialog))
        self.assertEqual(self.dialog.right_index, 2)
        self.assertEqual(self.dialog.title_right.cget("text"), "c.png")

        self.dialog.event_generate("<Left>", when="now")
        self.app.update()
        self.assertTrue(self.pump(self.dialog))
        self.assertEqual(self.dialog.right_index, 1)
        self.assertEqual(self.dialog.title_right.cget("text"), "b.png")

    def test_escape_closes_dialog(self):
        self.assertIn(self.dialog, self.app.winfo_children())
        self.dialog.event_generate("<Escape>", when="now")
        self.app.update()
        self.assertNotIn(self.dialog, self.app.winfo_children())

class InlineThread:
    """把清理流程的后台线程改成同步执行，便于断言结局。"""

    def __init__(self, target=None, daemon=None, name=None, args=(),
                 kwargs=None):
        self._target = target
        self._args = args or ()
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class CleanupOutcomes(TkCase):
    """清理流程三种结局：回收站成功、部分失败、必须永久删除。"""

    def setUp(self):
        super().setUp()
        import types
        patcher = mock.patch.object(gui.tools, "MANIFEST_DIR",
                                    self.tmp / "manifests")
        patcher.start()
        self.addCleanup(patcher.stop)
        threads = mock.patch.object(
            gui, "threading", types.SimpleNamespace(Thread=InlineThread))
        threads.start()
        self.addCleanup(threads.stop)
        self.answers = []

    def _page(self, count=2):
        paths = [self.png(f"f{i}.png", (20 + i * 40, 60, 90))
                 for i in range(count)]
        group = self.group(paths)
        page = gui.ResultsPage(self.app, self.app)
        page.result = models.ScanResult(root=self.tmp, groups=[group],
                                        file_count=len(paths))
        page._size_map = {str(p): p.stat().st_size for p in paths}
        return page, paths

    def _run(self, page, targets, recycle, deleted, failed,
             recycled=None, gone=None, capable=None):
        """跑一遍 _delete + on_deleted，返回各替身的调用记录。"""
        # 真实调用方传进来的是字符串路径（来自 selection），
        # 传 Path 会让记录里的 src 变成 WindowsPath，撤销时无法匹配
        targets = [str(p) for p in targets]
        deleted = [str(p) for p in deleted]
        failed = [str(p) for p in failed]
        recycled = [str(p) for p in (recycled or [])]
        gone = [str(p) for p in (gone or [])]
        planned = []
        real_save = gui.tools.save_manifest

        def spy_save(kind, root, records):
            planned.append(list(records))
            return real_save(kind, root, records)

        before = len(self.app.posts)
        with mock.patch.object(gui.tools, "save_manifest",
                               side_effect=spy_save),                 mock.patch.object(
                    core, "recycle_capable",
                    return_value=(capable if capable is not None
                                  else (list(targets), []))),                 mock.patch.object(core, "recycle_paths",
                                  return_value=(list(deleted),
                                                list(failed))) as rec,                 mock.patch.object(core, "permanent_delete",
                                  return_value=(list(deleted),
                                                list(failed))) as perm,                 mock.patch.object(core, "verify_recycled",
                                  return_value=(set(recycled or []),
                                                set(gone or []))) as verify,                 mock.patch.object(gui, "alert") as alert,                 mock.patch.object(gui, "confirm",
                                  side_effect=self.answers) as conf:
            page._delete(list(targets), recycle=recycle)
            payloads = [p[1] for p in self.app.posts[before:]
                        if p[0] == "deleted"]
            self.assertTrue(payloads, "清理流程没有产生结果消息")
            page.on_deleted(*payloads[-1])
        return {"planned": planned, "rec": rec, "perm": perm,
                "verify": verify, "alert": alert, "confirm": conf}

    def test_recycle_success_keeps_restorable_record(self):
        page, paths = self._page(2)
        keep, dup = paths
        info = self._run(page, [dup], True, deleted=[dup], failed=[],
                         recycled=[dup])

        self.assertFalse(page.busy)
        self.assertEqual(info["planned"][0][0]["act"], "recycle")
        self.assertEqual(info["planned"][0][0]["src"], str(dup))
        self.assertEqual(info["verify"].call_count, 1)
        self.assertIn("清理完成：成功 1 个，失败 0 个",
                      [m for _lvl, m in self.app.logs])
        self.assertIn("ok", [lvl for lvl, _m in self.app.logs
                             if "清理完成" in _m])

        path = page._last_cleanup
        self.assertIsNotNone(path)
        items = gui.tools.read_manifest(path)["items"]
        self.assertEqual(items, [{"act": "recycle", "src": str(dup),
                                  "size": dup.stat().st_size}])
        self.assertEqual(gui.tools.last_restorable_cleanup(), path)
        self.assertTrue(page.btn_undo_clean.enabled)
        self.assertEqual(info["alert"].call_count, 0)
        self.assertTrue(any("已清理 1 个文件" in m
                            for _lvl, m in self.app.toasts))

    def test_partial_failure_warns_and_only_records_success(self):
        page, paths = self._page(3)
        keep, dup1, dup2 = paths
        info = self._run(page, [dup1, dup2], True, deleted=[dup1],
                         failed=[dup2], recycled=[dup1])

        status = [(lvl, m) for lvl, m in self.app.logs if "清理完成" in m]
        self.assertEqual(status, [("warn", "清理完成：成功 1 个，失败 1 个")])
        items = gui.tools.read_manifest(page._last_cleanup)["items"]
        self.assertEqual([r["src"] for r in items], [str(dup1)])
        self.assertTrue(dup2.exists())
        self.assertEqual(info["alert"].call_count, 1)
        self.assertIn("1 个文件未能处理", info["alert"].call_args[0][2])
        self.assertTrue(any(len(g.items) == 2 for g in page.result.groups))
        self.assertTrue(page.btn_undo_clean.enabled)

    def test_unrecyclable_needs_permanent_delete(self):
        page, paths = self._page(2)
        keep, dup = paths
        self.answers = ["extra", "ok"]
        info = self._run(page, [dup], True, deleted=[dup], failed=[],
                         capable=([], [(str(dup), "FAT32 卷没有回收站")]))

        titles = [call[0][2] for call in info["confirm"].call_args_list]
        self.assertTrue(any("无法进入回收站" in t for t in titles))
        self.assertTrue(any("永久删除" in t for t in titles))
        self.assertEqual(info["rec"].call_count, 0)
        self.assertEqual(info["perm"].call_count, 1)
        self.assertEqual(info["verify"].call_count, 0)

        self.assertEqual(info["planned"][-1][0]["act"], "delete")
        items = gui.tools.read_manifest(page._last_cleanup)["items"]
        self.assertEqual(items[0]["act"], "delete")
        self.assertEqual(items[0]["src"], str(dup))
        self.assertFalse(page.btn_undo_clean.enabled)
        self.assertIsNone(gui.tools.last_restorable_cleanup())
        self.assertIn("warn", [lvl for lvl, m in self.app.logs
                               if m.startswith("永久删除：")])

    def test_missing_from_recycle_bin_recorded_as_permanent(self):
        page, paths = self._page(2)
        keep, dup = paths
        info = self._run(page, [dup], True, deleted=[dup], failed=[],
                         recycled=[], gone=[dup])

        self.assertIn(("err", "注意：1 个文件没有进入回收站，已被永久删除"),
                      self.app.logs)
        self.assertEqual(info["alert"].call_count, 1)
        self.assertIn("1 个文件未进入回收站", info["alert"].call_args[0][2])
        items = gui.tools.read_manifest(page._last_cleanup)["items"]
        self.assertEqual(items[0]["act"], "delete")
        self.assertFalse(page.btn_undo_clean.enabled)
        self.assertIsNone(gui.tools.last_restorable_cleanup())

@unittest.skipUnless(os.environ.get("DEDUP_TEST_RECYCLE") == "1",
                     "设置 DEDUP_TEST_RECYCLE=1 才会动真实回收站")
class UndoCleanup(TkCase):
    """还原上次清理：真实回收站往返，覆盖成功、部分失败、同名文件三种结局。"""

    def setUp(self):
        super().setUp()
        import types
        patcher = mock.patch.object(gui.tools, "MANIFEST_DIR",
                                    self.tmp / "manifests")
        patcher.start()
        self.addCleanup(patcher.stop)
        threads = mock.patch.object(
            gui, "threading", types.SimpleNamespace(Thread=InlineThread))
        threads.start()
        self.addCleanup(threads.stop)

    # ---------------------------------------------------------- 工具
    def _drain(self, path):
        """把还留在回收站里的测试文件捞回来删掉，不污染用户的回收站。"""
        try:
            if path.exists():
                path.unlink()
            core.restore_from_recycle([str(path)])
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def _recycle(self, *paths):
        """真的移进回收站，并按实际结果返回清理记录。"""
        sizes = {str(p): p.stat().st_size for p in paths}
        for path in paths:
            self.addCleanup(self._drain, path)
        deleted, failed = core.recycle_paths([str(p) for p in paths])
        self.assertEqual(failed, [])
        self.assertEqual(sorted(deleted), sorted(str(p) for p in paths))
        for path in paths:
            self.assertFalse(path.exists())
        return [{"act": "recycle", "src": str(p), "size": sizes[str(p)]}
                for p in paths]

    def _manifest(self, records):
        path = gui.tools.save_manifest("cleanup", self.tmp, records)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        return path

    def _page(self):
        page = gui.ResultsPage(self.app, self.app)
        page.result = models.ScanResult(root=self.tmp, groups=[],
                                        file_count=0)
        return page

    def _undo(self, page):
        before = len(self.app.posts)
        with mock.patch.object(gui, "confirm", return_value="ok") as conf,                 mock.patch.object(gui, "alert") as alert:
            page.undo_clean()
            payloads = [p[1] for p in self.app.posts[before:]
                        if p[0] == "cleanup_restored"]
            self.assertTrue(payloads, "还原流程没有产生结果消息")
            page.on_cleanup_restored(*payloads[-1])
        return {"confirm": conf, "alert": alert, "payload": payloads[-1]}

    # ---------------------------------------------------------- 用例
    def test_restore_success_puts_file_back(self):
        dup = self.png("dup.png", (12, 200, 90))
        content = dup.read_bytes()
        page = self._page()
        page._last_cleanup = self._manifest(self._recycle(dup))
        info = self._undo(page)

        self.assertTrue(dup.exists(), "撤销后文件没有回到原位置")
        self.assertEqual(dup.read_bytes(), content)
        self.assertEqual(info["payload"], ([str(dup)], []))
        self.assertIn(("ok", "还原完成：1 个成功，0 个失败"), self.app.logs)
        self.assertTrue(any("已还原 1 个文件" in m
                            for _lvl, m in self.app.toasts))
        self.assertIsNone(page._last_cleanup)
        self.assertFalse(page.busy)
        self.assertEqual(info["alert"].call_count, 0)
        title, detail = info["confirm"].call_args[0][2:4]
        self.assertEqual(title, "撤销上次清理")
        self.assertIn("从回收站还原到原位置", detail)
        self.assertIn("会跳过而不是覆盖", detail)

    def test_partial_failure_reports_what_stayed_missing(self):
        dup = self.png("dup.png", (12, 120, 200))
        ghost = self.png("ghost.png", (200, 120, 12))
        content = dup.read_bytes()
        records = self._recycle(dup)
        ghost.unlink()                      # 模拟它已被人从回收站清掉
        records.append({"act": "recycle", "src": str(ghost), "size": 64})
        page = self._page()
        page._last_cleanup = self._manifest(records)
        info = self._undo(page)

        self.assertTrue(dup.exists())
        self.assertEqual(dup.read_bytes(), content)
        self.assertFalse(ghost.exists())
        self.assertEqual(info["payload"], ([str(dup)], [str(ghost)]))
        self.assertIn(("warn", "还原完成：1 个成功，1 个失败"), self.app.logs)
        self.assertEqual(info["alert"].call_count, 1)
        title, detail = info["alert"].call_args[0][2:4]
        self.assertEqual(title, "1 个文件未还原")
        self.assertIn("ghost.png", detail)
        self.assertIsNone(page._last_cleanup)   # 有成功就作废这条记录

    def test_existing_file_is_never_overwritten(self):
        dup = self.png("dup.png", (200, 30, 30))
        original = dup.read_bytes()
        page = self._page()
        manifest = self._manifest(self._recycle(dup))
        page._last_cleanup = manifest
        dup.write_bytes(b"newer content")   # 原位置又出现同名文件
        info = self._undo(page)

        self.assertEqual(dup.read_bytes(), b"newer content")
        self.assertNotEqual(dup.read_bytes(), original)
        self.assertEqual(info["payload"], ([], [str(dup)]))
        self.assertIn(("warn", "还原完成：0 个成功，1 个失败"), self.app.logs)
        title, detail = info["alert"].call_args[0][2:4]
        self.assertEqual(title, "1 个文件未还原")
        self.assertIn("原位置已有同名文件", detail)
        self.assertEqual(page._last_cleanup, manifest)  # 没还原成功，记录留着
        self.assertFalse(page.btn_undo_clean.enabled)


class ResetAndPresetDefaults(TkCase):
    """初始化路径上的默认值：组重置/取消选择、旧方案载入、分段控件取值。"""

    def _page(self, sizes):
        import types
        paths = [self.png(f"r{i}.png", (30 + i * 40, 90, 140), size=size)
                 for i, size in enumerate(sizes)]
        group = self.group(paths)
        page = gui.ResultsPage(self.app, self.app)
        page.result = models.ScanResult(root=self.tmp, groups=[group],
                                        file_count=len(paths))
        page.keep_rule = "best"
        page.selection = set(group.default_selection())
        card = types.SimpleNamespace(group=group, refresh=lambda: None)
        return page, group, paths, card

    def test_reset_group_reapplies_keep_rule(self):
        page, group, paths, card = self._page([(20, 20), (200, 200),
                                               (60, 60)])
        expected = core._pick_keep(group.kind, group.items, "best",
                                   page.result.root)
        manual = (expected + 1) % len(paths)
        page.set_keep(card, manual)
        self.assertEqual(group.keep, manual)

        page.reset_group(card)
        self.assertEqual(group.keep, expected)      # 不是留在手动选的那项
        self.assertEqual(page.selection, group.default_selection())

    def test_clear_group_keeps_keep_and_drops_marks(self):
        page, group, paths, card = self._page([(20, 20), (200, 200),
                                               (60, 60)])
        page.set_keep(card, 1)
        page.selection = {str(paths[0]), str(paths[2])}

        page.clear_group(card)
        self.assertEqual(page.selection, set())     # 不是又勾回来
        self.assertEqual(group.keep, 1)             # 保留项不动

    def test_old_preset_does_not_enable_new_checks(self):
        scan = gui.ScanPage(self.app, self.app)
        self.app.settings["scan_presets"] = {
            "旧方案": {"strictness": "strict", "exclude": "cache"},
        }
        scan.apply_preset("旧方案")

        self.assertFalse(scan.sw_quick.value, "旧方案缺键不能把快速扫描打开")
        self.assertFalse(scan.sw_fuzzy.value, "旧方案缺键不能把模糊匹配打开")
        self.assertTrue(scan.sw_cache.value)
        self.assertTrue(scan.sw_recursive.value)
        self.assertEqual(scan.seg_strict.value, "strict")
        self.assertEqual(scan.exclude_var.get(), "cache")

    def test_segmented_snaps_unknown_value_to_nearest(self):
        seg = ui_kit.Segmented(
            self.app, self.app.theme,
            [(0, "不限"), (1 << 20, "1 MB"), (100 << 20, "100 MB")],
            value=1 << 30)
        self.assertEqual(seg.value, 100 << 20)
        seg.set_value(7)
        self.assertEqual(seg.value, 0)
        seg.set_value(None)
        self.assertEqual(seg.value, 0)

        text = ui_kit.Segmented(self.app, self.app.theme,
                                [("relaxed", "宽松"), ("strict", "严格")],
                                value="??")
        self.assertEqual(text.value, "relaxed")
        text.set_value("strict")
        self.assertEqual(text.value, "strict")
        seg.destroy()
        text.destroy()

    def test_bad_min_size_in_settings_is_snapped_on_startup(self):
        self.app.settings["min_size"] = 1 << 30
        scan = gui.ScanPage(self.app, self.app)
        self.assertIn(scan.seg_size.value, (0, 1 << 20, 10 << 20, 100 << 20))
        self.assertEqual(scan.seg_size.value, 100 << 20)


class GuiCoreContract(unittest.TestCase):
    """界面源码里引用的 core.* 必须真的导出了，否则运行到才炸。"""

    def test_core_symbols_used_by_gui_exist(self):
        source = (HERE / "dedup_gui.py").read_text(encoding="utf-8")
        used = set()
        pos = 0
        while True:
            hit = source.find("core.", pos)
            if hit < 0:
                break
            pos = hit + 5
            before = source[hit - 1] if hit else ""
            if before.isalnum() or before == "_":
                continue                      # 例如 dedup_core.py
            name = ""
            for ch in source[pos:]:
                if ch.isalnum() or ch == "_":
                    name += ch
                else:
                    break
            if name:
                used.add(name)
        self.assertTrue(used)
        missing = sorted(name for name in used if not hasattr(core, name))
        self.assertEqual(missing, [])

if __name__ == "__main__":
    unittest.main()
