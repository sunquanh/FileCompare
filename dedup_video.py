"""重复视频：ffmpeg 抽帧（OpenCV 兜底）+ 多帧签名 + 匹配帧占比。

旧实现固定取 5 个百分比位置的帧，并要求「所有帧的平均距离」低于阈值，
一段片头不同就会被平均稀释掉。改成：
  * 整段均匀取 N 帧（默认 15），逐帧找时间上最接近的候选帧；
  * 只有「单帧粗差/细差/颜色」同时通过才算该帧匹配；
  * 用「匹配帧占比 ≥ 阈值」判定，支持片头片尾略有差异的转码版本；
  * 时长容差改为 max(1.5 秒, 20%)，短视频不会因为 1 秒差异漏检。
"""
from __future__ import annotations

import json
import subprocess
import threading

try:
    import numpy as np
    from PIL import Image
except Exception:  # pragma: no cover
    np = None
    Image = None

from dedup_models import (check_cancel, cv2, find_tool, hidden_subprocess_kwargs,
                          ignore)
from dedup_photos import dhash64, phash256, popcount

FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")


def video_backend() -> str:
    if FFMPEG and FFPROBE:
        return "ffmpeg"
    if cv2 is not None:
        return "opencv"
    return ""


def available() -> bool:
    return bool(video_backend())


def probe(path) -> dict | None:
    if not (FFMPEG and FFPROBE):
        return None
    try:
        proc = subprocess.run(
            [FFPROBE, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, timeout=45, check=False,
            **hidden_subprocess_kwargs())
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout.decode("utf-8", "ignore") or "{}")
        duration = float(data.get("format", {}).get("duration") or 0)
        width = height = 0
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = int(stream.get("width") or 0)
                height = int(stream.get("height") or 0)
                break
        if duration <= 0:
            return None
        return {"dur": duration, "w": width, "h": height}
    except Exception:
        return None


# ---------------------------------------------------------------- 抽帧
def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_ppm(stream):
    magic = stream.readline().strip()
    if magic != b"P6":
        return None
    line = stream.readline()
    while line.startswith(b"#"):
        line = stream.readline()
    try:
        width, height = (int(x) for x in line.split())
    except Exception:
        return None
    stream.readline()  # maxval
    data = _read_exact(stream, width * height * 3)
    if len(data) < width * height * 3:
        return None
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def _ffmpeg_frames(path, count, width=320, timeout=90):
    duration = probe(path)
    if not duration:
        return None, None
    fps = max(count / max(duration["dur"], 0.1), 0.001)
    cmd = [
        FFMPEG, "-v", "error", "-nostdin", "-i", str(path), "-an",
        "-vf", f"fps={fps:.6f},scale={width}:-2",
        "-frames:v", str(count),
        "-f", "image2pipe", "-vcodec", "ppm", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL,
                                **hidden_subprocess_kwargs())
    except Exception:
        return None, None
    timer = threading.Timer(timeout, proc.kill)
    timer.daemon = True
    timer.start()
    frames = []
    try:
        stream = proc.stdout
        while len(frames) < count:
            frame = _read_ppm(stream)
            if frame is None:
                break
            frames.append(frame)
    except Exception as exc:
        ignore(__name__, exc)
    finally:
        timer.cancel()
        try:
            proc.stdout.close()
        except Exception as exc:
            ignore(__name__, exc)
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception as exc:
                ignore(__name__, exc)
    return frames, duration


def _cv2_frames(path, count, width=320):
    if cv2 is None:
        return None, None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None, None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = total / fps if fps > 0 else 0
        if duration <= 0:
            return None, None
        frames = []
        for i in range(count):
            fraction = (i + 0.5) / count
            cap.set(cv2.CAP_PROP_POS_MSEC, duration * fraction * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            if im.width > width:
                im = im.resize((width, max(1, int(im.height * width / im.width))),
                               Image.Resampling.BILINEAR)
            frames.append(np.asarray(im))
        return frames, {"dur": duration,
                        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)}
    finally:
        cap.release()


def grab_frame(path, fraction=0.15, width=640):
    """取一帧用于缩略图。返回 PIL.Image 或 None。"""
    if FFMPEG:
        duration = probe(path)
        if duration:
            ts = max(0.0, duration["dur"] * fraction)
            cmd = [FFMPEG, "-v", "error", "-nostdin", "-ss", f"{ts:.3f}",
                   "-i", str(path), "-frames:v", "1", "-f", "image2pipe",
                   "-vcodec", "ppm", "-"]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=45,
                                      check=False, **hidden_subprocess_kwargs())
                if proc.returncode == 0 and proc.stdout:
                    import io

                    from PIL import Image as _Image
                    arr = _read_ppm(io.BytesIO(proc.stdout))
                    if arr is not None:
                        im = _Image.fromarray(arr)
                        if im.width > width:
                            im = im.resize(
                                (width, max(1, int(im.height * width / im.width))),
                                Image.Resampling.LANCZOS)
                        return im
            except Exception as exc:
                ignore(__name__, exc)
    frames, _meta = _cv2_frames(path, 3, width)
    if frames:
        return Image.fromarray(frames[len(frames) // 3])
    return None


# ---------------------------------------------------------------- 签名
def _frame_signature(rgb: np.ndarray):
    gray = Image.fromarray(rgb).convert("L")
    color = rgb.reshape(-1, 3).mean(axis=0)
    return dhash64(gray), phash256(gray), (float(color[0]), float(color[1]),
                                           float(color[2]))


def signature(path, frames=15, cancel=None):
    check_cancel(cancel)
    if FFMPEG:
        raw, meta = _ffmpeg_frames(path, frames)
    else:
        raw, meta = _cv2_frames(path, frames)
    if not raw or not meta or len(raw) < 3:
        return None
    sig = []
    for index, frame in enumerate(raw):
        d8, ph, color = _frame_signature(frame)
        t = (index + 0.5) / len(raw)
        sig.append((t, d8, [int(x) for x in ph], color))
    return {"dur": float(meta["dur"]), "w": int(meta["w"]),
            "h": int(meta["h"]), "frames": sig}


def signature_to_cache(sig):
    return {
        "dur": sig["dur"], "w": sig["w"], "h": sig["h"],
        "frames": [[round(t, 4), int(d8), [int(x) for x in ph],
                    [round(float(c), 2) for c in color]]
                   for t, d8, ph, color in sig["frames"]],
    }


def signature_from_cache(data):
    frames = []
    for t, d8, ph, color in data["frames"]:
        frames.append((float(t), int(d8),
                       [int(x) for x in ph],
                       tuple(float(c) for c in color)))
    return {"dur": float(data["dur"]), "w": int(data["w"]),
            "h": int(data["h"]), "frames": frames}


def match_videos(sigs, dist=10, fine=96, ratio=0.6, dur_tol=0.2,
                 cancel=None, on_progress=None):
    """返回 [(i, j, 平均粗差, 平均细差, 匹配帧占比)]。"""
    items = [(i, s) for i, s in enumerate(sigs) if s]
    items.sort(key=lambda x: x[1]["dur"])
    out = []
    total = len(items)
    for a in range(total):
        check_cancel(cancel)
        ia, sa = items[a]
        for b in range(a + 1, total):
            ib, sb = items[b]
            tolerance = max(1.5, dur_tol * max(sa["dur"], sb["dur"]))
            if sb["dur"] - sa["dur"] > tolerance:
                break
            matched = 0
            compared = 0
            sum_c = 0.0
            sum_f = 0.0
            for ta, d8a, pha, ca in sa["frames"]:
                best = None
                best_gap = 0.09
                for tb, d8b, phb, cb in sb["frames"]:
                    gap = abs(tb - ta)
                    if gap <= best_gap:
                        d = int(popcount(np.uint64(d8a) ^ np.uint64(d8b)))
                        if best is None or d < best[0]:
                            best_gap = gap
                            best = (d, pha, phb, ca, cb)
                if best is None:
                    continue
                compared += 1
                d8, pha, phb, ca, cb = best
                diff = int(popcount(
                    np.array(pha, dtype=np.uint64) ^
                    np.array(phb, dtype=np.uint64)).sum())
                color = sum(abs(x - y) for x, y in zip(ca, cb)) / 3.0
                sum_c += d8
                sum_f += diff
                if d8 <= dist and diff <= fine and color <= 78:
                    matched += 1
            if compared < 3:
                continue
            matched_ratio = matched / compared
            if matched_ratio >= ratio:
                out.append((ia, ib, sum_c / compared, sum_f / compared,
                            matched_ratio))
        if on_progress is not None:
            on_progress(a + 1, total, "视频比对")
    return out
