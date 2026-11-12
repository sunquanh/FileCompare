"""重复音乐：标签解析 + 码率估算 + 可选 Chromaprint 声学指纹。

标签解析保留原来的手写 ID3v1/ID3v2/FLAC/Ogg/MP4/WAV 读取器（不引入新依赖），
新增两件事：
  * MP3 没有 TLEN 时，读第一个帧头估算时长与码率（保留项据此选音质更好的）；
  * 如果系统里有 `fpcalc`（Chromaprint），同一标签组再验一次声学指纹，
    避免「同名但不同版本（Live / Remix / 伴奏）」被误判成同一首。
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from dedup_models import (FileItem, find_tool, hidden_subprocess_kwargs,
                          ignore)

FPCALC = find_tool("fpcalc")

_BITRATES_V1L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256,
                  320, 0]
_BITRATES_V2L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144,
                  160, 0]
_SAMPLE_RATES = [44100, 48000, 32000, 0]


# ---------------------------------------------------------------- 标签读取
def _synchsafe(raw) -> int:
    if len(raw) < 4:
        return 0
    return ((raw[0] & 0x7F) << 21) | ((raw[1] & 0x7F) << 14) | \
           ((raw[2] & 0x7F) << 7) | (raw[3] & 0x7F)


def _decode_id3_text(payload: bytes) -> str:
    if not payload:
        return ""
    enc = payload[0]
    body = payload[1:]
    try:
        if enc == 1:
            text = body.decode("utf-16", "ignore")
        elif enc == 2:
            text = body.decode("utf-16-be", "ignore")
        elif enc == 3:
            text = body.decode("utf-8", "ignore")
        else:
            text = body.decode("latin-1", "ignore")
    except Exception:
        text = body.decode("utf-8", "ignore")
    return text.split("\x00")[0].strip().strip("\ufeff")


def _id3_tags(f) -> dict:
    head = f.read(10)
    if len(head) < 10 or head[:3] != b"ID3":
        return {}
    ver = head[3]
    flags = head[5]
    size = _synchsafe(head[6:10])
    body = f.read(max(0, min(size, 8 << 20)))
    if flags & 0x80:
        body = body.replace(b"\xff\x00", b"\xff")
    pos = 0
    if ver >= 3 and flags & 0x40:
        if ver == 3 and len(body) >= 4:
            pos = 4 + int.from_bytes(body[0:4], "big")
        elif len(body) >= 4:
            pos = _synchsafe(body[0:4])
    tags = {}
    while pos + 6 <= len(body):
        if ver == 2:
            fid = body[pos:pos + 3].decode("latin-1")
            if not fid.strip("\x00"):
                break
            fsize = int.from_bytes(body[pos + 3:pos + 6], "big")
            hdr = 6
        else:
            fid = body[pos:pos + 4].decode("latin-1")
            if not fid.strip("\x00"):
                break
            raw = body[pos + 4:pos + 8]
            fsize = _synchsafe(raw) if ver >= 4 else int.from_bytes(raw, "big")
            hdr = 10
        if fsize <= 0 or pos + hdr + fsize > len(body):
            break
        payload = body[pos + hdr:pos + hdr + fsize]
        pos += hdr + fsize
        if fid in ("TIT2", "TT2"):
            tags["title"] = _decode_id3_text(payload)
        elif fid in ("TPE1", "TP1"):
            tags["artist"] = _decode_id3_text(payload)
        elif fid in ("TALB", "TAL"):
            tags["album"] = _decode_id3_text(payload)
        elif fid in ("TLEN", "TLE"):
            try:
                tags["duration"] = float(_decode_id3_text(payload)) / 1000.0
            except ValueError:
                pass
    return tags


def _id3v1_tags(f) -> dict:
    try:
        f.seek(0, 2)
        if f.tell() < 128:
            return {}
        f.seek(-128, 2)
        raw = f.read(128)
    except Exception:
        return {}
    if len(raw) < 128 or raw[:3] != b"TAG":
        return {}

    def clean(b):
        return b.split(b"\x00", 1)[0].decode("latin-1", "ignore").strip()

    tags = {}
    if clean(raw[3:33]):
        tags["title"] = clean(raw[3:33])
    if clean(raw[33:63]):
        tags["artist"] = clean(raw[33:63])
    if clean(raw[63:93]):
        tags["album"] = clean(raw[63:93])
    return tags


def _vorbis_comments(data: bytes, pos: int) -> dict:
    tags = {}
    try:
        vlen = int.from_bytes(data[pos:pos + 4], "little")
        pos += 4 + vlen
        count = int.from_bytes(data[pos:pos + 4], "little")
        pos += 4
        for _ in range(min(max(count, 0), 128)):
            length = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
            item = data[pos:pos + length]
            pos += length
            if b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            key = key.decode("ascii", "ignore").upper()
            value = value.decode("utf-8", "ignore").strip()
            if key == "ARTIST" and "artist" not in tags:
                tags["artist"] = value
            elif key == "TITLE" and "title" not in tags:
                tags["title"] = value
            elif key == "ALBUM" and "album" not in tags:
                tags["album"] = value
    except Exception as exc:
        ignore(__name__, exc)
    return tags


def _flac_tags(f) -> dict:
    if f.read(4) != b"fLaC":
        return {}
    tags = {}
    while True:
        head = f.read(4)
        if len(head) < 4:
            break
        last = bool(head[0] & 0x80)
        btype = head[0] & 0x7F
        size = int.from_bytes(head[1:4], "big")
        data = f.read(min(size, 4 << 20))
        if btype == 0 and len(data) >= 18:
            info = int.from_bytes(data[10:18], "big")
            sample_rate = (info >> 44) & 0xFFFFF
            total = info & ((1 << 36) - 1)
            if sample_rate and total:
                tags["duration"] = total / sample_rate
                tags["sample_rate"] = float(sample_rate)
        elif btype == 4:
            tags.update(_vorbis_comments(data, 0))
        if last:
            break
    return tags


def _ogg_tags(f) -> dict:
    try:
        f.seek(0)
        data = f.read(262144)
    except Exception:
        return {}
    if not data.startswith(b"OggS"):
        return {}
    tags = {}
    idx = data.find(b"\x03vorbis")
    if idx >= 0:
        tags.update(_vorbis_comments(data, idx + 7))
    else:
        idx = data.find(b"OpusTags")
        if idx >= 0:
            tags.update(_vorbis_comments(data, idx + 8))
    rate = 0
    vid = data.find(b"\x01vorbis")
    if vid >= 0 and len(data) >= vid + 16:
        rate = int.from_bytes(data[vid + 12:vid + 16], "little")
    elif data.find(b"OpusHead") >= 0:
        rate = 48000
    if rate:
        try:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read()
            last = tail.rfind(b"OggS")
            if last >= 0 and len(tail) >= last + 14:
                granule = int.from_bytes(tail[last + 6:last + 14], "little")
                if granule:
                    tags["duration"] = granule / rate
        except Exception as exc:
            ignore(__name__, exc)
    return tags


def _mp4_atoms(f, start, end):
    f.seek(start)
    while f.tell() + 8 <= end:
        pos = f.tell()
        head = f.read(8)
        if len(head) < 8:
            break
        size = int.from_bytes(head[0:4], "big")
        typ = head[4:8]
        header = 8
        if size == 1:
            big = f.read(8)
            if len(big) < 8:
                break
            size = int.from_bytes(big, "big")
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            break
        yield typ, pos + header, pos + size
        f.seek(pos + size)


def _mp4_tags(f) -> dict:
    try:
        f.seek(0, 2)
        total = f.tell()
    except Exception:
        return {}
    tags = {}
    try:
        for typ, start, end in _mp4_atoms(f, 0, total):
            if typ != b"moov":
                continue
            for typ2, s2, e2 in _mp4_atoms(f, start, end):
                if typ2 == b"mvhd":
                    f.seek(s2)
                    data = f.read(min(32, e2 - s2))
                    if len(data) >= 20:
                        if data[0] == 1 and len(data) >= 32:
                            scale = int.from_bytes(data[20:24], "big")
                            dur = int.from_bytes(data[24:32], "big")
                        else:
                            scale = int.from_bytes(data[12:16], "big")
                            dur = int.from_bytes(data[16:20], "big")
                        if scale:
                            tags["duration"] = dur / scale
                elif typ2 == b"udta":
                    for typ3, s3, e3 in _mp4_atoms(f, s2, e2):
                        if typ3 != b"meta":
                            continue
                        for typ4, s4, e4 in _mp4_atoms(f, s3 + 4, e3):
                            if typ4 != b"ilst":
                                continue
                            for typ5, s5, e5 in _mp4_atoms(f, s4, e4):
                                key = {b"\xa9nam": "title", b"\xa9ART": "artist",
                                       b"\xa9alb": "album"}.get(typ5)
                                if not key:
                                    continue
                                for typ6, s6, e6 in _mp4_atoms(f, s5, e5):
                                    if typ6 != b"data":
                                        continue
                                    f.seek(s6)
                                    raw = f.read(min(4096, e6 - s6))
                                    if raw[:4] == b"\x00\x00\x00\x01":
                                        tags[key] = raw[8:].decode(
                                            "utf-8", "ignore").strip()
                                        break
    except Exception as exc:
        ignore(__name__, exc)
    return tags


def _wav_tags(f) -> dict:
    try:
        f.seek(0)
        head = f.read(12)
        if head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            return {}
        tags = {}
        byte_rate = 0
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            cid = h[:4]
            size = int.from_bytes(h[4:8], "little")
            if cid == b"fmt ":
                data = f.read(min(size, 4096))
                if len(data) >= 12:
                    byte_rate = int.from_bytes(data[8:12], "little")
            elif cid == b"data":
                if byte_rate:
                    tags["duration"] = size / byte_rate
                f.seek(size, 1)
            elif cid == b"LIST":
                data = f.read(min(size, 1 << 20))
                if data[:4] == b"INFO":
                    pos = 4
                    while pos + 8 <= len(data):
                        sid = data[pos:pos + 4]
                        slen = int.from_bytes(data[pos + 4:pos + 8], "little")
                        pos += 8
                        value = data[pos:pos + slen]
                        pos += slen + (slen & 1)
                        text = value.rstrip(b"\x00").decode(
                            "utf-8", "ignore").strip()
                        if sid == b"IART":
                            tags["artist"] = text
                        elif sid == b"INAM":
                            tags["title"] = text
                        elif sid == b"IPRD":
                            tags["album"] = text
                f.seek(size - min(size, 1 << 20), 1)
            else:
                f.seek(size, 1)
            if size & 1:
                f.seek(1, 1)
        return tags
    except Exception:
        return {}


def _mp3_header_info(f, size):
    """读第一个 MPEG 帧头，返回 (码率 kbps, 采样率, 时长)。"""
    try:
        f.seek(0)
        head = f.read(10)
        offset = 0
        if head[:3] == b"ID3":
            offset = 10 + _synchsafe(head[6:10])
            if head[5] & 0x10:
                offset += 10
        f.seek(offset)
        data = f.read(64 * 1024)
    except Exception:
        return 0.0, 0, 0.0
    index = data.find(b"\xff")
    while 0 <= index < len(data) - 4:
        b1 = data[index + 1]
        if (b1 & 0xE0) == 0xE0 and b1 & 24 != 8:
            version = (b1 >> 3) & 0x03
            layer = (b1 >> 1) & 0x03
            if version != 1 and layer == 1:
                bitrate_idx = (data[index + 2] >> 4) & 0x0F
                sample_idx = (data[index + 2] >> 2) & 0x03
                if bitrate_idx not in (0, 15) and sample_idx != 3:
                    table = (_BITRATES_V1L3 if version == 3
                             else _BITRATES_V2L3)
                    bitrate = table[bitrate_idx] * 1000
                    rate = _SAMPLE_RATES[sample_idx]
                    if version == 2:      # MPEG2
                        rate //= 2
                    elif version == 0:    # MPEG2.5
                        rate //= 4
                    audio = max(0, size - offset)
                    duration = audio * 8 / bitrate if bitrate else 0.0
                    return bitrate / 1000.0, rate, duration
        index = data.find(b"\xff", index + 1)
    return 0.0, 0, 0.0


def read_tags(p: Path) -> dict:
    p = Path(p)
    ext = p.suffix.lower()
    tags = {}
    try:
        with open(p, "rb") as f:
            head = f.read(12)
        with open(p, "rb") as f:
            if head[:3] == b"ID3":
                tags = _id3_tags(f)
                if not tags.get("title") or not tags.get("artist"):
                    f.seek(0)
                    for k, v in _id3v1_tags(f).items():
                        tags.setdefault(k, v)
            elif head[:4] == b"fLaC" or ext == ".flac":
                tags = _flac_tags(f)
            elif head[:4] == b"OggS":
                tags = _ogg_tags(f)
            elif head[:4] in (b"RIFF", b"RF64"):
                tags = _wav_tags(f)
            elif head[4:8] == b"ftyp" or ext in (".m4a", ".m4b", ".mp4", ".aac"):
                tags = _mp4_tags(f)
            elif ext == ".mp3":
                tags = _id3_tags(f)
                if not tags.get("title") or not tags.get("artist"):
                    f.seek(0)
                    for k, v in _id3v1_tags(f).items():
                        tags.setdefault(k, v)
        bitrate = 0.0
        if ext == ".mp3" or head[:3] == b"ID3":
            with open(p, "rb") as f:
                kbps, _rate, estimated = _mp3_header_info(f, p.stat().st_size)
            bitrate = kbps
            if estimated and not tags.get("duration"):
                tags["duration"] = estimated
        duration = float(tags.get("duration") or 0.0)
        if not bitrate and duration > 0:
            bitrate = p.stat().st_size * 8 / duration / 1000.0
        return {
            "artist": (tags.get("artist") or "").strip(),
            "title": (tags.get("title") or "").strip(),
            "album": (tags.get("album") or "").strip(),
            "duration": duration,
            "bitrate": float(bitrate),
        }
    except Exception:
        return {"artist": "", "title": "", "album": "", "duration": 0.0,
                "bitrate": 0.0}


# ---------------------------------------------------------------- 分组
def _norm_tag(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def audio_key(item: FileItem):
    artist = _norm_tag(item.audio_artist)
    title = _norm_tag(item.audio_title)
    if not artist or not title:
        return None
    return f"{artist}\x00{title}"


def cluster_audio(items, dur_tol=4.0):
    """按 (歌手, 标题) 聚组，时长差超过容差的拆成不同组。"""
    known = sorted([i for i in items if i.duration > 0],
                   key=lambda x: x.duration)
    unknown = [i for i in items if i.duration <= 0]
    groups, cur = [], []
    for item in known:
        if not cur:
            cur = [item]
            continue
        # 与组内首项比较：跟 cur[-1] 比会链式漂移，100/103/106 秒的三个
        # 版本会被并成一组（首尾差 6 秒 > 4 秒容差）。
        tol = max(dur_tol, cur[0].duration * 0.02)
        if item.duration - cur[0].duration <= tol:
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]
    if cur:
        groups.append(cur)
    if unknown:
        if groups:
            groups[0].extend(unknown)
        else:
            groups.append(unknown)
    return [g for g in groups if len(g) >= 2]


# ---------------------------------------------------------------- Chromaprint
def fingerprint_available() -> bool:
    return FPCALC is not None


def chromaprint(path, length=120):
    """返回 (指纹整数列表, 覆盖时长)；失败返回 None。"""
    if not FPCALC:
        return None
    try:
        proc = subprocess.run(
            [FPCALC, "-raw", "-json", "-length", str(length), str(path)],
            capture_output=True, timeout=90, check=False,
            **hidden_subprocess_kwargs())
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout.decode("utf-8", "ignore") or "{}")
        values = data.get("fingerprint") or []
        if not values:
            return None
        return [int(v) & 0xFFFFFFFF for v in values], float(
            data.get("duration") or 0.0)
    except Exception:
        return None


def fp_distance(first, second, max_shift=4):
    """对齐后取最小汉明距离，返回 0~1 的归一化差异（越小越像）。"""
    if not first or not second:
        return 1.0
    import numpy as np
    a = np.array(first, dtype=np.uint32)
    b = np.array(second, dtype=np.uint32)
    length = min(a.size, b.size)
    if length < 8:
        return 1.0
    best = 1.0
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            x, y = a[shift:shift + length - max_shift], b[:length - max_shift]
        else:
            x, y = a[:length - max_shift], b[-shift:length - max_shift - shift]
        if x.size < 8 or y.size < 8:
            continue
        n = min(x.size, y.size)
        diff = np.bitwise_count(x[:n] ^ y[:n]).sum() if hasattr(
            np, "bitwise_count") else np.unpackbits(
            (x[:n] ^ y[:n]).view(np.uint8)).sum()
        best = min(best, float(diff) / (32.0 * n))
    return best
