"""重复文件检测引擎（入口层）。

对界面提供稳定的 API：`scan_folder(options, ...) -> ScanResult`，
其余实现拆在 dedup_models / dedup_cache / dedup_exact / dedup_photos /
dedup_video / dedup_audio / dedup_names / dedup_files 里。

四类内容检测（全部只读，不修改任何文件）：
  完全重复  —— 分层哈希：大小 → 首尾 → 采样 → 全文件
  相似照片  —— 8 方向 dHash 粗筛 + pHash 复核 + 颜色直方图 + 完全链接聚类
  重复视频  —— ffmpeg 抽帧签名 + 匹配帧占比
  重复音乐  —— 标签(歌手/标题) + 时长，可选 Chromaprint 声学指纹
另外把「去掉副本标记后同名」作为独立提示，默认不勾选删除。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from dedup_audio import (
    audio_key,
    chromaprint,
    cluster_audio,
    fingerprint_available,
    fp_distance,
    read_tags,
)
from dedup_cache import HashCache
from dedup_exact import (
    ProgressThrottle,
    _full_hash,
    _quick_hash,
    _sample_hash,
    find_exact,
    parallel_map,
)
from dedup_files import (
    collect_files,
    export_report,
    export_report_json,
    find_duplicate_dirs,
    hardlink_replace,
    make_thumbnail,
    permanent_delete,
    recycle_capable,
    recycle_paths,
    restore_from_recycle,
    verify_recycled,
    restore_hardlinks,
)
from dedup_models import (
    AUDIO_EXT,
    CACHE_TAGS,
    KEEP_RULE_LABELS,
    KEEP_RULES,
    KIND_AUDIO,
    KIND_DIR,
    KIND_EXACT,
    KIND_NAME,
    KIND_ORDER,
    KIND_PHOTO,
    KIND_VIDEO,
    PHOTO_EXT,
    PRESETS,
    SKIP_DIRS,
    SKIP_PREFIX,
    VIDEO_EXT,
    DupGroup,
    FileItem,
    Image,
    ImageDraw,
    ImageOps,
    ScanCancelled,
    ScanOptions,
    ScanResult,
    app_dir,
    check_cancel,
    clusters_complete_link,
    cv2,
    format_date,
    format_duration,
    human_size,
    ignore,
    is_audio,
    is_photo,
    is_video,
    np,
    norm_path,
    pick_keep,
    should_skip_dir,
)
from dedup_names import find_name_groups, name_key
from dedup_photos import PhotoFP, match_photos, pair_similarity, probe_size
from dedup_photos import frame_fingerprints as photo_frame_fingerprints
from dedup_photos import _frames_from_cache, _frames_to_cache
from dedup_models import detect_bursts
from dedup_video import (
    match_videos,
    signature_from_cache,
    signature_to_cache,
    video_backend,
)
from dedup_video import signature as video_signature

_pick_keep = pick_keep


# 相似类（照片/视频）低于该相似度时默认不勾选，需人工对比确认后再删
SIMILARITY_AUTO_SELECT = 95


# 只在盘符根 / UNC 根 / POSIX 根下直接出现时才算系统区；
# “windows”“recovery”这类名字用户也可能给普通文件夹起，深处出现不算。
SYSTEM_DIR_NAMES = {
    "windows", "program files", "program files (x86)", "programdata",
    "system volume information", "$recycle.bin", "perflogs",
    "recovery", "system32",
}

# AppData 是 Windows 管理的用户配置目录，名字本身就有指向性，任意深度都算。
SYSTEM_DIR_NAMES_ANY_DEPTH = {
    "appdata",
}


def is_system_area(path) -> bool:
    """路径本身或其上级是 Windows 系统目录（扫描/清理有误删风险）。
    系统名只认根下的直接子级（C:/Windows、D:/Program Files），避免
    D:/backup/windows 这类普通文件夹被误拦；AppData 例外，任意深度都算；
    相对路径只认开头就是系统名的那一级。"""
    parts = [p.lower() for p in Path(path).parts]
    if not parts:
        return False
    if any(p in SYSTEM_DIR_NAMES_ANY_DEPTH for p in parts):
        return True
    first = parts[0]
    absolute = (len(first) >= 2 and first[1] == ":" and first[0].isalpha()) \
        or first.startswith(chr(92) * 2) or first.startswith("/")
    if absolute:
        return len(parts) > 1 and parts[1] in SYSTEM_DIR_NAMES
    return parts[0] in SYSTEM_DIR_NAMES


def is_filesystem_root(path) -> bool:
    """是不是盘符根目录（Windows 的 C 盘根目录）、UNC 根目录
    （服务器加共享名那一层）或根目录（POSIX 的 /）。"""
    text = str(path)
    if text.startswith(("\\\\", "//")):
        parts = [p for p in text[2:].replace("\\", "/").split("/") if p]
        return len(parts) == 2 and parts[0] not in ("?", ".")
    if len(text) >= 2 and text[1] == ":":
        return text[2:] in ("", "\\", "/")
    return text in ("/", "\\")


def video_available() -> bool:
    """本机是否有可用的视频解码后端（ffmpeg 或 OpenCV）。"""
    return bool(video_backend())
audio_fingerprint_available = fingerprint_available


# ---------------------------------------------------------------- 主流程
def scan_folder(options: ScanOptions, on_progress=None, on_log=None,
                cancel=None):
    """执行完整扫描，并自动保存/复用指纹缓存。"""
    cache = HashCache(enabled=bool(getattr(options, "cache", True)))
    try:
        return _scan_impl(options, on_progress=on_progress, on_log=on_log,
                          cancel=cancel, cache=cache)
    finally:
        cache.save()
        cache.close()


def _scan_impl(options: ScanOptions, on_progress=None, on_log=None,
               cancel=None, cache=None):
    """on_progress(stage, done, total, detail) / on_log(message)"""
    emit = on_progress or (lambda *a: None)
    say = on_log or (lambda *a: None)
    cancel = cancel or threading.Event()
    started = time.perf_counter()

    if np is None or Image is None:
        raise RuntimeError("缺少 numpy 或 Pillow，无法检测。")

    root = Path(options.root)
    keep_rule = getattr(options, "keep_rule", "best")
    quick = bool(getattr(options, "quick_scan", False))
    emit("collect", 0, 0, "正在遍历文件夹…")
    say(f"开始扫描：{root}")
    files, skipped_inodes, skipped_unreadable, skipped_empty = collect_files(
        options, cancel=cancel,
        on_progress=lambda d, t, m: emit("collect", d, t, m))
    if skipped_inodes:
        say(f"跳过 {skipped_inodes} 个硬链接/重复挂载（同一物理文件）")

    if not files:
        if skipped_unreadable:
            say(f"{skipped_unreadable} 个文件无法读取，已跳过")
        say("没有发现可检测的文件")
        return ScanResult(root=root, file_count=0,
                          elapsed=time.perf_counter() - started,
                          skipped_inodes=skipped_inodes,
                          skipped_unreadable=skipped_unreadable,
                          skipped_empty=skipped_empty)

    total_size = sum(f.size for f in files)
    photos = [f for f in files if is_photo(f.path)]
    videos = [f for f in files if is_video(f.path)]
    audios = [f for f in files if is_audio(f.path)]
    say(f"发现 {len(files):,} 个文件（照片 {len(photos):,} / 视频 {len(videos):,}"
        f" / 音乐 {len(audios):,}），共 {human_size(total_size)}")
    emit("collect", len(files), len(files), f"共 {len(files):,} 个文件")
    ref_files = sum(1 for f in files if f.protected)
    if ref_files:
        say(f"保护/参考目录：{ref_files:,} 个文件只作为保留基准参与比对，不会被清理")

    groups: list[DupGroup] = []
    consumed: set[str] = set()
    bad_photos = bad_videos = bad_audio = 0
    # ---- 重复目录（只按文件名+大小，内容未校验，默认不勾选，不占 consumed）
    # 参考/保护目录里的文件只作为保留基准，不参与「重复目录」提示
    dir_groups = find_duplicate_dirs([f for f in files if not f.protected])
    for members in dir_groups:
        group = DupGroup(KIND_DIR, members, 0,
                         "目录内文件的文件名+大小完全一致，内容未校验（默认不勾选）",
                         select_by_default=False)
        group.keep = pick_keep(group.kind, members, keep_rule, root)
        groups.append(group)
    if dir_groups:
        say(f"重复目录：{len(dir_groups)} 组")
    tiny_photos: list[str] = []
    notes: list[str] = []

    # ---- 完全重复
    exact_stats: dict = {}
    if options.exact:
        emit("exact", 0, 0, "正在分组比对文件指纹…")
        exact_groups = find_exact(
            files, options.workers, cancel=cancel,
            on_progress=lambda d, t, m: emit("exact", d, t, m), cache=cache,
            stats=exact_stats, quick=quick)
        for members in exact_groups:
            items = sorted(members, key=lambda x: (-x.size, str(x.path)))
            group = DupGroup(KIND_EXACT, items, 100, "字节完全一致")
            group.keep = pick_keep(group.kind, items, keep_rule, root)
            groups.append(group)
            consumed |= {str(i.path) for i in items}
        say(f"完全重复：{len(exact_groups)} 组")
        emit("exact", 1, 1, f"完成，发现 {len(exact_groups)} 组")
        if quick:
            notes.append("快速扫描：仅完全重复，已全量确认")
            say("快速扫描：已跳过照片/视频/音乐/同名检测")
    else:
        say("已跳过完全重复检测")
        emit("exact", 1, 1, "已跳过")

    # ---- 相似照片
    todo_photos = [f for f in photos if str(f.path) not in consumed]
    _bursts = detect_bursts([str(f.path) for f in todo_photos])
    burst_of = {}
    for _bi, _b in enumerate(_bursts):
        for _p in _b:
            burst_of[_p] = _bi
    if options.photos and todo_photos and not quick:
        emit("photo", 0, len(todo_photos), "正在计算照片指纹…")
        tag = CACHE_TAGS["photo"]

        stale_photo_lock = threading.Lock()
        stale_photo_logged = []

        def photo_work(item):
            hit = cache.get(item, tag)
            if hit:
                try:
                    return _frames_from_cache(hit)
                except ValueError as exc:
                    if "旧格式" in str(exc):
                        with stale_photo_lock:
                            if not stale_photo_logged:
                                stale_photo_logged.append(True)
                                say("照片指纹缓存是旧格式，正在重新生成（本次会比平时慢）")
                    ignore(__name__, exc)
                except Exception as exc:
                    ignore(__name__, exc)
            fps_list = photo_frame_fingerprints(item.path,
                                                options.max_image_edge)
            if fps_list:
                cache.put(item, tag, _frames_to_cache(fps_list))
            return fps_list

        throttle = ProgressThrottle(
            lambda d, t, m: emit("photo", d, t, f"照片指纹 {d:,}/{t:,}"))
        fps_nested = parallel_map(photo_work, todo_photos, options.workers,
                                cancel=cancel,
                                on_progress=lambda d, t, m: throttle(d, t, m))
        valid = []
        for item, fps_list in zip(todo_photos, fps_nested):
            if not fps_list:
                continue
            for fi, fp in enumerate(fps_list):
                valid.append((item, fi, fp))
        invalid = [todo_photos[i] for i in range(len(todo_photos))
                   if not fps_nested[i]]
        bad_photos = len(invalid)
        for item in invalid:
            size = probe_size(item.path)
            if size and (size[0] < 16 or size[1] < 16):
                tiny_photos.append(str(item.path))
        for item, fi, fp in valid:
            if fi == 0:
                item.width, item.height = fp.width, fp.height
        n_files = len({str(v[0].path) for v in valid})
        n_frames = len(valid)
        frame_note = (f"（共 {n_frames:,} 帧）" if n_frames != n_files else "")
        say(f"照片指纹完成：{n_files:,} 张可用{frame_note}，{bad_photos} 张跳过"
            + (f"（其中 {len(tiny_photos)} 张小于 16px）" if tiny_photos else ""))
        if tiny_photos:
            shown = "、".join(tiny_photos[:10])
            extra = f" 等 {len(tiny_photos)} 张" if len(tiny_photos) > 10 else ""
            say(f"小于 16px 的图片 {len(tiny_photos)} 张，已跳过：{shown}{extra}")

        if len(valid) >= 2:
            emit("photo", 0, 1, "正在比对相似照片…")
            pairs = match_photos(
                [v[2] for v in valid],
                dist=getattr(options, "img_dist", 8),
                fine=getattr(options, "img_fine", 42),
                hist_tol=getattr(options, "img_hist", 0.26),
                ncc_tol=getattr(options, "img_ncc", 0.88),
                layout_tol=getattr(options, "img_layout", 14.0),
                cancel=cancel,
                on_progress=lambda d, t, m: emit("photo", d, t, "照片比对"))
            edges = [(a, b, pair_similarity(fine, ncc))
                     for a, b, _c, fine, _h, ncc, _l in pairs
                     if str(valid[a][0].path) != str(valid[b][0].path)]
            clusters = clusters_complete_link(len(valid), edges)
            # 多帧：同一文件可能出现在多个帧簇里，先把文件集合
            # 有交集的帧簇合并（并查集），保证每文件最多进一个组。
            # 注意：合并后组的相似度取簇内所有帧对的最小值，多帧文件
            # 只要有一对帧较弱，整组就可能默认不勾选（有意为之，保守）。
            # 另注意：并查集合并把完全连接聚类改成文件级单连接——
            # 同一组里可能出现从未直接配对过的两个文件（都与第三个
            # 文件配对），是有意设计（保证每文件最多进一个组）。
            _parent = list(range(len(clusters)))
            def _find(x):
                while _parent[x] != x:
                    _parent[x] = _parent[_parent[x]]
                    x = _parent[x]
                return x
            def _union(a, b):
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    _parent[rb] = ra
            _file_to_cluster = {}
            for _ci, _c in enumerate(clusters):
                for _fi in _c:
                    _p = str(valid[_fi][0].path)
                    if _p in _file_to_cluster:
                        _union(_ci, _file_to_cluster[_p])
                    else:
                        _file_to_cluster[_p] = _ci
            _merged_map = {}
            for _ci, _c in enumerate(clusters):
                _merged_map.setdefault(_find(_ci), set()).update(_c)
            clusters = [sorted(s) for s in _merged_map.values()]
            clusters.sort(key=lambda c: -sum(
                {str(valid[i][0].path): valid[i][0].size for i in c}.values()))
            for cluster in clusters:
                items = []
                _seen = set()
                for i in cluster:
                    _p = str(valid[i][0].path)
                    if _p not in _seen:
                        _seen.add(_p)
                        items.append(valid[i][0])
                if len(items) < 2:
                    continue
                member = set(cluster)
                sel = [p for p in pairs if p[0] in member and p[1] in member]
                fine = max((p[3] for p in sel), default=0)
                ncc = min((p[5] for p in sel), default=1.0)
                sim = min((pair_similarity(p[3], p[5]) for p in sel),
                          default=0)
                group = DupGroup(
                    KIND_PHOTO, items, max(0, sim),
                    f"结构吻合 {ncc:.0%} · 细节差 {fine}/256")
                _bids = [burst_of.get(str(it.path)) for it in items]
                if len({b for b in _bids if b is not None}) < len(
                        [b for b in _bids if b is not None]):
                    group.note += "（连拍序列）"
                if group.similarity < SIMILARITY_AUTO_SELECT:
                    group.select_by_default = False
                    group.note += "（相似度低于 95%，默认不勾选）"
                group.keep = pick_keep(group.kind, items, keep_rule, root)
                groups.append(group)
                consumed |= {str(i.path) for i in items}
            say(f"相似照片：{len(clusters)} 组")
            if _bursts:
                _n = sum(len(b) for b in _bursts)
                say(f"连拍序列：{len(_bursts)} 组，共 {_n} 张")
            emit("photo", 1, 1, f"完成，发现 {len(clusters)} 组")
        else:
            emit("photo", 1, 1, "没有足够的照片")
    else:
        say("已跳过照片检测")
        emit("photo", 1, 1, "已跳过")

    # ---- 重复视频
    todo_videos = [f for f in videos if str(f.path) not in consumed]
    backend = video_backend()
    if options.videos and todo_videos and backend and not quick:
        emit("video", 0, len(todo_videos), "正在读取视频关键帧…")
        tag = CACHE_TAGS["video"]

        def video_work(item):
            hit = cache.get(item, tag)
            if hit:
                try:
                    return signature_from_cache(hit)
                except Exception as exc:
                    ignore(__name__, exc)
            sig = video_signature(item.path, 15, cancel=cancel)
            if sig:
                cache.put(item, tag, signature_to_cache(sig))
            return sig

        throttle = ProgressThrottle(
            lambda d, t, m: emit("video", d, t, f"视频关键帧 {d:,}/{t:,}"))
        sigs = parallel_map(video_work, todo_videos, max(2, options.workers // 2),
                            cancel=cancel,
                            on_progress=lambda d, t, m: throttle(d, t, m))
        for item, sig in zip(todo_videos, sigs):
            if sig:
                item.width, item.height = sig["w"], sig["h"]
                item.duration = sig["dur"]
        bad_videos = sum(1 for s in sigs if not s)
        say(f"视频解析完成：{len(todo_videos) - bad_videos:,} 个可用，"
            f"{bad_videos} 个跳过（后端：{backend}）")

        emit("video", 0, 1, "正在比对视频…")
        pairs = match_videos(
            sigs, dist=getattr(options, "vid_dist", 10),
            fine=getattr(options, "vid_fine", 96),
            ratio=getattr(options, "vid_ratio", 0.6), cancel=cancel,
            on_progress=lambda d, t, m: emit("video", d, t, "视频比对"))
        edges = [(a, b, fine) for a, b, _c, fine, _r in pairs]
        clusters = clusters_complete_link(len(sigs), edges)
        clusters.sort(key=lambda c: -sum(todo_videos[i].size for i in c))
        for cluster in clusters:
            items = [todo_videos[i] for i in cluster]
            member = set(cluster)
            sel = [p for p in pairs if p[0] in member and p[1] in member]
            ratio = max((p[4] for p in sel), default=0.0)
            fine = max((p[3] for p in sel), default=0.0)
            group = DupGroup(KIND_VIDEO, items, round(ratio * 100),
                             f"匹配 {ratio:.0%} 帧 · 位差 {fine:.0f}/256")
            if group.similarity < SIMILARITY_AUTO_SELECT:
                group.select_by_default = False
                group.note += "（相似度低于 95%，默认不勾选）"
            group.keep = pick_keep(group.kind, items, keep_rule, root)
            groups.append(group)
            consumed |= {str(i.path) for i in items}
        say(f"重复视频：{len(clusters)} 组")
        emit("video", 1, 1, f"完成，发现 {len(clusters)} 组")
    else:
        if options.videos and not backend:
            say("未找到 ffmpeg 或 OpenCV，已跳过视频检测")
            notes.append("视频检测需要 ffmpeg 或 OpenCV")
        else:
            say("已跳过视频检测")
        emit("video", 1, 1, "已跳过")

    # ---- 重复音乐
    todo_audio = [f for f in audios if str(f.path) not in consumed]
    if options.audio and todo_audio and not quick:
        emit("audio", 0, len(todo_audio), "正在读取音乐标签…")
        tag = CACHE_TAGS["audio"]

        def audio_work(item):
            hit = cache.get(item, tag)
            if hit and isinstance(hit, dict):
                return hit
            tags = read_tags(item.path)
            cache.put(item, tag, tags)
            return tags

        throttle = ProgressThrottle(
            lambda d, t, m: emit("audio", d, t, f"音乐标签 {d:,}/{t:,}"))
        tags_list = parallel_map(audio_work, todo_audio, options.workers,
                                 cancel=cancel,
                                 on_progress=lambda d, t, m: throttle(d, t, m))
        buckets: dict[str, list[FileItem]] = {}
        for item, tags in zip(todo_audio, tags_list):
            if not tags:
                bad_audio += 1
                continue
            item.audio_artist = tags.get("artist", "")
            item.audio_title = tags.get("title", "")
            item.audio_album = tags.get("album", "")
            item.duration = float(tags.get("duration") or 0.0)
            item.bitrate = float(tags.get("bitrate") or 0.0)
            key = audio_key(item)
            if key:
                buckets.setdefault(key, []).append(item)
            else:
                bad_audio += 1

        audio_groups = 0
        for key, items in buckets.items():
            if len(items) < 2:
                continue
            for chunk in cluster_audio(items):
                if len(chunk) > 1 and fingerprint_available():
                    fps = []
                    fp_throttle = ProgressThrottle(
                        lambda d, t, m: emit("audio", d, t,
                                           f"声学指纹 {d:,}/{t:,}"))
                    for i, it in enumerate(chunk):
                        check_cancel(cancel)
                        fps.append((it, chromaprint(it.path)))
                        fp_throttle(i + 1, len(chunk))
                    usable = [(it, fp) for it, fp in fps if fp]
                    if len(usable) >= 2:
                        base = usable[0][1][0]
                        far = max(fp_distance(base, fp[0]) for _, fp in usable[1:])
                        if far > 0.28:
                            say(f"声学指纹显示《{chunk[0].audio_title}》版本不同，跳过")
                            continue
                title = f"{chunk[0].audio_artist} - {chunk[0].audio_title}"
                verified = fingerprint_available()
                group = DupGroup(KIND_AUDIO,
                                 sorted(chunk, key=lambda x: str(x.path)),
                                 98 if len(chunk) > 2 else 99,
                                 f"标签相同：{title}"[:64],
                                 select_by_default=verified)
                if not verified:
                    group.similarity = 90
                    group.note = f"标签相同，未做声学比对（默认不勾选）：{title}"[:64]
                group.keep = pick_keep(group.kind, group.items, keep_rule, root)
                groups.append(group)
                audio_groups += 1
                consumed |= {str(i.path) for i in group.items}
        say(f"重复音乐：{audio_groups} 组"
            + (f"（{bad_audio} 个文件缺少歌手/标题标签，已跳过）"
               if bad_audio else "")
            + ("（已启用声学指纹复核）" if fingerprint_available() else ""))
        notes.append("音频声学指纹：" +
                     ("Chromaprint 可用" if fingerprint_available()
                      else "未安装 fpcalc，仅按标签判断"))
        emit("audio", 1, 1, f"完成，发现 {audio_groups} 组")
    else:
        say("已跳过音乐检测")
        emit("audio", 1, 1, "已跳过")

    # ---- 同名相似（默认不勾选，只作提示）
    if options.names and not quick:
        emit("name", 0, 1, "正在比对文件名…")
        todo_names = [f for f in files if str(f.path) not in consumed]
        name_groups, oversized, fuzzy_groups = find_name_groups(
            todo_names, fuzzy=bool(getattr(options, "names_fuzzy", False)),
            ratio=float(getattr(options, "name_ratio", 0.88)))
        for items in name_groups:
            group = DupGroup(KIND_NAME, items, 90,
                             "文件名相近，内容未校验（默认不勾选）"
                             if getattr(options, "names_fuzzy", False)
                             else "同名，内容未校验（默认不勾选）",
                             select_by_default=False)
            group.keep = pick_keep(group.kind, items, keep_rule, root)
            groups.append(group)
        extra = f"（{oversized} 组文件过多，已忽略）" if oversized else ""
        if fuzzy_groups:
            extra += f"（其中 {fuzzy_groups} 组来自文件名模糊匹配）"
        say(f"同名相似：{len(name_groups)} 组{extra}")
        emit("name", 1, 1, f"完成，发现 {len(name_groups)} 组")
    else:
        say("已跳过同名相似检测")
        emit("name", 1, 1, "已跳过")

    if cache is not None and cache.enabled:
        say(f"指纹缓存：复用 {cache.hits:,} 条，新算 {cache.misses:,} 条")

    groups.sort(key=lambda g: (KIND_ORDER.index(g.kind)
                               if g.kind in KIND_ORDER else 9, -g.wasted))
    skipped_unreadable += len(exact_stats.get("unreadable", ()))
    if skipped_unreadable:
        say(f"扫描中共 {skipped_unreadable} 个文件无法读取，已跳过")
    if skipped_empty:
        say(f"{skipped_empty} 个 0 字节文件，已跳过")
    result = ScanResult(
        root=root, groups=groups, file_count=len(files),
        total_size=total_size, elapsed=time.perf_counter() - started,
        bad_photos=bad_photos, bad_videos=bad_videos, bad_audio=bad_audio,
        skipped_inodes=skipped_inodes,
        skipped_unreadable=skipped_unreadable,
        skipped_empty=skipped_empty, tiny_photos=tiny_photos,
        video_backend=backend, notes=notes)
    say(f"扫描完成：{result.total_groups} 组，可释放 "
        f"{human_size(result.total_wasted)}，耗时 {result.elapsed:.1f} 秒")
    return result


__all__ = [
    "AUDIO_EXT",
    "CACHE_TAGS",
    "KEEP_RULES",
    "KEEP_RULE_LABELS",
    "KIND_AUDIO",
    "KIND_DIR",
    "KIND_EXACT",
    "KIND_NAME",
    "KIND_ORDER",
    "KIND_PHOTO",
    "KIND_VIDEO",
    "PHOTO_EXT",
    "PRESETS",
    "SKIP_DIRS",
    "SKIP_PREFIX",
    "VIDEO_EXT",
    "DupGroup",
    "FileItem",
    "Image",
    "ImageDraw",
    "ImageOps",
    "ScanCancelled",
    "ScanOptions",
    "ScanResult",
    "_full_hash",
    "_pick_keep",
    "_quick_hash",
    "_sample_hash",
    "app_dir",
    "audio_fingerprint_available",
    "cv2",
    "export_report",
    "export_report_json",
    "find_duplicate_dirs",
    "find_exact",
    "format_date",
    "format_duration",
    "hardlink_replace",
    "human_size",
    "is_audio",
    "is_photo",
    "is_video",
    "make_thumbnail",
    "name_key",
    "norm_path",
    "np",
    "permanent_delete",
    "recycle_capable",
    "recycle_paths",
    "verify_recycled",
    "restore_from_recycle",
    "restore_hardlinks",
    "scan_folder",
    "should_skip_dir",
    "video_available",
]
