# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：单文件、无控制台。

    python -m PyInstaller dedup_gui.spec --noconfirm

产物在 dist\\Dedup.exe，双击即可运行，不需要本机装 Python。
"""
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    collect_submodules("PIL")
    + ["numpy", "tkinter", "tkinter.ttk", "sqlite3",
       "dedup_core", "dedup_models", "dedup_cache", "dedup_exact",
       "dedup_photos", "dedup_names", "dedup_video", "dedup_audio",
       "dedup_files", "dedup_tools", "ui_kit"]
)

a = Analysis(
    ["dedup_gui.py"],
    pathex=[],
    binaries=[],
    datas=[("使用说明.txt", ".")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "cv2"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Dedup",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
