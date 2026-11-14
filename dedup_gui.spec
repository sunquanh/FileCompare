# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：单文件、无控制台。

    python -m PyInstaller dedup_gui.spec --noconfirm

产物在 dist\\Dedup.exe，双击即可运行，不需要本机装 Python。
"""
from PyInstaller.utils.hooks import collect_submodules

# onnxruntime 是可选的：没装就跳过（去水印还有不靠模型的镜像填充可用）
try:
    from PyInstaller.utils.hooks import collect_dynamic_libs
    # 只拿运行时需要的几个模块：collect_submodules 会拖进 onnx 的开发工具，
    # 它们依赖 pandas/fsspec，能把打包时间拉长十倍、体积胀一大圈。
    ort_modules = ["onnxruntime", "onnxruntime.capi",
                   "onnxruntime.capi._pybind_state",
                   "onnxruntime.capi.onnxruntime_pybind11_state"]
    ort_bins = collect_dynamic_libs("onnxruntime")
except Exception:  # pragma: no cover
    ort_modules = []
    ort_bins = []

hiddenimports = (
    collect_submodules("PIL")
    + ort_modules
    + ["numpy", "onnxruntime", "tkinter", "tkinter.ttk", "sqlite3",
       "dedup_core", "dedup_models", "dedup_cache", "dedup_exact",
       "dedup_photos", "dedup_photo_embed", "dedup_names", "dedup_video",
       "dedup_audio", "dedup_files", "dedup_tools", "dedup_watermark",
       "dedup_watermark_lama", "ui_kit"]
)

a = Analysis(
    ["dedup_gui.py"],
    pathex=[],
    binaries=ort_bins,
    # 模型不内嵌：198 MB 打进 exe 会变成每次启动都解包，改成放在 exe 同级 models\ 里
    datas=[("使用说明.txt", ".")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 这些是 onnxruntime 的“分析连带”拉进来的重量级包，运行时不靠它们：
    # 不排掉会把 torch / sympy / pandas 一起打包，体积和打包时间都会爆。
    excludes=["pytest", "cv2", "torch", "torchvision", "sympy", "pandas",
              "scipy", "sklearn", "matplotlib", "lxml", "fsspec",
              "urllib3", "requests", "networkx", "onnx", "tqdm",
              "IPython", "jupyter", "numba", "ipykernel"],
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
