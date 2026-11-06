"""一键打包：用 PyInstaller 按 dedup_gui.spec 打出 Windows 便携版。

    python tools/build_release.py

产物在 dist\\Dedup.exe，双击即可运行，不需要本机装 Python。
只做打包，不上传、不发版；发版走 .github/workflows/release.yml。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    spec = ROOT / "dedup_gui.spec"
    if not spec.exists():
        print(f"找不到打包配置：{spec}")
        return 1
    cmd = [sys.executable, "-m", "PyInstaller", str(spec), "--noconfirm"]
    print("执行：" + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("打包失败")
        return result.returncode
    exe = ROOT / "dist" / "Dedup.exe"
    if exe.exists():
        print(f"打包完成：{exe}（{exe.stat().st_size / 1048576:.1f} MB）")
    else:
        print("PyInstaller 退出了 0，但没找到 dist\\Dedup.exe")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
