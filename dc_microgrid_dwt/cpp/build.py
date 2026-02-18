#!/usr/bin/env python3
"""
Build script for the C++ DSP module.

Usage:
    python cpp/build.py

This compiles the C++ DSP core into a Python-importable shared library
(microgrid_dsp.so) and places it in the project root.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def find_pybind11_cmake():
    """Find pybind11's CMake directory."""
    try:
        import pybind11
        return pybind11.get_cmake_dir()
    except ImportError:
        print("ERROR: pybind11 not installed. Run: pip install pybind11")
        sys.exit(1)


def build():
    # Paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    build_dir = script_dir / "build"

    print("=" * 60)
    print("  DC Microgrid DSP Core — C++ Build")
    print("=" * 60)

    # Clean previous build
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    # Get pybind11 CMake path
    pybind11_dir = find_pybind11_cmake()
    print(f"  pybind11 CMake: {pybind11_dir}")

    # Configure
    print("\n[1/3] Configuring with CMake...")
    cmake_args = [
        "cmake",
        str(script_dir),
        f"-Dpybind11_DIR={pybind11_dir}",
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
    ]
    result = subprocess.run(cmake_args, cwd=str(build_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CMake configure FAILED:\n{result.stderr}")
        sys.exit(1)
    print("  Configure OK")

    # Build
    print("\n[2/3] Compiling C++ DSP core...")
    build_args = ["cmake", "--build", ".", "--config", "Release", "-j"]
    result = subprocess.run(build_args, cwd=str(build_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Build FAILED:\n{result.stderr}")
        sys.exit(1)
    print("  Compile OK")

    # Copy .so to project root
    print("\n[3/3] Installing module...")
    so_files = list(build_dir.glob("microgrid_dsp*.so")) + list(build_dir.glob("microgrid_dsp*.pyd"))
    if not so_files:
        print("  ERROR: No .so/.pyd file found after build")
        sys.exit(1)

    dest = project_root / so_files[0].name
    shutil.copy2(so_files[0], dest)
    print(f"  Installed: {dest}")

    # Verify import
    print("\n[✓] Verifying import...")
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); import microgrid_dsp as d; "
         "p = d.create_default_pipeline(); "
         "r = p.process_sample(380.0); "
         "print(f'  Pipeline OK: {r}')"],
        cwd=str(project_root), capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Import test FAILED:\n{result.stderr}")
    else:
        print(result.stdout.strip())

    print("\n" + "=" * 60)
    print("  Build complete! Module ready at:")
    print(f"  {dest}")
    print("=" * 60)


if __name__ == "__main__":
    build()
