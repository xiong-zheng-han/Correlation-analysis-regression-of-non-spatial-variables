"""Build script for packaging the correlation analyzer with PyInstaller."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Build the application."""
    print("=" * 60)
    print("Building Correlation Analyzer")
    print("=" * 60)

    # Check if pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        run_command([sys.executable, "-m", "uv", "add", "--dev", "pyinstaller"])
        run_command([sys.executable, "-m", "uv", "sync"])

    # Build with PyInstaller
    print("\nBuilding with PyInstaller...")
    spec_file = Path(__file__).parent / "build.spec"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        str(spec_file)
    ]

    exit_code = run_command(cmd)

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("Build successful!")
        print("Executable location: dist/相关性分析工具.exe")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Build failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
