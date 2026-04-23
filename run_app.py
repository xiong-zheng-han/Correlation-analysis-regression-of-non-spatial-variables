"""Entry point script for PyInstaller packaging."""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from correlation_analyzer.main import main

if __name__ == "__main__":
    main()
