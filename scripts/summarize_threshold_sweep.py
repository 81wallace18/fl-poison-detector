#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

tools_dir = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(tools_dir))
runpy.run_path(str(tools_dir / "summarize_threshold_sweep.py"), run_name="__main__")
