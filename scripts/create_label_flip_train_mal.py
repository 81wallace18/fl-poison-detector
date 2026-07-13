#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "tools/create_label_flip_train_mal.py"),
    run_name="__main__",
)
