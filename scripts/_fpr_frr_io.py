"""Compatibility import for the analysis helper moved to scripts/tools."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "tools/_fpr_frr_io.py"
_SPEC = importlib.util.spec_from_file_location("_fpr_frr_io_impl", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

load_fpr_frr = _MODULE.load_fpr_frr
normalize_columns = _MODULE.normalize_columns
summarize_fpr_frr = _MODULE.summarize_fpr_frr

__all__ = ["load_fpr_frr", "normalize_columns", "summarize_fpr_frr"]
