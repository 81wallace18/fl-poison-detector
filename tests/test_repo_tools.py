from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = ROOT / "scripts/check_runtime_sync.py"
ARCHIVE_SCRIPT = ROOT / "scripts/tools/archive_results.py"
MIRRORED_FILES = ("cc.py", "cc_mlp.py", "context_features.py", "features.py", "fl_save.py")


class RuntimeSyncTest(unittest.TestCase):
    def test_reports_matching_and_divergent_copies(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source = temp_path / "source"
            runtime = temp_path / "runtime"
            source.mkdir()
            runtime.mkdir()
            for filename in MIRRORED_FILES:
                (source / filename).write_text(f"# {filename}\n", encoding="utf-8")
                shutil.copy2(source / filename, runtime / filename)

            command = [
                sys.executable,
                str(SYNC_SCRIPT),
                "--source-dir",
                str(source),
                "--runtime-dir",
                str(runtime),
            ]
            self.assertEqual(subprocess.run(command, check=False).returncode, 0)
            (runtime / "features.py").write_text("# changed\n", encoding="utf-8")
            self.assertEqual(subprocess.run(command, check=False).returncode, 1)


class ArchiveResultsTest(unittest.TestCase):
    def test_archives_and_hardlinks_duplicate_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            (temp_path / "first").mkdir()
            (temp_path / "second").mkdir()
            (temp_path / "first/result.csv").write_text("same\n", encoding="utf-8")
            (temp_path / "second/result.csv").write_text("same\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ARCHIVE_SCRIPT),
                    "--archive-dir",
                    "archive",
                    "first",
                    "second",
                ],
                cwd=temp_path,
                check=True,
            )

            first = temp_path / "archive/first/result.csv"
            second = temp_path / "archive/second/result.csv"
            self.assertEqual(os.stat(first).st_ino, os.stat(second).st_ino)
            manifest = json.loads(
                (temp_path / "archive/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["deduplicated"]), 1)


class ScientificHelpersTest(unittest.TestCase):
    def test_label_flip_creates_expected_labels(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp)
            train = dataset / "train"
            train.mkdir()
            np.savez_compressed(
                train / "0.npz",
                data={"x": np.asarray([[1.0]]), "y": np.asarray([0, 4, 9])},
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/tools/create_label_flip_train_mal.py"),
                    "--dataset-dir",
                    str(dataset),
                ],
                check=True,
            )
            loaded = np.load(dataset / "train_mal/0.npz", allow_pickle=True)
            labels = loaded["data"].tolist()["y"].tolist()
            self.assertEqual(labels, [9, 5, 0])

    def test_fpr_csv_legacy_columns_are_normalized(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        import importlib.util

        path = ROOT / "scripts/tools/_fpr_frr_io.py"
        spec = importlib.util.spec_from_file_location("fpr_io_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame = pd.DataFrame({"Round": [1], "UploadFPR": [0.1], "FPR": [0.2]})
        normalized = module.normalize_columns(frame)
        self.assertIn("DetectionFPR", normalized.columns)
        self.assertIn("QuarantineFPR", normalized.columns)


if __name__ == "__main__":
    unittest.main()
