from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = ROOT / "scripts/_check_runtime_sync.py"
MIRRORED_FILES = ("cc_mlp.py", "context_features.py", "features.py", "fl_save.py")


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
                    str(ROOT / "scripts/create_label_flip_train_mal.py"),
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

        path = ROOT / "scripts/_fpr_frr_io.py"
        spec = importlib.util.spec_from_file_location("fpr_io_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame = pd.DataFrame({"Round": [1], "UploadFPR": [0.1], "FPR": [0.2]})
        normalized = module.normalize_columns(frame)
        self.assertIn("DetectionFPR", normalized.columns)
        self.assertIn("QuarantineFPR", normalized.columns)


class WorkflowCliTest(unittest.TestCase):
    def run_script(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / script), *args],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

    def test_dry_run_profiles(self) -> None:
        for script in ("run_full.sh", "rerun_cc7.sh"):
            for profile, dataset in (("mnist", "MNIST"), ("cifar10", "Cifar10")):
                result = self.run_script(script, profile, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"profile={profile}", result.stdout)
                self.assertIn(f"dataset={dataset}", result.stdout)

    def test_profile_is_required_and_validated(self) -> None:
        for script in ("run_full.sh", "rerun_cc7.sh"):
            missing = self.run_script(script)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("Uso:", missing.stderr)
            invalid = self.run_script(script, "unknown", "--dry-run")
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("Perfil invalido", invalid.stderr)


class NotebookHygieneTest(unittest.TestCase):
    def test_only_active_notebook_is_clean(self) -> None:
        import json

        notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual([path.name for path in notebooks], ["notebook_monza_analysis.ipynb"])
        notebook = json.loads(notebooks[0].read_text(encoding="utf-8"))
        source = "".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertNotIn("DistilBERT", source)
        self.assertNotIn("cc=6", source)
        self.assertNotIn("/home/", source)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
