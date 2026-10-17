"""Run provenance and backup regressions; no model or physical-performance claims."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from inspection_core import Settings
from inspection_store import (assert_run, create_backup, digest, fail_stage,
                              load_latest, load_stage, new_run, read_json,
                              restore_backup, seal_stage, start_stage)


PROJECT_SOURCE = Path(__file__).resolve().parents[1]


def copy_project(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for source in PROJECT_SOURCE.glob("*.py"):
        shutil.copy2(source, destination / source.name)
    for name in ("requirements.txt", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "VALIDATION.md"):
        source = PROJECT_SOURCE / name
        if source.exists():
            shutil.copy2(source, destination / name)
    (destination / "tests").mkdir(exist_ok=True)
    for source in (PROJECT_SOURCE / "tests").glob("*.py"):
        shutil.copy2(source, destination / "tests" / source.name)
    return destination


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = copy_project(self.base / "project")
        self.settings = Settings()
        self.run = new_run(self.root, self.settings)

    def completed_stage(self, name, payload=None):
        stage = start_stage(self.run, name, digest({"contract": 1}), digest({"plans": 1}))
        seal_stage(self.run, name, stage, payload or {"value": name})
        return stage

    def test_unchanged_run_is_reusable(self):
        manifest = assert_run(self.root, self.run, self.settings)
        self.assertEqual(manifest["run_id"], self.run.name)
        loaded_settings, loaded_run = load_latest(self.root)
        self.assertEqual(loaded_run, self.run)
        self.assertEqual(loaded_settings, self.settings)

    def test_each_run_has_a_unique_directory(self):
        self.assertNotEqual(new_run(self.root, self.settings), self.run)

    def test_settings_change_requires_new_run(self):
        for changed in (replace(self.settings, user_instruction="Aだけを点検して。"),
                        replace(self.settings, stale_threshold=1.0),
                        replace(self.settings, model_revision="a" * 40)):
            with self.subTest(settings=changed), self.assertRaises(ValueError):
                assert_run(self.root, self.run, changed)

    def test_live_source_edit_requires_new_run(self):
        path = self.root / "inspection_core.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            assert_run(self.root, self.run, self.settings)

    def test_new_run_cannot_snapshot_code_different_from_imported_modules(self):
        path = self.root / "inspection_core.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# not loaded into Python\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            new_run(self.root, self.settings)

    def test_snapshot_edit_is_detected(self):
        path = self.run / "source" / "inspection_core.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            assert_run(self.root, self.run, self.settings)

    def test_manifest_edit_is_detected(self):
        path = self.run / "manifest.json"
        data = read_json(path)
        data["settings"]["seed"] += 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            assert_run(self.root, self.run, self.settings)

    def test_redundant_settings_record_cannot_disagree_with_manifest(self):
        path = self.run / "settings.json"
        data = read_json(path)
        data["stale_threshold"] = 1.0
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            assert_run(self.root, self.run, self.settings)

    def test_environment_record_edit_is_detected(self):
        path = self.run / "environment.json"
        data = read_json(path)
        data["mujoco"] = "synthetic_changed_version"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            assert_run(self.root, self.run, self.settings)

    def test_changed_torch_version_blocks_computation_but_allows_reading(self):
        from run_project import generate_plans
        changed = dict(read_json(self.run / "environment.json"), torch="synthetic_changed_version")
        with patch("inspection_store.runtime_environment", return_value=changed):
            # A restored run can still be inspected without altering its evidence.
            loaded_settings, loaded_run = load_latest(self.root)
            self.assertEqual((loaded_settings, loaded_run), (self.settings, self.run))
            # Resume must stop before fetching a model or mixing new inference.
            with patch("inspection_ai.resolve_contract") as resolver:
                with self.assertRaisesRegex(ValueError, "torch"):
                    generate_plans(self.root, self.run, self.settings)
                resolver.assert_not_called()

    def test_new_attempt_invalidates_old_and_downstream_results(self):
        old = self.completed_stage("evaluation")
        self.completed_stage("comparison")
        self.completed_stage("demonstrations")
        self.completed_stage("publication")
        stage = start_stage(self.run, "evaluation", "contract", "plans")
        self.assertNotEqual(stage, old)
        self.assertTrue((old / "stage.json").exists(), "History should remain for audit")
        for name in ("evaluation", "comparison", "demonstrations", "publication"):
            with self.subTest(stage=name), self.assertRaises(ValueError):
                load_stage(self.run, name)
        self.assertEqual(read_json(stage / "stage.json")["status"], "running")

    def test_failed_attempt_cannot_reuse_previous_success(self):
        old = self.completed_stage("evaluation", {"success": True})
        current = start_stage(self.run, "evaluation", "contract", "plans")
        fail_stage(self.run, "evaluation", current, OSError("diagnostic image failure"))
        self.assertEqual(read_json(current / "stage.json")["status"], "failed")
        self.assertEqual(read_json(old / "stage.json")["payload"], {"success": True})
        with self.assertRaises(ValueError):
            load_stage(self.run, "evaluation")

    def test_keyboard_interrupt_is_explicit(self):
        stage = start_stage(self.run, "evaluation", "contract", "plans")
        fail_stage(self.run, "evaluation", stage, KeyboardInterrupt())
        self.assertEqual(read_json(stage / "stage.json")["status"], "interrupted")
        with self.assertRaises(ValueError):
            load_stage(self.run, "evaluation")

    def test_cannot_seal_old_or_completed_attempt(self):
        old = start_stage(self.run, "evaluation", "contract", "plans")
        current = start_stage(self.run, "evaluation", "contract", "plans")
        with self.assertRaises(ValueError):
            seal_stage(self.run, "evaluation", old, {"success": True})
        seal_stage(self.run, "evaluation", current, {"success": False})
        with self.assertRaises(ValueError):
            seal_stage(self.run, "evaluation", current, {"success": True})

    def test_completed_stage_tamper_is_detected(self):
        stage = self.completed_stage("evaluation", {"success": False})
        path = stage / "stage.json"
        data = read_json(path)
        data["payload"] = {"success": True}
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_stage(self.run, "evaluation")

    def test_backup_roundtrip_restores_run_source_results_and_entrypoint(self):
        contract = {"model_id": self.settings.model_id, "revision": "a" * 40,
                    "source": "diagnostic_fixture"}
        (self.run / "planner_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        self.completed_stage("evaluation", {"success": False, "source": "diagnostic_fixture"})
        archive = create_backup(self.root, self.run, self.base / "backup.zip")
        restored_root = self.base / "restored"
        restored = restore_backup(restored_root, archive)
        self.assertEqual(read_json(restored / "planner_contract.json"), contract)
        self.assertEqual(read_json(restored / "manifest.json"), read_json(self.run / "manifest.json"))
        self.assertEqual(load_stage(restored, "evaluation")["payload"]["success"], False)
        settings, selected_run = load_latest(restored_root)
        self.assertEqual(settings, self.settings)
        self.assertEqual(selected_run, restored)
        assert_run(restored_root, restored, settings)

    def test_backup_rejects_traversal_absolute_and_symlink_entries(self):
        for i, name in enumerate(("../escape.txt", "/escape.txt", "C:/escape.txt", "run/..\\escape.txt")):
            archive = self.base / f"bad-{i}.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, "test")
            with self.subTest(name=name), self.assertRaises(ValueError):
                restore_backup(self.base / "restored", archive)
        symlink_archive = self.base / "symlink.zip"
        with zipfile.ZipFile(symlink_archive, "w") as z:
            entry = zipfile.ZipInfo("link")
            entry.create_system = 3
            entry.external_attr = 0o120777 << 16
            z.writestr(entry, "../escape")
        with self.assertRaises(ValueError):
            restore_backup(self.base / "restored", symlink_archive)

    def test_corrupt_backup_is_rejected_before_extracting(self):
        archive = create_backup(self.root, self.run, self.base / "good.zip")
        corrupt = self.base / "corrupt.zip"
        with zipfile.ZipFile(archive) as src, zipfile.ZipFile(corrupt, "w") as dest:
            for name in src.namelist():
                data = src.read(name)
                if name == "run/settings.json":
                    data += b"\n "
                dest.writestr(name, data)
        restored_root = self.base / "restored"
        with self.assertRaises(ValueError):
            restore_backup(restored_root, corrupt)
        self.assertFalse((restored_root / "artifacts" / self.run.name).exists())


if __name__ == "__main__":
    unittest.main()
