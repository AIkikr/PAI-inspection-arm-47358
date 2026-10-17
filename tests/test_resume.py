"""Resume storage regressions using synthetic wire-format fixtures only.

These fixtures perform no model inference, physics or media generation. Their
temporary outputs are never submitted as project measurements.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from inspection_core import parse_and_validate, save_json
from inspection_store import (create_backup, digest, file_hash, load_latest,
                              load_stage, read_json, restore_backup)
from run_project import evaluate, compare_freshness_guard, record_demonstrations
import test_project


def resumable_fixture(settings, plan, directory, **kwargs):
    """Imitate sealed episode files so hash validation remains under test."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "plan.json", plan)
    _, check = parse_and_validate(plan["raw_output"], settings)
    artifacts = {"plan.json": file_hash(directory / "plan.json")}
    captures = []
    if check.accepted:
        image = directory / "capture_synthetic.png"
        image.write_bytes(b"SYNTHETIC HASH FIXTURE; NOT AN IMAGE")
        artifacts[image.name] = file_hash(image)
        captures.append(image.name)
    if kwargs.get("record"):
        clip = directory / "clip.mp4"
        clip.write_bytes(b"SYNTHETIC HASH FIXTURE; NOT A VIDEO")
        artifacts[clip.name] = file_hash(clip)
    result = {
        "status": "complete" if check.accepted else "rejected",
        # Mimic the runtime schema here; the fixture is not public evidence.
        "diagnostic": False, "accepted": check.accepted,
        "planner_source": plan.get("source"),
        "validation_reason": check.reason,
        "planned_targets": list(check.targets),
        "completed_targets": list(check.targets),
        "success": check.accepted, "failure": "", "sim_duration_s": 8.0,
        "captures": captures, "artifact_hashes": artifacts,
        "stale_moving_s": 0.0, "max_hold_drift_m": 0.0,
    }
    if kwargs.get("allow_reference"):
        from inspection_reference import reference_metadata
        result.update(reference_metadata(plan))
    save_json(directory / "result.json", result)
    return result


def terminal_failure_fixture(settings, plan, directory, **kwargs):
    result = resumable_fixture(settings, plan, directory, **kwargs)
    result.update(status="failed", success=False, completed_targets=[],
                  failure="deliberate fixture failure")
    save_json(Path(directory) / "result.json", result)
    return result


class ResumeTests(unittest.TestCase):
    setUp = test_project.ProjectTests.setUp
    plans = test_project.ProjectTests.plans

    def interrupt_after(self, count, *, terminal_failure=False):
        calls = []

        def execute(settings, plan, directory, **kwargs):
            calls.append(Path(directory))
            if len(calls) == count + 1:
                directory = Path(directory)
                directory.mkdir(parents=True, exist_ok=False)
                (directory / "capture_partial.png").write_bytes(b"UNCOMMITTED")
                save_json(directory / "result.json", {
                    "status": "interrupted", "success": False,
                    "failure": "deliberate interruption", "captures": [],
                    "artifact_hashes": {},
                })
                raise KeyboardInterrupt("deliberate resume fixture interruption")
            fn = terminal_failure_fixture if terminal_failure and len(calls) == 1 else resumable_fixture
            return fn(settings, plan, directory, **kwargs)

        return calls, execute

    def partial(self, fn, count, **kwargs):
        plans = self.plans()
        calls, execute = self.interrupt_after(count, **kwargs)
        with patch("inspection_sim.run_episode", side_effect=execute), self.assertRaises(KeyboardInterrupt):
            fn(self.root, self.run, self.settings, plans, resume=True)
        return plans, calls

    def test_evaluation_resume_reuses_only_sealed_prefix(self):
        plans, first_calls = self.partial(evaluate, 3)
        previous = [read_json(path / "result.json") for path in first_calls[:3]]
        previous_hashes = [file_hash(path / "result.json") for path in first_calls[:3]]
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as sim:
            payload = evaluate(self.root, self.run, self.settings, plans, resume=True)
        self.assertEqual(sim.call_count, 21)
        self.assertEqual(len(payload["records"]), 24)
        for index, row in enumerate(payload["records"][:3]):
            self.assertEqual(self.run / row["result_path"], first_calls[index] / "result.json")
            self.assertEqual(row["result_sha256"], previous_hashes[index])
            self.assertEqual(read_json(self.run / row["result_path"]), previous[index])
        interrupted_result = first_calls[3] / "result.json"
        self.assertNotIn(interrupted_result, [self.run / r["result_path"] for r in payload["records"]])
        self.assertEqual(read_json(interrupted_result)["status"], "interrupted")

    def test_terminal_failure_is_retained_instead_of_retrying_until_success(self):
        plans, first_calls = self.partial(evaluate, 1, terminal_failure=True)
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as sim:
            payload = evaluate(self.root, self.run, self.settings, plans, resume=True)
        self.assertEqual(sim.call_count, 23)
        row = payload["records"][0]
        self.assertFalse(row["success"])
        self.assertEqual(row["failure"], "deliberate fixture failure")
        self.assertEqual(self.run / row["result_path"], first_calls[0] / "result.json")
        self.assertEqual(payload["summary"]["runtime_failure_n"], 1)

    def test_completed_resume_does_not_execute_any_episode(self):
        plans = self.plans()
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture):
            first = evaluate(self.root, self.run, self.settings, plans, resume=True)
        first_stage = load_stage(self.run, "evaluation")
        with patch("inspection_sim.run_episode", side_effect=AssertionError("completed episode must stay cached")):
            second = evaluate(self.root, self.run, self.settings, plans, resume=True)
        self.assertEqual(first, second)
        self.assertEqual(load_stage(self.run, "evaluation")["_path"], first_stage["_path"])

    def test_fresh_attempt_does_not_fall_back_to_old_success_on_failure(self):
        plans = self.plans()
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture):
            evaluate(self.root, self.run, self.settings, plans, resume=True)
        previous = load_stage(self.run, "evaluation")
        with patch("inspection_sim.run_episode", side_effect=OSError("fresh failure")) as sim, self.assertRaises(OSError):
            evaluate(self.root, self.run, self.settings, plans, resume=False)
        self.assertEqual(sim.call_count, 1)
        with self.assertRaises(ValueError):
            load_stage(self.run, "evaluation")
        self.assertTrue((self.run / previous["_path"]).exists())

    def test_changed_cached_result_is_rejected_before_resuming(self):
        plans, calls = self.partial(evaluate, 2)
        result = read_json(calls[0] / "result.json")
        result["success"] = False
        save_json(calls[0] / "result.json", result)
        with patch("inspection_sim.run_episode", side_effect=AssertionError("corruption must block")), self.assertRaises(ValueError):
            evaluate(self.root, self.run, self.settings, plans, resume=True)

    def test_missing_cached_capture_is_rejected_before_resuming(self):
        plans, calls = self.partial(evaluate, 2)
        (calls[0] / "capture_synthetic.png").unlink()
        with patch("inspection_sim.run_episode", side_effect=AssertionError("missing evidence must block")), self.assertRaises(ValueError):
            evaluate(self.root, self.run, self.settings, plans, resume=True)

    def test_completed_resume_still_verifies_cached_artifacts(self):
        plans = self.plans()
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture):
            payload = evaluate(self.root, self.run, self.settings, plans, resume=True)
        first = self.run / payload["records"][0]["result_path"]
        (first.parent / "capture_synthetic.png").write_bytes(b"CORRUPTED")
        with patch("inspection_sim.run_episode", side_effect=AssertionError("corruption must block")), self.assertRaises(ValueError):
            evaluate(self.root, self.run, self.settings, plans, resume=True)

    def test_restore_backup_then_resume_keeps_completed_prefix(self):
        plans, calls = self.partial(evaluate, 4)
        archive = create_backup(self.root, self.run, self.root.parent / "resume_backup.zip")
        restored_root = self.root.parent / "restored"
        restored_run = restore_backup(restored_root, archive)
        settings, latest = load_latest(restored_root)
        self.assertEqual(latest, restored_run)
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as sim:
            payload = evaluate(restored_root, restored_run, settings, plans, resume=True)
        self.assertEqual(sim.call_count, 20)
        for index, row in enumerate(payload["records"][:4]):
            original = calls[index] / "result.json"
            self.assertEqual(row["result_path"], original.relative_to(self.run).as_posix())
            self.assertEqual(file_hash(restored_run / row["result_path"]), file_hash(original))

    def test_reordered_checkpoint_prefix_is_rejected_even_when_rehashed(self):
        plans, _ = self.partial(evaluate, 3)
        current = read_json(self.run / "current.json")["evaluation"]
        path = self.run / current["path"] / "progress.json"
        progress = read_json(path)
        progress["records"][0], progress["records"][1] = progress["records"][1], progress["records"][0]
        progress["content_hash"] = digest({k: v for k, v in progress.items() if k != "content_hash"})
        save_json(path, progress)
        with patch("inspection_sim.run_episode", side_effect=AssertionError("prefix mismatch must block")), self.assertRaises(ValueError):
            evaluate(self.root, self.run, self.settings, plans, resume=True)

    def test_comparison_resume_runs_only_remaining_conditions(self):
        from inspection_reference import make_control_reference
        plans, calls = self.partial(compare_freshness_guard, 5)
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as sim:
            payload = compare_freshness_guard(self.root, self.run, self.settings, plans, resume=True)
        self.assertEqual(sim.call_count, 7)
        self.assertEqual(len(payload["records"]), 12)
        self.assertEqual(payload["reference_plan"], make_control_reference())
        self.assertTrue(all(row["planner_source"] == "control_reference" for row in payload["records"]))
        self.assertEqual(self.run / payload["records"][0]["result_path"], calls[0] / "result.json")
        for scenario in ("normal", "outage", "delayed", "reordered"):
            rows = [r for r in payload["records"] if r["scenario"] == scenario]
            self.assertEqual({r["mode"] for r in rows}, {"off", "receipt", "age"})
            self.assertEqual(len({r["trace_hash"] for r in rows}), 1)

    def test_completed_control_comparison_resume_reuses_fixed_reference(self):
        plans = self.plans()
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture):
            first = compare_freshness_guard(self.root, self.run, self.settings, plans, resume=True)
        with patch("inspection_sim.run_episode", side_effect=AssertionError("completed comparison must stay cached")):
            second = compare_freshness_guard(self.root, self.run, self.settings, plans, resume=True)
        self.assertEqual(first, second)
        self.assertFalse(second["ai_accuracy_evaluated"])

    def test_demonstration_resume_reuses_completed_clips(self):
        plans, calls = self.partial(record_demonstrations, 2)
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as sim:
            payload = record_demonstrations(self.root, self.run, self.settings, plans, resume=True)
        self.assertEqual(sim.call_count, 2)
        self.assertEqual(len(payload["records"]), 4)
        self.assertEqual(self.run / payload["records"][0]["result_path"], calls[0] / "result.json")
        self.assertEqual(self.run / payload["records"][0]["clip_path"], calls[0] / "clip.mp4")

    def test_checkpoint_is_called_after_each_completed_evaluation(self):
        plans = self.plans()
        events = []
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture):
            evaluate(self.root, self.run, self.settings, plans, resume=True, checkpoint=events.append)
        self.assertGreaterEqual(len(events), 26, "start, each of 24 episodes, and terminal checkpoint")
        self.assertTrue(all(isinstance(event, str) and event for event in events))

    def test_checkpoint_upload_failure_stops_work_and_keeps_local_prefix(self):
        plans = self.plans()
        events = []

        def failed_upload(event):
            events.append(event)
            if event == "evaluation:n02":
                raise OSError("synthetic persistent-storage upload failure")

        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as sim, self.assertRaises(OSError):
            evaluate(self.root, self.run, self.settings, plans, resume=True, checkpoint=failed_upload)
        self.assertEqual(sim.call_count, 2, "Save failure must stop before the next episode")
        current = read_json(self.run / "current.json")["evaluation"]
        self.assertEqual(current["status"], "failed")
        progress = read_json(self.run / current["path"] / "progress.json")
        self.assertEqual([r["id"] for r in progress["records"]], ["n01", "n02"])
        sealed_paths = [r["result_path"] for r in progress["records"]]
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as resumed:
            payload = evaluate(self.root, self.run, self.settings, plans, resume=True)
        self.assertEqual(resumed.call_count, 22)
        self.assertEqual([r["result_path"] for r in payload["records"][:2]], sealed_paths)
        self.assertEqual(events, ["evaluation:start", "evaluation:n01", "evaluation:n02"])

    def test_persistent_checkpoint_restore_in_new_root_resumes_remaining_work(self):
        from inspection_checkpoint import latest_checkpoint, restore_checkpoint, save_checkpoint

        plans = self.plans()
        vault = self.root.parent / "synthetic_persistent_vault"
        self.assertIsNone(latest_checkpoint(vault))

        def persist(event):
            save_checkpoint(self.root, self.run, vault, event, keep=3)

        calls, execute = self.interrupt_after(3)
        with patch("inspection_sim.run_episode", side_effect=execute), self.assertRaises(KeyboardInterrupt):
            evaluate(self.root, self.run, self.settings, plans, resume=True, checkpoint=persist)
        receipt = latest_checkpoint(vault)
        self.assertEqual(receipt["event"], "evaluation:n03")
        restored_root = self.root.parent / "fresh_runtime_root"
        settings, restored_run, restored_receipt = restore_checkpoint(restored_root, vault)
        self.assertEqual(restored_receipt["checkpoint_id"], receipt["checkpoint_id"])
        self.assertEqual(restored_run.name, self.run.name)
        current = read_json(restored_run / "current.json")["evaluation"]
        self.assertEqual(current["status"], "running", "Last upload predates the simulated runtime interruption")
        with patch("inspection_sim.run_episode", side_effect=resumable_fixture) as resumed:
            payload = evaluate(restored_root, restored_run, settings, plans, resume=True)
        self.assertEqual(resumed.call_count, 21)
        self.assertEqual(len(payload["records"]), 24)
        for index, record in enumerate(payload["records"][:3]):
            original = calls[index] / "result.json"
            self.assertEqual(record["result_path"], original.relative_to(self.run).as_posix())
            self.assertEqual(file_hash(restored_run / record["result_path"]), file_hash(original))
        self.assertFalse((restored_run / calls[3].relative_to(self.run)).exists(),
                         "The partial fourth episode was never committed to persistent storage")


if __name__ == "__main__":
    unittest.main()
