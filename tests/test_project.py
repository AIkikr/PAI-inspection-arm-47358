"""Orchestration regression tests with explicitly synthetic, temporary fixtures.

FixturePlanner imitates the real planner wire format solely to test cache and
stage logic. Its output is never a Qwen benchmark, persisted project result or
submission artifact. Diagnostic-tagged inference is separately tested to reject.
"""
from __future__ import annotations

import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import inspection_ai
from inspection_core import Settings, parse_and_validate, save_json
from inspection_store import digest, load_stage, new_run, read_json, seal_stage, start_stage
from run_project import (compare_freshness_guard, evaluate, generate_plans,
                         load_contract, load_plans, record_demonstrations)
from test_store import copy_project


class FixturePlanner:
    calls = []
    close_calls = 0
    fail_at = None
    diagnostic = False
    force_order = None

    def __init__(self, settings, contract, require_gpu=True, log_path=None):
        self.contract = contract

    def infer(self, text):
        if self.fail_at is not None and len(type(self).calls) == self.fail_at:
            raise RuntimeError("deliberate diagnostic interruption")
        type(self).calls.append(text)
        order = self.force_order or ["A", "B"]
        import json
        return {"source": "llm", "diagnostic": self.diagnostic, "input": text,
                "raw_output": json.dumps({"decision": "execute", "targets": order, "finish": "home"}),
                "model_id": self.contract["model_id"], "revision": self.contract["revision"],
                "contract_hash": digest(self.contract), "inference_wall_s": 0.0,
                "generated_tokens": 1}

    def close(self):
        type(self).close_calls += 1


def synthetic_episode(settings, plan, directory, **kwargs):
    """A storage fixture; no simulation, model inference, image capture or video."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _, check = parse_and_validate(plan["raw_output"], settings)
    result = {"status": "complete", "diagnostic": True,
              "planner_source": plan.get("source"),
              "accepted": check.accepted, "validation_reason": check.reason,
              "planned_targets": list(check.targets), "completed_targets": list(check.targets),
              "success": check.accepted, "failure": "", "sim_duration_s": 8.0,
              "captures": [], "artifacts": []}
    if kwargs.get("allow_reference"):
        from inspection_reference import reference_metadata
        result.update(reference_metadata(plan))
    save_json(directory / "result.json", result)
    return result


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = copy_project(Path(self.temp.name) / "project")
        self.settings = Settings()
        self.run = new_run(self.root, self.settings)
        request = inspection_ai.planner_request(self.settings)
        self.contract = dict(request, revision="a" * 40, request_hash=digest(request),
                             note="SYNTHETIC SOFTWARE TEST FIXTURE; NOT MODEL INFERENCE")
        FixturePlanner.calls = []
        FixturePlanner.close_calls = 0
        FixturePlanner.fail_at = None
        FixturePlanner.diagnostic = False
        FixturePlanner.force_order = None
        self.addCleanup(patch.stopall)
        patch("inspection_ai.resolve_contract", return_value=self.contract).start()
        patch("inspection_ai.QwenPlanner", FixturePlanner).start()
        self.stdout = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.stdout)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def plans(self):
        return generate_plans(self.root, self.run, self.settings)

    def test_exact_contract_cache_resumes_without_model_loading(self):
        first = self.plans()
        self.assertEqual(len(first), 25)
        self.assertEqual(len(FixturePlanner.calls), 25)
        with patch("inspection_ai.QwenPlanner", side_effect=AssertionError("must not load model")):
            second = self.plans()
        self.assertEqual(first, second)
        self.assertEqual(load_plans(self.root, self.run, self.settings), first)

    def test_changed_revision_cannot_reuse_same_run(self):
        self.plans()
        with self.assertRaises(ValueError):
            generate_plans(self.root, self.run, replace(self.settings, model_revision="b" * 40))
        self.assertEqual(len(FixturePlanner.calls), 25)

    def test_changed_prompt_cannot_reuse_same_run(self):
        self.plans()
        with patch("inspection_ai.SYSTEM_PROMPT", inspection_ai.SYSTEM_PROMPT + "\nCHANGED"), self.assertRaises(ValueError):
            self.plans()
        self.assertEqual(len(FixturePlanner.calls), 25)

    def test_mixed_revision_cache_is_rejected_even_if_content_rehashed(self):
        self.plans()
        path = self.run / "plans" / "n03.json"
        record = read_json(path)
        record["revision"] = "b" * 40
        record["content_hash"] = digest({k: v for k, v in record.items() if k != "content_hash"})
        save_json(path, record)
        with self.assertRaises(ValueError):
            self.plans()

    def test_diagnostic_inference_is_rejected_before_cache_publication(self):
        FixturePlanner.diagnostic = True
        with self.assertRaises(ValueError):
            self.plans()
        self.assertFalse((self.run / "plans_manifest.json").exists())
        self.assertFalse((self.run / "plans" / "n01.json").exists())
        attempts = list((self.run / "inference_attempts").glob("*.json"))
        self.assertEqual(len(attempts), 1)
        self.assertEqual(read_json(attempts[0])["status"], "failed")

    def test_partial_inference_retry_keeps_contract_and_only_completes_missing(self):
        FixturePlanner.fail_at = 2
        with self.assertRaises(RuntimeError):
            self.plans()
        self.assertEqual(len(list((self.run / "plans").glob("*.json"))), 2)
        locked = load_contract(self.run)
        FixturePlanner.fail_at = None
        with patch("inspection_ai.resolve_contract", side_effect=AssertionError("must keep pinned SHA")):
            plans = self.plans()
        self.assertEqual(len(plans), 25)
        self.assertEqual(len(FixturePlanner.calls), 25)
        self.assertEqual(load_contract(self.run), locked)
        self.assertEqual(FixturePlanner.close_calls, 2)

    def test_new_inference_invalidates_completed_stages(self):
        plans = self.plans()
        stage = start_stage(self.run, "evaluation", digest(self.contract), digest(plans))
        seal_stage(self.run, "evaluation", stage, {"diagnostic": True})
        (self.run / "plans" / "custom.json").unlink()
        self.plans()
        with self.assertRaises(ValueError):
            load_stage(self.run, "evaluation")

    def test_each_run_uses_its_own_pinned_model_contract(self):
        self.plans()
        newer = new_run(self.root, self.settings)
        newer_contract = dict(self.contract, revision="b" * 40)
        with patch("inspection_ai.resolve_contract", return_value=newer_contract):
            generate_plans(self.root, newer, self.settings)
        self.assertEqual(load_contract(self.run)["revision"], "a" * 40)
        self.assertEqual(load_contract(newer)["revision"], "b" * 40)
        self.assertEqual({p["revision"] for p in load_plans(self.root, self.run, self.settings).values()}, {"a" * 40})

    def test_memory_plan_edit_cannot_bypass_saved_plan_verification(self):
        plans = self.plans()
        changed = copy.deepcopy(plans)
        changed["n03"]["raw_output"] = "{}"
        with self.assertRaises(ValueError):
            evaluate(self.root, self.run, self.settings, changed)

    def test_failed_evaluation_retry_cannot_load_previous_success(self):
        plans = self.plans()
        with patch("inspection_sim.run_episode", side_effect=synthetic_episode):
            evaluate(self.root, self.run, self.settings, plans)
        old_stage = load_stage(self.run, "evaluation")
        with patch("inspection_sim.run_episode", side_effect=OSError("deliberate image fixture failure")), self.assertRaises(OSError):
            evaluate(self.root, self.run, self.settings, plans)
        with self.assertRaises(ValueError):
            load_stage(self.run, "evaluation")
        current = read_json(self.run / "current.json")["evaluation"]
        self.assertEqual(current["status"], "failed")
        self.assertNotEqual(current["path"] + "/stage.json", old_stage["_path"])
        self.assertTrue((self.run / old_stage["_path"]).exists())

    def test_instruction_mismatch_is_not_presented_as_task_success(self):
        FixturePlanner.force_order = ["B", "A"]
        plans = self.plans()
        with patch("inspection_sim.run_episode", side_effect=synthetic_episode):
            payload = record_demonstrations(self.root, self.run, self.settings, plans)
        normal = next(r for r in payload["records"] if r["id"] == "normal")
        self.assertTrue(normal["success"], "Synthetic controller completed the wrong plan")
        self.assertFalse(normal["semantic_match"])
        self.assertFalse(normal["task_success"])
        custom = next(r for r in payload["records"] if r["id"] == "custom")
        self.assertTrue(custom["task_success"])
        self.assertEqual(custom["expected"], ["B", "A"])

    def test_custom_demo_checks_declared_expected_order(self):
        plans = self.plans()  # Fixture emits A->B while declared user expectation is B->A.
        with patch("inspection_sim.run_episode", side_effect=synthetic_episode):
            payload = record_demonstrations(self.root, self.run, self.settings, plans)
        custom = next(r for r in payload["records"] if r["id"] == "custom")
        self.assertTrue(custom["success"])
        self.assertFalse(custom["semantic_match"])
        self.assertFalse(custom["task_success"])

    def test_wrong_llm_plan_does_not_skip_or_supply_control_comparison(self):
        from inspection_reference import make_control_reference
        FixturePlanner.force_order = ["B"]
        plans = self.plans()
        original = copy.deepcopy(plans)
        with patch("inspection_sim.run_episode", side_effect=synthetic_episode) as sim:
            comparison = compare_freshness_guard(self.root, self.run, self.settings, plans)
            self.assertEqual(sim.call_count, 12)
            self.assertTrue(all(call.args[1] == make_control_reference() for call in sim.call_args_list))
            self.assertTrue(all(call.kwargs.get("allow_reference") is True for call in sim.call_args_list))
        with patch("inspection_sim.run_episode", side_effect=synthetic_episode) as sim:
            evaluation = evaluate(self.root, self.run, self.settings, plans)
            self.assertTrue(all(not call.kwargs.get("allow_reference") for call in sim.call_args_list))
        n03 = next(row for row in evaluation["records"] if row["id"] == "n03")
        self.assertFalse(n03["language_exact"])
        self.assertFalse(n03["end_to_end_success"])
        self.assertEqual(plans, original)
        self.assertEqual(load_plans(self.root, self.run, self.settings), original)
        self.assertTrue(comparison["available"])
        self.assertEqual(comparison["planner_source"], "control_reference")
        self.assertFalse(comparison["ai_accuracy_evaluated"])
        self.assertEqual(comparison["reference_plan_hash"], digest(make_control_reference()))
        self.assertTrue(all(r["reference_plan_hash"] == comparison["reference_plan_hash"]
                            and r["planner_source"] == "control_reference"
                            for r in comparison["records"]))

    def test_comparison_replays_identical_trace_for_each_monitor(self):
        plans = self.plans()
        with patch("inspection_sim.run_episode", side_effect=synthetic_episode):
            payload = compare_freshness_guard(self.root, self.run, self.settings, plans)
        self.assertTrue(payload["available"])
        self.assertEqual(len(payload["records"]), 12)
        for scenario in ("normal", "outage", "delayed", "reordered"):
            rows = [r for r in payload["records"] if r["scenario"] == scenario]
            self.assertEqual({r["mode"] for r in rows}, {"off", "receipt", "age"})
            self.assertEqual(len({r["trace_hash"] for r in rows}), 1)
            self.assertEqual(len({r["trace_file_sha256"] for r in rows}), 1)


if __name__ == "__main__":
    unittest.main()
