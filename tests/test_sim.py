"""State-machine regression tests with an explicit fake plant, not physics tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from inspection_core import Settings, HOME_XYZ, inverse_kinematics, forward_kinematics
from inspection_network import make_trace
from inspection_sim import run_episode


class FakePlant:
    """Only verifies orchestration; never used by production run_episode."""
    def __init__(self, settings, run_dir, record):
        self.settings, self.run_dir = settings, Path(run_dir)
        self.data = SimpleNamespace(qpos=inverse_kinematics(HOME_XYZ), qvel=np.zeros(4), time=0.)
        self.frames_written = 0
        (self.run_dir / "scene.xml").write_text("TEST DOUBLE", encoding="utf-8")

    def tip(self):
        return forward_kinematics(self.data.qpos)

    def physics_step(self, q):
        self.data.qvel = (q-self.data.qpos)/self.settings.physics_dt
        self.data.qpos = q.copy()
        self.data.time += self.settings.physics_dt

    def capture(self, name):
        filename = "capture_"+name+".png"
        Image.new("RGB", (8, 8)).save(self.run_dir/filename)
        return filename

    def contact_snapshot(self):
        return 0, None, False

    def close(self):
        pass


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.settings = Settings()
        self.plan = {"source": "diagnostic", "input": "A", "raw_output": json.dumps({"decision":"execute", "targets":["A"], "finish":"home"})}
        self.mock = patch("inspection_sim.InspectionSimulator", FakePlant)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.tmp.cleanup()

    def run_it(self, path="attempt", **kwargs):
        return run_episode(self.settings, self.plan, self.root/path, allow_diagnostic=True, **kwargs)

    def test_attempt_is_immutable_and_assets_are_explicit(self):
        result = self.run_it()
        self.assertTrue(result["success"], result)
        self.assertTrue(result["diagnostic"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_targets"], ["A"])
        self.assertEqual(len(result["captures"]), 1)
        self.assertIn(result["captures"][0], result["artifact_hashes"])
        with self.assertRaises(FileExistsError):
            self.run_it()

    def test_production_refuses_diagnostic_planner(self):
        result = run_episode(self.settings, self.plan, self.root/"denied")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["success"])
        self.assertFalse((self.root/"denied"/"scene.xml").exists())

    def test_diagnostic_flag_cannot_masquerade_as_llm_source(self):
        plan = dict(self.plan, source="llm", diagnostic=True)
        result = run_episode(self.settings, plan, self.root/"false_llm")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["success"])
        self.assertFalse((self.root/"false_llm"/"scene.xml").exists())

    def test_reference_requires_explicit_separate_permission(self):
        from inspection_reference import make_control_reference
        for label, options in (("ordinary", {}), ("diagnostic", {"allow_diagnostic": True}),
                               ("mixed", {"allow_reference": True, "allow_diagnostic": True})):
            result = run_episode(self.settings, make_control_reference(), self.root/label, **options)
            self.assertEqual(result["status"], "failed")
            self.assertFalse((self.root/label/"scene.xml").exists())

    def test_reference_is_valid_control_evidence_without_llm_claim(self):
        from inspection_reference import make_control_reference, reference_metadata
        plan = make_control_reference()
        result = run_episode(self.settings, plan, self.root/"reference", allow_reference=True)
        self.assertTrue(result["success"], result)
        self.assertFalse(result["diagnostic"])
        self.assertEqual(result["planned_targets"], ["A", "B"])
        self.assertTrue(all(result[k] == v for k, v in reference_metadata(plan).items()))
        self.assertEqual(result["planner_source"], "control_reference")
        self.assertFalse(result["ai_accuracy_evaluated"])

    def test_reference_tampering_fails_even_with_recalculated_hash(self):
        from inspection_reference import make_control_reference, validate_control_reference
        from inspection_store import digest
        alterations = [dict(raw_output='{"decision":"execute","targets":["B"],"finish":"home"}'),
                       dict(source="llm"), dict(purpose="evaluation"), dict(diagnostic=True),
                       dict(ai_accuracy_evaluated=True), dict(extra="not predeclared")]
        for index, changes in enumerate(alterations):
            plan = dict(make_control_reference(), **changes)
            plan["content_hash"] = digest({k:v for k,v in plan.items() if k != "content_hash"})
            with self.assertRaises(ValueError):
                validate_control_reference(plan)
            result = run_episode(self.settings, plan, self.root/("tampered_"+str(index)), allow_reference=True)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["success"])
            self.assertFalse((self.root/("tampered_"+str(index))/"scene.xml").exists())

    def test_single_delivery_burst_cannot_resume(self):
        trace = make_trace(self.settings, "normal")
        packets = trace["packets"]
        initial = [p for p in packets if p["capture_time"] < 1.]
        burst = [dict(p, receive_time=2.) for p in packets if p["sequence_id"] in (18, 19, 20)]
        continuous = [p for p in packets if p["capture_time"] >= 2.3]
        trace["packets"] = initial + burst + continuous
        for mode in ("receipt", "age"):
            result = self.run_it("burst_"+mode, trace=trace, monitor_mode=mode)
            self.assertTrue(result["success"], result)
            resume = next(e for e in result["events"] if e["event"] == "resume")
            # A burst at t=2 is three IDs, but only one delivery tick.
            # Additional arrivals at 2.32 and 2.42 are needed for recovery.
            self.assertGreaterEqual(resume["sim_time_s"], 2.42-1e-8)
            self.assertGreaterEqual(resume["distinct_delivery_ticks"], 3)
            self.assertGreaterEqual(resume["new_observation_delivery_span_s"], .2-1e-8)

    def test_no_initial_observation_never_moves(self):
        trace = make_trace(self.settings, "normal")
        trace["packets"] = []
        for mode in ("off", "receipt", "age"):
            result = self.run_it(mode, trace=trace, monitor_mode=mode)
            self.assertEqual(result["failure"], "initial_observation_timeout")
            self.assertEqual(result["captures"], [])
            self.assertFalse(any(e["event"] == "start" for e in result["events"]))

    def test_no_recovery_timeout_with_old_duplicates(self):
        trace = make_trace(self.settings, "normal")
        trace["packets"] = [p for p in trace["packets"] if p["capture_time"] < 1.]
        last = trace["packets"][-1]
        trace["packets"] += [dict(last, receive_time=2.+i*.1) for i in range(150)]
        result = self.run_it(trace=trace)
        self.assertEqual(result["failure"], "recovery_timeout")
        self.assertFalse(any(e["event"] == "resume" for e in result["events"]))
        self.assertGreater(result["rejected_observations"], 0)

    def test_same_trace_reordered_recovers_and_stops_with_dwell(self):
        trace = make_trace(self.settings, "reordered")
        result = self.run_it(trace=trace)
        self.assertTrue(result["success"], result)
        hold = next(e for e in result["events"] if e["event"] == "hold")
        resume = next(e for e in result["events"] if e["event"] == "resume")
        self.assertGreaterEqual(hold["hold_command_to_stationary_s"], .15-1e-6)
        self.assertGreaterEqual(resume["distinct_new_observations"], 3)
        self.assertLessEqual(resume["joint_speed_norm"], .02)
        self.assertGreaterEqual(result["rejected_observations"], 1)
        self.assertEqual(result["metric_sample_hz"], 200)

    def test_coordinates_used_from_adopted_observation(self):
        trace = make_trace(self.settings, "normal")
        for packet in trace["packets"]:
            packet["targets"]["A"]["xyz"][0] = .46
        result = self.run_it(trace=trace)
        goal = next(e for e in result["events"] if e["event"] == "segment_plan" and e["segment"] == "DESCEND")
        self.assertEqual(goal["goal_xyz"], [.46, -.2, .31])
        self.assertIsNotNone(goal["observation"]["sequence_id"])

    def test_success_then_capture_failure_cannot_reuse_success(self):
        success = self.run_it("first")
        with patch.object(FakePlant, "capture", side_effect=OSError("injected write failure")):
            failed = self.run_it("second")
        stored = json.loads((self.root/"second"/"result.json").read_text(encoding="utf-8"))
        self.assertTrue(success["success"])
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(stored["success"])
        self.assertEqual(stored["captures"], [])
        self.assertFalse(any(name.startswith("capture_") for name in stored["artifact_hashes"]))

    def test_rejection_has_no_old_captures(self):
        self.run_it("first")
        self.plan["raw_output"] = '{"decision":"execute","targets":["C"],"finish":"home"}'
        result = self.run_it("rejected")
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["captures"], [])
        self.assertEqual(list((self.root/"rejected").glob("capture_*.png")), [])

    def test_keyboard_interrupt_is_persisted_then_propagated(self):
        with patch.object(FakePlant, "physics_step", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_it("interrupted")
        result = json.loads((self.root/"interrupted"/"result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "interrupted")
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
