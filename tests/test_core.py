"""計画検査・運動学の単体テスト。AI推論／物理動作の検証ではない。"""
import json
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from inspection_core import *
from inspection_sim import scene_xml, build_segments, segment_goal

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings()
    def check(self, targets, decision="execute"):
        return parse_and_validate(json.dumps({"decision":decision,"targets":targets,"finish":"home"}), self.s)[1]
    def test_defaults(self):
        self.s.validate()
    def test_valid_order(self):
        self.assertEqual(self.check(["B","A"]).targets, ("B","A"))
    def test_repeated_target(self):
        self.assertTrue(self.check(["B","A","B"]).accepted)
    def test_forbidden_whole_plan(self):
        r = self.check(["A","C"])
        self.assertFalse(r.accepted); self.assertEqual(r.targets, ())
        self.assertEqual(r.reason, "forbidden_target:C")
    def test_unknown_whole_plan(self):
        self.assertEqual(self.check(["A","Z"]).reason, "unknown_target:Z")
    def test_clarify(self):
        self.assertEqual(self.check([], "clarify").reason, "clarification_requested")
    def test_empty_execute(self):
        self.assertFalse(self.check([]).accepted)
    def test_too_many_targets(self):
        self.assertFalse(self.check(["A"]*5).accepted)
    def test_bad_schema(self):
        for value in ['{}', '[]', 'true', '{"decision":"execute"}', '```json\n{}\n```']:
            self.assertFalse(parse_and_validate(value, self.s)[1].accepted)
    def test_code_is_not_executed(self):
        self.assertFalse(parse_and_validate("__import__('os').system('echo bad')", self.s)[1].accepted)
    def test_duplicate_json_keys(self):
        raw = '{"decision":"execute","targets":["C"],"targets":["A"],"finish":"home"}'
        self.assertFalse(parse_and_validate(raw, self.s)[1].accepted)
    def test_bad_target_type(self):
        for value in [[1], [True], ["../A"], [{"target":"A"}]]:
            self.assertFalse(self.check(value).accepted)
    def test_ik_fk_roundtrip(self):
        points = [HOME_XYZ]+[v["xyz"] for v in self.s.targets.values()]
        for p in points:
            np.testing.assert_allclose(forward_kinematics(inverse_kinematics(p)), p, atol=1e-10)
    def test_grid_ik_fk(self):
        for x in np.linspace(.4,.53,5):
            for y in np.linspace(-.24,.24,5):
                for z in [.28,.31,.35,.48]:
                    p = np.array([x,y,z]); q = inverse_kinematics(p)
                    np.testing.assert_allclose(forward_kinematics(q), p, atol=1e-10)
                    self.assertAlmostEqual(float(q[1:].sum()), 0.)
    def test_unreachable(self):
        for p in [[2,0,.3], [float('nan'),0,.3], [0,0,0]]:
            with self.assertRaises(ValueError): inverse_kinematics(p)
    def test_smoothstep_bounds(self):
        self.assertEqual(smoothstep5(-1),0); self.assertEqual(smoothstep5(2),1)
        self.assertAlmostEqual(smoothstep5(.5),.5)
    def test_segment_targets(self):
        seq = build_segments(("B","A"),self.s)
        self.assertEqual([s["target"] for s in seq if s["inspect"]],["B","A"])
        xyz, _ = segment_goal(seq[-1], {'targets':self.s.targets})
        np.testing.assert_allclose(xyz, HOME_XYZ)
    def test_xml_well_formed(self):
        xml = ET.fromstring(scene_xml(self.s))
        self.assertEqual(len(xml.findall("./actuator/motor")),4)
        self.assertIsNotNone(xml.find(".//camera[@name='inspection_camera']"))
    def test_fps_clock_ratio(self):
        self.assertEqual(round(1/self.s.fps/self.s.physics_dt),10)
        self.assertEqual(round(self.s.control_dt/self.s.physics_dt),4)
    def test_evaluation_counts(self):
        cases=evaluation_cases()
        self.assertEqual(sum(c["kind"]=="normal" for c in cases),14)
        self.assertEqual(sum(c["kind"]=="reject" for c in cases),10)
        self.assertEqual(sum(c['split']=='holdout' for c in cases),12)
    def test_draft_flag(self):
        self.assertTrue(self.s.is_draft)
        self.s.account="demo_id";self.s.own_changes="実施した変更";self.s.reflection="実測に基づく考察"
        self.assertFalse(self.s.is_draft)

if __name__ == "__main__":
    unittest.main()
