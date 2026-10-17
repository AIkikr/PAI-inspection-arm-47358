"""Software-only tests; synthetic packets do not establish physical performance."""
import copy
import unittest
from inspection_core import Settings
from inspection_network import ObservationReceiver, make_trace, trace_hash


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.trace = make_trace(self.settings, "normal")

    def test_trace_deterministic_and_modes_share_it(self):
        self.assertEqual(trace_hash(self.trace), trace_hash(make_trace(self.settings, "normal")))
        self.assertNotEqual(trace_hash(self.trace), trace_hash(make_trace(self.settings, "outage")))

    def test_no_initial_observation(self):
        receiver = ObservationReceiver(self.settings, dict(self.trace, packets=[]))
        receiver.step(0.)
        for mode in ("off", "receipt", "age"):
            self.assertFalse(receiver.healthy(0., mode))

    def test_delayed_packet_recently_received_but_old(self):
        packet = copy.deepcopy(self.trace["packets"][0])
        packet["receive_time"] = 2.
        receiver = ObservationReceiver(self.settings, dict(self.trace, packets=[packet]))
        receiver.step(2.)
        self.assertTrue(receiver.healthy(2., "receipt"))
        self.assertFalse(receiver.healthy(2., "age"))
        self.assertEqual(receiver.age(2.), 2.)

    def test_duplicates_and_out_of_order_do_not_refresh(self):
        old, new = copy.deepcopy(self.trace["packets"][0]), copy.deepcopy(self.trace["packets"][5])
        old["receive_time"] = 1.
        duplicate = dict(new, receive_time=1.2)
        receiver = ObservationReceiver(self.settings, dict(self.trace, packets=[new, old, duplicate]))
        receiver.step(2.)
        self.assertEqual(receiver.accepted_count, 1)
        self.assertEqual(receiver.rejected_count, 2)
        self.assertEqual(receiver.provenance()["sequence_id"], 5)
        self.assertAlmostEqual(receiver.receipt_age(2.), 2. - new["receive_time"])

    def test_invalid_future_and_invalid_xyz(self):
        base = self.trace["packets"][0]
        packets = [dict(base, capture_time=10.), dict(base, sequence_id=True),
                   dict(base, receive_time=float("nan")), dict(base, capture_time=-1.)]
        invalid = copy.deepcopy(base)
        invalid["targets"]["A"]["xyz"] = [.9, 0, .3]
        packets.append(invalid)
        receiver = ObservationReceiver(self.settings, dict(self.trace, packets=packets))
        receiver.step(3.)
        self.assertEqual(receiver.accepted_count, 0)
        self.assertEqual(receiver.rejected_count, len(packets))

    def test_reordered_recovery_never_rewinds_capture_time(self):
        receiver = ObservationReceiver(self.settings, make_trace(self.settings, "reordered"))
        caps = []
        for i in range(600):
            receiver.step(i*.02)
            if receiver.observation is not None:
                caps.append(receiver.observation["capture_time"])
        self.assertEqual(caps, sorted(caps))
        self.assertGreater(receiver.rejected_count, 0)
        self.assertTrue(receiver.healthy(11.98, "age"))

    def test_trace_immutable_when_replayed(self):
        digest = trace_hash(self.trace)
        receiver = ObservationReceiver(self.settings, self.trace)
        receiver.step(1.)
        receiver.observation["targets"]["A"]["xyz"][0] = .45
        self.assertEqual(digest, trace_hash(self.trace))


if __name__ == "__main__":
    unittest.main()
