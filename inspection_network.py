"""Deterministic timestamped observation delivery on one simulation clock.

The robot observes static panels. Every policy receives the same trace and uses
exactly the same validation/adoption rule; only motion gating differs.
"""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
from inspection_core import inverse_kinematics

MODES = ("off", "receipt", "age")
SCENARIOS = ("normal", "outage", "delayed", "reordered")


def trace_hash(trace: dict) -> str:
    payload = json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_trace(settings, scenario: str = "normal") -> dict:
    """Build once, save, then replay unchanged for each monitoring mode.

    Delayed messages retain their true capture time. Reordered recovery injects
    old and duplicate messages after newer messages; they cannot refresh health.
    No stochastic sampling is needed for this small, deterministic experiment.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown communication scenario: {scenario}")
    period = float(getattr(settings, "observation_period_s", .1))
    start, end = float(settings.fault_start), float(settings.fault_end)
    latency = .02
    # Fixed independently of the tested threshold to keep condition comparable.
    bad_latency = 1.2
    packets = []
    n = int(math.ceil((settings.max_episode_s + bad_latency + 1) / period))
    xyz = {name: {"xyz": list(spec["xyz"])} for name, spec in settings.targets.items()}
    for seq in range(n):
        capture = round(seq * period, 8)
        if scenario == "outage" and start <= capture < end:
            continue
        delay = bad_latency if scenario == "delayed" and start <= capture < end else latency
        if scenario == "reordered" and start <= capture < end:
            # Most messages are lost; selected old messages arrive during recovery.
            if seq % 7 != 0:
                continue
            delay = end - capture + .15 + .03 * (seq % 3)
        packets.append({"sequence_id": seq, "capture_time": capture,
                        "receive_time": round(capture + delay, 8), "targets": copy.deepcopy(xyz)})
    if scenario == "reordered":
        for packet in list(packets):
            if end <= packet["capture_time"] < end + .4:
                duplicate = copy.deepcopy(packet)
                duplicate["receive_time"] = round(packet["receive_time"] + .21, 8)
                packets.append(duplicate)
    packets.sort(key=lambda p: (p["receive_time"], p["sequence_id"]))
    return {"schema_version": 1, "scenario": scenario, "clock": "simulation_seconds",
            "observation_period_s": period, "base_latency_s": latency,
            "delayed_latency_s": bad_latency if scenario == "delayed" else None,
            "fault_start_s": None if scenario == "normal" else start,
            "fault_end_s": None if scenario == "normal" else end,
            "targets_are_static": True, "packets": packets}


class ObservationReceiver:
    """Adopt only valid strictly newer observations, never arrival-order overwrite.

    capture_time <= receive_time <= now on a shared simulation clock. Permission
    is checked against Settings, never taken from a sensor packet. Duplicate,
    malformed and out-of-order packets do not refresh receipt freshness either.
    """
    def __init__(self, settings, trace: dict):
        if trace.get("clock") != "simulation_seconds" or not isinstance(trace.get("packets"), list):
            raise ValueError("Trace must contain packets on the simulation_seconds clock")
        self.settings = settings
        self.trace = copy.deepcopy(trace)
        self.observation: dict | None = None
        self.accepted_count = 0
        self.rejected_count = 0
        self.event_log: list[dict] = []
        self._pending = []
        self._cursor = 0
        for packet in trace["packets"]:
            receive = packet.get("receive_time") if isinstance(packet, dict) else None
            if isinstance(receive, bool) or not isinstance(receive, (int, float)) or not math.isfinite(receive):
                self.rejected_count += 1
                self.event_log.append({"event": "observation_rejected", "reason": "invalid_receive_time", "sim_time_s": 0.0})
            else:
                self._pending.append(copy.deepcopy(packet))
        self._pending.sort(key=lambda p: p["receive_time"])

    def _reason(self, packet: dict, now: float) -> str | None:
        seq, cap, rec = packet.get("sequence_id"), packet.get("capture_time"), packet.get("receive_time")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            return "invalid_sequence_id"
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (cap, rec)):
            return "invalid_timestamp"
        if cap < 0 or rec < 0 or cap > rec + 1e-9 or cap > now + 1e-9 or rec > now + 1e-9:
            return "invalid_or_future_timestamp"
        targets = packet.get("targets")
        if not isinstance(targets, dict) or set(targets) != set(self.settings.targets):
            return "invalid_targets"
        try:
            coords = []
            for name, spec in targets.items():
                p = np.asarray(spec["xyz"], dtype=float)
                if p.shape != (3,) or not np.isfinite(p).all():
                    return "invalid_coordinates"
                if not (.40 <= p[0] <= .53 and abs(p[1]) <= .24 and .28 <= p[2] <= .35):
                    return "coordinates_out_of_scope"
                inverse_kinematics(p)
                inverse_kinematics([p[0], p[1], .48])
                coords.append(p[:2])
            if any(np.linalg.norm(a-b) < .15 for i, a in enumerate(coords) for b in coords[i+1:]):
                return "panel_spacing_out_of_scope"
        except (KeyError, TypeError, ValueError):
            return "invalid_coordinates"
        if self.observation is not None:
            if seq <= self.observation["sequence_id"] or cap <= self.observation["capture_time"] + 1e-9:
                return "duplicate_or_out_of_order"
        return None

    def step(self, now: float) -> list[dict]:
        adopted = []
        while self._cursor < len(self._pending) and self._pending[self._cursor]["receive_time"] <= now + 1e-9:
            packet = self._pending[self._cursor]
            self._cursor += 1
            reason = self._reason(packet, now)
            event = {"sim_time_s": round(now, 8), "sequence_id": packet.get("sequence_id"),
                     "capture_time": packet.get("capture_time"), "receive_time": packet.get("receive_time")}
            if reason:
                self.rejected_count += 1
                self.event_log.append(dict(event, event="observation_rejected", reason=reason))
                continue
            self.observation = packet
            self.accepted_count += 1
            adopted.append(packet)
            self.event_log.append(dict(event, event="observation_adopted"))
        return adopted

    def age(self, now: float) -> float:
        return math.inf if self.observation is None else max(0., now - self.observation["capture_time"])

    def receipt_age(self, now: float) -> float:
        return math.inf if self.observation is None else max(0., now - self.observation["receive_time"])

    def healthy(self, now: float, mode: str) -> bool:
        if mode not in MODES:
            raise ValueError(f"Unknown monitor mode: {mode}")
        if self.observation is None:
            return False
        if mode == "off":
            return True
        value = self.age(now) if mode == "age" else self.receipt_age(now)
        return value <= self.settings.stale_threshold + 1e-9

    def provenance(self) -> dict | None:
        if self.observation is None:
            return None
        return {key: self.observation[key] for key in ("sequence_id", "capture_time", "receive_time")}
