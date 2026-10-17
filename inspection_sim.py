"""自作4自由度点検アーム。実行時はトルク指令とmj_stepで物理を進める。"""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from inspection_network import ObservationReceiver, make_trace, trace_hash, MODES
from inspection_core import (
    BASE_Z, L1, L2, TOOL_X, TOOL_Z, HOME_XYZ, Settings,
    inverse_kinematics, forward_kinematics, smoothstep5,
    parse_and_validate, save_json,
)


def scene_xml(settings: Settings) -> str:
    panels = []
    for name, spec in settings.targets.items():
        x, y, _ = spec["xyz"]
        rgba = {"A":"0.15 0.62 0.70 1", "B":"0.95 0.64 0.19 1", "C":"0.78 0.24 0.26 1"}[name]
        panels.append(f'''
        <geom name="panel_{name}" type="box" pos="{x} {y} 0.115"
              size="0.062 0.062 0.025" rgba="{rgba}"/>
        <geom name="indicator_{name}" type="cylinder" pos="{x+0.025} {y} 0.143"
              size="0.012 0.003" rgba="0.82 0.98 0.91 1"/>
        <geom name="line_{name}" type="box" pos="{x-0.025} {y} 0.143"
              size="0.007 0.033 0.003" rgba="0.10 0.14 0.18 1"/>
        ''')
    # 床・パネル・アームは衝突有効。土台は装飾で衝突対象から外す。
    return f'''<mujoco model="original_four_dof_inspection_arm">
      <compiler angle="radian"/>
      <option timestep="{settings.physics_dt}" integrator="implicitfast" gravity="0 0 -9.81"/>
      <size njmax="1000" nconmax="200"/>
      <visual>
        <global offwidth="1024" offheight="640"/>
        <quality shadowsize="2048"/>
        <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
      </visual>
      <statistic center="0.28 0 0.3" extent="1.25"/>
      <default>
        <joint damping="0.8" armature="0.02" limited="true"/>
        <geom friction="0.7 0.1 0.1" condim="3" density="850"/>
        <motor ctrllimited="true" ctrlrange="-70 70"/>
      </default>
      <worldbody>
        <light pos="0.0 -0.5 1.8" dir="0.2 0.2 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.91 0.93 0.95 1"/>
        <geom name="workbench" type="box" pos="0.47 0 0.045" size="0.30 0.39 0.045" rgba="0.30 0.35 0.40 1"/>
        <geom name="pedestal" type="cylinder" pos="0 0 0.18" size="0.075 0.18"
              rgba="0.16 0.20 0.26 1" contype="0" conaffinity="0"/>
        {''.join(panels)}
        <body name="yaw_body" pos="0 0 {BASE_Z}">
          <joint name="yaw" axis="0 0 1" range="-1.5 1.5"/>
          <geom name="yaw_geom" type="sphere" size="0.05" mass="0.5" rgba="0.18 0.26 0.36 1"/>
          <body name="upper_body">
            <joint name="shoulder" axis="0 1 0" range="-2.2 1.2"/>
            <geom name="upper_geom" type="capsule" fromto="0 0 0 {L1} 0 0" size="0.027" mass="0.8" rgba="0.12 0.45 0.63 1"/>
            <body name="forearm_body" pos="{L1} 0 0">
              <joint name="elbow" axis="0 1 0" range="0.15 2.7"/>
              <geom name="elbow_geom" type="sphere" size="0.036" mass="0.15" rgba="0.18 0.26 0.36 1"/>
              <geom name="forearm_geom" type="capsule" fromto="0 0 0 {L2} 0 0" size="0.023" mass="0.5" rgba="0.17 0.59 0.72 1"/>
              <body name="wrist_body" pos="{L2} 0 0">
                <joint name="wrist" axis="0 1 0" range="-2.2 2.2"/>
                <geom name="camera_case" type="box" pos="0.035 0 0" size="0.035 0.031 0.022" mass="0.17" rgba="0.13 0.17 0.22 1"/>
                <geom name="camera_lens" type="cylinder" pos="{TOOL_X} 0 -0.024" size="0.012 0.003" mass="0.03" rgba="0.05 0.1 0.15 1"/>
                <site name="camera_tip" pos="{TOOL_X} 0 {TOOL_Z}" size="0.004" rgba="0.9 0.9 0.2 0"/>
                <camera name="inspection_camera" pos="{TOOL_X} 0 {TOOL_Z}" fovy="58"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <contact>
        <exclude body1="yaw_body" body2="upper_body"/>
        <exclude body1="upper_body" body2="forearm_body"/>
        <exclude body1="forearm_body" body2="wrist_body"/>
      </contact>
      <actuator>
        <motor name="yaw_motor" joint="yaw"/>
        <motor name="shoulder_motor" joint="shoulder"/>
        <motor name="elbow_motor" joint="elbow"/>
        <motor name="wrist_motor" joint="wrist"/>
      </actuator>
    </mujoco>'''


@lru_cache(maxsize=24)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        f"/usr/share/fonts/opentype/noto/NotoSansCJK-{'Bold' if bold else 'Regular'}.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=0)
    raise RuntimeError("表示用フォントがありません。セットアップセルを実行してください。")


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_width: int,
                 fnt: ImageFont.FreeTypeFont, fill: str = "#243449", line_height: int = 25,
                 max_lines: int = 6) -> int:
    x, y = xy
    lines, current = [], ""
    for ch in str(text):
        if ch == "\n":
            lines.append(current); current = ""
        elif current and draw.textlength(current+ch, font=fnt) > max_width:
            lines.append(current); current = ch
        else:
            current += ch
    if current:
        lines.append(current)
    for line in lines[:max_lines]:
        draw.text((x, y), line, font=fnt, fill=fill); y += line_height
    return y


def information_card(title: str, lines: list[str], footer: str = "SIMULATION / 1x simulation time") -> np.ndarray:
    img = Image.new("RGB", (1024, 640), "#eef2f5")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1024, 14), fill="#176c86")
    draw_wrapped(d, title, (55, 68), 920, font(34, True), line_height=48, max_lines=2)
    y = 196
    for line in lines:
        y = draw_wrapped(d, line, (58, y), 907, font(23), line_height=34, max_lines=4) + 22
    d.line((55, 581, 969, 581), fill="#b9c8d3", width=2)
    draw_wrapped(d, footer, (55, 598), 920, font(16), line_height=20, max_lines=1)
    return np.asarray(img)


class InspectionSimulator:
    def __init__(self, settings: Settings, run_dir: Path, record: bool):
        # 描画バックエンドは必ずこのimportより前に設定する（setup_colab.py）。
        import mujoco
        self.mj = mujoco
        self.settings, self.run_dir = settings, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        xml = scene_xml(settings)
        (self.run_dir / "scene.xml").write_text(xml, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # qposの直接設定は初期化のみ。動作中はPDトルクとmj_stepを使用。
        self.data.qpos[:] = inverse_kinematics(HOME_XYZ)
        mujoco.mj_forward(self.model, self.data)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "camera_tip")
        if self.model.nq != 4 or self.model.nu != 4:
            raise RuntimeError("想定と異なるロボットモデルです。")
        # 手書きIK/FKとMJCFの軸・長さの対応を起動時に検査。
        if np.linalg.norm(self.tip()-HOME_XYZ) > 1e-6:
            raise RuntimeError("MJCFと運動学の不一致。計算結果を成功として扱いません。")
        self.main_renderer = None
        self.wrist_renderer = None
        self.writer = None
        self.latest_wrist = np.full((192, 256, 3), 235, np.uint8)
        self.frames_written = 0
        self.record = record
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.28, 0.0, 0.30]
        self.camera.distance = 1.45
        self.camera.azimuth = 135
        self.camera.elevation = -32
        if record:
            import imageio.v2 as imageio
            self.main_renderer = mujoco.Renderer(self.model, height=480, width=720)
            self.writer = imageio.get_writer(
                str(self.run_dir / "clip.mp4"), format="FFMPEG", mode="I", fps=settings.fps,
                codec="libx264", pixelformat="yuv420p", quality=7, macro_block_size=16)

    def tip(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def physics_step(self, q_target: np.ndarray) -> None:
        kp = np.array([100.0, 110.0, 70.0, 18.0])
        kd = np.array([3.5, 4.5, 2.5, 1.0])
        # 毎物理ステップでPDとバイアス補償を更新。制御目標は50Hzで更新。
        tau = kp*(q_target-self.data.qpos)-kd*self.data.qvel+self.data.qfrc_bias
        self.data.ctrl[:] = np.clip(tau, -70.0, 70.0)
        self.mj.mj_step(self.model, self.data)
        # mj_step後の最新qposで先端位置や描画を更新。時間は進めない。
        self.mj.mj_forward(self.model, self.data)

    def contact_problem(self) -> bool:
        # モデル上の接触監視であり、実機安全性の保証ではない。
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if (b1 != 0 or b2 != 0) and c.dist < -0.001:
                return True
        return False

    def contact_snapshot(self) -> tuple[int, float | None, bool]:
        """Robot-involving contacts; abnormal only if penetration exceeds 1 mm."""
        distances = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.model.geom_bodyid[c.geom1] != 0 or self.model.geom_bodyid[c.geom2] != 0:
                distances.append(float(c.dist))
        return len(distances), min(distances) if distances else None, any(d < -.001 for d in distances)

    def wrist_image(self) -> np.ndarray:
        if self.wrist_renderer is None:
            self.wrist_renderer = self.mj.Renderer(self.model, height=192, width=256)
        self.wrist_renderer.update_scene(self.data, camera="inspection_camera")
        return np.asarray(self.wrist_renderer.render()).copy()

    def capture(self, name: str) -> str:
        path = self.run_dir / f"capture_{name}.png"
        image = self.wrist_image()
        Image.fromarray(image).save(path)
        return path.name

    def video_frame(self, instruction: str, plan: list[str], state: str, target: str,
                    age: float, error: float, done: list[str], guard: bool, note: str) -> None:
        if self.writer is None:
            return
        self.main_renderer.update_scene(self.data, camera=self.camera)
        # 3Dのラベルは描画用の追加geom。物理モデルには混ぜない。
        scene = self.main_renderer.scene
        for name, spec in self.settings.targets.items():
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            x, y, _ = spec["xyz"]
            self.mj.mjv_initGeom(geom, self.mj.mjtGeom.mjGEOM_LABEL,
                                np.zeros(3), np.array([x, y-0.055, 0.20]),
                                np.eye(3).reshape(9), np.array([0.13, 0.18, 0.25, 1.0]))
            geom.label = name + (" (blocked)" if not spec["allowed"] else "")
            scene.ngeom += 1
        world = self.main_renderer.render()
        if self.frames_written % 4 == 0:  # 手先プレビューは5Hz。物理は200Hzのまま。
            self.latest_wrist = self.wrist_image()
        img = Image.new("RGB", (1024, 640), "#eef2f5")
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, 1024, 61), fill="#143549")
        d.text((20, 13), "AI × ROBOTICS  |  設備点検アーム", font=font(26, True), fill="white")
        img.paste(Image.fromarray(world), (16, 78))
        img.paste(Image.fromarray(self.latest_wrist), (752, 78))
        d.text((752, 277), "手先カメラ / 5 Hz", font=font(17), fill="#243449")
        labels = [
            ("STATE", state), ("TARGET", target or "-"),
            ("AGE", f"{age:.2f} s"), ("ERROR", f"{error*1000:.1f} mm"),
            ("DONE", ", ".join(done) or "-"),
            ("GUARD", "ON" if guard else "OFF (comparison)"),
        ]
        y = 318
        for key, value in labels:
            d.text((754, y), key, font=font(14, True), fill="#526b7c")
            draw_wrapped(d, value, (834, y-2), 169, font(16), line_height=21, max_lines=2)
            y += 36
        draw_wrapped(d, instruction, (20, 570), 730, font(18), line_height=24, max_lines=2)
        draw_wrapped(d, f"計画: {' → '.join(plan) or 'なし'}  | {note}", (20, 617), 980,
                     font(14), line_height=18, max_lines=1)
        d.text((752, 570), f"SIM {self.data.time:05.2f} s / 1x", font=font(17, True), fill="#176c86")
        self.writer.append_data(np.asarray(img))
        self.frames_written += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close(); self.writer = None
        if self.main_renderer is not None:
            self.main_renderer.close(); self.main_renderer = None
        if self.wrist_renderer is not None:
            self.wrist_renderer.close(); self.wrist_renderer = None


def build_segments(targets: tuple[str, ...], settings: Settings) -> list[dict]:
    """Only task labels are prebuilt. Coordinates come from adopted observations."""
    segments = []
    for index, target in enumerate(targets):
        for label, duration, inspect in [
            ("APPROACH", 1.6, False), ("DESCEND", 1.4, False),
            ("INSPECT", settings.dwell_s, True), ("LIFT", 1.3, False),
        ]:
            segments.append({"label": label, "duration": duration, "inspect": inspect,
                             "target": target, "index": index})
    segments.append({"label": "HOME", "duration": 1.8, "inspect": False,
                     "target": "", "index": -1})
    return segments


def segment_goal(segment: dict, observation: dict) -> tuple[np.ndarray, np.ndarray]:
    if segment["label"] == "HOME":
        xyz = HOME_XYZ.copy()
    else:
        xyz = np.asarray(observation["targets"][segment["target"]]["xyz"], dtype=float).copy()
        if segment["label"] in ("APPROACH", "LIFT"):
            xyz[2] = .48
    return xyz, inverse_kinematics(xyz)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_episode(settings: Settings, planner_result: dict, run_dir: Path,
                record: bool = False, trace: dict | None = None,
                monitor_mode: str = "age", allow_diagnostic: bool = False,
                allow_reference: bool = False) -> dict[str, Any]:
    """Execute one immutable attempt. Existing directories are never reused.

    A real renderer is required for capture even when record=False. No image or
    controller fallback exists. Exceptions leave an explicit failed result;
    KeyboardInterrupt additionally propagates so Colab interruption is respected.
    Diagnostics must be explicitly enabled and are marked in every result.
    A control reference requires a separate explicit opt-in and exact validation;
    it never substitutes for LLM output in evaluation or demonstrations.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "running", "success": False, "failure": "", "accepted": False,
        "validation_reason": "not_validated", "diagnostic": bool(allow_diagnostic),
        "planner_source": planner_result.get("source"), "monitor_mode": monitor_mode,
        "planned_targets": [], "completed_targets": [], "captures": [], "artifact_hashes": {},
        "events": [], "run_dir": run_dir.name, "sim_duration_s": 0., "duration_s": 0.,
        "video_frames": 0, "video_duration_s": None, "hold_s": 0., "stale_s": 0.,
        "stale_moving_s": 0., "unprotected_stale_s": 0., "max_hold_drift_m": 0.,
        "first_fault_to_hold_command_s": None, "first_threshold_exceedance_to_detection_s": None,
        "first_detection_to_hold_command_s": None, "first_hold_command_to_stationary_s": None,
        "inspection_error_mean_m": None, "contact_samples": 0,
        "min_robot_contact_distance_m": None, "contact_abnormal_threshold_m": -.001,
        "metric_sample_hz": None, "trajectory_log_hz": None,
        "resume_pose_error_limit_rad": .03,
        "scope": "静止対象の位置到達と画像取得。AIは指示解釈、監視は規則ベース。実機安全保証・故障診断ではない。",
    }
    events = result["events"]
    captures = result["captures"]
    completed = result["completed_targets"]
    files = []
    sim = None
    receiver = None
    interrupted = None
    active_hold = None
    inspection_errors = []
    save_json(run_dir / "result.json", result)
    try:
        settings.validate()
        result["metric_sample_hz"] = round(1 / settings.physics_dt)
        result["trajectory_log_hz"] = round(1 / settings.control_dt)
        if monitor_mode not in MODES:
            raise ValueError(f"monitor_mode must be one of {MODES}")
        if allow_reference or planner_result.get("source") == "control_reference":
            if not allow_reference or allow_diagnostic:
                raise ValueError("制御比較の参照計画には allow_reference=True が必要です。診断との混用はできません。")
            from inspection_reference import reference_metadata
            result.update(reference_metadata(planner_result))
            result["scope"] = "固定 A→B 計画による独立した監視方式比較。AIの言語正解率は評価しない。実機安全保証ではない。"
        elif (planner_result.get("source") != "llm" or planner_result.get("diagnostic", False)) and not allow_diagnostic:
            raise ValueError("本実行にはLLMの実出力が必要です。診断はallow_diagnostic=Trueで明示してください。")
        trace = make_trace(settings, "normal") if trace is None else trace
        digest = trace_hash(trace)
        result.update(observation_trace_sha256=digest, communication_scenario=trace.get("scenario"),
                      freshness_guard=monitor_mode != "off", fault_injected=trace.get("scenario") != "normal")
        receiver = ObservationReceiver(settings, trace)
        plan, check = parse_and_validate(planner_result.get("raw_output", ""), settings)
        result.update(accepted=check.accepted, validation_reason=check.reason, planned_targets=list(check.targets))
        save_json(run_dir / "plan.json", {"planner": planner_result, "parsed": plan,
                                         "accepted": check.accepted, "reason": check.reason})
        save_json(run_dir / "observation_trace.json", trace)
        files.extend(["plan.json", "observation_trace.json"])
        sim = InspectionSimulator(settings, run_dir, record)
        files.append("scene.xml")
        segments = build_segments(check.targets, settings) if check.accepted else []
        q_hold = sim.data.qpos.copy()
        q_start = q_hold.copy()
        q_des = q_hold.copy()
        goal_xyz, goal_q = HOME_XYZ.copy(), q_hold.copy()
        goal_observation = None
        seg_index = 0
        elapsed = stable = 0.
        waiting_initial = check.accepted
        holding = False
        health_since = None
        recovery_ids: set[int] = set()
        recovery_delivery_ticks: set[float] = set()
        wait_since = 0.
        low_speed_since = 0.
        terminal_since = None
        task_complete_time = None
        logged_fault_boundaries = set()
        state, target, note, error = "INIT", "", "", 0.
        phys_per_ctrl = int(round(settings.control_dt/settings.physics_dt))
        phys_per_frame = int(round(1./settings.fps/settings.physics_dt))
        physics_index = 0
        resume_speed = float(getattr(settings, "resume_speed_rad_s", .02))
        stop_dwell = float(getattr(settings, "stop_dwell_s", .15))
        min_observations = int(getattr(settings, "recovery_observations", 3))
        max_wait = float(getattr(settings, "max_wait_s", 12.))

        def plan_current(t: float, reason: str):
            nonlocal goal_xyz, goal_q, q_start, elapsed, stable, goal_observation
            goal_xyz, goal_q = segment_goal(segments[seg_index], receiver.observation)
            goal_observation = receiver.provenance()
            q_start = sim.data.qpos.copy()
            elapsed = stable = 0.
            events.append({"event": "segment_plan", "reason": reason, "sim_time_s": round(t, 6),
                           "segment": segments[seg_index]["label"], "target": segments[seg_index]["target"],
                           "observation": goal_observation, "goal_xyz": goal_xyz.tolist()})

        csv_fields = ["sim_time_s", "state", "target", "adopted_sequence_id", "used_sequence_id",
                      "capture_time", "receive_time", "observation_age_s", "receipt_age_s", "healthy", "holding",
                      "tip_error_m", "tip_x", "tip_y", "tip_z", "joint_speed_norm", "q0", "q1", "q2", "q3"]
        files.append("trajectory.csv")
        with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            while sim.data.time < settings.max_episode_s - 1e-8:
                t = float(sim.data.time)
                for boundary in ("fault_start_s", "fault_end_s"):
                    event_time = trace.get(boundary)
                    if event_time is not None and boundary not in logged_fault_boundaries and t >= event_time - 1e-8:
                        logged_fault_boundaries.add(boundary)
                        current = segments[seg_index] if check.accepted and seg_index < len(segments) else {}
                        events.append({"event": boundary.removesuffix("_s"), "sim_time_s": round(t, 6),
                                       "scheduled_time_s": event_time, "segment": current.get("label"),
                                       "target": current.get("target"), "state": state,
                                       "joint_speed_norm": float(np.linalg.norm(sim.data.qvel)),
                                       "observation": receiver.provenance()})
                new_observations = receiver.step(t)
                age = receiver.age(t)
                receipt_age = receiver.receipt_age(t)
                healthy = receiver.healthy(t, monitor_mode)
                terminal = not check.accepted or bool(result["failure"]) or result["success"]
                if terminal:
                    if terminal_since is None:
                        terminal_since = t
                        q_hold = sim.data.qpos.copy()
                    state = "COMPLETE" if result["success"] else ("REJECTED" if not check.accepted else "FAILED")
                    target, error = "", 0.
                    note = check.reason if not check.accepted else (result["failure"] or "生成された計画を実行完了")
                    q_des = q_hold.copy()
                    terminal_seconds = (3. if not check.accepted else 1.) if record else 0.
                    if t - terminal_since >= terminal_seconds - 1e-8:
                        break
                else:
                    segment = segments[seg_index]
                    target = segment["target"]
                    error = float(np.linalg.norm(sim.tip()-goal_xyz))
                    if not waiting_initial and not holding and not healthy:
                        holding = True
                        wait_since = t
                        q_hold = sim.data.qpos.copy()
                        health_since = None
                        recovery_ids.clear()
                        recovery_delivery_ticks.clear()
                        timestamp_key = "capture_time" if monitor_mode == "age" else "receive_time"
                        threshold_time = receiver.observation[timestamp_key] + settings.stale_threshold
                        active_hold = {"event": "hold", "sim_time_s": round(t, 6),
                                       "segment": segment["label"], "target": target,
                                       "joint_speed_norm": float(np.linalg.norm(sim.data.qvel)),
                                       "observation": receiver.provenance(), "threshold_exceedance_time_s": threshold_time,
                                       "threshold_exceedance_to_detection_s": max(0., t-threshold_time),
                                       "detection_to_hold_command_s": 0., "stationary_time_s": None,
                                       "hold_command_to_stationary_s": None, "max_tip_displacement_m": 0.,
                                       "tip_at_hold": sim.tip().tolist(), "low_speed_since": None}
                        events.append(active_hold)
                        if result["first_detection_to_hold_command_s"] is None:
                            result["first_threshold_exceedance_to_detection_s"] = max(0., t-threshold_time)
                            result["first_detection_to_hold_command_s"] = 0.
                            fault_time = trace.get("fault_start_s")
                            if fault_time is not None and t >= fault_time:
                                result["first_fault_to_hold_command_s"] = t-fault_time
                    if waiting_initial or holding:
                        # New accepted IDs only. A burst of old/duplicate packets cannot count.
                        if healthy:
                            if health_since is None:
                                health_since = t
                                recovery_ids.clear()
                                recovery_delivery_ticks.clear()
                            if new_observations:
                                recovery_delivery_ticks.add(round(t, 8))
                            recovery_ids.update(p["sequence_id"] for p in new_observations)
                        else:
                            health_since = None
                            recovery_ids.clear()
                            recovery_delivery_ticks.clear()
                        delivery_span = (max(recovery_delivery_ticks)-min(recovery_delivery_ticks)
                                         if recovery_delivery_ticks else 0.)
                        low_speed_ready = low_speed_since is not None and t - low_speed_since >= stop_dwell - 1e-8
                        observed_ready = (monitor_mode == "off" and receiver.observation is not None) or (
                            healthy and health_since is not None and t-health_since >= settings.recovery_fresh_s - 1e-8
                            and len(recovery_ids) >= min_observations
                            and len(recovery_delivery_ticks) >= min_observations
                            and delivery_span >= settings.recovery_fresh_s - 1e-8)
                        pose_error = float(np.linalg.norm(sim.data.qpos-q_hold))
                        if observed_ready and low_speed_ready and pose_error <= result["resume_pose_error_limit_rad"]:
                            was_initial = waiting_initial
                            waiting_initial = holding = False
                            events.append({"event": "start" if was_initial else "resume", "sim_time_s": round(t, 6),
                                           "segment": segment["label"], "observation": receiver.provenance(),
                                           "distinct_new_observations": len(recovery_ids),
                                           "distinct_delivery_ticks": len(recovery_delivery_ticks),
                                           "new_observation_delivery_span_s": round(delivery_span, 8),
                                           "joint_speed_norm": float(np.linalg.norm(sim.data.qvel)),
                                           "hold_pose_error_norm_rad": pose_error})
                            plan_current(t, "start" if was_initial else "resume")
                            active_hold = None
                        elif t - wait_since >= max_wait:
                            result["failure"] = "initial_observation_timeout" if waiting_initial else "recovery_timeout"
                    if waiting_initial or holding:
                        state = "WAIT_OBSERVATION" if waiting_initial else "HOLD_STALE"
                        note = "有効な新観測と低速状態を確認中" if waiting_initial else "情報の復旧・低速継続を確認中"
                        q_des = q_hold.copy()
                    else:
                        state, note = segment["label"], "採用済み観測から目標を生成"
                        error = float(np.linalg.norm(sim.tip()-goal_xyz))
                        speed = float(np.linalg.norm(sim.data.qvel))
                        if segment["inspect"]:
                            q_des = goal_q.copy()
                            valid_pose = error < settings.position_tolerance_m and speed < .05
                            stable = stable + settings.control_dt if valid_pose else 0.
                            ready = stable >= segment["duration"] - 1e-8
                        else:
                            q_des = q_start + (goal_q-q_start)*smoothstep5(elapsed/segment["duration"])
                            ready = elapsed >= segment["duration"] and error < settings.position_tolerance_m and speed < .07
                        elapsed += settings.control_dt
                        if elapsed > segment["duration"] + 5.:
                            result["failure"] = "segment_timeout:" + segment["label"]
                        if ready and not result["failure"]:
                            if segment["inspect"]:
                                filename = sim.capture(f"{segment['index']:02d}_{target}")
                                if Path(filename).name != filename or not (run_dir / filename).is_file():
                                    raise RuntimeError("capture did not produce the referenced image file")
                                captures.append(filename)
                                files.append(filename)
                                completed.append(target)
                                inspection_errors.append(error)
                                events.append({"event": "capture", "target": target, "sim_time_s": round(t, 6),
                                               "error_m": error, "file": filename, "used_observation": goal_observation,
                                               "latest_observation": receiver.provenance()})
                            seg_index += 1
                            if seg_index == len(segments):
                                result["success"] = True
                                task_complete_time = t
                            else:
                                plan_current(t, "next_segment")
                obs = receiver.observation or {}
                writer.writerow(dict(zip(csv_fields, [round(t, 6), state, target, obs.get("sequence_id"),
                    None if goal_observation is None else goal_observation["sequence_id"], obs.get("capture_time"),
                    obs.get("receive_time"), None if math.isinf(age) else round(age, 6),
                    None if math.isinf(receipt_age) else round(receipt_age, 6), int(healthy), int(holding),
                    round(error, 7), *sim.tip().tolist(), float(np.linalg.norm(sim.data.qvel)), *sim.data.qpos.tolist()])))
                for _ in range(phys_per_ctrl):
                    sim.physics_step(q_des)
                    physics_index += 1
                    pt = float(sim.data.time)
                    speed = float(np.linalg.norm(sim.data.qvel))
                    if not np.isfinite(sim.data.qpos).all() or not math.isfinite(speed) or speed > 25:
                        raise RuntimeError("物理計算が不安定です。成功として出力しません。")
                    if speed <= resume_speed:
                        if low_speed_since is None:
                            low_speed_since = pt
                    else:
                        low_speed_since = None
                    count, distance, abnormal = sim.contact_snapshot()
                    result["contact_samples"] += count
                    if distance is not None:
                        old = result["min_robot_contact_distance_m"]
                        result["min_robot_contact_distance_m"] = distance if old is None else min(old, distance)
                    if abnormal and not result["failure"]:
                        result["failure"] = "robot_contact_penetration_exceeds_1mm"
                        result["success"] = False
                    # Metrics sample every 5 ms, independently of 50 Hz logging.
                    active = check.accepted and terminal_since is None and task_complete_time is None
                    observation_is_old = receiver.observation is not None and receiver.age(pt) > settings.stale_threshold + 1e-9
                    if active and observation_is_old:
                        result["stale_s"] += settings.physics_dt
                        if speed > resume_speed:
                            result["stale_moving_s"] += settings.physics_dt
                        if not holding and not waiting_initial:
                            result["unprotected_stale_s"] += settings.physics_dt
                    if holding and active_hold is not None:
                        result["hold_s"] += settings.physics_dt
                        drift = float(np.linalg.norm(sim.tip()-np.asarray(active_hold["tip_at_hold"])))
                        result["max_hold_drift_m"] = max(result["max_hold_drift_m"], drift)
                        active_hold["max_tip_displacement_m"] = max(active_hold["max_tip_displacement_m"], drift)
                        if speed <= resume_speed:
                            if active_hold["low_speed_since"] is None:
                                active_hold["low_speed_since"] = pt
                            if (active_hold["stationary_time_s"] is None and
                                pt-active_hold["low_speed_since"] >= stop_dwell - 1e-8):
                                active_hold["stationary_time_s"] = pt
                                active_hold["hold_command_to_stationary_s"] = pt-active_hold["sim_time_s"]
                                if result["first_hold_command_to_stationary_s"] is None:
                                    result["first_hold_command_to_stationary_s"] = active_hold["hold_command_to_stationary_s"]
                        else:
                            active_hold["low_speed_since"] = None
                    if record and physics_index % phys_per_frame == 0:
                        sim.video_frame(planner_result.get("input", ""), list(check.targets), state, target,
                                        receiver.age(pt), error, completed, monitor_mode != "off",
                                        f"{monitor_mode}: {note}")
        if check.accepted and not result["success"] and not result["failure"]:
            result["failure"] = "episode_timeout"
        result["task_completion_time_s"] = task_complete_time
        result["status"] = "rejected" if not check.accepted else ("complete" if result["success"] and not result["failure"] else "failed")
        result["success"] = result["status"] == "complete"
    except KeyboardInterrupt as exc:
        result.update(status="interrupted", success=False, failure="KeyboardInterrupt")
        interrupted = exc
    except Exception as exc:
        result.update(status="failed", success=False, failure=f"{type(exc).__name__}: {exc}")
    finally:
        if sim is not None:
            result["sim_duration_s"] = round(float(sim.data.time), 6)
            result["duration_s"] = result["sim_duration_s"]
            result["video_frames"] = sim.frames_written
            result["video_duration_s"] = sim.frames_written / settings.fps if record else None
            try:
                sim.close()
            except Exception as exc:
                result.update(status="failed", success=False, failure="resource_close_error: " + str(exc))
            if record and (run_dir / "clip.mp4").is_file():
                files.append("clip.mp4")
                result["clip_file"] = "clip.mp4"
        if receiver is not None:
            save_json(run_dir / "observation_events.json", receiver.event_log)
            files.append("observation_events.json")
            result["accepted_observations"] = receiver.accepted_count
            result["rejected_observations"] = receiver.rejected_count
        if inspection_errors:
            result["inspection_error_mean_m"] = float(np.mean(inspection_errors))
        for metric in ("hold_s", "stale_s", "stale_moving_s", "unprotected_stale_s", "max_hold_drift_m"):
            result[metric] = round(result[metric], 9)
        result["artifact_hashes"] = {name: _file_hash(run_dir / name) for name in dict.fromkeys(files) if (run_dir / name).is_file()}
        save_json(run_dir / "result.json", result)
    if interrupted is not None:
        raise interrupted
    return result
