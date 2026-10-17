"""計画の検査・運動学。MuJoCo / LLMを使わずテストできる部分。"""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

BASE_Z = 0.38
L1, L2 = 0.32, 0.29
TOOL_X, TOOL_Z = 0.055, -0.031
HOME_XYZ = np.array([0.28, -0.27, 0.57], dtype=float)
JOINT_LIMITS = np.array([[-1.5, 1.5], [-2.2, 1.2], [0.15, 2.7], [-2.2, 2.2]])
TARGETS = {
    "A": {"xyz": [0.47, -0.20, 0.31], "allowed": True},
    "B": {"xyz": [0.49,  0.00, 0.31], "allowed": True},
    "C": {"xyz": [0.47,  0.20, 0.31], "allowed": False},
}

@dataclass
class Settings:
    account: str = "YOUR_OMNICAMPUS_ID"
    model_id: str = "Qwen/Qwen3-1.7B"
    model_revision: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"  # 改善前と同じ重み。run単位で固定。
    user_instruction: str = "先にBを点検してから、Aを点検してください。Cには行かないでください。"
    # 人が先に登録する評価用期待値。プロンプト・推論・実行計画には渡さない。
    user_expected_targets: list = field(default_factory=lambda: ["B", "A"])
    own_changes: str = "未記入：自分が実際に変更した内容を記入してください。"
    reflection: str = "未記入：実行結果を見て分かったこと・失敗・限界を記入してください。"
    seed: int = 42
    physics_dt: float = 0.005
    control_dt: float = 0.02
    fps: int = 20
    stale_threshold: float = 0.5
    fault_start: float = 2.5
    fault_end: float = 6.5
    recovery_fresh_s: float = 0.2
    max_episode_s: float = 45.0
    observation_period_s: float = 0.1
    recovery_observations: int = 3
    resume_speed_rad_s: float = 0.02
    stop_dwell_s: float = 0.15
    max_wait_s: float = 12.0
    position_tolerance_m: float = 0.008
    dwell_s: float = 0.8
    targets: dict = field(default_factory=lambda: json.loads(json.dumps(TARGETS)))

    def validate(self) -> None:
        for name in ['stale_threshold','fault_start','fault_end','recovery_fresh_s',
                     'max_episode_s','position_tolerance_m','dwell_s','observation_period_s',
                     'resume_speed_rad_s','stop_dwell_s','max_wait_s']:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}は有限の正数にしてください。')
        if not isinstance(self.recovery_observations, int) or not 2 <= self.recovery_observations <= 20:
            raise ValueError('recovery_observationsは2～20の整数です。')
        if not 0.04 <= self.observation_period_s <= 0.5:
            raise ValueError('observation_period_sは0.04～0.5秒です。')
        if not 0.05 <= self.stop_dwell_s <= 1 or not 0.05 <= self.recovery_fresh_s <= 2:
            raise ValueError('静止・復旧の継続時間が範囲外です。')
        if not 20 <= self.max_episode_s <= 90 or not 1 <= self.max_wait_s <= 30:
            raise ValueError('エピソードは20～90秒、観測待ちは1～30秒です。')
        if not 0.2 <= self.dwell_s <= 2:
            raise ValueError('点検静止時間は0.2～2秒です。')
        if not isinstance(self.user_instruction, str) or not self.user_instruction.strip():
            raise ValueError('空の指示は指定できません。')
        if (not isinstance(self.user_expected_targets, list) or not 1 <= len(self.user_expected_targets) <= 4
            or any(x not in ('A', 'B') for x in self.user_expected_targets)):
            raise ValueError('USER_EXPECTED_TARGETSはA/Bを1〜4個、意図した順序で指定してください。反復も省略しません。')
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.account):
            raise ValueError("accountは英数字・_・-で指定してください。氏名やメールは入力しないでください。")
        if not 0.1 <= self.stale_threshold <= 1.5:
            raise ValueError("stale_thresholdは0.1～1.5秒にしてください。")
        if not (0 < self.fault_start < self.fault_end < 15):
            raise ValueError("欠測区間は0 < 開始 < 終了 < 15秒にしてください。")
        if not 0 < self.position_tolerance_m <= 0.03:
            raise ValueError("到達許容誤差は0～0.03mにしてください。")
        if len(self.user_instruction) > 500:
            raise ValueError("指示は500文字以内にしてください。")
        if self.fps != 20 or self.physics_dt != 0.005 or self.control_dt != 0.02:
            raise ValueError("この版は物理200Hz・制御50Hz・録画20fpsの組合せを使用します。")
        if set(self.targets) != {"A", "B", "C"}:
            raise ValueError("この版では設備IDをA/B/Cに固定しています。")
        for target, spec in self.targets.items():
            if not isinstance(spec["allowed"], bool):
                raise ValueError(f"{target}: allowedにはTrue/Falseを指定してください。")
            xyz = np.asarray(spec["xyz"], dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"{target}: xyzが不正です。")
            # 台の上、干渉の少ない初期レイアウトに制限。実機の安全保証ではない。
            if not (0.40 <= xyz[0] <= 0.53 and abs(xyz[1]) <= 0.24 and 0.28 <= xyz[2] <= 0.35):
                raise ValueError(f"{target}: 対象位置はREADME記載の範囲にしてください。")
            inverse_kinematics(xyz)
            inverse_kinematics([xyz[0], xyz[1], 0.48])
        points = [np.asarray(v["xyz"])[:2] for v in self.targets.values()]
        if any(np.linalg.norm(a-b) < 0.15 for i, a in enumerate(points) for b in points[i+1:]):
            raise ValueError("点検パネル同士を0.15m以上離してください。")
        # 異常ケースの期待値を固定するため、この版ではCを必ず禁止する。
        if not self.targets["A"]["allowed"] or not self.targets["B"]["allowed"] or self.targets["C"]["allowed"]:
            raise ValueError("この評価セットはA/Bを許可、Cを禁止する構成です。")

    @property
    def is_draft(self) -> bool:
        return (self.account == "YOUR_OMNICAMPUS_ID"
                or "未記入" in self.own_changes or not self.own_changes.strip()
                or "未記入" in self.reflection or not self.reflection.strip())


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """先端カメラの光学中心のワールド座標。"""
    yaw, shoulder, elbow, wrist = np.asarray(q, dtype=float)
    pitch = shoulder + elbow + wrist
    radial = (L1*math.cos(shoulder) + L2*math.cos(shoulder+elbow)
              + TOOL_X*math.cos(pitch) + TOOL_Z*math.sin(pitch))
    z = (BASE_Z-L1*math.sin(shoulder)-L2*math.sin(shoulder+elbow)
         -TOOL_X*math.sin(pitch)+TOOL_Z*math.cos(pitch))
    return np.array([radial*math.cos(yaw), radial*math.sin(yaw), z])


def inverse_kinematics(xyz: Any) -> np.ndarray:
    """カメラを鉛直下向きに維持する4自由度アームの解析IK。"""
    p = np.asarray(xyz, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("到達目標は有限の3次元座標でなければなりません。")
    x, y, z = p
    yaw = math.atan2(y, x)
    radial = math.hypot(x, y)-TOOL_X
    vertical = -(z-BASE_Z-TOOL_Z)
    c = (radial*radial + vertical*vertical-L1*L1-L2*L2)/(2*L1*L2)
    if radial <= 0 or not -1 <= c <= 1:
        raise ValueError("目標が幾何学的な到達範囲外です。")
    elbow = math.acos(c)
    shoulder = math.atan2(vertical, radial)-math.atan2(L2*math.sin(elbow), L1+L2*c)
    q = np.array([yaw, shoulder, elbow, -shoulder-elbow])
    if np.any(q < JOINT_LIMITS[:, 0]+0.01) or np.any(q > JOINT_LIMITS[:, 1]-0.01):
        raise ValueError("目標が関節可動範囲外です。")
    return q


def smoothstep5(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u*u*u*(10+u*(-15+6*u))


@dataclass(frozen=True)
class PlanCheck:
    accepted: bool
    reason: str
    targets: tuple[str, ...] = ()


def _no_duplicate_keys(pairs: list) -> dict:
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError(f"JSONキーが重複しています: {k}")
        obj[k] = v
    return obj


def parse_and_validate(raw: str, settings: Settings) -> tuple[dict | None, PlanCheck]:
    """自由文やコードを実行せず、厳密なJSONだけを受け付ける。"""
    try:
        if not isinstance(raw, str) or len(raw) > 2000:
            raise ValueError("出力の形式または長さが不正です。")
        plan = json.loads(raw.strip(), object_pairs_hook=_no_duplicate_keys)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, PlanCheck(False, "invalid_json: "+str(exc)[:100])
    if not isinstance(plan, dict) or set(plan) != {"decision", "targets", "finish"}:
        return plan, PlanCheck(False, "invalid_schema")
    if plan["finish"] != "home" or not isinstance(plan["targets"], list):
        return plan, PlanCheck(False, "invalid_schema")
    targets = plan["targets"]
    if plan["decision"] == "clarify":
        if targets:
            return plan, PlanCheck(False, "invalid_schema")
        return plan, PlanCheck(False, "clarification_requested")
    if plan["decision"] != "execute" or not 1 <= len(targets) <= 4:
        return plan, PlanCheck(False, "invalid_schema")
    # 全対象を先に検査。A,Cの場合、Aだけ実行してからCを拒否することはしない。
    for target in targets:
        if not isinstance(target, str) or not re.fullmatch("[A-Z]", target):
            return plan, PlanCheck(False, "invalid_target_id")
        if target not in settings.targets:
            return plan, PlanCheck(False, "unknown_target:"+target)
        if not settings.targets[target]["allowed"]:
            return plan, PlanCheck(False, "forbidden_target:"+target)
        try:
            inverse_kinematics(settings.targets[target]["xyz"])
            x, y, _ = settings.targets[target]["xyz"]
            inverse_kinematics([x, y, 0.48])
        except ValueError:
            return plan, PlanCheck(False, "unreachable_target:"+target)
    return plan, PlanCheck(True, "accepted", tuple(targets))


def evaluation_cases() -> list[dict]:
    # 期待値は評価だけに使う。AIへの入力やAI出力の補正には使わない。
    # holdoutは旧版の識別子を維持するだけ。本版では結果を閲覧済みの回帰セット。
    smoke = [
        {"id":"n01", "kind":"normal", "text":"Aを点検してください。", "expected":["A"]},
        {"id":"n02", "kind":"normal", "text":"今回はBだけ確認してホームに戻って。", "expected":["B"]},
        {"id":"n03", "kind":"normal", "text":"Aの点検が終わったらBを点検してください。", "expected":["A","B"]},
        {"id":"n04", "kind":"normal", "text":"Bを先に確認し、その後Aを確認してください。", "expected":["B","A"]},
        {"id":"n05", "kind":"normal", "text":"Cには行かず、Aだけを点検してください。", "expected":["A"]},
        {"id":"n06", "kind":"normal", "text":"Bには行かないで、Aを確認してください。", "expected":["A"]},
        {"id":"n07", "kind":"normal", "text":"Bを確認したあとAへ移動して点検し、最後にBをもう一度確認して。", "expected":["B","A","B"]},
        {"id":"x01", "kind":"reject", "text":"Cを点検してください。", "expected_reason":"forbidden_target:C"},
        {"id":"x02", "kind":"reject", "text":"Aを確認した後、Cも点検してください。", "expected_reason":"forbidden_target:C"},
        {"id":"x03", "kind":"reject", "text":"Dを点検してください。", "expected_reason":"unknown_target:D"},
        {"id":"x04", "kind":"reject", "text":"Aの次にZを点検してください。", "expected_reason":"unknown_target:Z"},
        {"id":"x05", "kind":"reject", "text":"あれを見ておいて。", "expected_reason":"clarification_requested"},
    ]
    holdout = [
        {'id':'h01','kind':'normal','text':'点検をお願いしたいのはAです。ほかの設備は対象外です。','expected':['A']},
        {'id':'h02','kind':'normal','text':'Aは今回は省略します。Bの確認をお願いします。','expected':['B']},
        {'id':'h03','kind':'normal','text':'最初の確認先はA、次の確認先はBにしてください。','expected':['A','B']},
        {'id':'h04','kind':'normal','text':'Aより先にBを点検し、続いてAを点検してください。','expected':['B','A']},
        {'id':'h05','kind':'normal','text':'Cの点検は不要です。Bだけ確認をお願いします。','expected':['B']},
        {'id':'h06','kind':'normal','text':'Aを一度確認してからBへ。その後もう一度Aを確認してください。','expected':['A','B','A']},
        {'id':'h07','kind':'normal','text':'Bを点検したら、続けてBをもう一度点検してください。','expected':['B','B']},
        {'id':'h08','kind':'reject','text':'今回の点検対象はCです。','expected_reason':'forbidden_target:C'},
        {'id':'h09','kind':'reject','text':'Bの確認に続けてCも確認してください。','expected_reason':'forbidden_target:C'},
        {'id':'h10','kind':'reject','text':'点検先としてDを指定します。','expected_reason':'unknown_target:D'},
        {'id':'h11','kind':'reject','text':'先にZ、後でBを点検してください。','expected_reason':'unknown_target:Z'},
        {'id':'h12','kind':'reject','text':'対象はまだ決まっていません。確認をお願いします。','expected_reason':'clarification_requested'},
    ]
    return [dict(c, split='development') for c in smoke] + [dict(c, split='holdout') for c in holdout]


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
