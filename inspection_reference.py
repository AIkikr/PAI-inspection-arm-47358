"""Predeclared control experiment; never an LLM answer or a fallback answer."""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)

from inspection_store import digest


def make_control_reference():
    """Return an independently declared, immutable-by-validation A -> B plan."""
    record = {
        "reference_id": "guard_ablation_fixed_ab_v1",
        "source": "control_reference",
        "diagnostic": False,
        "purpose": "guard_ablation",
        "input": "監視方式の比較専用：固定順序 A → B。AIの言語正解率は評価しない。",
        "raw_output": '{"decision":"execute","targets":["A","B"],"finish":"home"}',
        "ai_accuracy_evaluated": False,
        "note": "事前固定した制御比較用計画。AI出力の修正・代替ではない。",
    }
    return dict(record, content_hash=digest(record))


def validate_control_reference(record):
    """Require the exact declared record, not merely a self-consistent hash."""
    if not isinstance(record, dict) or record != make_control_reference():
        raise ValueError("制御比較の参照計画が事前固定した A→B 計画と一致しません。")
    return record


def reference_metadata(record):
    validate_control_reference(record)
    return {
        "planner_source": "control_reference",
        "reference_plan_hash": digest(record),
        "reference_id": record["reference_id"],
        "plan_purpose": "guard_ablation",
        "ai_accuracy_evaluated": False,
    }
