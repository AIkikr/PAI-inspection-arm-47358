"""Immutable publication from verified run records; never discover captures by glob.

Generated materials describe measured outputs, not an assurance of real-world safety.
The public entry point rejects diagnostic runs and rechecks files before sealing.
"""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)

import copy
import json
import math
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from xml.sax.saxutils import escape

from inspection_core import Settings, save_json
from inspection_store import read_json, digest, file_hash, assert_run, load_stage

STAGES = ("evaluation", "comparison", "demonstrations")


def contained(base: Path, relative: str) -> Path:
    """Resolve only explicitly listed, regular files below base (no symlinks)."""
    base = Path(base).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in str(relative):
        raise ValueError(f"Invalid artifact path: {relative}")
    path = base / rel
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symbolic artifact: {relative}")
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Artifact escapes run: {relative}")
    return resolved


def require_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Missing SHA256: {path.name}")
    if file_hash(path) != expected:
        raise ValueError(f"Artifact changed after execution: {path.name}")


def verify_record(run: Path, record: dict, *, allow_diagnostic: bool = False) -> dict:
    """Verify sealed result, every declared artifact, and declared capture membership."""
    result_path = contained(run, record["result_path"])
    require_hash(result_path, record["result_sha256"])
    actual = read_json(result_path)
    if not allow_diagnostic and (record.get("diagnostic") or actual.get("diagnostic")):
        raise ValueError("Diagnostic evidence cannot enter a submission.")
    # A stage may add evaluation labels but cannot overwrite any simulator result.
    for key, value in actual.items():
        if key in record and record[key] != value:
            raise ValueError(f"Stage/result disagreement: {key}")
    if actual.get("status") in {"running", "interrupted"}:
        raise ValueError("Incomplete episode cannot enter a submission.")
    if actual.get("status") == "failed" and (actual.get("success") or not actual.get("failure")):
        raise ValueError("A failed episode must record failure and cannot claim success.")
    artifacts = actual.get("artifact_hashes")
    if not isinstance(artifacts, dict):
        raise ValueError("Result lacks explicit artifact hashes.")
    verified = {}
    for name, sha in artifacts.items():
        path = contained(result_path.parent, name)
        require_hash(path, sha)
        verified[path.relative_to(Path(run).resolve()).as_posix()] = sha
    for name in actual.get("captures", []):
        if name not in artifacts:
            raise ValueError("Capture missing from explicit artifact manifest.")
    clip = record.get("clip_path")
    if clip and clip not in verified:
        raise ValueError("Demo clip missing from result artifact manifest.")
    verified[record["result_path"]] = record["result_sha256"]
    return {"result":actual, "files":verified,
            "captures":[(result_path.parent / name).relative_to(Path(run).resolve()).as_posix()
                        for name in actual.get("captures", [])]}


def select_evidence(root: Path, run: Path, settings: Settings) -> dict:
    """Freeze one compatible completed stage set and verify only its explicit files."""
    root, run = Path(root), Path(run).resolve()
    manifest = assert_run(root, run, settings)
    if manifest.get("diagnostic"):
        raise ValueError("Diagnostic run cannot be published.")
    from run_project import load_contract, load_plans
    contract_path = contained(run, "planner_contract.json")
    contract = load_contract(run)
    plans = load_plans(root,run,settings)
    if not re.fullmatch(r"[0-9a-f]{40}", str(contract.get("revision", ""))):
        raise ValueError("The run has no immutable model revision.")
    contract_hash = digest(contract)
    selected = {name:load_stage(run, name) for name in STAGES}
    plan_hashes = {stage.get("plan_set_hash") for stage in selected.values()}
    if len(plan_hashes) != 1 or next(iter(plan_hashes)) != digest(plans):
        raise ValueError("Evaluation, comparison and video use different AI plans.")
    plan_manifest = read_json(run/"plans_manifest.json")
    files = {"plans/"+key+".json":sha for key,sha in plan_manifest["files"].items()}
    for rel in ["manifest.json","manifest.sha256","planner_contract.json","planner_contract.sha256",
                "plans_manifest.json","settings.json","environment.json","environment.sha256"]:
        files[rel] = file_hash(contained(run,rel))
    captures, selected_inputs = [], {}
    for name, stage in selected.items():
        if (stage.get("status") != "complete" or stage.get("run_id") != manifest["run_id"]
                or stage.get("contract_hash") != contract_hash):
            raise ValueError(f"Incompatible or incomplete stage: {name}")
        stage_path = contained(run, stage["_path"])
        require_hash(stage_path, stage["_sha256"])
        selected_inputs[name] = {"path":stage["_path"], "sha256":stage["_sha256"],
                                "stage_id":stage["stage_id"]}
        files[stage["_path"]] = stage["_sha256"]
        payload = stage["payload"]
        if payload.get("diagnostic"):
            raise ValueError("Diagnostic stage cannot be published.")
        if name == "comparison" and payload.get("records"):
            from inspection_reference import validate_control_reference, reference_metadata
            reference = payload.get("reference_plan", {})
            validate_control_reference(reference)
            origin = reference_metadata(reference)
            if any(payload.get(key) != value for key, value in origin.items()):
                raise ValueError("Comparison reference provenance is inconsistent.")
        for record in payload.get("records", []):
            checked = verify_record(run, record)
            verify_planner_origin(run, name, record, checked["result"], payload, plans)
            files.update(checked["files"])
            if name == "demonstrations":
                captures.extend(checked["captures"])
            if record.get("trace_path"):
                trace_path=contained(run,record["trace_path"])
                require_hash(trace_path,record["trace_file_sha256"])
                files[record["trace_path"]]=record["trace_file_sha256"]
        # Trace files or per-stage CSVs must also be explicit hash-addressed entries.
        for rel, sha in payload.get("files", {}).items():
            require_hash(contained(run, rel), sha)
            files[rel] = sha
    return {"manifest":manifest, "contract":contract, "contract_hash":contract_hash,
            "plan_set_hash":next(iter(plan_hashes)), "stages":selected,
            "selected_inputs":selected_inputs, "files":files, "captures":captures,
            "diagnostic":False}


def verify_planner_origin(run: Path, stage_name: str, record: dict, actual: dict,
                          payload: dict, plans: dict) -> None:
    """Control references are valid only for the explicitly separated ablation."""
    plan_path = contained(run, record["result_path"]).parent / "plan.json"
    require_hash(plan_path, actual.get("artifact_hashes", {}).get("plan.json"))
    planner = read_json(plan_path).get("planner", {})
    if stage_name == "comparison":
        from inspection_reference import validate_control_reference, reference_metadata
        reference = payload.get("reference_plan", {})
        validate_control_reference(reference)
        if planner != reference:
            raise ValueError("Comparison episode used a different control reference.")
        for key, value in reference_metadata(reference).items():
            if actual.get(key) != value or record.get(key) != value or payload.get(key) != value:
                raise ValueError("Comparison reference provenance is inconsistent: " + key)
    else:
        if actual.get("planner_source") != "llm" or planner.get("source") != "llm":
            raise ValueError("Evaluation and demos require actual LLM output, never a control reference.")
        key = record.get("id") if stage_name == "evaluation" else record.get("source_plan_id")
        if key not in plans or planner != plans[key]:
            raise ValueError("Episode plan differs from the verified LLM inference: " + str(key))


def rejection_label(record: dict) -> str:
    if record.get("accepted"):
        return "実行許可／拒否失敗"
    return "実行抑止／理由一致" if record.get("reason_correct") else "実行抑止／理由不一致"


def comparison_label(comparison: dict) -> str:
    records = comparison.get("records", [])
    if not comparison.get("available") or not records:
        return "監視方式の比較未実施: " + str(comparison.get("reason", "記録なし"))
    completed = sum(r.get("success") is True for r in records)
    return f"監視比較 {len(records)}/12条件を実行・タスク完了 {completed}/{len(records)}。固定A→B計画、AI精度とは別評価。"


def result_card_lines(evidence: dict) -> list[str]:
    """Every claim on the final card is derived from the selected measurements."""
    from inspection_baseline import baseline_comparison
    evaluation = _payload(evidence, "evaluation")
    summary = evaluation_summary(evaluation["records"])
    baseline = baseline_comparison(evaluation)
    before = next((m for m in baseline.get("metrics", []) if m["key"] == "end_to_end_success_n"), None)
    task_line = (f"同じ既存指示の達成: 改善前 {before['before']}/{before['total']} → 今回 {before['after']}/{before['total']}。"
                 if baseline.get("available") and before else
                 f"正常指示達成 {summary['end_to_end_success_n']}/{summary['normal_n']}。改善前との比較は成立せず。")
    return [task_line,
            f"異常指示: 抑止 {summary['blocked_n']}/{summary['invalid_n']} / 理由一致 {summary['correct_rejection_reason_n']}/{summary['invalid_n']}",
            comparison_label(_payload(evidence, "comparison")),
            "既に確認した24指示の回帰評価。未使用指示への汎化・実機安全は未検証。"]


def semantic_label(demo: dict) -> str:
    """Never call a physically completed wrong plan an achieved instruction."""
    if demo.get("expected_reason"):
        return ("指示達成（期待した理由で実行を抑止）" if demo.get("task_success") is True else
                rejection_label(dict(demo, reason_correct=False))+" / "+str(demo.get("validation_reason","理由未記録")))
    if demo.get("expected") is None:
        return ("生成された計画の実行完了（指示との一致は未判定）" if demo.get("success") else
                "生成された計画は未完了 / " + str(demo.get("failure") or demo.get("validation_reason")))
    if demo.get("task_success") is True:
        return "指示達成（期待値と実行結果が一致）"
    if demo.get("success"):
        return "計画の実行完了 / 指示達成は不一致"
    if not demo.get("accepted"):
        return "実行抑止 / " + str(demo.get("validation_reason", "理由未記録"))
    return "未完了 / " + str(demo.get("failure") or "期待値との不一致")


def _payload(evidence, name):
    return evidence["stages"][name]["payload"]


def evaluation_summary(records: list[dict]) -> dict:
    normal = [r for r in records if r.get("kind") == "normal"]
    invalid = [r for r in records if r.get("kind") != "normal"]
    return {"normal_n":len(normal), "invalid_n":len(invalid),
            "language_exact_n":sum(r.get("language_exact") is True for r in normal),
            "end_to_end_success_n":sum(r.get("end_to_end_success") is True for r in normal),
            "blocked_n":sum(not r.get("accepted", True) for r in invalid),
            "correct_rejection_reason_n":sum(r.get("reason_correct") is True for r in invalid)}


def inspect_video(path: Path, *, expected_frames: int | None = None, fps: int = 20) -> dict:
    """Decode actual file; do not accept a stored duration as proof."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path), format="FFMPEG")
    count = 0
    try:
        meta = reader.get_meta_data()
        actual_fps = float(meta["fps"])
        duration = float(meta.get("duration", float("nan")))
        if not math.isfinite(duration) or duration <= 0 or abs(actual_fps-fps) > 0.01:
            raise ValueError("Cannot verify actual video duration/FPS with FFmpeg.")
        for frame in reader:
            if frame.shape[:2] != (640, 1024):
                raise ValueError("Unexpected video dimensions.")
            count += 1
    finally:
        reader.close()
    if expected_frames is not None and count != expected_frames:
        raise ValueError("Decoded frame count differs from episode record.")
    if abs(duration-count/actual_fps) > max(0.1, 2/fps):
        raise ValueError("FFmpeg duration differs from decoded frame count.")
    return {"frames":count, "fps":actual_fps, "duration_s":duration,
            "decoded_duration_s":count/actual_fps, "width":1024, "height":640,
            "sha256":file_hash(path), "simulation_playback_speed":1}


def create_submission_video(run: Path, output: Path, settings: Settings, evidence: dict) -> dict:
    import imageio.v2 as imageio
    from inspection_sim import information_card
    demos = _payload(evidence, "demonstrations")["records"]
    summary = evaluation_summary(_payload(evidence, "evaluation")["records"])
    if len(demos) != 4:
        raise ValueError("The complete submission video requires four recorded scenes.")
    fps = settings.fps
    # The leading/trailing cards may add time; clips are never sped up or clipped.
    frame_total = (6+4*3+8)*fps + sum(int(d["video_frames"]) for d in demos)
    padding = max(0, 60*fps-frame_total)
    expected = frame_total+padding
    if not 60*fps <= expected <= 180*fps:
        raise ValueError(f"動画予定尺 {expected/fps:.1f} 秒が60～180秒の範囲外です。")
    count = 0
    with imageio.get_writer(str(output), format="FFMPEG", mode="I", fps=fps,
                            codec="libx264", pixelformat="yuv420p", quality=7,
                            macro_block_size=16) as writer:
        def card(title, lines, n):
            nonlocal count
            frame = information_card(title, lines, "動作はシミュレーション時間の1倍速 / AI計画は事前推論")
            for _ in range(n):
                writer.append_data(frame)
                count += 1
        card("自然言語と情報鮮度による設備点検", [
            f"OmniCampus: {settings.account}", "AIが指示を対象列へ変換。独立した監視・制御機構が実行する。",
            "点検は到達・静止・画像保存。画像診断と実機安全保証は対象外。",
            f"run: {evidence['manifest']['run_id']}",
        ], 6*fps)
        for demo in demos:
            verify_record(run, demo)
            card(demo["title"], [demo["instruction"], semantic_label(demo),
                 "指示達成と、生成された計画の物理的な実行完了を分けて記録。"], 3*fps)
            reader = imageio.get_reader(str(contained(run, demo["clip_path"])), format="FFMPEG")
            written = 0
            try:
                if abs(float(reader.get_meta_data()["fps"])-fps) > 0.01:
                    raise ValueError("素材FPSが一致しません。自動変速は行いません。")
                for frame in reader:
                    if frame.shape[:2] != (640, 1024):
                        raise ValueError("素材の解像度が一致しません。")
                    writer.append_data(frame)
                    count += 1
                    written += 1
            finally:
                reader.close()
            if written != demo["video_frames"]:
                raise ValueError("素材のフレーム数が実行ログと一致しません。")
        card("測定結果と適用範囲", result_card_lines(evidence), 8*fps+padding)
    if count != expected:
        raise ValueError("Internal video frame accounting mismatch.")
    return inspect_video(output, expected_frames=expected, fps=fps)


def create_report(run: Path, output: Path, settings: Settings, evidence: dict,
                  video: dict, own_changes: str, reflection: str) -> Path:
    """Paginated Japanese report. QA may call this directly with diagnostic=True."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    fn = "HeiseiKakuGo-W5"
    navy, teal = colors.HexColor("#143549"), colors.HexColor("#176c86")
    styles = {"body":ParagraphStyle("jp", fontName=fn, fontSize=9.0, leading=14.4,
                  wordWrap="CJK", textColor=navy, spaceAfter=7),
              "small":ParagraphStyle("jpSmall", fontName=fn, fontSize=7.5, leading=11.2,
                  wordWrap="CJK", textColor=navy, spaceAfter=4),
              "h1":ParagraphStyle("jpTitle", fontName=fn, fontSize=21, leading=30,
                  wordWrap="CJK", textColor=navy, spaceAfter=17),
              "h2":ParagraphStyle("jpSection", fontName=fn, fontSize=12.5, leading=19,
                  wordWrap="CJK", textColor=teal, spaceBefore=7, spaceAfter=7)}
    def p(text, style="body"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])
    def table(rows, widths):
        t = Table([[p(x,"small") for x in row] for row in rows], colWidths=widths,
                  hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e2edf1")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f1f5f7")]),
            ("LINEBELOW",(0,0),(-1,0),0.6,teal)]))
        return t
    width = A4[0]-84
    records = _payload(evidence,"evaluation")["records"]
    summary = evaluation_summary(records)
    comparison = _payload(evidence,"comparison")
    from inspection_baseline import baseline_comparison
    baseline = baseline_comparison(_payload(evidence,"evaluation"))
    contract = evidence["contract"]
    diagnostic = evidence.get("diagnostic",False)
    draft = not (settings.account != "YOUR_OMNICAMPUS_ID" and own_changes.strip()
                 and reflection.strip() and "未記入" not in own_changes+reflection)
    story = [p("自然言語と情報鮮度による\n設備点検アーム", "h1"),
             p(f"OmniCampus: {settings.account}"),
             p("診断用サンプル / 性能実績・提出物に使用不可" if diagnostic else
               ("下書き / 本人の記入・内容確認が必要" if draft else "実行ログに基づく説明資料"),"small"),
             p("1. 目的と実装範囲","h2"),
             p("AIが日本語の指示を点検対象と順序に変換し、自作4自由度アームがMuJoCo上で指定位置へ移動する。"
               "点検は到達・静止・手先画像の保存まで。画像診断や実機運用は実装しない。"
               "許可されない計画は実行前に検査し、対象観測が古くなったときは独立したルールで動作を保留する。"),
             p("実測結果","h2"), table([
               ["測定項目","結果","判定"],
               ["正常指示の言語一致",f"{summary['language_exact_n']}/{summary['normal_n']}","対象と順序が期待値に完全一致"],
               ["正常指示の達成",f"{summary['end_to_end_success_n']}/{summary['normal_n']}","言語一致・到達・画像保存・ホーム復帰"],
               ["異常指示の実行抑止",f"{summary['blocked_n']}/{summary['invalid_n']}","形式不正による抑止も含む"],
               ["異常指示の理由一致",f"{summary['correct_rejection_reason_n']}/{summary['invalid_n']}","期待した拒否理由に一致"],
             ],[140,54,width-194]),
             p(f"動画 {video['duration_s']:.2f}秒 / {video['fps']:.0f}fps。動作部分はシミュレーション時間の1倍速。","small"),
             p("結果の読み方","h2"),p("AIの誤解釈は正解に置換しない。物理的に計画を完走しても、指示との順序が違えば指示達成とはしない。"
               "自由入力も、事前登録した期待順序で指示達成を判定する。"
               "旧開発用12指示と既存評価（回帰）12指示を分けて記録する。24指示はいずれも改善前に結果を確認済みであり、未使用データではない。"
               "固定配置の小規模評価であり、汎用性能や統計的な頑健性は主張しない。"),
             PageBreak(),p("2. 構成と通信観測の扱い","h1"),
             table([["層","担当する処理"],
               ["AI","Qwenの事前学習済み重みで推論。JSONのdecision / targets / finishだけを出力。"],
               ["実行条件","JSON・ID・権限・到達可能性を検査。未知IDや禁止対象があれば全計画を拒否。"],
               ["観測","取得時刻・受信時刻・連番を持つ観測を配送。重複・逆順・不正時刻のデータは採用しない。"],
               ["制御","採用した観測に由来する座標でIK・軌道を構成。PDトルクとmj_stepで物理時間を進める。"],
               ["監視","監視なし / 受信途絶 / 取得時刻の鮮度の3方式。同じ通信系列・初期状態・固定A→B参照計画を使用。AI精度とは独立した制御実験。"],
             ],[73,width-73]),
             p("開始・保留・再開","h2"),p("有効な観測がない間は移動を開始しない。保留中は現在姿勢を維持し、関節・カメラの取得と物理計算を継続する。"
               "復旧は新しい観測の継続と低速条件を満たしてから判断し、採用観測と現在姿勢から再計画する。古い到着だけで再開しない。"
               "全時計をシミュレーション時間に統一し、実機の時計同期誤差は対象外とする。"),
             p("鮮度と到達の定義","h2"),p(f"情報の古さは現在時刻から採用中の観測の取得時刻を引いた値。鮮度閾値は{settings.stale_threshold:.2f}秒。"
               f"物理200Hz・制御50Hz・動画20fps。先端誤差{settings.position_tolerance_m*1000:.1f}mm未満と低速が{settings.dwell_s:.1f}秒続くと画像を保存する。"
               "位置・権限はシミュレータから与え、画像認識で推定しない。"),
             PageBreak(),p("3. 同一通信系列による比較","h1"),
             p(comparison_label(comparison)),
             p("この節だけは固定A→B計画（control_reference）を使用する。AIが誤解釈した計画の救済や成功への置換ではなく、監視方式のみを変える独立した制御比較である。AIの評価・動作デモには実際のLLM出力を使う。","small")]
    rows = [["通信条件 / 方式","完了","所要時間 s","保留 s","古い情報下の移動 s","保留中変位 mm"]]
    modes = {"off":"監視なし","receipt":"受信途絶","age":"取得時刻"}
    for r in comparison.get("records",[]):
        rows.append([str(r.get("scenario",""))+" / "+modes.get(r.get("mode"),str(r.get("mode",""))),
            "○" if r.get("success") else "未完了", _number(r.get("sim_duration_s")),
            _number(r.get("hold_s")),_number(r.get("stale_moving_s")),
            _number(None if r.get("max_hold_drift_m") is None else r["max_hold_drift_m"]*1000)])
    if len(rows)>1:
        story.append(table(rows,[145,38,65,52,90,width-390]))
    else:
        story.append(p("比較未実施: "+str(comparison.get("reason","記録なし"))))
    story += [p("指標の範囲","h2"),p("古い情報下の移動時間は鮮度閾値超過中の関節運動時間。保留中の最大変位は保留開始位置からの最大距離で、"
              "経路に沿った停止距離ではない。保留しない方式の変位0は安全を意味しない。完了率・所要時間も併記し、停止し続ける方式が有利にならないようにする。"),
              p("故障開始→保留指令、閾値超過→検出、検出→保留切替、保留指令→継続的低速の4区間をログに記録する。"
                "検出と保留が同じ制御周期なら切替時間0秒は正常。開始時刻と動作区間・速度もログで確認する。測定周期は結果JSONの定義に従う。","small"),
              p("対象は静止しており、衝突を発生させる比較ではない。監視だけを変更し、観測採用規則は全方式に共通。通信系列のSHA256を各結果に残す。","small"),
              PageBreak(),p("4. 改善前との比較・旧開発指示","h1")]
    if baseline.get("available"):
        story += [table([["同じ24指示での指標","改善前","今回","差（件）"]] + [
            [m["label"], f"{m['before']}/{m['total']}", f"{m['after']}/{m['total']}", f"{m['delta']:+d}"]
            for m in baseline["metrics"]], [width-165,55,55,55]),
            p("改善前 run: " + baseline["source_run_id"] + "\n元ZIP SHA256: " + baseline["source_zip_sha256"], "small"),
            p(baseline["note"], "small")]
        if contract.get("revision") != baseline["same_model_revision_expected"]:
            story.append(p("今回のモデルrevisionは改善前と異なる。差にはモデル変更も含まれるため、プロンプト変更だけの効果とは扱わない。", "small"))
    else:
        story.append(p("改善前との数値比較なし: " + str(baseline.get("reason", "条件が一致しないため")), "small"))
    story.append(p("以下は旧開発用12指示の再評価。次ページの12指示も既に結果を確認した既存評価（回帰）であり、未使用指示への汎化性能を表さない。", "small"))
    for index,split in enumerate(["development","holdout"]):
        if index:
            story += [PageBreak(),p("5. 既存評価（回帰）の結果","h1")]
        subset = [r for r in records if r.get("split")==split]
        split_summary=evaluation_summary(subset)
        story.append(p(f"正常指示: 言語一致 {split_summary['language_exact_n']}/{split_summary['normal_n']} / 指示達成 {split_summary['end_to_end_success_n']}/{split_summary['normal_n']}。"
                       f"異常指示: 理由一致 {split_summary['correct_rejection_reason_n']}/{split_summary['invalid_n']}。","small"))
        rows = [["ID","指示","計画検査","評価"]]
        for r in subset:
            normal = r.get("kind")=="normal"
            outcome = ("指示達成" if r.get("end_to_end_success") else "不一致 / 未完了") if normal else rejection_label(r)
            rows.append([r.get("id",""),r.get("instruction",""),r.get("validation_reason",""),outcome])
        story.append(table(rows,[30,width-195,95,70]))
        story.append(p("AIの出力は期待値で修正しない。JSONエラーで抑止できた場合と、正しい理由で拒否した場合を区別する。","small"))
    captures = evidence.get("captures",[])[:2]
    if captures:
        story += [p("この試行に列挙・検証された手先画像","h2"),
                  Table([[Image(str(contained(run,name)),width=172,height=129) for name in captures]],hAlign="LEFT")]
    env = read_json(run/"environment.json") if (run/"environment.json").is_file() else {}
    versions = " / ".join(str(k)+": "+str(v) for k,v in env.items() if k in {"python","torch","mujoco","transformers","numpy"})
    story += [PageBreak(),p("6. 再現性・考察・限界","h1"),p("本人が実際に変更した内容","h2"),p(own_changes),
              p("実行結果に基づく考察","h2"),p(reflection),
              p("実験の識別","h2"),
              p(f"提出名義: {settings.account}\n実験開始時のaccount: {evidence['manifest'].get('settings',{}).get('account',settings.account)}。提出名義のみ変更しても、実験条件・測定値・原本は変更しない。", "small"),
              p(f"run: {evidence['manifest']['run_id']}\nmodel: {contract['model_id']}\nrevision: {contract['revision']}\n"
                f"contract SHA256: {evidence['contract_hash']}\nplan set SHA256: {evidence['plan_set_hash']}\n{versions}","small"),
              p("設定・実験時ソース・モデルの確定revision・プロンプト・通信系列をrun単位で固定する。再試行は新しい保存先を作る。"
                "PDFとZIPには選択した試行の結果に列挙された成果物だけを収録し、ファイルSHA256を検査する。設定やコードを変える場合は新runにする。","small"),
              p("限界と参考研究","h2"),p("AIは限定した対象列への変換を担い、鮮度監視はルールベースである。SayCanの価値関数によるスキル選択や、"
                "Code as Policiesの自由なコード生成を再現したものではない。先行研究から高位判断と既存制御の分離を参考にした。"
                "AoI研究の取得時刻に基づく鮮度定義を、保留・再開の比較へ適用する。"),
              p("接触判定は設定したしきい値の異常検出であり、無接触・実機安全を保証しない。プロセス停止に対するハードウェアwatchdog、"
                "非常停止装置、動的障害物回避、時計同期、故障診断は対象外。","small")]
    for citation in ["SayCan: Ahn et al., 2022. https://arxiv.org/abs/2204.01691",
                     "Code as Policies: Liang et al., 2022. https://arxiv.org/abs/2209.07753",
                     "Age of Information: Yates et al., 2021. https://arxiv.org/abs/2007.08564",
                     "Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B",
                     "MuJoCo 3.3.7: https://mujoco.readthedocs.io/en/3.3.7/"]:
        story.append(p(citation,"small"))
    def footer(canvas, doc):
        canvas.setStrokeColor(teal); canvas.line(42,32,A4[0]-42,32)
        canvas.setFont(fn,8); canvas.setFillColor(navy)
        canvas.drawString(42,20,"DIAGNOSTIC - NOT FOR SUBMISSION" if diagnostic else "PAI | SIMULATION STUDY")
        canvas.drawRightString(A4[0]-42,20,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=43,
                      title="PAI 設備点検アーム",author="").build(story,onFirstPage=footer,onLaterPages=footer)
    return output


def _number(value):
    return "-" if value is None else f"{float(value):.3f}"


def make_checks(evidence: dict, settings: Settings, own_changes: str, reflection: str,
                video: dict, pdf_ok: bool, source_ok: bool, hashes_ok: bool) -> dict:
    """The top-level readiness is the AND of every named mandatory check."""
    evaluations = _payload(evidence,"evaluation").get("records",[])
    demos = _payload(evidence,"demonstrations").get("records",[])
    comparisons = _payload(evidence,"comparison").get("records",[])
    summary = evaluation_summary(evaluations)
    checks = {
        "account_recorded":settings.account != "YOUR_OMNICAMPUS_ID",
        "own_changes_recorded":bool(own_changes.strip()) and "未記入" not in own_changes,
        "reflection_recorded":bool(reflection.strip()) and "未記入" not in reflection,
        "non_diagnostic_evidence":not evidence.get("diagnostic",False),
        "evaluation_complete":summary["normal_n"] == 14 and summary["invalid_n"] == 10 and
            all(sum(r.get("split")==split for r in evaluations)==12 for split in ["development","holdout"]),
        "at_least_one_correct_task":summary["end_to_end_success_n"] > 0,
        "no_runtime_failed_trials":all(not r.get("failure") for r in evaluations+comparisons+demos),
        "comparison_complete":bool(_payload(evidence,"comparison").get("available")) and len(comparisons)==12,
        "four_recorded_demos":len(demos)==4,
        "expected_demos_match":bool(demos) and all(r.get("task_success") is True for r in demos if r.get("expected") is not None or r.get("expected_reason")),
        "a_verified_expected_demo":any(r.get("expected") is not None and r.get("task_success") is True for r in demos),
        "video_60_to_180_seconds":60 <= video.get("duration_s",-1) <= 180,
        "video_decoded_and_verified":bool(video.get("frames",0)>0 and video.get("sha256")),
        "pdf_created_and_readable":bool(pdf_ok),
        "source_zip_created_and_verified":bool(source_ok),
        "evidence_hashes_verified":bool(hashes_ok),
    }
    return {"required":checks,"ready_for_manual_review":all(checks.values()),
            "note":"機械的な確認のみ。講座の合格判定ではない。本人が内容・個人情報・公開設定・最新の提出要項を確認する。"}


def create_source_archive(run: Path, output: Path, settings: Settings, evidence: dict, notebook: Path,
                          publication_identity: dict | None = None) -> None:
    """Whitelist execution-time snapshot and referenced evidence, never current root."""
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as archive:
        for rel, sha in evidence["manifest"]["source_hashes"].items():
            path = contained(Path(run)/"source",rel)
            require_hash(path,sha)
            archive.write(path,rel)
        archive.write(notebook,"PAI_Inspection_Colab_v3.ipynb")
        # Export is self-contained but excludes weights, videos, tokens and unrelated trials.
        for rel, sha in evidence["files"].items():
            require_hash(contained(run,rel),sha)
            if Path(rel).suffix.lower() in {".json",".csv",".xml",".png",".sha256"}:
                archive.write(contained(run,rel),"results/"+rel)
        archive.writestr("results/selected_inputs.json",json.dumps(evidence["selected_inputs"],ensure_ascii=False,indent=2))
        if publication_identity is not None:
            archive.writestr("results/publication_identity.json",json.dumps(publication_identity,ensure_ascii=False,indent=2))


def create_submission(root: Path, run: Path, settings: Settings,
                      own_changes: str, reflection: str, submission_account: str | None = None) -> dict:
    """Create a new immutable publication; failures never resurrect older successes."""
    from inspection_store import start_stage, seal_stage, fail_stage
    from notebook_builder import build_notebook
    from pypdf import PdfReader
    root, run = Path(root), Path(run).resolve()
    # Invalidate an older publication before any input verification can fail.
    stage = start_stage(run,"publication","pending","pending")
    folder = Path(stage)
    try:
        evidence = select_evidence(root,run,settings)
        # Identity is presentation metadata, never a rewrite of the experiment.
        account = settings.account if submission_account is None else submission_account
        if not isinstance(account, str):
            raise ValueError("提出アカウント名は文字列で指定してください。")
        render_settings = replace(settings, account=account)
        render_settings.validate()
        identity = {"experiment_account":settings.account,"publication_account":account,
                    "run_id":evidence["manifest"]["run_id"],
                    "note":"提出名義だけを別記。保存済み実験設定・推論・測定結果は変更していない。"}
        stage_data=read_json(folder/"stage.json")
        stage_data.update(contract_hash=evidence["contract_hash"],plan_set_hash=evidence["plan_set_hash"])
        save_json(folder/"stage.json",stage_data)
        # Author text lives in this publication; it does not change run settings.
        if len(own_changes)>500 or len(reflection)>800:
            raise ValueError("本文の変更内容は500字、考察は800字以内にしてください。詳細は別途READMEへ記載できます。")
        video_path = folder/f"PAI最終課題_{account}.mp4"
        pdf_path = folder/f"説明資料_PAI最終課題_{account}.pdf"
        notebook = folder/"PAI_Inspection_Colab_v3.ipynb"
        source_zip = folder/"github_source.zip"
        video = create_submission_video(run,video_path,render_settings,evidence)
        save_json(folder/"video_metadata.json",video)
        create_report(run,pdf_path,render_settings,evidence,video,own_changes,reflection)
        page_count = len(PdfReader(str(pdf_path)).pages)
        pdf_ok = 4 <= page_count <= 10 and pdf_path.stat().st_size>1000
        build_notebook(run/"source",notebook,settings=render_settings,model_revision=evidence["contract"]["revision"])
        create_source_archive(run,source_zip,render_settings,evidence,notebook,publication_identity=identity)
        with zipfile.ZipFile(source_zip) as archive:
            source_ok = archive.testzip() is None and "PAI_Inspection_Colab_v3.ipynb" in archive.namelist()
        # Recheck the selected set, not whichever stage happens to be newest now.
        for rel, sha in evidence["files"].items():
            require_hash(contained(run,rel),sha)
        assert_run(root,run,settings)
        checks = make_checks(evidence,render_settings,own_changes,reflection,video,pdf_ok,source_ok,True)
        save_json(folder/"pre_submission_checks.json",checks)
        save_json(folder/"publication_inputs.json",{
            **identity,
            "selected_inputs":evidence["selected_inputs"],"contract_hash":evidence["contract_hash"],
            "plan_set_hash":evidence["plan_set_hash"],"own_changes":own_changes,"reflection":reflection,
            "pdf_pages":page_count,"evidence_files":evidence["files"],
        })
        bundle = folder/f"PAI_submission_{account}.zip"
        bundle_members = [source_zip,pdf_path,video_path,folder/"pre_submission_checks.json",
                          folder/"publication_inputs.json",folder/"video_metadata.json"]
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
            for path in bundle_members:
                archive.write(path,path.name)
            archive.writestr("提出手順.txt", "PDF・動画・チェック結果を本人が確認してください。\n"
                "github_source.zipを展開してpublic GitHubへ登録。MP4をYouTubeへ限定公開以上でアップロード。\n"
                f"動画タイトル: PAI最終課題_{account}\n"
                "OmniCampusへPDF・GitHub URL・YouTube URLを提出。提出直前に最新要項を確認。自動公開はしません。\n")
        output_files = bundle_members+[bundle,notebook]
        payload = {"bundle":bundle.relative_to(run).as_posix(),"pdf":pdf_path.relative_to(run).as_posix(),
            "video":video_path.relative_to(run).as_posix(),"source_zip":source_zip.relative_to(run).as_posix(),
            "checks":checks,"selected_inputs":evidence["selected_inputs"],**identity,
            "files":{p.relative_to(run).as_posix():file_hash(p) for p in output_files}}
        seal_stage(run,"publication",folder,payload)
        return {**payload,"bundle":str(bundle),"pdf":str(pdf_path),"video":str(video_path),"source_zip":str(source_zip)}
    except BaseException as exc:
        fail_stage(run,"publication",folder,exc)
        raise
