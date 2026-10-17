"""One immutable run, real inference, and independently versioned stage attempts."""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)
import csv
from pathlib import Path
import time
from inspection_resume import open_resumable_stage, save_progress, checkpoint_event
from inspection_core import Settings, save_json, parse_and_validate
from inspection_store import (assert_run, digest, file_hash, read_json, unique_id,
    start_stage, seal_stage, fail_stage, invalidate_stages, new_run, load_latest,
    create_backup, restore_backup, assert_execution_environment)

def _requests(manifest):
    return [{'id':c['id'],'text':c['text']} for c in manifest['cases']] + [
        {'id':'custom','text':manifest['settings']['user_instruction']}]

def load_contract(run):
    run = Path(run)
    contract = read_json(run/'planner_contract.json')
    if digest(contract) != (run/'planner_contract.sha256').read_text().strip():
        raise ValueError('モデル・プロンプトの固定記録が変更されています。')
    return contract

def _check_plan(record, req, contract):
    ch = digest(contract)
    expected = digest({'id':req['id'],'input':req['text'],'contract_hash':ch})
    if (record.get('source') != 'llm' or record.get('diagnostic',False)
        or record.get('revision') != contract['revision']
        or record.get('model_id') != contract['model_id']
        or record.get('contract_hash') != ch or record.get('input') != req['text']
        or record.get('cache_key') != expected):
        raise ValueError('推論キャッシュのモデル・入力・プロンプトが一致しません: '+req['id'])
    if digest({k:v for k,v in record.items() if k!='content_hash'}) != record.get('content_hash'):
        raise ValueError('推論キャッシュの内容が変更されています: '+req['id'])

def generate_plans(root, run, settings, *, checkpoint=None):
    from inspection_ai import planner_request, resolve_contract, QwenPlanner
    run = Path(run)
    manifest = assert_run(root,run,settings)
    assert_execution_environment(run)
    request = planner_request(settings)
    if request != manifest['planner_request']:
        raise ValueError('プロンプト等がrunと異なります。新runが必要です。')
    if (run/'planner_contract.json').exists():
        contract = load_contract(run)
        if contract['request_hash'] != digest(request):
            raise ValueError('モデル指定・推論条件が変わっています。新runが必要です。')
    else:
        contract = resolve_contract(settings)
        if contract['request_hash'] != digest(request): raise ValueError('モデル解決時の条件不一致')
        save_json(run/'planner_contract.json',contract)
        (run/'planner_contract.sha256').write_text(digest(contract),encoding='ascii')
    requests = _requests(manifest)
    plans, missing = {}, []
    for req in requests:
        path = run/'plans'/(req['id']+'.json')
        if path.exists():
            record = read_json(path)
            _check_plan(record,req,contract)
            plans[req['id']] = record
        else: missing.append(req)
    if missing: invalidate_stages(run)
    planner = None
    attempt_path = run/'inference_attempts'/(unique_id('infer')+'.json')
    save_json(attempt_path,{'status':'running','contract_hash':digest(contract),'pending':[x['id'] for x in missing]})
    try:
        checkpoint_event(checkpoint,'inference:start')
        if missing:
            planner = QwenPlanner(settings,contract,require_gpu=True,
                                  log_path=attempt_path.with_suffix('.worker.log'))
        for req in missing:
            record = planner.infer(req['text'])
            record['cache_key'] = digest({'id':req['id'],'input':req['text'],'contract_hash':digest(contract)})
            record['content_hash'] = digest(record)
            _check_plan(record,req,contract)
            save_json(run/'plans'/(req['id']+'.json'),record)
            plans[req['id']] = record
            print(req['id'],record['raw_output'])
            checkpoint_event(checkpoint,'inference:'+req['id'])
        plan_manifest = {'contract_hash':digest(contract),'plan_set_hash':digest(plans),
                         'files':{req['id']:file_hash(run/'plans'/(req['id']+'.json')) for req in requests}}
        save_json(run/'plans_manifest.json',plan_manifest)
        save_json(attempt_path,{'status':'complete','contract_hash':digest(contract),'plan_set_hash':digest(plans),
                                'inferred':len(missing),'cached':len(plans)-len(missing)})
        checkpoint_event(checkpoint,'inference:complete')
    except BaseException as exc:
        save_json(attempt_path,{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                               'error_type':type(exc).__name__,'error':str(exc)[:500]})
        raise
    finally:
        if planner is not None: planner.close()
    return plans

def load_plans(root,run,settings):
    run = Path(run)
    manifest = assert_run(root,run,settings)
    contract = load_contract(run)
    if contract['request_hash'] != digest(manifest['planner_request']): raise ValueError('runの推論条件不一致')
    pm = read_json(run/'plans_manifest.json')
    plans = {}
    for req in _requests(manifest):
        path = run/'plans'/(req['id']+'.json')
        if file_hash(path) != pm['files'].get(req['id']): raise ValueError('推論ファイルのハッシュ不一致')
        record = read_json(path)
        _check_plan(record,req,contract)
        plans[req['id']] = record
    if digest(plans) != pm['plan_set_hash'] or pm['contract_hash'] != digest(contract):
        raise ValueError('異なる推論結果が混在しています。')
    return plans

def _prepare(root,run,settings,plans):
    assert_execution_environment(run)
    actual = load_plans(root,run,settings)
    if plans is not None and digest(plans) != digest(actual):
        raise ValueError('メモリ内の計画と保存済み計画が異なります。')
    return actual, digest(load_contract(run)), digest(actual)

def _trace(run,settings,scenario):
    from inspection_network import make_trace, trace_hash
    path = Path(run)/'traces'/(scenario+'.json')
    expected = make_trace(settings,scenario)
    if path.exists():
        actual = read_json(path)
        if trace_hash(actual) != trace_hash(expected): raise ValueError('固定通信系列が変更されています。')
    else: save_json(path,expected)
    return expected

def _record(run,directory,result):
    path = Path(directory)/'result.json'
    if read_json(path) != result: raise ValueError('試行結果と保存内容が一致しません。')
    return dict(result, result_path=path.relative_to(run).as_posix(),result_sha256=file_hash(path))

def _summary(records):
    normal = [r for r in records if r['kind']=='normal']
    invalid = [r for r in records if r['kind']=='reject']
    return {'normal_n':len(normal),'language_exact_n':sum(r['language_exact'] for r in normal),
            'end_to_end_success_n':sum(r['end_to_end_success'] for r in normal),
            'normal_false_rejection_n':sum(not r['accepted'] for r in normal),
            'invalid_n':len(invalid),'blocked_n':sum(r['execution_blocked'] for r in invalid),
            'correct_rejection_reason_n':sum(r['reason_correct'] for r in invalid),
            'runtime_failure_n':sum(bool(r.get('failure')) for r in records)}

def evaluate(root,run,settings,plans=None,*,resume=False,checkpoint=None):
    from inspection_sim import run_episode
    run = Path(run)
    plans,ch,ph = _prepare(root,run,settings,plans)
    cases = read_json(run/'manifest.json')['cases']
    directory,records,cached = open_resumable_stage(run,'evaluation',ch,ph,[c['id'] for c in cases],resume=resume)
    if cached is not None: return cached
    try:
        checkpoint_event(checkpoint,'evaluation:start')
        for case in cases[len(records):]:
            ep = directory/'episodes'/case['id']
            result = run_episode(settings,plans[case['id']],ep,record=False,
                                 trace=_trace(run,settings,'normal'),monitor_mode='age')
            row = dict(_record(run,ep,result),id=case['id'],kind=case['kind'],split=case['split'],instruction=case['text'])
            if case['kind']=='normal':
                row.update(expected=case['expected'],language_exact=result['accepted'] and result['planned_targets']==case['expected'])
                row['end_to_end_success']=bool(row['language_exact'] and result['success'] and result['completed_targets']==case['expected'])
            else:
                row.update(expected_reason=case['expected_reason'],execution_blocked=not result['accepted'],
                           reason_correct=result['validation_reason']==case['expected_reason'])
            records.append(row)
            save_progress(directory,records)
            print(f'[{len(records)}/{len(cases)}]',case['id'],row.get('end_to_end_success',row.get('reason_correct')),result.get('failure',''))
            checkpoint_event(checkpoint,'evaluation:'+case['id'])
        summary=_summary(records)
        summary['splits']={s:_summary([r for r in records if r['split']==s]) for s in ['development','holdout']}
        payload={'summary':summary,'records':records,'note':'旧開発用12件と既存評価12件の回帰評価。いずれも改善前に結果を確認済みであり、未見データではない。固定初期条件の小規模評価。'}
        seal_stage(run,'evaluation',directory,payload)
        checkpoint_event(checkpoint,'evaluation:complete')
        return payload
    except BaseException as exc:
        fail_stage(run,'evaluation',directory,exc); raise

def compare_freshness_guard(root,run,settings,plans=None,*,resume=False,checkpoint=None):
    from inspection_sim import run_episode
    from inspection_network import trace_hash
    from inspection_reference import make_control_reference, reference_metadata, validate_control_reference
    run=Path(run)
    plans,ch,ph=_prepare(root,run,settings,plans)
    reference=make_control_reference()
    origin=reference_metadata(reference)
    conditions=[(s,m) for s in ['normal','outage','delayed','reordered'] for m in ['off','receipt','age']]
    directory,records,cached=open_resumable_stage(run,'comparison',ch,ph,[s+'_'+m for s,m in conditions],resume=resume)
    def verify_origin(rows):
        for row in rows:
            if any(row.get(key)!=value for key,value in origin.items()):
                raise ValueError('比較結果と事前固定した参照計画の出所が一致しません。')
    if cached is not None:
        validate_control_reference(cached.get('reference_plan'))
        if any(cached.get(key)!=value for key,value in origin.items()):
            raise ValueError('比較stageの参照計画が一致しません。')
        verify_origin(cached['records'])
        return cached
    try:
        verify_origin(records)
        checkpoint_event(checkpoint,'comparison:start')
        for scenario,mode in conditions[len(records):]:
            trace=_trace(run,settings,scenario)
            ep=directory/'episodes'/(scenario+'_'+mode)
            result=run_episode(settings,reference,ep,trace=trace,monitor_mode=mode,allow_reference=True)
            row=dict(_record(run,ep,result),scenario=scenario,mode=mode,trace_hash=trace_hash(trace),
                     trace_path='traces/'+scenario+'.json',trace_file_sha256=file_hash(run/'traces'/(scenario+'.json')))
            # The runner must establish provenance; never relabel an LLM result here.
            verify_origin([row])
            records.append(row)
            save_progress(directory,records)
            print(f'[{len(records)}/{len(conditions)}]',scenario,mode,'固定参照計画の実行完了',result['success'],'時間',result['sim_duration_s'])
            checkpoint_event(checkpoint,'comparison:'+scenario+'_'+mode)
        payload=dict(available=True,records=records,reference_plan=reference,**origin,
                     note='3方式×4条件。同一通信系列・事前固定 A→B 参照計画・初期姿勢による独立した制御比較。AIの言語正解率は評価しない。AI推論の失敗は言語評価にそのまま残す。時刻はシミュレーション時計。')
        seal_stage(run,'comparison',directory,payload)
        checkpoint_event(checkpoint,'comparison:complete')
        return payload
    except BaseException as exc:
        fail_stage(run,'comparison',directory,exc); raise

def record_demonstrations(root,run,settings,plans=None,*,resume=False,checkpoint=None):
    from inspection_sim import run_episode
    run=Path(run)
    plans,ch,ph=_prepare(root,run,settings,plans)
    definitions=[('normal','A → B の指示','n03','normal',['A','B'],None),
                 ('custom','自分の指示','custom','normal',list(settings.user_expected_targets),None),
                 ('forbidden','禁止対象 C を拒否','x01','normal',None,'forbidden_target:C'),
                 ('stale','遅れて届く観測を監視','n03','delayed',['A','B'],None)]
    directory,records,cached=open_resumable_stage(run,'demonstrations',ch,ph,[d[0] for d in definitions],resume=resume)
    if cached is not None: return cached
    try:
        checkpoint_event(checkpoint,'demonstrations:start')
        for name,title,key,scenario,expected,expected_reason in definitions[len(records):]:
            ep=directory/'episodes'/name
            result=run_episode(settings,plans[key],ep,record=True,trace=_trace(run,settings,scenario),monitor_mode='age')
            semantic=None if expected is None else bool(result['accepted'] and result['planned_targets']==expected)
            task=None if semantic is None else bool(semantic and result['success'] and result['completed_targets']==expected)
            if expected_reason:
                semantic=(not result['accepted'] and result['validation_reason']==expected_reason)
                task=bool(semantic and not result.get('failure'))
            row=dict(_record(run,ep,result),id=name,title=title,instruction=plans[key]['input'],source_plan_id=key,
                     expected=expected,expected_reason=expected_reason,semantic_match=semantic,task_success=task,
                     clip_path=(ep/'clip.mp4').relative_to(run).as_posix())
            records.append(row)
            save_progress(directory,records)
            print(f'[{len(records)}/{len(definitions)}]',name,'計画実行',result['success'],'指示達成',task)
            checkpoint_event(checkpoint,'demonstrations:'+name)
        payload={'records':records}
        seal_stage(run,'demonstrations',directory,payload)
        checkpoint_event(checkpoint,'demonstrations:complete')
        return payload
    except BaseException as exc:
        fail_stage(run,'demonstrations',directory,exc); raise

def control_preflight(root,run,settings):
    import numpy as np
    from inspection_core import HOME_XYZ,inverse_kinematics,smoothstep5
    from inspection_sim import InspectionSimulator
    assert_run(root,run,settings)
    assert_execution_environment(run)
    directory=Path(run)/'preflight'/unique_id('attempt')
    started=time.perf_counter()
    sim=InspectionSimulator(settings,directory,record=True)
    q0=inverse_kinematics(HOME_XYZ); q1=inverse_kinematics(settings.targets['A']['xyz'])
    try:
        for k in range(round(6/settings.physics_dt)):
            sim.physics_step(q0+(q1-q0)*smoothstep5(float(sim.data.time)/2.5))
            if sim.contact_problem(): raise RuntimeError('設定した接触異常判定に該当しました。')
            if not np.isfinite(sim.data.qpos).all() or np.linalg.norm(sim.data.qvel)>25:
                raise RuntimeError('物理計算が不安定です。')
            error=float(np.linalg.norm(sim.tip()-settings.targets['A']['xyz']))
            if (k+1)%10==0: sim.video_frame('制御・撮影テスト / AI未使用',['A'],'CONTROL_TEST','A',0,error,[],True,'提出動画ではありません')
        if error>=settings.position_tolerance_m: raise RuntimeError('到達許容誤差を超えています。')
        capture=sim.capture('A')
        result={'status':'complete','diagnostic':True,'ai_used':False,'final_error_m':error,
                'video_frames':sim.frames_written,'capture':capture,'clip':str(directory/'clip.mp4'),
                'wall_time_s':time.perf_counter()-started}
        result['render_frames_per_wall_s']=sim.frames_written/max(result['wall_time_s'],1e-9)
        save_json(directory/'check.json',result)
        return result
    except BaseException as exc:
        save_json(directory/'check.json',{'status':'failed','diagnostic':True,'error_type':type(exc).__name__,'error':str(exc)[:500]})
        raise
    finally: sim.close()
