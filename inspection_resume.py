"""Resume at sealed episode boundaries, never at a half-written physical state."""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)
from pathlib import Path
from inspection_core import save_json
from inspection_store import (read_json, digest, file_hash, verified_path,
                              start_stage, load_stage)


def checkpoint_event(callback, event):
    # A failed persistent save must stop computation instead of implying protection.
    if callback is not None:
        callback(event)


def _key(stage, row):
    return row['scenario']+'_'+row['mode'] if stage == 'comparison' else row['id']


def _verified_file(base, name, sha):
    path = verified_path(base, name)
    if not path.is_file() or file_hash(path) != sha:
        raise ValueError('再開用の確定済み成果物が欠落・変更しています: '+str(name))
    return path


def verify_records(run, name, records, expected_keys, *, complete=False):
    if not isinstance(records, list):
        raise ValueError('再開記録の形式が不正です。')
    keys = [_key(name, row) for row in records]
    if keys != list(expected_keys)[:len(keys)] or len(keys) > len(expected_keys):
        raise ValueError('再開記録の条件・実行順序が一致しません。')
    if complete and keys != list(expected_keys):
        raise ValueError('完了stageの実験件数が一致しません。')
    for row in records:
        result_path = _verified_file(run, row['result_path'], row['result_sha256'])
        result = read_json(result_path)
        if result.get('status') not in ('complete', 'rejected', 'failed'):
            raise ValueError('中断・未完了のエピソードを確定結果として再利用できません。')
        if result['status'] == 'failed' and (result.get('success') or not result.get('failure')):
            raise ValueError('失敗結果の記録が矛盾しています。')
        if any(key not in row or row[key] != value for key, value in result.items()):
            raise ValueError('再開記録とエピソード結果が一致しません。')
        artifacts = result.get('artifact_hashes')
        if not isinstance(artifacts, dict):
            raise ValueError('確定済み成果物のハッシュ一覧がありません。')
        for relative, sha in artifacts.items():
            _verified_file(result_path.parent, relative, sha)
        if any(capture not in artifacts for capture in result.get('captures', [])):
            raise ValueError('画像が確定済み成果物に登録されていません。')
        if row.get('clip_path'):
            clip = verified_path(run, row['clip_path'])
            if clip.parent != result_path.parent or clip.name not in artifacts:
                raise ValueError('録画とエピソードの対応が不正です。')
        if row.get('trace_path'):
            _verified_file(run, row['trace_path'], row['trace_file_sha256'])


def save_progress(directory, records):
    header = read_json(Path(directory)/'stage.json')
    progress = {key:header[key] for key in ('stage','contract_hash','plan_set_hash')}
    progress['records'] = records
    progress['content_hash'] = digest(progress)
    save_json(Path(directory)/'progress.json', progress)


def open_resumable_stage(run, name, contract_hash, plan_set_hash, expected_keys, *, resume=False):
    """Return (new_directory, sealed_prefix, cached_payload_or_None).

    Continuations get a new attempt; the sealed prefix explicitly refers to
    prior immutable episode paths. Terminal failures remain failures.
    """
    run = Path(run)
    records, previous = [], None
    current = read_json(run/'current.json').get(name)
    if resume and current:
        old_dir = verified_path(run, current['path'])
        header = read_json(old_dir/'stage.json')
        if (header.get('stage') != name or header.get('run_id') != run.name or
            header.get('contract_hash') != contract_hash or header.get('plan_set_hash') != plan_set_hash):
            raise ValueError('再開元のモデル・計画・stageが現在の条件と異なります。')
        if current['status'] == 'complete':
            stage = load_stage(run, name)
            payload = stage['payload']
            keys = [] if name == 'comparison' and payload.get('available') is False else expected_keys
            verify_records(run, name, payload['records'], keys, complete=True)
            print(name, '完了済みの結果を検証して再利用:', len(payload['records']), '件')
            return None, [], payload
        path = old_dir/'progress.json'
        if path.is_file():
            progress = read_json(path)
            if digest({k:v for k,v in progress.items() if k != 'content_hash'}) != progress.get('content_hash'):
                raise ValueError('再開用progressのハッシュが一致しません。')
            if any(progress.get(k) != header[k] for k in ('stage','contract_hash','plan_set_hash')):
                raise ValueError('再開用progressの実験条件が一致しません。')
            records = progress['records']
            verify_records(run, name, records, expected_keys)
        previous = current['path']
    directory = start_stage(run, name, contract_hash, plan_set_hash)
    if previous:
        header = read_json(directory/'stage.json')
        header.update(resumed_from=previous, reused_episode_count=len(records))
        save_json(directory/'stage.json', header)
    save_progress(directory, records)
    if records:
        print(name, '確定済み', len(records), '件を引き継ぎ、残り', len(expected_keys)-len(records), '件を実行します。')
    return directory, records, None
