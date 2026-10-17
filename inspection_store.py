"""Immutable experiment provenance, fresh stage attempts and verified backups."""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)
import hashlib
import json
import platform
import shutil
import stat
import uuid
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import importlib.metadata
from inspection_core import Settings, evaluation_cases, save_json

STAGES = ('evaluation', 'comparison', 'demonstrations', 'publication')

def runtime_environment():
    versions = {'python':platform.python_version(), 'platform':platform.platform()}
    for name in ['numpy','torch','transformers','accelerate','huggingface-hub','mujoco','Pillow','imageio','imageio-ffmpeg','reportlab']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'NOT_INSTALLED'
    return versions

def assert_execution_environment(run):
    """Do not mix calculations from different package versions in one run."""
    _manifest(run)
    saved, current = read_json(Path(run)/'environment.json'), runtime_environment()
    changed = [key for key in current if key != 'platform' and saved.get(key) != current[key]]
    if changed:
        raise ValueError('実行環境のバージョンが保存時と異なります（'+', '.join(changed)+'）。同じ環境へ戻すか、新runで全件を再実行してください。')
    return current

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def unique_id(prefix):
    return prefix+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]

def source_files(root):
    root = Path(root)
    files = list(root.glob('*.py')) + list((root/'tests').glob('*.py'))
    files += [root/n for n in ('README.md','操作手順.md','LICENSE','THIRD_PARTY_NOTICES.md','requirements.txt','VALIDATION.md') if (root/n).is_file()]
    return sorted(files)

def source_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in source_files(root)}

def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('不正な相対パスです。')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ('..', '.') for x in value.split('/')):
        raise ValueError('保存先の外へ出るパスは使用できません。')
    return p.as_posix()

def verified_path(base, relative):
    base = Path(base).resolve()
    path = (base/safe_relative(relative)).resolve()
    if not path.is_relative_to(base):
        raise ValueError('保存先の外を参照しています。')
    return path

def _manifest(run):
    run = Path(run)
    m = read_json(run/'manifest.json')
    if digest(m) != (run/'manifest.sha256').read_text().strip():
        raise ValueError('run manifestが変更されています。新runを作成してください。')
    if read_json(run/'settings.json') != m['settings']:
        raise ValueError('設定記録とrun manifestが一致しません。')
    if not (run/'environment.sha256').is_file() or file_hash(run/'environment.json') != (run/'environment.sha256').read_text().strip():
        raise ValueError('環境記録が変更されています。')
    for name, sha in m['source_hashes'].items():
        p = verified_path(run/'source', name)
        if not p.is_file() or file_hash(p) != sha:
            raise ValueError('実験開始時のソースsnapshotが変更されています: '+name)
    return m

def new_run(root, settings):
    from inspection_ai import planner_request
    from inspection_provenance import assert_loaded_sources
    root = Path(root).resolve()
    settings.validate()
    hashes = source_hashes(root)
    assert_loaded_sources(hashes)
    run = root/'artifacts'/unique_id('run')
    run.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        dest = run/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, dest)
    manifest = {'manifest_version':3,'run_id':run.name,'settings':asdict(settings),
                'evaluation_scope':'same_cases_regression_not_unseen',
                'source_hashes':hashes,'cases':evaluation_cases(),
                'planner_request':planner_request(settings),
                'network_scenarios':['normal','outage','delayed','reordered'],
                'monitor_modes':['off','receipt','age'],
                'clock':'simulation_seconds', 'created_utc':datetime.now(timezone.utc).isoformat()}
    save_json(run/'manifest.json', manifest)
    (run/'manifest.sha256').write_text(digest(manifest), encoding='ascii')
    save_json(run/'settings.json', asdict(settings))
    save_json(run/'environment.json', runtime_environment())
    (run/'environment.sha256').write_text(file_hash(run/'environment.json'),encoding='ascii')
    save_json(run/'current.json', {})
    save_json(root/'latest_run.json', {'relative_run_dir':run.relative_to(root).as_posix()})
    return run

def assert_run(root, run, settings):
    from inspection_provenance import assert_loaded_sources
    settings.validate()
    m = _manifest(run)
    if digest(m['settings']) != digest(asdict(settings)):
        raise ValueError('設定が実験開始時と異なります。変更した条件では新runを作成してください。')
    if m['source_hashes'] != source_hashes(root):
        raise ValueError('コード・説明ファイルが実験開始時と異なります。新runが必要です。')
    assert_loaded_sources(m['source_hashes'])
    return m

def load_latest(root):
    root = Path(root).resolve()
    info = read_json(root/'latest_run.json')
    run = verified_path(root, info['relative_run_dir'])
    if not run.is_relative_to((root/'artifacts').resolve()):
        raise ValueError('artifacts外のrunは指定できません。')
    m = _manifest(run)
    settings = Settings(**m['settings'])
    assert_run(root, run, settings)
    return settings, run

def invalidate_stages(run):
    save_json(Path(run)/'current.json', {})

def start_stage(run, name, contract_hash, plan_set_hash):
    if name not in STAGES: raise ValueError('不明なstage')
    run = Path(run)
    m = _manifest(run)
    stage_dir = run/'stages'/name/unique_id('attempt')
    stage_dir.mkdir(parents=True, exist_ok=False)
    data = {'stage_id':stage_dir.name,'stage':name,'run_id':m['run_id'],
            'contract_hash':contract_hash,'plan_set_hash':plan_set_hash,'status':'running'}
    save_json(stage_dir/'stage.json', data)
    current = read_json(run/'current.json') if (run/'current.json').exists() else {}
    for key in STAGES[STAGES.index(name):]: current.pop(key, None)
    current[name] = {'path':stage_dir.relative_to(run).as_posix(), 'status':'running'}
    save_json(run/'current.json', current)
    return stage_dir

def seal_stage(run, name, stage_dir, payload):
    run, stage_dir = Path(run), Path(stage_dir)
    current = read_json(run/'current.json')
    if current.get(name, {}).get('path') != stage_dir.relative_to(run).as_posix():
        raise ValueError('このstageは現行の試行ではありません。')
    data = read_json(stage_dir/'stage.json')
    if data['status'] != 'running': raise ValueError('完了済みstageは上書きできません。')
    data.update(status='complete', payload=payload)
    save_json(stage_dir/'stage.json', data)
    current[name] = {'path':stage_dir.relative_to(run).as_posix(), 'status':'complete',
                     'sha256':file_hash(stage_dir/'stage.json')}
    save_json(run/'current.json', current)
    return load_stage(run, name)

def fail_stage(run, name, stage_dir, exc):
    run, stage_dir = Path(run), Path(stage_dir)
    data = read_json(stage_dir/'stage.json')
    data.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                error_type=type(exc).__name__, error=str(exc)[:500])
    save_json(stage_dir/'stage.json', data)
    current = read_json(run/'current.json')
    if current.get(name, {}).get('path') == stage_dir.relative_to(run).as_posix():
        current[name] = {'path':stage_dir.relative_to(run).as_posix(), 'status':data['status'],
                         'sha256':file_hash(stage_dir/'stage.json')}
        save_json(run/'current.json', current)

def load_stage(run, name):
    run = Path(run)
    _manifest(run)
    entry = read_json(run/'current.json').get(name)
    if not entry or entry['status'] != 'complete':
        raise ValueError(f'{name}の現行試行は未完了です。該当セルを実行してください。')
    path = verified_path(run, entry['path'])/'stage.json'
    if file_hash(path) != entry['sha256']:
        raise ValueError('stageの記録が変更されています。')
    data = read_json(path)
    if data['status'] != 'complete' or data['run_id'] != run.name or data['stage'] != name:
        raise ValueError('stageとrunの対応が不正です。')
    return dict(data, _path=path.relative_to(run).as_posix(), _sha256=file_hash(path))

def create_backup(root, run, path):
    root, run, path = Path(root), Path(run), Path(path)
    m = _manifest(run)
    assert_run(root, run, Settings(**m['settings']))
    if path.resolve().is_relative_to(run.resolve()):
        raise ValueError('バックアップ先はrunの外にしてください。')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    files = {p.relative_to(run).as_posix():file_hash(p) for p in sorted(run.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    manifest = {'format':'pai-backup-v2','run_id':run.name,'files':files}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('backup_manifest.json',json.dumps(manifest,ensure_ascii=False))
        for name in files: z.write(run/name,'run/'+name)
    return path

def restore_backup(root, archive):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)): raise ValueError('ZIPに重複パスがあります。')
        if sum(i.file_size for i in infos) > 2*1024**3: raise ValueError('バックアップが大きすぎます。')
        for i in infos:
            safe_relative(i.filename)
            if stat.S_ISLNK(i.external_attr >> 16): raise ValueError('リンクは復元しません。')
        bm = json.loads(z.read('backup_manifest.json'))
        if bm['format'] != 'pai-backup-v2': raise ValueError('対応しないバックアップです。')
        run_id = safe_relative(bm['run_id'])
        if '/' in run_id or not run_id.startswith('run_'): raise ValueError('不正なrun ID')
        expected = {'backup_manifest.json'} | {'run/'+safe_relative(n) for n in bm['files']}
        if set(names) != expected: raise ValueError('ZIPの目録と内容が一致しません。')
        for name, sha in bm['files'].items():
            if hashlib.sha256(z.read('run/'+name)).hexdigest() != sha:
                raise ValueError('バックアップ破損: '+name)
        m = json.loads(z.read('run/manifest.json'))
        if digest(m) != z.read('run/manifest.sha256').decode().strip() or m['run_id'] != run_id:
            raise ValueError('manifest不一致')
        for name, sha in m['source_hashes'].items():
            safe_relative(name)
            if bm['files'].get('source/'+name) != sha: raise ValueError('source不一致')
            live = root/name
            if live.exists() and file_hash(live) != sha:
                raise ValueError('復元元と現在のコードが異なります。空の別ROOTへ復元してください。')
        dest = root/'artifacts'/run_id
        if dest.exists(): raise FileExistsError('同じrunが既にあります。末尾の再開セルを使ってください。')
        dest.mkdir(parents=True, exist_ok=False)
        for name in bm['files']:
            target = verified_path(dest,name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read('run/'+name))
    for name in m['source_hashes']:
        live = verified_path(root, name)
        if not live.exists():
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest/'source'/name, live)
    _manifest(dest)
    save_json(root/'latest_run.json',{'relative_run_dir':dest.relative_to(root).as_posix()})
    return dest
