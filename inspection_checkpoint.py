"""Verified filesystem checkpoints for an explicitly mounted persistent drive.

Only a committed receipt makes an archive discoverable. A kernel/runtime may die
at any point; an unfinished upload never replaces the previous checkpoint. These
snapshots resume completed work units, not a live MuJoCo state or a model forward.
"""
from __future__ import annotations
from inspection_provenance import register_module
register_module(__name__, __file__)

import json
import os
import re
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

from inspection_core import Settings
from inspection_store import (assert_run, create_backup, digest, file_hash,
                              read_json, restore_backup, unique_id)

_RUN = re.compile(r"run_[A-Za-z0-9_-]+\Z")
_CHECKPOINT = re.compile(r"checkpoint_\d{8}T\d{6}_\d{6}Z_[0-9a-f]{8}\Z")


def _run_id(value):
    if not isinstance(value, str) or not _RUN.fullmatch(value):
        raise ValueError('不正なcheckpoint run IDです。')
    return value


def _inside(base, path):
    base, path = Path(base).resolve(), Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(base):
        raise ValueError('checkpoint保存先の外部やリンクは参照できません。')
    return path


def _atomic_json(path, value):
    """Commit metadata after its complete byte stream has been written."""
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    content = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False).encode('utf-8')
    with partial.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    if path.read_bytes() != content:
        raise IOError('checkpoint受領記録の読み戻し確認に失敗しました。')


def _receipt(path, vault):
    path = _inside(vault, path)
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError('checkpoint受領記録が不正です。')
    identifier = value.get('checkpoint_id')
    if not isinstance(identifier, str) or not _CHECKPOINT.fullmatch(identifier):
        raise ValueError('checkpoint IDが不正です。')
    if value.get('version') != 1 or value.get('status') != 'committed':
        raise ValueError('未対応または未確定のcheckpointです。')
    if path.name != identifier + '.json' or path.parent.name != _run_id(value.get('run_id')):
        raise ValueError('checkpoint保存先と受領記録が一致しません。')
    if value.get('archive') != identifier + '.zip':
        raise ValueError('checkpoint ZIP名が不正です。')
    if not isinstance(value.get('archive_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', value['archive_sha256']):
        raise ValueError('checkpoint ZIPのハッシュが不正です。')
    signature = value.get('receipt_sha256')
    if digest({key: item for key, item in value.items() if key != 'receipt_sha256'}) != signature:
        raise ValueError('checkpoint受領記録のハッシュが一致しません。')
    timestamp = datetime.fromisoformat(value['created_utc'])
    if timestamp.tzinfo is None or not isinstance(value.get('event'), str):
        raise ValueError('checkpointの時刻または保存イベントが不正です。')
    archive = _inside(vault, path.parent / value['archive'])
    return value, archive


def _candidates(vault, run_id=None):
    vault = Path(vault).resolve()
    if not vault.is_dir():
        raise FileNotFoundError('永続保存フォルダがありません。Driveのマウントと保存先を確認してください。')
    directories = [vault / _run_id(run_id)] if run_id is not None else list(vault.glob('run_*'))
    candidates = []
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir() or not _RUN.fullmatch(directory.name):
            continue
        candidates.extend(directory.glob('checkpoint_*.json'))
    return sorted(candidates, key=lambda item: (item.name, item.parent.name), reverse=True)


def latest_checkpoint(vault, run_id=None):
    """Find newest readable, hash-verified snapshot; explicitly report fallback.

    Archives are hashed from newest to oldest only until one succeeds. Orphan ZIPs
    and *.partial files have no committed receipt and are intentionally ignored.
    """
    if run_id is not None:
        _run_id(run_id)
    if not Path(vault).exists():
        return None
    candidates = _candidates(vault, run_id)
    if not candidates:
        return None
    problems = []
    for path in candidates:
        try:
            receipt, archive = _receipt(path, vault)
            if not archive.is_file() or file_hash(archive) != receipt['archive_sha256']:
                raise ValueError('ZIPが欠落しているかハッシュが一致しません。')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f'{path.parent.name}/{path.name}: {exc}')
            continue
        if problems:
            warnings.warn('新しいcheckpointを利用できないため、直前の検証済みcheckpointへ戻ります。'
                          + '\n'.join(problems), RuntimeWarning, stacklevel=2)
        return dict(receipt, archive_path=str(archive), receipt_path=str(path),
                    warnings=problems)
    detail = '\n'.join(problems)
    raise FileNotFoundError('復元可能なcheckpointがありません。' + ('\n' + detail if detail else ''))


def _prune(vault, run_id, keep, already_verified=None):
    """Delete only old files explicitly named by this module's valid receipts."""
    retained = 0
    for path in _candidates(vault, run_id):
        try:
            receipt, archive = _receipt(path, vault)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not archive.is_file():
            continue
        if retained < keep:
            # Verify only enough recent snapshots to retain a working fallback.
            if receipt['checkpoint_id'] != already_verified and file_hash(archive) != receipt['archive_sha256']:
                continue
            retained += 1
            continue
        # Removing the receipt first prevents discovery of a ZIP being removed.
        path.unlink()
        archive.unlink()


def save_checkpoint(root, run, vault, event, keep=3):
    """Save and verify a full snapshot. Failure is propagated, never hidden.

    Call only between completed inference/trial/stage work units. Do not write
    run files concurrently. A returned receipt means the mounted filesystem was
    read back successfully; remote service availability cannot be guaranteed.
    """
    root, run, vault = Path(root).resolve(), Path(run).resolve(), Path(vault).resolve()
    run_id = _run_id(run.name)
    if not run.is_relative_to(root / 'artifacts'):
        raise ValueError('checkpoint対象はROOT/artifacts内のrunに限定します。')
    if vault.is_relative_to(run):
        raise ValueError('永続保存先をrunの中に設定できません。')
    if not isinstance(event, str) or not event.strip() or len(event) > 256:
        raise ValueError('保存イベントには1〜256文字の説明を指定してください。')
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 2:
        raise ValueError('少なくとも直近2件のcheckpointを保持してください。')
    manifest = read_json(run / 'manifest.json')
    assert_run(root, run, Settings(**manifest['settings']))
    identifier = unique_id('checkpoint')
    local = root / 'backups' / (identifier + '.zip')
    destination = _inside(vault, vault / run_id)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (identifier + '.zip')
    partial = archive.with_name(archive.name + '.partial')
    try:
        create_backup(root, run, local)
        expected = file_hash(local)
        with local.open('rb') as source, partial.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(partial, archive)
        if file_hash(archive) != expected:
            raise IOError('checkpoint ZIPの読み戻し確認に失敗しました。保存完了とは扱いません。')
        receipt = {'version': 1, 'status': 'committed', 'checkpoint_id': identifier,
                   'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'event': event, 'archive': archive.name, 'archive_sha256': expected,
                   'archive_bytes': archive.stat().st_size}
        receipt['receipt_sha256'] = digest(receipt)
        receipt_path = destination / (identifier + '.json')
        _atomic_json(receipt_path, receipt)
        # Retention failure does not invalidate the new, verified checkpoint.
        retention_warnings = []
        try:
            _prune(vault, run_id, keep, already_verified=identifier)
        except OSError as exc:
            retention_warnings.append('保存は完了しましたが旧checkpointの整理に失敗しました: ' + str(exc))
            warnings.warn(retention_warnings[-1], RuntimeWarning, stacklevel=2)
        return dict(receipt, archive_path=str(archive), receipt_path=str(receipt_path),
                    warnings=retention_warnings)
    finally:
        # The local build is disposable. Incomplete remote files are not receipts.
        if local.exists():
            local.unlink()


def restore_checkpoint(root, vault, run_id=None):
    """Restore a validated checkpoint without replacing a local run implicitly."""
    receipt = latest_checkpoint(vault, run_id)
    if receipt is None:
        raise FileNotFoundError('保存済みcheckpointがありません。最初の実行では新runを作成してください。')
    root = Path(root).resolve()
    destination = root / 'artifacts' / receipt['run_id']
    if destination.exists():
        raise FileExistsError('同名runがローカルにあります。RESUME_LOCALで再開するか、空の別ROOTへ復元してください。'
                             'Driveのcheckpointでローカル結果を自動上書きしません。')
    # Expand locally: ZIP's many small seeks are expensive on a Drive mount.
    local = root / 'backups' / (unique_id('restore') + '.zip')
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(receipt['archive_path']).open('rb') as source, local.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if file_hash(local) != receipt['archive_sha256']:
            raise IOError('checkpoint復元用ZIPのローカルコピーが破損しています。')
        run = restore_backup(root, local)
    finally:
        if local.exists():
            local.unlink()
    settings = Settings(**read_json(run / 'manifest.json')['settings'])
    assert_run(root, run, settings)
    return settings, run, receipt
