"""Filesystem checkpoint recovery diagnostics; no cloud or AI performance claims."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from inspection_checkpoint import latest_checkpoint, restore_checkpoint, save_checkpoint
from inspection_core import Settings
from inspection_store import digest, new_run, read_json

PROJECT_SOURCE = Path(__file__).resolve().parents[1]


def copy_project(destination):
    destination.mkdir(parents=True)
    for source in PROJECT_SOURCE.glob('*.py'):
        shutil.copy2(source, destination / source.name)
    for name in ('README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'requirements.txt', 'VALIDATION.md'):
        source = PROJECT_SOURCE / name
        if source.exists():
            shutil.copy2(source, destination / name)
    (destination / 'tests').mkdir()
    for source in (PROJECT_SOURCE / 'tests').glob('*.py'):
        shutil.copy2(source, destination / 'tests' / source.name)
    return destination


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = copy_project(self.base / 'project')
        self.settings = Settings()
        self.run = new_run(self.root, self.settings)
        self.vault = self.base / 'drive_vault'

    def save(self, event='diagnostic', **kwargs):
        return save_checkpoint(self.root, self.run, self.vault, event, **kwargs)

    def test_saved_snapshot_restores_in_empty_root(self):
        marker = self.run / 'partial_progress.json'
        marker.write_text('{"completed": 3}', encoding='utf-8')
        receipt = self.save('three units completed')
        settings, restored, info = restore_checkpoint(self.base / 'restored', self.vault)
        self.assertEqual(settings, self.settings)
        self.assertEqual(read_json(restored / marker.name), {'completed': 3})
        self.assertEqual(info['checkpoint_id'], receipt['checkpoint_id'])
        self.assertEqual(info['warnings'], [])
        self.assertTrue((self.base / 'restored' / 'latest_run.json').is_file())
        self.assertEqual(list((self.root / 'backups').glob('checkpoint_*.zip')), [])

    def test_interrupted_copy_keeps_previous_checkpoint(self):
        old = self.save()
        original_copy = shutil.copyfileobj
        def interrupt_upload(source, target, *args, **kwargs):
            if str(getattr(target, 'name', '')).endswith('.partial'):
                target.write(b'partial upload')
                raise OSError('connection lost')
            return original_copy(source, target, *args, **kwargs)
        with patch('inspection_checkpoint.shutil.copyfileobj', side_effect=interrupt_upload):
            with self.assertRaisesRegex(OSError, 'connection lost'):
                self.save('interrupted')
        self.assertEqual(latest_checkpoint(self.vault)['checkpoint_id'], old['checkpoint_id'])
        self.assertTrue(list((self.vault / self.run.name).glob('*.partial')))

    def test_missing_receipt_does_not_commit_uploaded_zip(self):
        old = self.save()
        with patch('inspection_checkpoint._atomic_json', side_effect=OSError('receipt interrupted')):
            with self.assertRaisesRegex(OSError, 'receipt interrupted'):
                self.save('uncommitted')
        self.assertEqual(latest_checkpoint(self.vault)['checkpoint_id'], old['checkpoint_id'])
        self.assertEqual(len(list((self.vault / self.run.name).glob('*.zip'))), 2)

    def test_readback_mismatch_fails_loudly_without_committing(self):
        old = self.save()
        from inspection_checkpoint import file_hash as original_hash
        def wrong_remote_hash(path):
            return '0' * 64 if Path(path).is_relative_to(self.vault) else original_hash(path)
        with patch('inspection_checkpoint.file_hash', side_effect=wrong_remote_hash):
            with self.assertRaisesRegex(IOError, '読み戻し'):
                self.save('corrupt transfer')
        self.assertEqual(latest_checkpoint(self.vault)['checkpoint_id'], old['checkpoint_id'])

    def test_corrupt_newest_falls_back_with_visible_warning(self):
        old = self.save('old')
        current = self.save('new')
        Path(current['archive_path']).write_bytes(b'broken')
        with self.assertWarnsRegex(RuntimeWarning, '直前'):
            info = latest_checkpoint(self.vault)
        self.assertEqual(info['checkpoint_id'], old['checkpoint_id'])
        self.assertEqual(len(info['warnings']), 1)

    def test_missing_newest_archive_falls_back(self):
        old = self.save('old')
        current = self.save('new')
        Path(current['archive_path']).unlink()
        with self.assertWarns(RuntimeWarning):
            info = latest_checkpoint(self.vault)
        self.assertEqual(info['checkpoint_id'], old['checkpoint_id'])

    def test_malformed_newest_receipt_falls_back(self):
        old = self.save('old')
        current = self.save('new')
        Path(current['receipt_path']).write_text('{', encoding='utf-8')
        with self.assertWarns(RuntimeWarning):
            info = latest_checkpoint(self.vault)
        self.assertEqual(info['checkpoint_id'], old['checkpoint_id'])

    def test_retention_keeps_three_and_unrelated_files(self):
        directory = self.vault / self.run.name
        self.save()
        unrelated = directory / 'personal_notes.zip'
        unrelated.write_bytes(b'not managed')
        for _ in range(3):
            self.save()
        self.assertEqual(len(list(directory.glob('checkpoint_*.json'))), 3)
        self.assertEqual(len(list(directory.glob('checkpoint_*.zip'))), 3)
        self.assertEqual(unrelated.read_bytes(), b'not managed')

    def test_retention_preserves_usable_fallbacks_if_recent_corrupt(self):
        first = self.save()
        second = self.save()
        Path(second['archive_path']).write_bytes(b'corrupt')
        self.save(keep=2)
        self.assertTrue(Path(first['archive_path']).is_file())
        self.assertTrue(Path(first['receipt_path']).is_file())

    def test_restore_does_not_overwrite_existing_local_run(self):
        self.save()
        with self.assertRaisesRegex(FileExistsError, 'RESUME_LOCAL'):
            restore_checkpoint(self.root, self.vault)

    def test_source_change_blocks_save_and_existing_checkpoint_survives(self):
        old = self.save()
        (self.root / 'inspection_core.py').write_text('# changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'コード'):
            self.save()
        self.assertEqual(latest_checkpoint(self.vault)['checkpoint_id'], old['checkpoint_id'])

    def test_selecting_run_is_explicit_and_retention_does_not_cross_runs(self):
        first = self.save()
        other = new_run(self.root, self.settings)
        for _ in range(4):
            save_checkpoint(self.root, other, self.vault, 'other run')
        self.assertEqual(latest_checkpoint(self.vault, self.run.name)['checkpoint_id'], first['checkpoint_id'])
        self.assertTrue(Path(first['archive_path']).is_file())

    def test_receipt_path_escape_is_rejected(self):
        receipt = self.save()
        path = Path(receipt['receipt_path'])
        value = read_json(path)
        value['archive'] = '../external.zip'
        value['receipt_sha256'] = digest({key: item for key, item in value.items() if key != 'receipt_sha256'})
        path.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaisesRegex(FileNotFoundError, 'ZIP名'):
            latest_checkpoint(self.vault)

    def test_invalid_run_id_and_retention_are_rejected(self):
        self.save()
        with self.assertRaises(ValueError):
            latest_checkpoint(self.vault, '../escape')
        with self.assertRaises(ValueError):
            self.save(keep=1)

    def test_no_committed_snapshot_returns_none(self):
        self.assertIsNone(latest_checkpoint(self.vault))
        self.vault.mkdir()
        self.assertIsNone(latest_checkpoint(self.vault))
        with self.assertRaisesRegex(FileNotFoundError, '保存済み'):
            restore_checkpoint(self.root, self.vault)


if __name__ == '__main__':
    unittest.main()
