"""Actual child-process protocol tests with a diagnostic stub, never model inference."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inspection_ai import QwenPlanner, TORCH_ONLY_ENV, planner_request
from inspection_core import Settings
from inspection_store import digest


STUB = r'''
import json, os, sys, time
mode = sys.argv[1]
request = json.loads(sys.stdin.readline())
if mode == 'init_error':
    print('diagnostic initialization failure', file=sys.stderr, flush=True)
    print(json.dumps({'type':'error','error_type':'RuntimeError','error':'test GPU unavailable'}), flush=True)
    raise SystemExit(1)
if mode == 'timeout':
    time.sleep(60)
print(json.dumps({'type':'ready','protocol':1}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'close':
        break
    if mode == 'crash':
        print('LLVM ERROR: diagnostic duplicate options', file=sys.stderr, flush=True)
        os._exit(37)
    result = {'source':'diagnostic_stub','diagnostic':True,'input':request['text'],
              'pid':os.getpid(),'environment':{key:os.environ.get(key) for key in
              ('USE_TF','USE_FLAX','USE_TORCH','USE_TORCH_XLA','FORCE_TF_AVAILABLE')}}
    print(json.dumps({'type':'result','request_id':request['request_id'],'result':result}), flush=True)
'''


class AIIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.stub = self.base / 'stub_worker.py'
        self.stub.write_text(STUB, encoding='utf-8')
        self.settings = Settings()
        request = planner_request(self.settings)
        self.contract = dict(request, revision='a'*40, request_hash=digest(request))

    def planner(self, mode='normal', **kwargs):
        with patch('inspection_ai._worker_command', return_value=[sys.executable, '-u', str(self.stub), mode]):
            return QwenPlanner(self.settings, self.contract, log_path=self.base/'worker.log', **kwargs)

    def test_multiple_requests_use_same_isolated_worker_with_torch_only_flags(self):
        planner = self.planner()
        try:
            first, second = planner.infer('Aを点検'), planner.infer('Bを点検')
            self.assertEqual(first['pid'], second['pid'])
            self.assertNotEqual(first['pid'], os.getpid())
            self.assertEqual(first['environment'], TORCH_ONLY_ENV)
            self.assertEqual(second['input'], 'Bを点検')
        finally:
            planner.close()
        self.assertIsNotNone(planner._process.poll())
        planner.close()

    def test_native_child_exit_is_runtime_error_and_parent_survives(self):
        planner = self.planner('crash')
        with self.assertRaisesRegex(RuntimeError, 'exit code: 37') as caught:
            planner.infer('Aを点検')
        self.assertIn('LLVM ERROR', str(caught.exception))
        self.assertTrue(planner._closed)
        self.assertGreater(os.getpid(), 0)

    def test_initialization_error_includes_saved_log(self):
        with self.assertRaisesRegex(RuntimeError, 'test GPU unavailable') as caught:
            self.planner('init_error')
        self.assertIn('diagnostic initialization failure', str(caught.exception))
        self.assertTrue((self.base/'worker.log').is_file())

    def test_startup_timeout_terminates_worker(self):
        with self.assertRaisesRegex(RuntimeError, 'タイムアウト'):
            self.planner('timeout', startup_timeout_s=0.1)

    def test_parent_module_import_does_not_load_ml_backends(self):
        root = Path(__file__).resolve().parents[1]
        code = 'import sys,json; import inspection_ai; print(json.dumps([x for x in ("torch","transformers","tensorflow","jax","mujoco") if x in sys.modules]))'
        output = subprocess.check_output([sys.executable, '-c', code], cwd=root, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(json.loads(output.strip()), [])

    def test_actual_worker_separates_native_and_python_stdout_from_protocol(self):
        root = Path(__file__).resolve().parents[1]
        launcher = self.base / 'diagnostic_worker_launcher.py'
        launcher.write_text('''
import os, runpy, sys, types
class DiagnosticPlanner:
    def __init__(self, settings, contract, require_gpu=True):
        print('diagnostic Python loader output', flush=True)
        os.write(1, b'diagnostic native loader output\\n')
    def infer(self, text):
        print('diagnostic inference output', flush=True)
        return {'source':'diagnostic_stub', 'diagnostic':True, 'input':text}
    def close(self):
        print('diagnostic close output', flush=True)
ai = types.ModuleType('inspection_ai')
ai.LocalQwenPlanner = DiagnosticPlanner
sys.modules['inspection_ai'] = ai
core = types.ModuleType('inspection_core')
core.Settings = lambda **values: types.SimpleNamespace(**values)
sys.modules['inspection_core'] = core
sys.path.insert(0, sys.argv[1])
runpy.run_path(os.path.join(sys.argv[1], 'inspection_ai_worker.py'), run_name='__main__')
''', encoding='utf-8')
        with patch('inspection_ai._worker_command', return_value=[sys.executable, '-u', str(launcher), str(root)]):
            planner = QwenPlanner(self.settings, self.contract, log_path=self.base/'actual_worker.log')
        try:
            self.assertEqual(planner.infer('Aを点検')['input'], 'Aを点検')
        finally:
            planner.close()
        log = (self.base/'actual_worker.log').read_text(encoding='utf-8')
        self.assertIn('diagnostic Python loader output', log)
        self.assertIn('diagnostic native loader output', log)
        self.assertIn('diagnostic close output', log)


if __name__ == '__main__':
    unittest.main()
