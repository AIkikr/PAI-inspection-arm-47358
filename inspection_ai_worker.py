"""Fresh Python worker: load Qwen once, communicate through JSON lines only."""
from __future__ import annotations

import json
import os
import sys
import traceback

# Set backend selection before importing any third-party ML package.
os.environ.update({'USE_TF':'0', 'USE_FLAX':'0', 'USE_TORCH':'1',
                   'USE_TORCH_XLA':'0', 'FORCE_TF_AVAILABLE':'0'})


def main():
    # Duplicate the protocol pipe, then route even native fd-1 output to the log.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def emit(value):
        protocol.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        protocol.flush()

    planner = None
    try:
        from inspection_provenance import register_module
        register_module(__name__, __file__)
        from inspection_ai import LocalQwenPlanner
        from inspection_core import Settings
        first = sys.stdin.readline()
        if not first:
            raise RuntimeError('初期化要求がありません。')
        request = json.loads(first)
        if request.get('op') != 'init':
            raise ValueError('最初の要求はinitでなければなりません。')
        settings = Settings(**request['settings'])
        planner = LocalQwenPlanner(settings, request['contract'], require_gpu=request.get('require_gpu', True))
        emit({'type':'ready', 'protocol':1})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('op') == 'close':
                break
            if request.get('op') != 'infer':
                raise ValueError('不明なworker操作です。')
            result = planner.infer(request['text'])
            emit({'type':'result', 'request_id':request['request_id'], 'result':result})
        return 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            emit({'type':'error', 'error_type':type(exc).__name__, 'error':str(exc)[:2000]})
        except (OSError, ValueError):
            pass
        return 1
    finally:
        if planner is not None:
            try:
                planner.close()
            except BaseException:
                traceback.print_exc(file=sys.stderr)
        protocol.close()


if __name__ == '__main__':
    raise SystemExit(main())
