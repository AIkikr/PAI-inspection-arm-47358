"""Remember module bytes at import time to reject stale in-memory source."""
from __future__ import annotations
import hashlib
from pathlib import Path

_IMPORTED = {}

def register_module(name, path):
    p = Path(path)
    _IMPORTED[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

def assert_loaded_sources(source_hashes):
    for name, loaded_sha in _IMPORTED.items():
        if name in source_hashes and source_hashes[name] != loaded_sha:
            raise ValueError('メモリ上の旧コードと保存済みソースが異なります。ランタイムを再起動してください: '+name)

register_module(__name__, __file__)
