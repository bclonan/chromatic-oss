import json
import os
from experiments.common import write_json


def test_atomic_results_retry_reader_lock(monkeypatch, tmp_path):
    calls = []
    original = os.replace
    def replace(source, target):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("simulated transient Windows reader lock")
        return original(source, target)
    monkeypatch.setattr(os, "replace", replace)
    target = tmp_path / "result.json"
    write_json(target, {"real": 42})
    assert json.loads(target.read_text()) == {"real": 42}
    assert len(calls) == 2
    assert len(list(tmp_path.iterdir())) == 1
