from __future__ import annotations

import json
import platform

import numpy as np
import pytest

from f1_strategy.contracts import WorkbenchError
from f1_strategy.identity import lock_digest
from f1_strategy.pace.artifacts import load_model
from f1_strategy.pace.bundle import _runtime_versions, publish_bundle
from f1_strategy.pace.challenger import load_challenger


def _training(fit=(), calibration=()):
    return {"training_snapshot_ids": list(fit), "tuning_snapshot_ids": [],
            "selection_snapshot_ids": [], "calibration_snapshot_ids": list(calibration)}


def _source(identifier, roles=None):
    result = {"snapshot_id": identifier, "session_id": "SYN-2022-01-R",
              "source_ids": ["synthetic:test"], "synthetic": True,
              "max_used_utc": "2022-01-01T00:00:00+00:00"}
    if roles is not None:
        result["roles"] = roles
    return result


def _base(identifier):
    return {"model_kind": "hierarchical", "lock_digest": lock_digest(),
            "versions": {**_runtime_versions(), "platform": platform.platform()},
            "data_cutoff_utc": "2023-01-01T00:00:00+00:00",
            "training_spec": _training([identifier]), "lineage": [_source(identifier, ["fit"])]}


def _hgb(base, identifier):
    return {"model_kind": "hgb", "lock_digest": lock_digest(), "base_model_id": base,
            "data_cutoff_utc": "2024-01-01T00:00:00+00:00",
            "training_spec": _training([identifier]), "lineage": [_source(identifier)]}


def _no_arrays(*args, **kwargs):
    pytest.fail("numeric arrays read before rejecting model")


@pytest.mark.parametrize("field", ["lock", "version"])
def test_incompatible_serving_bundle_rejected_before_arrays(tmp_path, monkeypatch, field):
    metadata = _base("a" * 64)
    if field == "lock":
        metadata["lock_digest"] = "0" * 64
    else:
        metadata["versions"]["numpy"] = "0.0.invalid"
    model_id, _ = publish_bundle(tmp_path, metadata, {"data": np.array([1.])})
    monkeypatch.setattr(np, "load", _no_arrays)
    with pytest.raises(WorkbenchError):
        load_model(tmp_path, model_id)


def test_incomplete_base_lineage_rejected_before_arrays(tmp_path, monkeypatch):
    source = "a" * 64
    base_id, _ = publish_bundle(tmp_path, _base(source), {"data": np.array([1.])})
    metadata = _hgb(base_id, source)
    metadata["lineage"] = []
    model_id, _ = publish_bundle(tmp_path, metadata, {"data": np.array([1.])})
    monkeypatch.setattr(np, "load", _no_arrays)
    with pytest.raises(WorkbenchError):
        load_challenger(tmp_path, model_id)


def test_missing_base_training_source_rejected_before_arrays(tmp_path, monkeypatch):
    first, second = "a" * 64, "c" * 64
    base_id, _ = publish_bundle(tmp_path, _base(first), {"data": np.array([1.])})
    metadata = _hgb(base_id, first)
    metadata["training_spec"] = _training([first, second])
    metadata["lineage"].append(_source(second))
    model_id, _ = publish_bundle(tmp_path, metadata, {"data": np.array([1.])})
    monkeypatch.setattr(np, "load", _no_arrays)
    with pytest.raises(WorkbenchError):
        load_model(tmp_path, model_id)

def test_cyclic_dependency_manifest_rejected_before_arrays(tmp_path, monkeypatch):
    source = "a" * 64
    model_id, directory = publish_bundle(tmp_path, _hgb("b" * 64, source), {"data": np.array([1.])})
    manifest = directory / "manifest.json"
    document = json.loads(manifest.read_text())
    document["identity"]["metadata"]["base_model_id"] = model_id
    manifest.write_text(json.dumps(document))
    monkeypatch.setattr(np, "load", _no_arrays)
    with pytest.raises(WorkbenchError):
        load_challenger(tmp_path, model_id)
