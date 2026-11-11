"""Load verified numeric pace artifacts from the confined local registry."""
from __future__ import annotations

from datetime import datetime

from ..contracts import WorkbenchError
from ..identity import check_id
from .bundle import inspect_bundle, read_bundle
from .model import HierarchicalPaceModel
from .rolling_model import RollingPaceModel


def inspect_model_lineage(root, model_id) -> list[dict]:
    """Inspect all serving dependencies and return normalized declared sources."""
    active = set()

    def inspect(identifier):
        if identifier in active:
            raise ValueError("cyclic model dependency")
        active.add(identifier)
        metadata = inspect_bundle(root, identifier)
        kind = metadata["model_kind"]
        if kind not in ("rolling", "hierarchical", "hgb"):
            raise ValueError("unknown model kind")
        cutoff = datetime.fromisoformat(metadata["data_cutoff_utc"])
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("naive cutoff")
        sources = metadata["lineage"]
        if not isinstance(sources, list):
            raise ValueError("invalid lineage")
        by_id = {}
        for entry in sources:
            identifier_source = check_id(entry["snapshot_id"])
            used = datetime.fromisoformat(entry["max_used_utc"])
            if (identifier_source in by_id or used.tzinfo is None or used.utcoffset() is None
                    or used >= cutoff or not isinstance(entry["session_id"], str)
                    or not entry["session_id"] or type(entry["synthetic"]) is not bool
                    or not isinstance(entry["source_ids"], list) or not entry["source_ids"]
                    or any(not isinstance(s, str) or not s for s in entry["source_ids"])):
                raise ValueError("invalid lineage source")
            by_id[identifier_source] = entry
        training = metadata.get("training_spec")
        role_fields = {"fit": "training_snapshot_ids", "tuning": "tuning_snapshot_ids",
                       "selection": "selection_snapshot_ids", "calibration": "calibration_snapshot_ids"}
        if training is not None:
            roles = {}
            for role, field in role_fields.items():
                identifiers = training[field]
                if not isinstance(identifiers, list) or len(set(identifiers)) != len(identifiers):
                    raise ValueError("invalid training roles")
                for source_id in identifiers:
                    check_id(source_id)
                    roles.setdefault(source_id, set()).add(role)
            if set(roles) != set(by_id):
                raise ValueError("incomplete lineage")
            if any("calibration" in r and len(r) != 1 for r in roles.values()):
                raise ValueError("overlapping calibration")
        else:
            if kind != "rolling":
                raise ValueError("missing training provenance")
            roles = {key: set(entry["roles"]) for key, entry in by_id.items()}
        normalized = []
        for source_id, entry in by_id.items():
            declared = entry.get("roles")
            if declared is not None and (not isinstance(declared, list) or len(set(declared)) != len(declared)
                                         or set(declared) != roles[source_id]):
                raise ValueError("inconsistent roles")
            if not roles[source_id] or not roles[source_id] <= set(role_fields):
                raise ValueError("invalid roles")
            normalized.append({**entry, "roles": sorted(roles[source_id])})
        if kind == "hgb":
            base_metadata, base_sources = inspect(metadata["base_model_id"])
            if base_metadata["model_kind"] != "hierarchical":
                raise ValueError("invalid challenger base")
            base_cutoff = datetime.fromisoformat(base_metadata["data_cutoff_utc"])
            expected_base = {source_id for source_id, source_roles in roles.items()
                             if source_roles & {"fit", "tuning", "selection"}}
            if {source["snapshot_id"] for source in base_sources} != expected_base:
                raise ValueError("incomplete base lineage")
            if base_cutoff > cutoff:
                raise ValueError("base cutoff")
            for source in base_sources:
                source_id = source["snapshot_id"]
                if (source_id not in by_id or "calibration" in roles[source_id]
                        or set(source["roles"]) != roles[source_id]
                        or {k: v for k, v in source.items() if k != "roles"}
                        != {k: v for k, v in by_id[source_id].items() if k != "roles"}):
                    raise ValueError("base provenance mismatch")
        active.remove(identifier)
        return metadata, normalized

    try:
        return inspect(model_id)[1]
    except (KeyError, ValueError, TypeError, OverflowError):
        raise WorkbenchError("corrupt_artifact", "Invalid model dependency lineage") from None


def _check_target(metadata, lineage, target_start_utc):
    if target_start_utc is None:
        return
    if target_start_utc.tzinfo is None or target_start_utc.utcoffset() is None:
        raise WorkbenchError("invalid_input", "Target start must have a timezone")
    if (datetime.fromisoformat(metadata["data_cutoff_utc"]) > target_start_utc
            or any(datetime.fromisoformat(source["max_used_utc"]) >= target_start_utc for source in lineage)):
        raise WorkbenchError("future_artifact", "Model data does not precede the target race")


def _inspect_for_load(root, model_id, target_start_utc):
    lineage = inspect_model_lineage(root, model_id)
    metadata = inspect_bundle(root, model_id)
    _check_target(metadata, lineage, target_start_utc)
    return metadata

def load_model(root, model_id, *, target_start_utc=None):
    metadata = _inspect_for_load(root, model_id, target_start_utc)
    if metadata["model_kind"] == "hgb":
        from .challenger import load_challenger
        return load_challenger(root, model_id, target_start_utc=target_start_utc)
    metadata, arrays = read_bundle(root, model_id)
    try:
        if metadata["model_kind"] not in ("hierarchical", "rolling"):
            raise ValueError("unknown model kind")
        cutoff = datetime.fromisoformat(metadata["data_cutoff_utc"])
        if cutoff.tzinfo is None:
            raise ValueError("naive cutoff")
        if target_start_utc is not None:
            if target_start_utc.tzinfo is None:
                raise WorkbenchError("invalid_input", "Target start must have a timezone")
            if cutoff > target_start_utc:
                raise WorkbenchError("future_artifact", "Model cutoff is after the target race start")
            for source in metadata["lineage"]:
                if datetime.fromisoformat(source["max_used_utc"]) >= target_start_utc:
                    raise WorkbenchError("future_artifact", "Model includes data not preceding the target race")
        if metadata["model_kind"] == "rolling":
            if arrays:
                raise ValueError("unexpected rolling arrays")
            return RollingPaceModel(model_id, metadata)
        model = HierarchicalPaceModel(model_id, metadata, arrays)
        for variant in set(metadata["variant_indices"]):
            if arrays[f"fit_{variant}"].shape != (len(model.feature_names) + 1,):
                raise ValueError("coefficient shape")
            if arrays[f"bootstrap_{variant}"].shape != (200, len(model.feature_names) + 1):
                raise ValueError("bootstrap shape")
        return model
    except (ValueError, KeyError, TypeError, IndexError):
        raise WorkbenchError("corrupt_artifact", "Invalid pace model metadata or numeric arrays") from None
