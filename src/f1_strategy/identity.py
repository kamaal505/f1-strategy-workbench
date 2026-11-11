"""Canonical identities, finite JSON and confined artifact paths."""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
import re
import numpy as np
from pydantic import BaseModel
from .contracts import WorkbenchError


def json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkbenchError("nonfinite_payload", "Nonfinite values require explicit null and reason")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_id(identifier: str) -> str:
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise WorkbenchError("invalid_id", "Expected a lowercase SHA-256 identity")
    return identifier


def confined(root: Path, member: str, *, must_exist: bool = True) -> Path:
    relative = Path(member)
    if relative.is_absolute() or not relative.parts or any(p in ("..", ".") for p in relative.parts):
        raise WorkbenchError("unsafe_path", "Artifact member escapes approved root")
    if root.is_symlink():
        raise WorkbenchError("unsafe_path", "Artifact root may not be a symlink")
    base = root.resolve()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise WorkbenchError("unsafe_path", "Symlink artifact members are forbidden")
    if not current.resolve().is_relative_to(base):
        raise WorkbenchError("unsafe_path", "Artifact member escapes approved root")
    if must_exist and not current.exists():
        raise WorkbenchError("missing_artifact", "Required immutable artifact is unavailable")
    return current


def lock_digest() -> str:
    p = Path(__file__).resolve()
    for parent in p.parents:
        candidate = parent / "uv.lock"
        if candidate.is_file():
            return file_digest(candidate)
    embedded = files("f1_strategy").joinpath("assets/lock.sha256")
    if embedded.is_file():
        return check_id(embedded.read_text().strip())
    raise WorkbenchError("missing_lock", "Installed lock identity unavailable")


def core_fingerprint() -> str:
    root = files("f1_strategy")
    values = []
    def visit(directory, prefix=""):
        for item in sorted(directory.iterdir(), key=lambda x: x.name):
            name = prefix + item.name
            if item.is_dir() and item.name != "__pycache__":
                visit(item, name + "/")
            elif item.is_file() and (name.endswith(".py") or name.startswith(("assets/rules/", "assets/events/"))):
                values.append((name, hashlib.sha256(item.read_bytes()).hexdigest()))
    visit(root)
    return digest(values)


def seal_payload(payload: dict) -> dict:
    payload = json_value(payload)
    payload["analysis_id"] = digest(payload["identity_inputs"])
    payload.pop("payload_checksum", None)
    payload["payload_checksum"] = digest(payload)
    return payload


def verify_payload(payload: dict) -> None:
    check_id(payload.get("analysis_id"))
    check_id(payload.get("payload_checksum"))
    if digest(payload["identity_inputs"]) != payload["analysis_id"]:
        raise WorkbenchError("corrupt_analysis", "Analysis identity mismatch")
    if digest({k: v for k, v in payload.items() if k != "payload_checksum"}) != payload["payload_checksum"]:
        raise WorkbenchError("corrupt_analysis", "Analysis checksum mismatch")
