"""Content-addressed, pickle-free storage for numeric model bundles."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..contracts import WorkbenchError
from ..identity import canonical, check_id, confined, digest, file_digest
import platform
import re


def _runtime_versions():
    import pandas
    import scipy
    import sklearn
    return {"numpy": np.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "python": platform.python_version()}


def inspect_bundle(root: Path, model_id: str) -> dict:
    """Validate the manifest and serving compatibility without reading array data."""
    from ..identity import lock_digest

    check_id(model_id)
    directory = confined(root, f"models/{model_id}")
    manifest = confined(directory, "manifest.json")
    confined(directory, "arrays.npz")
    try:
        document = json.loads(manifest.read_bytes())
        if set(document) != {"model_id", "identity", "arrays_file_sha256"}:
            raise ValueError("manifest schema")
        identity = document["identity"]
        if (not isinstance(identity, dict) or set(identity) != {"metadata", "arrays"}
                or not isinstance(identity["metadata"], dict) or not isinstance(identity["arrays"], dict)
                or document["model_id"] != model_id or digest(identity) != model_id
                or not isinstance(document["arrays_file_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", document["arrays_file_sha256"])):
            raise ValueError("manifest identity")
        for name, spec in identity["arrays"].items():
            if (not isinstance(name, str) or not name or not name.replace("_", "").isalnum()
                    or not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "sha256"}
                    or not isinstance(spec["dtype"], str) or np.dtype(spec["dtype"]).kind not in "biuf"
                    or not isinstance(spec["shape"], list)
                    or any(type(size) is not int or size < 0 for size in spec["shape"])
                    or not isinstance(spec["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
                raise ValueError("array schema")
        metadata = identity["metadata"]
        kind = metadata.get("model_kind")
        if kind is not None:
            if kind not in ("rolling", "hierarchical", "hgb") or metadata.get("lock_digest") != lock_digest():
                raise ValueError("incompatible model lock")
            versions = metadata.get("versions")
            if kind == "hierarchical" or versions is not None:
                runtime = _runtime_versions()
                if (not isinstance(versions, dict)
                        or any(versions.get(key) != value for key, value in runtime.items() if key != "python")
                        or not isinstance(versions.get("python"), str)
                        or versions["python"].split(".")[:2] != runtime["python"].split(".")[:2]
                        or not isinstance(versions.get("platform"), str)):
                    raise ValueError("incompatible model versions")
        return metadata
    except (OSError, ValueError, KeyError, TypeError, EOFError, OverflowError):
        raise WorkbenchError("corrupt_artifact", "Invalid or incompatible model bundle") from None



def array_identity(arrays: dict[str, np.ndarray]) -> dict:
    result = {}
    for name in sorted(arrays):
        if not name or not name.replace("_", "").isalnum():
            raise WorkbenchError("invalid_model", "Invalid numeric array name")
        array = np.asarray(arrays[name])
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise WorkbenchError("invalid_model", "Model arrays must be finite numeric values")
        contiguous = np.ascontiguousarray(array)
        result[name] = {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape),
                        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()}
    return result


def publish_bundle(root: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> tuple[str, Path]:
    identity = {"metadata": metadata, "arrays": array_identity(arrays)}
    identifier = digest(identity)
    models = root / "models"
    if root.is_symlink() or models.is_symlink():
        raise WorkbenchError("unsafe_path", "Symlink model root forbidden")
    models.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=models))
    try:
        np.savez_compressed(stage / "arrays.npz", **arrays)
        document = {"model_id": identifier, "identity": identity, "arrays_file_sha256": file_digest(stage / "arrays.npz")}
        (stage / "manifest.json").write_bytes(canonical(document))
        destination = models / identifier
        if destination.exists():
            read_bundle(root, identifier)
        else:
            try:
                os.rename(stage, destination)
            except FileExistsError:
                read_bundle(root, identifier)
        return identifier, destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def read_bundle(root: Path, identifier: str) -> tuple[dict, dict[str, np.ndarray]]:
    inspect_bundle(root, identifier)
    check_id(identifier)
    directory = confined(root, f"models/{identifier}")
    manifest = confined(directory, "manifest.json")
    archive = confined(directory, "arrays.npz")
    try:
        document = json.loads(manifest.read_bytes())
        identity = document["identity"]
        if document["model_id"] != identifier or digest(identity) != identifier:
            raise ValueError("identity mismatch")
        if file_digest(archive) != document["arrays_file_sha256"]:
            raise ValueError("file checksum mismatch")
        with np.load(archive, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        if array_identity(arrays) != identity["arrays"]:
            raise ValueError("array identity mismatch")
        return identity["metadata"], arrays
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        raise WorkbenchError("corrupt_artifact", "Invalid or corrupt model bundle") from None
