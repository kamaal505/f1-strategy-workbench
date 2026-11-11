"""Fail-closed public authorization and bounded, single-child coordination."""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .contracts import DecisionResult, PublicUseAuthorization, ScenarioSpec, IngestRequest, SnapshotManifest, WorkbenchError
from .identity import canonical, check_id, confined, digest, file_digest, verify_payload

CURATED = frozenset({'2023-07-R', '2023-15-R', '2024-03-R'})
MAX_BYTES = 256 * 1024 * 1024
MAX_STDIN = 256 * 1024
MAX_STDOUT = 8 * 1024 * 1024
MAX_STDERR = 1024 * 1024


def load_authorizations() -> tuple[PublicUseAuthorization, ...]:
    name = os.environ.get('F1_STRATEGY_AUTHORIZATION_FILE')
    if not name:
        return ()
    try:
        path = Path(name)
        if path.is_symlink() or path.stat().st_size > MAX_STDIN:
            raise ValueError
        data = json.loads(path.read_bytes())
        if not isinstance(data, list):
            raise ValueError
        return tuple(PublicUseAuthorization.model_validate_json(canonical(item)) for item in data)
    except (OSError, ValueError, TypeError):
        raise WorkbenchError('authorization_required', 'Authorization records unavailable or invalid') from None


def authorize_lineage(lineage, scopes: tuple[str, ...], *, authorizations=None, now=None) -> None:
    records = load_authorizations() if authorizations is None else authorizations
    now = now or datetime.now(timezone.utc)
    allowed = {'source_access', 'snapshot_retention', 'public_display', 'derived_report_download', 'model_server_use', 'public_asset_distribution'}
    if not scopes or not set(scopes) <= allowed or not isinstance(lineage, (list, tuple)) or not lineage:
        raise WorkbenchError('authorization_required', 'Complete source lineage and scopes are required')
    for item in lineage:
        if not isinstance(item, dict):
            raise WorkbenchError('authorization_required', 'Invalid source lineage')
        session, sources, synthetic = item.get('session_id'), item.get('source_ids'), item.get('synthetic')
        if not isinstance(session, str) or not isinstance(sources, (list, tuple)) or not sources or not all(isinstance(s, str) and s for s in sources) or type(synthetic) is not bool:
            raise WorkbenchError('authorization_required', 'Incomplete source lineage')
        labels = session.startswith('SYN-') and all(s.startswith('synthetic:') for s in sources)
        if synthetic and labels:
            continue
        if synthetic or session.startswith('SYN-') or any(s.startswith('synthetic:') for s in sources):
            raise WorkbenchError('authorization_required', 'Inconsistent synthetic source labels')
        for source in sources:
            for scope in scopes:
                if not any(source in r.source_ids and session in r.session_ids and scope in r.scopes and r.confirmed_utc <= now and (r.expires_utc is None or r.expires_utc > now) for r in records):
                    raise WorkbenchError('authorization_required', 'Source permission missing or expired')


def _snapshot_manifest(root: Path, identifier: str) -> SnapshotManifest:
    from .data.snapshots import identity_document
    check_id(identifier)
    path = confined(root, f'snapshots/{identifier}/manifest.json')
    try:
        if path.stat().st_size > MAX_STDIN:
            raise ValueError
        manifest = SnapshotManifest.model_validate_json(path.read_bytes())
        if (manifest.snapshot_id != identifier or
                digest(identity_document(manifest.model_dump(mode='json'))) != identifier or
                digest(manifest.event_metadata) != manifest.event_metadata_digest):
            raise ValueError
        return manifest
    except (OSError, ValueError, TypeError):
        raise WorkbenchError('corrupt_snapshot', 'Invalid snapshot manifest') from None


def _lineage(root: Path, snapshot_id: str, model_id: str):
    from .pace.artifacts import inspect_model_lineage
    manifest = _snapshot_manifest(root, snapshot_id)
    focal = {'snapshot_id': snapshot_id, 'session_id': manifest.session_id, 'source_ids': list(manifest.source_ids), 'synthetic': manifest.synthetic}
    return manifest, [focal], inspect_model_lineage(root, model_id)


def _validate_entry(root, entry, authorizations=None):
    if not isinstance(entry, dict) or set(entry) - {'session_id', 'snapshot_id', 'model_id', 'driver_number', 'completed_lap', 'scenario'}:
        raise WorkbenchError('invalid_input', 'Invalid catalog entry')
    session = entry.get('session_id')
    if session not in CURATED and not (isinstance(session, str) and session.startswith('SYN-')):
        raise WorkbenchError('invalid_input', 'Session is not in public catalog')
    if not isinstance(entry.get('driver_number'), str) or not entry['driver_number'].isdigit() or type(entry.get('completed_lap')) is not int or entry['completed_lap'] < 1:
        raise WorkbenchError('invalid_input', 'Invalid catalog selection')
    manifest, focal, model = _lineage(root, entry.get('snapshot_id'), entry.get('model_id'))
    if manifest.session_id != session:
        raise WorkbenchError('invalid_input', 'Catalog session does not match snapshot')
    authorize_lineage(focal, ('public_display', 'public_asset_distribution'), authorizations=authorizations)
    authorize_lineage(model, ('model_server_use', 'public_asset_distribution'), authorizations=authorizations)
    if 'scenario' in entry:
        ScenarioSpec.model_validate_json(canonical(entry['scenario']))
    from .data.snapshots import open_snapshot
    snap = open_snapshot(root, entry['snapshot_id'])
    authorize_lineage(focal, ('public_display', 'public_asset_distribution'), authorizations=authorizations)
    authorize_lineage(model, ('model_server_use', 'public_asset_distribution'), authorizations=authorizations)
    # Only after permission has been established may model arrays be read.
    from .pace.artifacts import load_model
    from datetime import timedelta
    load_model(root, entry['model_id'], target_start_utc=snap.manifest.stream_origin_utc + timedelta(microseconds=snap.manifest.race_start_ns / 1000))
    authorize_lineage(focal, ('public_display', 'public_asset_distribution'), authorizations=authorizations)
    authorize_lineage(model, ('model_server_use', 'public_asset_distribution'), authorizations=authorizations)
    return {**entry, 'source_lineage': focal + model}


def public_catalog(root: Path, *, coordinator=None) -> tuple[dict, ...]:
    result = []
    coordinator = coordinator or Coordinator(root)
    try:
        registry = confined(root, 'public_registry.json')
        if registry.stat().st_size > MAX_STDIN:
            raise ValueError
        entries = json.loads(registry.read_bytes())
        if not isinstance(entries, list):
            raise ValueError
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                with coordinator.lease((entry.get('snapshot_id'),)):
                    result.append(_validate_entry(root, entry))
            except (WorkbenchError, ValueError, TypeError, OSError):
                continue
    except (WorkbenchError, OSError, ValueError, TypeError):
        pass
    # Demo index is server-created, never a visitor-supplied path.
    try:
        index = json.loads(confined(root, 'index.json').read_bytes())
        snap_id = index['target_snapshot_id']
        with coordinator.lease((snap_id,)):
            session = _snapshot_manifest(root, snap_id).session_id
            entry = {'session_id': session, 'snapshot_id': snap_id, 'model_id': index['model_id'], 'driver_number': index['driver_number'], 'completed_lap': index['completed_lap']}
            if session.startswith('SYN-'):
                result.append(_validate_entry(root, entry))
    except (WorkbenchError, OSError, ValueError, TypeError, KeyError):
        pass
    return tuple(result)


def materialize_packaged_demo():
    """Validate installed demo assets and retain a bounded writable runtime copy.

    The caller retains the TemporaryDirectory owner for its cached lifetime.
    Installed wheel resources are never a worker's mutable staging directory.
    """
    from importlib.resources import files
    from .data.snapshots import open_snapshot
    from .pace.artifacts import load_model
    source = Path(str(files("f1_strategy").joinpath("assets/demo")))
    entries = public_catalog(source)
    if len(entries) != 1 or not entries[0]["session_id"].startswith("SYN-"):
        raise WorkbenchError("missing_demo", "Verified packaged synthetic demo unavailable")
    index = json.loads(confined(source, "index.json").read_bytes())
    ids = tuple(index["training_snapshot_ids"]) + tuple(index["calibration_snapshot_ids"]) + (index["target_snapshot_id"],)
    for identifier in ids:
        if not open_snapshot(source, identifier).manifest.synthetic:
            raise WorkbenchError("authorization_required", "Packaged demo contains a real source")
    load_model(source, index["model_id"])
    owner = tempfile.TemporaryDirectory(prefix="f1-strategy-demo-")
    destination = Path(owner.name)
    copied = 0
    try:
        for member in sorted(source.rglob("*")):
            relative = member.relative_to(source)
            checked = confined(source, relative.as_posix())
            if checked.is_dir():
                continue
            if not checked.is_file():
                raise WorkbenchError("unsafe_path", "Non-file demo member forbidden")
            target = confined(destination, relative.as_posix(), must_exist=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            with checked.open("rb") as reader, target.open("xb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    copied += len(block)
                    if copied >= MAX_BYTES:
                        raise WorkbenchError("capacity_exceeded", "Packaged demo exceeds runtime quota")
                    writer.write(block)
        return owner, destination
    except BaseException:
        owner.cleanup()
        raise


class Coordinator:
    def __init__(self, root: Path):
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise WorkbenchError('unsafe_path', 'Symlink root forbidden')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.active = None
        self.leases = Counter()
        self.lru = OrderedDict()
        self.states = OrderedDict()
        self.analyses = OrderedDict()

    @contextmanager
    def lease(self, snapshot_ids: tuple[str, ...]):
        with self.lock:
            for sid in snapshot_ids:
                check_id(sid)
            for sid in snapshot_ids:
                self.leases[sid] += 1
                self.lru.pop(sid, None)
                self.lru[sid] = None
        try:
            yield
        finally:
            with self.lock:
                for sid in snapshot_ids:
                    self.leases[sid] -= 1
                    if not self.leases[sid]:
                        del self.leases[sid]

    def report_bytes(self, result: DecisionResult) -> bytes:
        payload = result.model_dump(mode='json')
        verify_payload(payload)
        lineage = list(result.source_lineage)
        authorize_lineage(lineage, ('derived_report_download',))
        with self.lock:
            if self.active is not None:
                raise WorkbenchError('busy', 'Another hosted job is running')
            self._reserve()
            job = uuid.uuid4().hex
            self.active = job
            deadline = time.monotonic() + 90
        stage = None
        try:
            with self.lease((result.snapshot_id,)):
                staging = self.root / 'staging'
                if staging.is_symlink():
                    raise WorkbenchError('unsafe_path', 'Symlink staging forbidden')
                staging.mkdir(exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix='report-', dir=staging))
                size = 0
                baseline_size = self._size()
                encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
                with (stage / 'analysis.json').open('xb') as handle:
                    for part in encoder.iterencode(payload):
                        chunk = part.encode('utf-8')
                        size += len(chunk)
                        if size > MAX_STDOUT or baseline_size + size >= MAX_BYTES:
                            raise WorkbenchError('capacity_exceeded', 'Report input exceeds capacity')
                        if time.monotonic() >= deadline:
                            raise WorkbenchError('deadline_exceeded', 'Hosted job deadline exceeded')
                        handle.write(chunk)
                response = self._child('report', {}, stage, deadline, job)
                if set(response) != {'path', 'sha256'} or response['path'] != f'output/{result.analysis_id}/report.html':
                    raise WorkbenchError('worker_failed', 'Invalid report response')
                report = confined(stage, response['path'])
                if not report.is_file() or report.stat().st_size > MAX_BYTES or file_digest(report) != response['sha256']:
                    raise WorkbenchError('corrupt_report', 'Invalid report output')
                data = report.read_bytes()
                with self.lock:
                    self._reserve()
                    authorize_lineage(lineage, ('derived_report_download',))
                    if self.active != job or time.monotonic() >= deadline:
                        raise WorkbenchError('deadline_exceeded', 'Hosted job deadline exceeded')
                    return data
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
            with self.lock:
                if self.active == job:
                    self.active = None

    def check_capacity(self):
        with self.lock:
            self._reserve()

    def _size(self):
        total = 0
        for name in ('cache', 'staging', 'snapshots', 'reports'):
            base = self.root / name
            if base.is_symlink():
                raise WorkbenchError('unsafe_path', 'Symlink cache forbidden')
            for path, dirs, files in os.walk(base, followlinks=False):
                for item in dirs + files:
                    if (Path(path) / item).is_symlink():
                        raise WorkbenchError('unsafe_path', 'Symlink cache member forbidden')
                for item in files:
                    total += (Path(path) / item).stat().st_size
        return total

    def _reserve(self, protected=()):
        if self._size() < MAX_BYTES:
            return
        for sid in list(self.lru):
            if sid not in self.leases and sid not in protected:
                path = confined(self.root, f'snapshots/{sid}', must_exist=False)
                if path.is_dir():
                    shutil.rmtree(path)
                self.lru.pop(sid, None)
                if self._size() < MAX_BYTES:
                    return
        raise WorkbenchError('capacity_exceeded', 'Hosted cache capacity exceeded')

    def run_job(self, kind, payload, *, source_lineage, authorizations=None):
        if kind not in ('compare', 'ingest') or not isinstance(payload, dict):
            raise WorkbenchError('invalid_input', 'Unsupported job')
        if kind == 'compare':
            if set(payload) - {'snapshot_id', 'model_id', 'driver_number', 'completed_lap', 'scenario', 'n_draws', 'seed'} or not {'snapshot_id', 'model_id', 'driver_number', 'completed_lap', 'scenario'} <= set(payload):
                raise WorkbenchError('invalid_input', 'Invalid comparison request')
            check_id(payload['snapshot_id']); check_id(payload['model_id'])
            if not isinstance(payload['driver_number'], str) or not payload['driver_number'].isdigit() or type(payload['completed_lap']) is not int or payload['completed_lap'] < 1 or type(payload.get('n_draws', 200)) is not int or not 100 <= payload.get('n_draws', 200) <= 1000 or type(payload.get('seed', 2026)) is not int:
                raise WorkbenchError('invalid_input', 'Invalid comparison controls')
            ScenarioSpec.model_validate_json(canonical(payload['scenario']))
            ids = (payload['snapshot_id'],)
        else:
            if set(payload) != {'year', 'round_number'}:
                raise WorkbenchError('invalid_input', 'Invalid acquisition request')
            request = IngestRequest.model_validate(payload)
            session = f'{request.year}-{request.round_number:02d}-R'
            if session not in CURATED:
                raise WorkbenchError('invalid_input', 'Session is not curated')
            ids = ()
        with self.lock:
            if self.active is not None:
                raise WorkbenchError('busy', 'Another hosted job is running')
            self._reserve(ids)
            job = uuid.uuid4().hex
            self.active = job
            deadline = time.monotonic() + (90 if kind == 'compare' else 180)
        stage = None
        try:
            with self.lease(ids):
                if kind == 'compare':
                    registered_sessions = set()
                    try:
                        registry = json.loads(confined(self.root, 'public_registry.json').read_bytes())
                        if isinstance(registry, list):
                            registered_sessions = {e.get('session_id') for e in registry if isinstance(e, dict) and isinstance(e.get('session_id'), str) and (e['session_id'] in CURATED or e['session_id'].startswith('SYN-')) and e.get('snapshot_id') == payload['snapshot_id'] and e.get('model_id') == payload['model_id']}
                    except (WorkbenchError, OSError, ValueError, TypeError):
                        pass
                    demo_match = False
                    if not registered_sessions:
                        try:
                            demo = json.loads(confined(self.root, 'index.json').read_bytes())
                            demo_match = demo.get('synthetic') is True and demo.get('target_snapshot_id') == payload['snapshot_id'] and demo.get('model_id') == payload['model_id']
                        except (WorkbenchError, OSError, ValueError, TypeError):
                            pass
                    if not registered_sessions and not demo_match:
                        raise WorkbenchError('invalid_input', 'Selection is not in the public registry')
                    manifest, focal, model = _lineage(self.root, payload['snapshot_id'], payload['model_id'])
                    if registered_sessions and manifest.session_id not in registered_sessions:
                        raise WorkbenchError('invalid_input', 'Registry session does not match snapshot')
                    if demo_match and not manifest.synthetic:
                        raise WorkbenchError('invalid_input', 'Demo must be synthetic')
                    if not isinstance(source_lineage, (tuple, list)) or list(source_lineage) != focal + model:
                        raise WorkbenchError('authorization_required', 'Lineage mismatch')
                    authorize_lineage(focal, ('public_display',), authorizations=authorizations)
                    authorize_lineage(model, ('model_server_use',), authorizations=authorizations)
                    if manifest.session_id not in CURATED and not (manifest.synthetic and manifest.session_id.startswith('SYN-')):
                        raise WorkbenchError('invalid_input', 'Session is not in the public registry')
                    authorize_lineage(focal, ('public_asset_distribution',), authorizations=authorizations)
                    authorize_lineage(model, ('public_asset_distribution',), authorizations=authorizations)
                    from .data.snapshots import open_snapshot
                    open_snapshot(self.root, payload['snapshot_id'])
                    authorize_lineage(focal, ('public_display', 'public_asset_distribution'), authorizations=authorizations)
                    authorize_lineage(model, ('model_server_use', 'public_asset_distribution'), authorizations=authorizations)
                else:
                    focal = [{'session_id': session, 'source_ids': [f'tracinginsights:{session}'], 'synthetic': False}]
                    model = []
                    if not isinstance(source_lineage, (tuple, list)) or list(source_lineage) != focal:
                        raise WorkbenchError('authorization_required', 'Lineage mismatch')
                    authorize_lineage(focal, ('source_access', 'snapshot_retention'), authorizations=authorizations)
                staging = self.root / 'staging'
                if staging.is_symlink():
                    raise WorkbenchError('unsafe_path', 'Symlink staging forbidden')
                staging.mkdir(exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix='job-', dir=staging))
                response = self._child(kind, payload, stage, deadline, job)
                with self.lock:
                    if self.active != job or time.monotonic() >= deadline:
                        raise WorkbenchError('deadline_exceeded', 'Hosted job deadline exceeded')
                    self._reserve()
                    authorize_lineage(focal, ('public_display',) if kind == 'compare' else ('source_access', 'snapshot_retention'), authorizations=authorizations)
                    if model:
                        authorize_lineage(model, ('model_server_use',), authorizations=authorizations)
                        authorize_lineage(focal, ('public_asset_distribution',), authorizations=authorizations)
                        authorize_lineage(model, ('public_asset_distribution',), authorizations=authorizations)
                    if kind == 'ingest':
                        from .data.snapshots import open_snapshot
                        sid = check_id(response['snapshot_id'])
                        snap = open_snapshot(stage, sid)
                        if snap.manifest.session_id != session or snap.manifest.synthetic or list(snap.manifest.source_ids) != focal[0]['source_ids']:
                            raise WorkbenchError('corrupt_snapshot', 'Acquired snapshot lineage mismatch')
                        target = confined(self.root, f'snapshots/{sid}', must_exist=False)
                        target.parent.mkdir(exist_ok=True)
                        if target.exists():
                            open_snapshot(self.root, sid)
                        else:
                            self._reserve()
                            authorize_lineage(focal, ('source_access', 'snapshot_retention'), authorizations=authorizations)
                            if self.active != job or time.monotonic() >= deadline:
                                raise WorkbenchError('deadline_exceeded', 'Hosted job deadline exceeded')
                            os.rename(stage / 'snapshots' / sid, target)
                        self.lru[sid] = None
                        self._reserve((sid,))
                        authorize_lineage(focal, ('source_access', 'snapshot_retention'), authorizations=authorizations)
                        if self.active != job or time.monotonic() >= deadline:
                            raise WorkbenchError('deadline_exceeded', 'Hosted job deadline exceeded')
                    return response
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
            with self.lock:
                if self.active == job:
                    self.active = None

    def _child(self, kind, payload, stage, deadline, job):
        request = canonical({'kind': kind, 'payload': payload, 'job_id': job})
        if len(request) > MAX_STDIN:
            raise WorkbenchError('capacity_exceeded', 'Request too large')
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT') if k in os.environ}
        env.update({'F1_WORKER_ROOT': str(self.root), 'F1_WORKER_STAGE': str(stage), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        proc = subprocess.Popen([sys.executable, '-m', 'f1_strategy.report_writer' if kind == 'report' else 'f1_strategy.worker'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True)
        selector = selectors.DefaultSelector()
        try:
            # The bounded request is sent without blocking on a child that never reads.
            for stream, event in ((proc.stdin, selectors.EVENT_WRITE), (proc.stdout, selectors.EVENT_READ), (proc.stderr, selectors.EVENT_READ)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, event)
            sent = 0
            out, err = bytearray(), bytearray()
            while selector.get_map():
                if time.monotonic() >= deadline:
                    raise WorkbenchError('deadline_exceeded', 'Hosted job deadline exceeded')
                if self._size() >= MAX_BYTES:
                    raise WorkbenchError('capacity_exceeded', 'Hosted cache capacity exceeded')
                for key, _ in selector.select(min(0.5, max(0, deadline - time.monotonic()))):
                    stream = key.fileobj
                    if stream is proc.stdin:
                        try:
                            n = os.write(stream.fileno(), request[sent:sent + 65536])
                            sent += n
                        except BrokenPipeError:
                            sent = len(request)
                        if sent == len(request):
                            selector.unregister(stream); stream.close()
                    else:
                        chunk = os.read(stream.fileno(), 65536)
                        if not chunk:
                            selector.unregister(stream); stream.close()
                        else:
                            target = out if stream is proc.stdout else err
                            target.extend(chunk)
                            if len(target) > (MAX_STDOUT if stream is proc.stdout else MAX_STDERR):
                                raise WorkbenchError('capacity_exceeded', 'Worker output limit exceeded')
            try:
                returncode = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise WorkbenchError('deadline_exceeded', 'Hosted job deadline exceeded') from None
            if returncode != 0:
                raise WorkbenchError('worker_failed', 'Hosted worker failed')
            try:
                result = json.loads(out)
                if result['job_id'] != job:
                    raise ValueError
                if 'error' in result:
                    raise WorkbenchError(result['error'], 'Hosted job could not complete')
                if not isinstance(result['result'], dict):
                    raise ValueError
                return result['result']
            except (ValueError, KeyError, TypeError):
                raise WorkbenchError('worker_failed', 'Invalid worker response') from None
        finally:
            selector.close()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            # A worker can exit before descendants; the group must not outlive it.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream and not stream.closed:
                    stream.close()


def deployment_check(*, root: Path) -> dict:
    records = load_authorizations()
    catalog = public_catalog(root)
    return {'mode': os.environ.get('F1_STRATEGY_MODE', 'public'), 'authorization_records': len(records), 'public_catalog': [{'session_id': e['session_id'], 'synthetic': e['session_id'].startswith('SYN-')} for e in catalog], 'real_data_catalog_available': any(e['session_id'] in CURATED for e in catalog), 'provider_prerequisites': ['Operator-controlled GitHub repository and connected Streamlit account', 'Applicable source, display and distribution permissions', 'Provider install, browser and capacity verification'], 'cache_limit_bytes': MAX_BYTES, 'monitor_interval_seconds': 0.5}
