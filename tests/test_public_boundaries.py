"""Hosted permissions, artifacts, process isolation and real synthetic execution."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from f1_strategy.contracts import PublicUseAuthorization, WorkbenchError
from f1_strategy.deployment import Coordinator, authorize_lineage, public_catalog


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    from f1_strategy.synthetic import generate_demo
    root = tmp_path_factory.mktemp('hosted-demo')
    generate_demo(root)
    index = json.loads((root / 'index.json').read_text())
    return root, index


def authorization(**overrides):
    now = datetime.now(timezone.utc)
    args = dict(source_ids=('tracinginsights:2023-07-R',), session_ids=('2023-07-R',),
                scopes=('public_display', 'model_server_use', 'derived_report_download'),
                expires_utc=now + timedelta(hours=1), confirmed_utc=now - timedelta(hours=1),
                permission_reference_id='operator-reference')
    args.update(overrides)
    return PublicUseAuthorization(**args)


def test_authorization_full_sources_and_synthetic_labels():
    real = [{'session_id': '2023-07-R', 'source_ids': ['tracinginsights:2023-07-R'], 'synthetic': False}]
    authorize_lineage(real, ('public_display',), authorizations=(authorization(),))
    for records, scopes, lineage in [
        ((), ('public_display',), real),
        ((authorization(expires_utc=datetime.now(timezone.utc) - timedelta(seconds=1)),), ('public_display',), real),
        ((authorization(session_ids=('2023-15-R',)),), ('public_display',), real),
        ((authorization(),), ('public_asset_distribution',), real),
        ((authorization(scopes=('public_display',)),), ('model_server_use',), real),
        ((authorization(scopes=('public_display',)),), ('derived_report_download',), real),
        ((authorization(),), ('public_display',), real + [{'session_id':'2023-15-R','source_ids':['tracinginsights:2023-15-R'],'synthetic':False}]),
        ((), ('public_display',), [{'session_id':'2023-07-R','source_ids':['synthetic:fake'],'synthetic':True}]),
        ((), ('public_display',), [{'session_id':'SYN-2023-07-R','source_ids':['tracinginsights:2023-07-R'],'synthetic':True}]),
    ]:
        with pytest.raises(WorkbenchError):
            authorize_lineage(lineage, scopes, authorizations=records)
    authorize_lineage([{'session_id':'SYN-2023-07-R','source_ids':['synthetic:v1:SYN-2023-07-R'],'synthetic':True}], ('public_display',), authorizations=())


def test_hosted_ingest_checks_permissions_and_isolates_child(tmp_path, monkeypatch):
    lineage = [{'session_id': '2023-07-R', 'source_ids': ['tracinginsights:2023-07-R'], 'synthetic': False}]
    payload = {'year': 2023, 'round_number': 7}
    coordinator = Coordinator(tmp_path)
    for key in ('F1_USERNAME', 'F1_PASSWORD', 'FASTF1_AUTH_METHOD',
                'F1_SOURCE_ACCESS_CONFIRMED', 'F1_SNAPSHOT_RETENTION_CONFIRMED',
                'F1_STRATEGY_AUTHORIZATION_FILE'):
        monkeypatch.setenv(key, 'private-value')

    class ChildStarted(Exception):
        pass

    def intercept(*args, **kwargs):
        assert not any(key in kwargs['env'] for key in (
            'F1_USERNAME', 'F1_PASSWORD', 'FASTF1_AUTH_METHOD',
            'F1_SOURCE_ACCESS_CONFIRMED', 'F1_SNAPSHOT_RETENTION_CONFIRMED',
            'F1_STRATEGY_AUTHORIZATION_FILE'))
        raise ChildStarted

    monkeypatch.setattr(subprocess, 'Popen', intercept)
    for scopes in ((), ('source_access',), ('snapshot_retention',)):
        records = (authorization(scopes=scopes),) if scopes else ()
        with pytest.raises(WorkbenchError):
            coordinator.run_job('ingest', payload, source_lineage=lineage, authorizations=records)
    with pytest.raises(ChildStarted):
        coordinator.run_job('ingest', payload, source_lineage=lineage,
                            authorizations=(authorization(scopes=('source_access', 'snapshot_retention')),))


def test_registry_rejects_untrusted_paths_and_manifests(tmp_path):
    (tmp_path / 'public_registry.json').write_text(json.dumps([{'session_id':'2023-07-R','snapshot_id':'../secret','model_id':'0'*64,'driver_number':'44','completed_lap':20}]))
    assert public_catalog(tmp_path) == ()


def test_corrupt_model_manifest_is_not_catalogued(demo, tmp_path):
    import shutil
    root, index = demo
    sid, mid = index['target_snapshot_id'], index['model_id']
    (tmp_path / 'snapshots').mkdir()
    (tmp_path / 'models').mkdir()
    shutil.copytree(root / 'snapshots' / sid, tmp_path / 'snapshots' / sid)
    shutil.copytree(root / 'models' / mid, tmp_path / 'models' / mid)
    manifest = tmp_path / 'models' / mid / 'manifest.json'
    data = json.loads(manifest.read_text())
    data['identity']['metadata']['model_kind'] = 'tampered'
    manifest.write_text(json.dumps(data))
    (tmp_path / 'index.json').write_text(json.dumps(index))
    assert public_catalog(tmp_path) == ()
    (tmp_path / 'public_registry.json').write_text('{invalid')
    assert public_catalog(tmp_path) == ()
    (tmp_path / 'public_registry.json').unlink()
    (tmp_path / 'public_registry.json').symlink_to(tmp_path / 'elsewhere')
    assert public_catalog(tmp_path) == ()


def comparison(demo):
    from f1_strategy.data.snapshots import open_snapshot
    from f1_strategy.replay import build_race_state
    from f1_strategy.strategy.plans import load_scenario
    root, index = demo
    entry = next(e for e in public_catalog(root) if e['snapshot_id'] == index['target_snapshot_id'])
    state = build_race_state(open_snapshot(root, index['target_snapshot_id']), driver_number=index['driver_number'], completed_lap=index['completed_lap'])
    scenario = load_scenario(root / 'scenario.json', state).model_dump(mode='json')
    payload = {'snapshot_id':index['target_snapshot_id'], 'model_id':index['model_id'], 'driver_number':index['driver_number'], 'completed_lap':index['completed_lap'], 'scenario':scenario, 'n_draws':100, 'seed':2026}
    return entry, payload


def test_real_synthetic_compare_child(demo):
    root, _ = demo
    entry, payload = comparison(demo)
    result = Coordinator(root).run_job('compare', payload, source_lineage=entry['source_lineage'])
    decision = result['decision']
    assert decision['snapshot_id'] == payload['snapshot_id']
    assert decision['model_id'] == payload['model_id']
    assert decision['delta_stay_minus_pit_s'] < 0
    assert set(decision['selected_plan_ids']) == {'pit_now', 'stay_out'}


def test_restricted_jobs_and_inputs(demo):
    root, _ = demo
    entry, payload = comparison(demo)
    coordinator = Coordinator(root)
    for kind, request in [('train', {}), ('ingest', {'year':2022, 'round_number':1}), ('compare', {**payload, 'url':'https://example.invalid'}), ('compare', {**payload, 'model_id':'../../etc/passwd'}), ('compare', {**payload, 'scenario_path':'/etc/passwd'})]:
        with pytest.raises((WorkbenchError, ValueError)):
            coordinator.run_job(kind, request, source_lineage=entry['source_lineage'])


def test_timeout_kills_group_and_releases_lease(demo, monkeypatch):
    root, _ = demo
    entry, payload = comparison(demo)
    coordinator = Coordinator(root)
    original_popen = subprocess.Popen
    child = []
    code = "import os,signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    def spawn(*args, **kwargs):
        proc = original_popen([sys.executable, '-c', code], **kwargs)
        child.append(proc)
        return proc
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    actual = coordinator._child
    def limited(kind, request, stage, deadline, job):
        return actual(kind, request, stage, time.monotonic() + 0.7, job)
    monkeypatch.setattr(coordinator, '_child', limited)
    with pytest.raises(WorkbenchError) as error:
        coordinator.run_job('compare', payload, source_lineage=entry['source_lineage'])
    assert error.value.code == 'deadline_exceeded'
    assert child[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(child[0].pid, 0)
    assert not coordinator.leases
    assert coordinator.active is None


def test_denied_catalog_never_reads_parquet(demo, tmp_path, monkeypatch):
    import shutil
    import pyarrow.parquet as pq
    from f1_strategy.data.snapshots import identity_document
    from f1_strategy.identity import digest
    root, index = demo
    shutil.copytree(root / 'models', tmp_path / 'models')
    source = root / 'snapshots' / index['target_snapshot_id']
    manifest = json.loads((source / 'manifest.json').read_bytes())
    manifest.update(session_id='2023-07-R', synthetic=False, source_ids=['tracinginsights:2023-07-R'])
    sid = digest(identity_document(manifest))
    manifest['snapshot_id'] = sid
    destination = tmp_path / 'snapshots' / sid
    shutil.copytree(source, destination)
    (destination / 'manifest.json').write_text(json.dumps(manifest))
    entry = {'session_id': '2023-07-R', 'snapshot_id': sid, 'model_id': index['model_id'], 'driver_number': '44', 'completed_lap': 20}
    (tmp_path / 'public_registry.json').write_text(json.dumps([entry]))
    def forbidden(*args, **kwargs):
        pytest.fail('Parquet was read without public permission')
    monkeypatch.setattr(pq, 'ParquetFile', forbidden)
    monkeypatch.delenv('F1_STRATEGY_AUTHORIZATION_FILE', raising=False)
    from f1_strategy.deployment import _lineage
    _, focal, _ = _lineage(tmp_path, sid, index['model_id'])
    with pytest.raises(WorkbenchError) as error:
        authorize_lineage(focal, ('public_display',), authorizations=())
    assert error.value.code == 'authorization_required'
    assert public_catalog(tmp_path) == ()


def test_report_writer_quota_and_deadline_release_slot(demo, monkeypatch):
    from f1_strategy.contracts import DecisionResult
    import f1_strategy.deployment as deployment
    root, _ = demo
    entry, payload = comparison(demo)
    coordinator = Coordinator(root)
    result = DecisionResult.model_validate_json(json.dumps(coordinator.run_job('compare', payload, source_lineage=entry['source_lineage'])['decision']))
    html = coordinator.report_bytes(result)
    assert b'<html' in html.lower() and result.analysis_id.encode() in html
    assert coordinator.active is None and not coordinator.leases
    actual = coordinator._child
    def limited(kind, request, stage, deadline, job):
        return actual(kind, request, stage, time.monotonic() + 0.01, job)
    monkeypatch.setattr(coordinator, '_child', limited)
    with pytest.raises(WorkbenchError) as error:
        coordinator.report_bytes(result)
    assert error.value.code == 'deadline_exceeded'
    assert coordinator.active is None and not coordinator.leases
    monkeypatch.setattr(coordinator, '_child', actual)
    monkeypatch.setattr(deployment, 'MAX_BYTES', coordinator._size() + 1024 * 1024)
    with pytest.raises(WorkbenchError) as error:
        coordinator.report_bytes(result)
    assert error.value.code == 'capacity_exceeded'
    assert coordinator.active is None and not coordinator.leases
    assert not list((root / 'staging').glob('report-*'))
