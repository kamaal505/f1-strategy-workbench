from __future__ import annotations

import json

import pytest

from f1_strategy.contracts import IngestRequest, WorkbenchError
from f1_strategy.data.acquire import acquire_session
from f1_strategy.data.archive import Fetcher, columns, entries, LAPS
from f1_strategy.data.snapshots import open_snapshot
from f1_strategy.data.query import prefix_table
from f1_strategy.replay import build_race_state


def payloads():
    return {
        'session_info': {'t0_date': '2023-01-01T00:00:00', 'session_start_time': 10, 'total_laps': 50,
                         'track_status': {'Time': [0], 'Status': ['1'], 'Message': ['Green']},
                         'session_status': {'Time': [10], 'Status': ['Started']}},
        'drivers': {'drivers': [{'driver': 'HAM'}]},
        'session_laptimes': {'lap': [1, 2], 'sesT': [100, 195], 'lST': [10, 100],
                             'time': [90, 95], 'dNum': ['44', '44'], 'drv': ['HAM', 'HAM'],
                             'life': [5, 6], 'compound': ['MEDIUM', 'MEDIUM'],
                             'pin': ['None', 190], 'pout': [None, 200],
                             'fresh': [False, None], 'del': [False, True],
                             'unknown': ['None', {'future': 'field'}]},
        'weather': {'wT': [50], 'wR': [False]},
        'rcm': {'time': ['2023-01-01T00:01:00'], 'msg': ['message']},
    }


def test_archive_prefix_preserves_prior_use_and_hides_future_pit_exit(tmp_path, monkeypatch, socket_disabled):
    records = payloads()
    records['session_laptimes']['fresh'] = ['False', None]
    def fetch(self, url, **kwargs):
        return json.dumps(records[url.rsplit('/', 1)[-1][:-5]], separators=(',', ':')).encode()
    monkeypatch.setattr(Fetcher, 'get', fetch)
    monkeypatch.delenv('F1_SOURCE_ACCESS_CONFIRMED', raising=False)
    monkeypatch.delenv('F1_SNAPSHOT_RETENTION_CONFIRMED', raising=False)
    ref = acquire_session(IngestRequest(year=2023, round_number=7), root=tmp_path)
    snapshot = open_snapshot(tmp_path, ref.snapshot_id)
    laps = snapshot.table('laps').to_pylist()
    assert [r['fresh_tyre'] for r in laps] == [False, None]
    state = build_race_state(snapshot, driver_number='44', completed_lap=1)
    assert state.drivers['44'].tyre_age_laps == 5
    visible_pits = prefix_table(snapshot, 'pit_events', 195_000_000_000).to_pylist()
    assert [row['kind'] for row in visible_pits] == ['entry']
    with pytest.raises(WorkbenchError) as error:
        build_race_state(snapshot, driver_number='44', completed_lap=2)
    assert error.value.code == 'focal_in_pit'


def test_ambiguous_join_and_unequal_arrays_fail():
    data = payloads()['session_laptimes']
    data['dNum'] = ['44', '55']
    with pytest.raises(WorkbenchError, match='Ambiguous'):
        entries({'drivers': [{'driver': 'HAM'}]}, columns(data, LAPS))
    data['dNum'] = ['44']
    with pytest.raises(WorkbenchError, match='lengths'):
        columns(data, LAPS)


def test_required_empty_stream_fails_before_publication(tmp_path, monkeypatch):
    records = payloads()
    records['session_info']['track_status'] = {'Time': [], 'Status': [], 'Message': []}
    monkeypatch.setattr(Fetcher, 'get', lambda self, url, **kwargs: json.dumps(records[url.rsplit('/', 1)[-1][:-5]]).encode())
    with pytest.raises(WorkbenchError) as exc:
        acquire_session(IngestRequest(year=2023, round_number=7), root=tmp_path)
    assert exc.value.code == 'missing_stream'
    assert not (tmp_path / 'snapshots').exists()


def test_oversized_response_is_rejected():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count): return b'x' * count
    class Opener:
        def open(self, *args, **kwargs): return Response()
    fetcher = Fetcher()
    fetcher.opener = Opener()
    with pytest.raises(WorkbenchError) as exc:
        fetcher.get('https://raw.githubusercontent.com/TracingInsights/2023/ref/file', limit=8)
    assert exc.value.code == 'capacity_exceeded'


def test_partial_telemetry_retains_slice_without_fabricated_clock(tmp_path, monkeypatch):
    records = payloads()
    def fetch(self, url, **kwargs):
        if url.endswith('1_tel.json'):
            return json.dumps({'tel': {'time': [0, 0.5], 'speed': [100, 110], 'brake': [0, 1], 'drs': [1, 0], 'x': [1, 2]}}).encode()
        if url.endswith('2_tel.json'):
            return None
        return json.dumps(records[url.rsplit('/', 1)[-1][:-5]]).encode()
    monkeypatch.setattr(Fetcher, 'get', fetch)
    ref = acquire_session(IngestRequest(year=2023, round_number=7, telemetry=True), root=tmp_path)
    snapshot = open_snapshot(tmp_path, ref.snapshot_id)
    samples = snapshot.table('car_samples').to_pylist()
    assert [row['release_ns'] for row in samples] == [None, None]
    assert [row['DRS'] for row in samples] == [True, False]
    assert snapshot.manifest.availability['car_samples'].reason is not None
    assert prefix_table(snapshot, 'car_samples', 1_000_000_000_000).num_rows == 0
