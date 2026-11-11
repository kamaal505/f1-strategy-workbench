"""Acquire pinned public archive inputs into immutable offline snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pyarrow as pa

from ..contracts import IngestRequest, SnapshotRef, WorkbenchError, REPLAY_NOTICE
from ..identity import digest, lock_digest
from ..pace.corpus import load_corpus
from .archive import (REFS, Fetcher, LAPS, WEATHER, MESSAGES, TRACK, SESSION,
                      boolean, columns, decode, entries, number, raw_table)
from .normalize import normalize_streams, quality_report
from .snapshots import publish_snapshot


def _telemetry(fetcher, base, session_id, laps, entries_by_code, tables, availability, quality):
    """Retain processed lap slices; slice-relative Time cannot establish event time."""
    samples = {'car': [], 'position': []}
    attempted = missing = 0
    partial_reason = None
    slice_quality = {}
    channels = {'car': {'rpm': 'RPM', 'speed': 'Speed', 'gear': 'nGear', 'throttle': 'Throttle', 'brake': 'Brake', 'drs': 'DRS', 'distance': 'Distance'},
                'position': {'x': 'X', 'y': 'Y', 'z': 'Z'}}
    pairs = sorted({(str(row.Driver), int(row.LapNumber)) for row in laps.itertuples()
                    if isinstance(row.Driver, str) and row.Driver in entries_by_code and pd.notna(row.LapNumber) and row.LapNumber > 0})
    for code, lap in pairs:
        url = f'{base}/{quote(code, safe="")}/{lap}_tel.json'
        try:
            data = fetcher.get(url, optional=True, limit=4 * 1024 * 1024)
        except WorkbenchError as exc:
            if exc.code in ('capacity_exceeded', 'deadline_exceeded', 'acquisition_failed'):
                partial_reason = 'Some processed lap files unavailable or acquisition limit reached'
                break
            raise
        attempted += 1
        if data is None:
            missing += 1
            continue
        obj = decode(data).get('tel')
        if not isinstance(obj, dict):
            raise WorkbenchError('invalid_source', 'Invalid telemetry archive')
        mapping = {'time': ('Time', 'number')}
        mapping.update({k: (v, 'bool' if k in ('brake', 'drs') else 'number') for group in channels.values() for k, v in group.items()})
        frame = columns(obj, mapping, required=('time',))
        cadence = frame['Time'].dropna().diff().dropna()
        slice_quality[f'{code}/{lap}'] = {
            'rows': len(frame),
            'missing_channels': sorted(k for group in channels.values() for k in group if k not in obj),
            'null_counts': {name: int(frame[name].isna().sum()) for group in channels.values() for name in group.values()},
            'median_cadence_s': float(cadence.median()) if len(cadence) else None,
        }
        tables[f'source/telemetry/{code}/{lap}'] = raw_table(url, data)
        for kind, group in channels.items():
            part = frame[['Time'] + list(group.values())].copy()
            count = len(part)
            part.insert(0, 'driver_number', entries_by_code[code])
            part.insert(0, 'lap_number', lap)
            part.insert(0, 'time_basis', 'unknown')
            part.insert(0, 'release_ns', pd.array([None] * count, dtype='Int64'))
            part.insert(0, 'event_ns', pd.array([None] * count, dtype='Int64'))
            part.insert(0, 'source_row_id', [f'telemetry:{code}:{lap}:{i}' for i in range(count)])
            part.insert(0, 'session_id', session_id)
            samples[kind].append(pa.Table.from_pandas(part, preserve_index=False))
    for kind in samples:
        name = f'{kind}_samples'
        count = sum(t.num_rows for t in samples[kind])
        if samples[kind]:
            tables[name] = pa.concat_tables(samples[kind], promote_options='permissive')
        reason = 'Replay-timed telemetry unavailable: source time is slice-relative'
        if partial_reason:
            reason += '; ' + partial_reason
        elif missing:
            reason += f'; {missing} of {attempted} requested lap files missing'
        if not count:
            reason += '; no telemetry files available'
        availability[name] = {'status': 'available' if count else 'unavailable', 'reason': reason, 'rows': count}
    quality['telemetry'] = {'requested_laps': len(pairs), 'attempted_laps': attempted, 'missing_laps': missing,
                            'time_basis': 'Slice-relative Time; absolute event and release times unavailable',
                            'source': 'Processed merged/interpolated car and position telemetry; DRS is boolean',
                            'slices': slice_quality}


def acquire_session(request: IngestRequest, *, root: Path) -> SnapshotRef:
    session_id = f'{request.year}-{request.round_number:02d}-R'
    matches = [e for e in load_corpus()['events'] if e['session_id'] == session_id]
    if len(matches) != 1:
        raise WorkbenchError('missing_metadata', 'Race not present in verified corpus')
    base = f'https://raw.githubusercontent.com/TracingInsights/{request.year}/{REFS[request.year]}/{quote(matches[0]["event_name"], safe="")}/Race'
    fetcher = Fetcher()
    raw = {}
    parsed = {}
    for stem in ('session_info', 'drivers', 'session_laptimes', 'weather', 'rcm'):
        url = f'{base}/{stem}.json'
        data = fetcher.get(url, optional=stem in ('weather', 'rcm'))
        if data is not None:
            raw[stem] = (url, data)
            parsed[stem] = decode(data)
    info = parsed['session_info']
    frames = {
        'laps': columns(parsed['session_laptimes'], LAPS, required=('lap', 'sesT', 'dNum', 'drv')),
        'track_status': columns(info.get('track_status'), TRACK, required=('Time', 'Status')),
        'session_status': columns(info.get('session_status'), SESSION, required=('Time', 'Status')),
    }
    if 'weather' in parsed:
        frames['weather'] = columns(parsed['weather'], WEATHER, required=('wT',))
    if 'rcm' in parsed:
        frames['messages'] = columns(parsed['rcm'], MESSAGES, required=('time',))
    for name in ('laps', 'track_status', 'session_status'):
        if frames[name].empty:
            raise WorkbenchError('missing_stream', f'Required stream {name} missing or empty')
    driver_entries = entries(parsed['drivers'], frames['laps'])
    try:
        origin = pd.Timestamp(info['t0_date'])
        if pd.isna(origin):
            raise ValueError()
        origin = origin.tz_localize('UTC') if origin.tzinfo is None else origin.tz_convert('UTC')
        distance = number(info['total_laps'], True)
        start = number(info['session_start_time'])
        started = frames['session_status'].loc[frames['session_status']['Status'].astype(str).str.lower().eq('started'), 'Time'].dropna()
        if not distance or distance < 1 or start is None or start < 0 or started.empty:
            raise ValueError()
        start_ns = int(started.iloc[0].value)
    except (ValueError, KeyError, TypeError, OverflowError):
        raise WorkbenchError('missing_metadata', 'Archive race origin, start or distance unavailable') from None
    tables = normalize_streams(session_id, origin, frames)
    for stem, (url, data) in raw.items():
        tables[f'source/{stem}'] = raw_table(url, data)
    availability = {}
    for name in ('laps', 'track_status', 'session_status', 'weather', 'messages'):
        count = len(frames[name]) if name in frames else 0
        availability[name] = {'status': ('available' if count else 'empty') if name in frames else 'unavailable',
                              'reason': None if name in frames else 'Archive stream unavailable', 'rows': count}
    quality = quality_report(tables)
    canonical_laps = tables['laps'].to_pandas()
    delta = ((canonical_laps.lap_end_ns - canonical_laps.lap_start_ns) / 1e9 - canonical_laps.lap_time_s).abs()
    affected = canonical_laps.loc[delta > .01, 'driver_number'].dropna()
    quality['lap_clock_duration_mismatch'] = {'count': int((delta > .01).sum()), 'drivers': sorted(set(affected.astype(str)))}
    quality['race_start_metadata_mismatch'] = {'count': int(abs(start_ns / 1e9 - start) > .001), 'drivers': []}
    if request.telemetry:
        _telemetry(fetcher, base, session_id, frames['laps'], {e['driver_code']: e['driver_number'] for e in driver_entries}, tables, availability, quality)
    else:
        for name in ('car_samples', 'position_samples'):
            availability[name] = {'status': 'unavailable', 'reason': 'Not requested', 'rows': 0}
    event_metadata = {'layout_id': f'event:{session_id}', 'rules_provenance': 'unverified', 'nomination_provenance': 'unverified',
                      'race_date': origin.date().isoformat(), 'source_reference': 'TracingInsights archive session metadata; event-specific sporting applicability unverified',
                      'archive_commit': REFS[request.year], 'archive_base_url': base}
    metadata = dict(schema_version=1, session_id=session_id, synthetic=False, year=request.year, round_number=request.round_number,
                    session='R', stream_origin_utc=origin.to_pydatetime(), race_start_ns=start_ns, scheduled_laps=distance,
                    entries=tuple(driver_entries), event_metadata=event_metadata, event_metadata_digest=digest(event_metadata),
                    acquired_utc=datetime.now(timezone.utc), source_ids=(f'tracinginsights:{session_id}',),
                    source_endpoints=tuple(url for url, _ in raw.values()), versions={name: version(name) for name in ('pandas', 'pyarrow', 'numpy')},
                    lock_digest=lock_digest(), load_options={'laps': True, 'weather': True, 'messages': True, 'telemetry': request.telemetry},
                    availability=availability, quality=quality,
                    caveats=(REPLAY_NOTICE, 'Original receipt timestamps unavailable.', 'TracingInsights processed archive; mixed upstream F1, FastF1, Jolpica and OpenF1 lineage. Source code licensing does not grant data rights.', 'Final deletion and classification fields are retrospective.'))
    return publish_snapshot(root=root, metadata=metadata, tables=tables)
