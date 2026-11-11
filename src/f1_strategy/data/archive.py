"""Pinned TracingInsights JSON decoding and bounded anonymous retrieval."""
from __future__ import annotations

import hashlib
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd
import pyarrow as pa

from ..contracts import WorkbenchError

REFS = {2022: 'df396d091c4a40a3a6b733631da3497f8c91d89f', 2023: '737c84d3ae14b9883a671ca248d015b69cf1cb58', 2024: 'd1543a71370d9b617b627319f8ad80b2f85dfd5e'}

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

class Fetcher:
    def __init__(self, *, seconds=180, max_bytes=256 * 1024 * 1024, max_requests=1600):
        self.deadline = time.monotonic() + seconds
        self.remaining = max_bytes
        self.requests = max_requests
        self.opener = build_opener(_NoRedirect())

    def get(self, url, *, optional=False, limit=8 * 1024 * 1024):
        if not url.startswith('https://raw.githubusercontent.com/TracingInsights/'):
            raise WorkbenchError('acquisition_failed', 'Invalid archive host')
        remaining_time = self.deadline - time.monotonic()
        if remaining_time <= 0 or self.requests <= 0 or self.remaining <= 0:
            raise WorkbenchError('capacity_exceeded', 'Archive acquisition limit exceeded')
        self.requests -= 1
        try:
            with self.opener.open(Request(url, headers={'User-Agent': 'f1-strategy-workbench'}), timeout=min(20, remaining_time)) as response:
                data = response.read(min(limit, self.remaining) + 1)
        except HTTPError as exc:
            if optional and exc.code == 404:
                return None
            raise WorkbenchError('acquisition_failed', 'Archive request failed') from None
        except (URLError, OSError, TimeoutError):
            raise WorkbenchError('acquisition_failed', 'Archive request failed') from None
        if len(data) > limit or len(data) > self.remaining:
            raise WorkbenchError('capacity_exceeded', 'Archive payload limit exceeded')
        self.remaining -= len(data)
        if time.monotonic() > self.deadline:
            raise WorkbenchError('deadline_exceeded', 'Archive acquisition deadline exceeded')
        return data

def decode(data):
    try:
        value = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, UnicodeError):
        raise WorkbenchError('invalid_source', 'Invalid archive JSON') from None

def raw_table(url, data):
    return pa.table({'url': [url], 'sha256': [hashlib.sha256(data).hexdigest()], 'json': pa.array([data], type=pa.binary())})

def null(value):
    return None if value is None or value == 'None' or value == '' else value

def number(value, integer=False):
    value = null(value)
    if value is None:
        return None
    if isinstance(value, bool):
        raise WorkbenchError('invalid_source', 'Invalid numeric archive value')
    try:
        result = float(value)
        if not math.isfinite(result) or (integer and not result.is_integer()):
            raise ValueError()
        return int(result) if integer else result
    except (ValueError, TypeError, OverflowError):
        raise WorkbenchError('invalid_source', 'Invalid numeric archive value') from None

def boolean(value):
    value = null(value)
    if value is None or isinstance(value, bool):
        return value
    if value in (0, 1, '0', '1', 'True', 'False'):
        return value in (1, '1', 'True')
    raise WorkbenchError('invalid_source', 'Invalid boolean archive value')

def columns(obj, mapping, required=()):
    if not isinstance(obj, dict):
        raise WorkbenchError('invalid_source', 'Invalid archive stream')
    for key in required:
        if key not in obj or not isinstance(obj[key], list):
            raise WorkbenchError('missing_stream', 'Required archive field missing')
    lengths = {len(v) for v in obj.values() if isinstance(v, list)}
    if len(lengths) > 1:
        raise WorkbenchError('invalid_source', 'Archive parallel array lengths differ')
    count = next(iter(lengths), 0)
    output = {}
    for src, (dest, kind) in mapping.items():
        values = obj.get(src, [None] * count)
        if not isinstance(values, list) or len(values) != count:
            raise WorkbenchError('invalid_source', 'Invalid archive parallel array')
        if kind == 'time':
            output[dest] = pd.to_timedelta([number(v) for v in values], unit='s')
        elif kind == 'number':
            output[dest] = [number(v) for v in values]
        elif kind == 'int':
            output[dest] = [number(v, True) for v in values]
        elif kind == 'bool':
            output[dest] = [boolean(v) for v in values]
        else:
            output[dest] = [null(v) for v in values]
    return pd.DataFrame(output)

LAPS = {k: (v, t) for k, v, t in [
 ('time','LapTime','time'),('lap','LapNumber','int'),('compound','Compound','str'),('stint','Stint','int'),('life','TyreLife','int'),
 ('s1','Sector1Time','time'),('s2','Sector2Time','time'),('s3','Sector3Time','time'),('sesT','Time','time'),('lST','LapStartTime','time'),
 ('dNum','DriverNumber','str'),('drv','Driver','str'),('pin','PitInTime','time'),('pout','PitOutTime','time'),('fresh','FreshTyre','bool'),
 ('del','Deleted','bool'),('delR','DeletedReason','str'),('ff1G','FastF1Generated','bool'),('iacc','IsAccurate','bool'),('pos','Position','int'),('status','TrackStatus','str')]}
WEATHER = {k:(v,t) for k,v,t in [('wT','Time','time'),('wAT','AirTemp','number'),('wH','Humidity','number'),('wP','Pressure','number'),('wR','Rainfall','bool'),('wTT','TrackTemp','number'),('wWD','WindDirection','number'),('wWS','WindSpeed','number')]}
MESSAGES = {k:(v,t) for k,v,t in [('time','Utc','str'),('cat','Category','str'),('msg','Message','str'),('status','Status','str'),('flag','Flag','str'),('scope','Scope','str'),('sector','Sector','int'),('dNum','RacingNumber','str'),('lap','Lap','int')]}
TRACK = {'Time':('Time','time'), 'Status':('Status','str'), 'Message':('Message','str')}
SESSION = {'Time':('Time','time'), 'Status':('Status','str')}

def entries(drivers, laps):
    pairs = {}
    reverse = {}
    for code, num in zip(laps['Driver'], laps['DriverNumber']):
        if code is None or num is None:
            continue
        code, num = str(code), str(num)
        pairs.setdefault(code, set()).add(num)
        reverse.setdefault(num, set()).add(code)
    result = []
    seen = set()
    seen_codes = set()
    for driver in drivers.get('drivers', []):
        code, num = null(driver.get('driver')), null(driver.get('dn'))
        if not isinstance(code, str) or not code.isascii() or not code.isalnum():
            raise WorkbenchError('invalid_source', 'Invalid driver identity')
        candidates = pairs.get(code, set())
        if num is None:
            if len(candidates) != 1:
                raise WorkbenchError('invalid_source', 'Ambiguous driver identity')
            num = next(iter(candidates))
        num = str(num)
        if not num.isdigit() or (candidates and candidates != {num}) or len(reverse.get(num, set())) > 1 or num in seen or code in seen_codes:
            raise WorkbenchError('invalid_source', 'Conflicting driver identity')
        seen.add(num)
        seen_codes.add(code)
        result.append({'driver_number': num, 'driver_code': code})
    if not result:
        raise WorkbenchError('missing_stream', 'Driver identities unavailable')
    return sorted(result, key=lambda x: (int(x['driver_number']), x['driver_code']))
