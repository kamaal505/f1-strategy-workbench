"""Private registered worker protocol; parent owns authorization and publication."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import contextlib
import json
from pathlib import Path
import sys


def main():
    from .contracts import IngestRequest, ScenarioSpec, WorkbenchError
    from .identity import canonical
    from .deployment import MAX_STDIN
    job = None
    try:
        raw = sys.stdin.buffer.read(MAX_STDIN + 1)
        if len(raw) > MAX_STDIN:
            raise WorkbenchError('capacity_exceeded', 'Request too large')
        request = json.loads(raw)
        if set(request) != {'kind', 'payload', 'job_id'} or not isinstance(request['job_id'], str):
            raise WorkbenchError('invalid_input', 'Invalid request')
        job = request['job_id']
        payload = request['payload']
        root = Path(os.environ['F1_WORKER_ROOT'])
        stage = Path(os.environ['F1_WORKER_STAGE'])
        # Dependencies may write to stdout. Capture it as stderr, never protocol output.
        with contextlib.redirect_stdout(sys.stderr):
            if request['kind'] == 'compare':
                if set(payload) - {'snapshot_id', 'model_id', 'driver_number', 'completed_lap', 'scenario', 'n_draws', 'seed'}:
                    raise WorkbenchError('invalid_input', 'Invalid comparison request')
                from .data.snapshots import open_snapshot
                from .replay import build_race_state
                from .pace.artifacts import load_model
                from .strategy.simulate import compare_strategies
                snapshot = open_snapshot(root, payload['snapshot_id'])
                state = build_race_state(snapshot, driver_number=payload['driver_number'], completed_lap=payload['completed_lap'])
                model = load_model(root, payload['model_id'], target_start_utc=state.race_start_utc)
                scenario = ScenarioSpec.model_validate_json(canonical(payload['scenario']))
                decision = compare_strategies(state, model, scenario, n_draws=payload.get('n_draws', 200), seed=payload.get('seed', 2026))
                result = {'decision': decision.model_dump(mode='json')}
            elif request['kind'] == 'ingest':
                from .data.acquire import acquire_session
                ref = acquire_session(IngestRequest.model_validate(payload), root=stage)
                result = {'snapshot_id': ref.snapshot_id, 'session_id': ref.session_id}
            else:
                raise WorkbenchError('invalid_input', 'Unsupported job')
        response = {'job_id': job, 'result': result}
    except WorkbenchError as exc:
        response = {'job_id': job, 'error': exc.code}
    except Exception:
        response = {'job_id': job, 'error': 'worker_failed'}
    sys.stdout.buffer.write(canonical(response))
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
