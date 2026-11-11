"""Fixed private renderer for a coordinator-owned report stage."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    from .contracts import DecisionResult, WorkbenchError
    from .deployment import MAX_STDIN, MAX_STDOUT
    from .identity import canonical, file_digest, verify_payload
    job = None
    try:
        raw = sys.stdin.buffer.read(MAX_STDIN + 1)
        if len(raw) > MAX_STDIN:
            raise WorkbenchError('capacity_exceeded', 'Request too large')
        request = json.loads(raw)
        if set(request) != {'kind', 'payload', 'job_id'} or request['kind'] != 'report' or request['payload'] != {} or not isinstance(request['job_id'], str):
            raise WorkbenchError('invalid_input', 'Invalid request')
        job = request['job_id']
        stage = Path(os.environ['F1_WORKER_STAGE'])
        analysis = stage / 'analysis.json'
        if analysis.is_symlink() or analysis.stat().st_size > MAX_STDOUT:
            raise WorkbenchError('capacity_exceeded', 'Report input exceeds capacity')
        payload = json.loads(analysis.read_bytes())
        verify_payload(payload)
        result = DecisionResult.model_validate_json(canonical(payload))
        with contextlib.redirect_stdout(sys.stderr):
            from .reporting import render_report
            report = render_report(result, output_dir=stage / 'output')
        response = {'job_id': job, 'result': {'path': f'output/{result.analysis_id}/report.html', 'sha256': file_digest(report.path)}}
    except WorkbenchError as exc:
        response = {'job_id': job, 'error': exc.code}
    except Exception:
        response = {'job_id': job, 'error': 'worker_failed'}
    sys.stdout.buffer.write(canonical(response))
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
