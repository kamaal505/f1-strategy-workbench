"""Render a sealed analysis without loading its source snapshots or models."""
from __future__ import annotations

import html
import json
from importlib.resources import files
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, select_autoescape

from .contracts import DecisionResult, ReportRef, REPLAY_NOTICE, WorkbenchError
from .identity import canonical, check_id, confined, digest, file_digest, verify_payload
from .payloads import load_analysis, save_analysis


def forecast_chart(payload: dict) -> go.Figure:
    figure = go.Figure()
    for branch, segments in payload.get("forecast_series", {}).items():
        for index, segment in enumerate(segments):
            laps, means = segment.get("race_laps", []), segment.get("mean_s", [])
            bounds = segment.get("interval80_s", [])
            if len(bounds) == len(laps) and bounds and all(len(b) == 2 and None not in b for b in bounds):
                figure.add_trace(go.Scatter(x=laps, y=[b[0] for b in bounds], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                figure.add_trace(go.Scatter(x=laps, y=[b[1] for b in bounds], mode="lines", line=dict(width=0), fill="tonexty", opacity=0.18, showlegend=False, hoverinfo="skip"))
            label = html.escape(f"{branch.replace('_', ' ')} · {segment.get('compound', 'unknown')}")
            figure.add_trace(go.Scatter(x=laps, y=means, mode="lines+markers", name=label, legendgroup=html.escape(branch)))
    figure.update_layout(title="Conditional lap pace · shaded 80% interval", xaxis_title="Race lap", yaxis_title="Seconds", template="plotly_white")
    return figure

def delta_chart(payload: dict) -> go.Figure:
    figure = go.Figure()
    draws = payload.get("simulation_series", {}).get("paired_delta_s", [])
    if draws:
        figure.add_trace(go.Histogram(x=draws, name="Paired draws"))
        figure.add_vline(x=0, line_dash="dash")
    figure.update_layout(title="Paired time difference: stay out minus pit now", xaxis_title="Seconds · positive favours pit now", yaxis_title="Draws", template="plotly_white")
    return figure


def render_report(result: DecisionResult, *, output_dir: Path) -> ReportRef:
    payload = result.model_dump(mode="json")
    verify_payload(payload)
    check_id(result.analysis_id)
    output_dir = Path(output_dir)
    directory = confined(output_dir, result.analysis_id, must_exist=False)
    directory.mkdir(parents=True, exist_ok=True)
    report = confined(directory, "report.html", must_exist=False)
    manifest = confined(directory, "manifest.json", must_exist=False)
    analysis = confined(directory, "analysis.json", must_exist=False)
    if report.exists() or manifest.exists():
        if not (report.is_file() and manifest.is_file() and analysis.is_file()):
            raise WorkbenchError("corrupt_report", "Incomplete existing report; refusing overwrite")
        try:
            existing = json.loads(manifest.read_bytes())
            valid = (load_analysis(analysis).payload_checksum == result.payload_checksum and
                     existing == {"analysis_id": result.analysis_id, "payload_checksum": result.payload_checksum,
                                  "analysis_sha256": file_digest(analysis), "report_sha256": file_digest(report)})
        except (OSError, ValueError, KeyError):
            valid = False
        if not valid:
            raise WorkbenchError("corrupt_report", "Existing report differs; refusing overwrite")
        return ReportRef(analysis_id=result.analysis_id, path=report, analysis_path=analysis, manifest_path=manifest)
    analysis = save_analysis(result, output_dir=output_dir)
    template = files("f1_strategy").joinpath("assets", "report.html.j2").read_text(encoding="utf-8")
    environment = Environment(autoescape=select_autoescape(default=True))
    charts = [forecast_chart(payload).to_html(full_html=False, include_plotlyjs=True),
              delta_chart(payload).to_html(full_html=False, include_plotlyjs=False)]
    html = environment.from_string(template).render(result=payload, notice=REPLAY_NOTICE, charts=charts,
                                                    pretty=lambda x: json.dumps(x, indent=2, ensure_ascii=False))
    try:
        with report.open("x", encoding="utf-8") as handle:
            handle.write(html)
        info = {"analysis_id": result.analysis_id, "payload_checksum": result.payload_checksum,
                "analysis_sha256": file_digest(analysis), "report_sha256": file_digest(report)}
        with manifest.open("xb") as handle:
            handle.write(canonical(info))
    except FileExistsError:
        raise WorkbenchError("corrupt_report", "Concurrent report publication; refusing overwrite") from None
    return ReportRef(analysis_id=result.analysis_id, path=report, analysis_path=analysis, manifest_path=manifest)
