"""Local operator CLI. Core modules, not the CLI, own analytical calculations."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import functools
import json
from pathlib import Path
import sys
import typer
from pydantic import ValidationError
from .contracts import WorkbenchError
from .identity import canonical

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

def emit(value):
    typer.echo(canonical(value).decode())

def guarded(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except WorkbenchError as e:
            codes = {"invalid": 2, "missing": 3, "acquisition": 3, "auth": 3, "permission": 3,
                     "corrupt": 4, "future": 4, "unsafe": 4}
            state_inputs = {"generated_focal_completion", "race_not_started", "focal_in_pit"}
            exit_code = 2 if e.code in state_inputs else next((v for k, v in codes.items() if e.code.startswith(k)), 5)
            typer.echo(canonical({"error": e.code, "detail": e.detail}).decode(), err=True)
            raise typer.Exit(exit_code) from None
        except (ValidationError, ValueError, TypeError) as e:
            typer.echo(canonical({"error": "invalid_input", "detail": "Input does not match the validated contract"}).decode(), err=True)
            raise typer.Exit(2) from None
        except FileNotFoundError:
            typer.echo('{"error":"missing_input","detail":"Required local input is unavailable"}', err=True)
            raise typer.Exit(3) from None
        except Exception:
            typer.echo('{"error":"internal_error","detail":"Analysis failed; no result was published"}', err=True)
            raise typer.Exit(5) from None
    return wrapped

@app.command()
@guarded
def synthetic(output_root: Path = typer.Option(Path("data/demo"))):
    from .synthetic import generate_demo
    emit(generate_demo(output_root))

@app.command("snapshot-list")
@guarded
def snapshot_list(root: Path = typer.Option(Path("data"))):
    from .data.snapshots import list_snapshots
    emit(list_snapshots(root))

@app.command("snapshot-check")
@guarded
def snapshot_check(snapshot: str = typer.Option(...), root: Path = typer.Option(Path("data"))):
    from .data.snapshots import open_snapshot
    snap = open_snapshot(root, snapshot)
    emit({"snapshot_id": snap.ref.snapshot_id, "session_id": snap.ref.session_id, "valid": True,
          "availability": snap.manifest.availability, "quality": snap.manifest.quality})

@app.command()
@guarded
def ingest(year: int = typer.Option(...), round: int = typer.Option(...),
           telemetry: bool = typer.Option(False), root: Path = typer.Option(Path("data"))):
    from .contracts import IngestRequest
    from .data.acquire import acquire_session
    emit(acquire_session(IngestRequest(year=year, round_number=round, telemetry=telemetry), root=root))

@app.command()
@guarded
def train(spec: Path = typer.Option(...), root: Path = typer.Option(Path("data"))):
    from .contracts import TrainingSpec
    from .pace.hierarchical import fit_pace
    emit(fit_pace(TrainingSpec.model_validate_json(spec.read_bytes()), root=root))

@app.command()
@guarded
def replay(snapshot: str = typer.Option(...), driver: str = typer.Option(...),
           lap: int = typer.Option(...), root: Path = typer.Option(Path("data"))):
    from .data.snapshots import open_snapshot
    from .replay import build_race_state
    from .payloads import prefix_payload
    emit(prefix_payload(build_race_state(open_snapshot(root, snapshot), driver_number=driver, completed_lap=lap)))

@app.command()
@guarded
def compare(snapshot: str = typer.Option(...), model: str = typer.Option(...),
            driver: str = typer.Option(...), lap: int = typer.Option(...),
            scenario: Path = typer.Option(...), draws: int = typer.Option(1000, min=100, max=5000),
            seed: int = typer.Option(2026), root: Path = typer.Option(Path("data")),
            output_dir: Path = typer.Option(Path("reports"))):
    from .data.snapshots import open_snapshot
    from .replay import build_race_state
    from .pace.artifacts import load_model
    from .strategy.plans import load_scenario
    from .strategy.simulate import compare_strategies
    from .payloads import save_analysis
    state = build_race_state(open_snapshot(root, snapshot), driver_number=driver, completed_lap=lap)
    fitted = load_model(root, model, target_start_utc=state.race_start_utc)
    result = compare_strategies(state, fitted, load_scenario(scenario, state), n_draws=draws, seed=seed)
    path = save_analysis(result, output_dir=output_dir)
    emit({"analysis_id": result.analysis_id, "analysis_path": str(path.resolve()), "decision": result})

@app.command()
@guarded
def report(analysis: Path = typer.Option(...), output_dir: Path = typer.Option(Path("reports"))):
    from .payloads import load_analysis
    from .reporting import render_report
    emit(render_report(load_analysis(analysis), output_dir=output_dir))

@app.command()
@guarded
def backtest(spec: Path = typer.Option(...), root: Path = typer.Option(Path("data"))):
    from .contracts import EvaluationSpec
    from .pace.validation import run_backtest
    emit(run_backtest(EvaluationSpec.model_validate_json(spec.read_bytes()), root=root))

@app.command("deployment-check")
@guarded
def deployment_check(root: Path = typer.Option(Path("data"))):
    from .deployment import deployment_check as check
    emit(check(root=root))

if __name__ == "__main__":
    app()
