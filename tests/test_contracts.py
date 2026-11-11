import pytest
from pydantic import ValidationError
from f1_strategy.contracts import IngestRequest, PitAssumptions, ScenarioSpec, TyreSet, TrainingSpec


@pytest.mark.parametrize("changes", [{"year": 2025}, {"year": "2023"}, {"round_number": 0}, {"session": "S"}, {"telemetry": "false"}])
def test_ingestion_rejects_unsupported_or_coerced_controls(changes):
    with pytest.raises(ValidationError):
        IngestRequest(**dict({"year": 2023, "round_number": 7}, **changes))


@pytest.mark.parametrize("changes", [{"service_s": 0.0}, {"net_transit_sd_s": -1.0}, {"outlap_warmup_s": float("nan")}, {"absolute_lane_duration_s": 0.0}])
def test_pit_uncertainty_and_units_boundaries(changes):
    with pytest.raises(ValidationError):
        PitAssumptions(**changes)


def test_consumable_set_ids_are_unique_across_cars():
    tyre = TyreSet(set_id="same", compound="HARD", age_laps=0, provenance="assumed")
    with pytest.raises(ValidationError):
        ScenarioSpec(name="invalid", inventory_by_driver={"44": (tyre,), "1": (tyre,)}, inventory_provenance="assumed")


def test_set_life_cannot_begin_exhausted_beyond_limit():
    with pytest.raises(ValidationError):
        TyreSet(set_id="old", compound="HARD", age_laps=19, life_limit_laps=18, provenance="operator_documented")
