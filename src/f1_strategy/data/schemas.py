"""Central column types: no object-per-sample validation."""
import pyarrow as pa

COMMON = [pa.field("session_id", pa.string()), pa.field("source_row_id", pa.string()),
          pa.field("event_ns", pa.int64()), pa.field("release_ns", pa.int64()), pa.field("time_basis", pa.string())]
def schema(fields):
    return pa.schema(COMMON + [pa.field(n, t) for n, t in fields])
S = pa.string(); I = pa.int32(); N = pa.int64(); F = pa.float64(); B = pa.bool_()
SCHEMAS = {
    "laps": schema([("driver_number", S), ("driver_code", S), ("lap_number", I), ("lap_start_ns", N), ("lap_end_ns", N),
        ("lap_time_s", F), ("sector1_s", F), ("sector2_s", F), ("sector3_s", F), ("source_stint", I),
        ("compound", S), ("tyre_age_laps", I), ("fresh_tyre", B), ("source_track_status", S), ("is_accurate", B),
        ("fastf1_generated", B), ("final_deleted", B), ("final_deleted_reason", S), ("source_position", I)]),
    "pit_events": schema([("driver_number", S), ("kind", S), ("source_lap_number", I)]),
    "weather": schema([("air_temp_c", F), ("track_temp_c", F), ("humidity_pct", F), ("pressure_mbar", F),
        ("rainfall", B), ("wind_direction_deg", F), ("wind_speed_ms", F)]),
    "track_status": schema([("source_status", S), ("status", S), ("message", S)]),
    "session_status": schema([("status", S)]),
    "messages": schema([("utc", pa.timestamp("ns", tz="UTC")), ("category", S), ("message", S), ("status", S),
        ("flag", S), ("scope", S), ("sector", I), ("driver_number", S), ("lap_number", I)]),
    "stints": schema([("driver_number", S), ("stint_id", S), ("start_ns", N), ("end_ns", N), ("compound", S),
        ("starting_tyre_age", I), ("observed_laps", I), ("source_ids", pa.list_(S))]),
}

def table_from_rows(name, rows):
    return pa.Table.from_pylist(rows, schema=SCHEMAS[name])
