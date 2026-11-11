"""Request-scoped, parameterized queries over already verified snapshots."""
from __future__ import annotations

from contextlib import contextmanager

import duckdb
import pyarrow as pa

from ..contracts import Snapshot, WorkbenchError


@contextmanager
def snapshot_query(snapshot: Snapshot):
    """Own and close one in-memory connection; callers hold their catalog lease."""
    connection = duckdb.connect(database=":memory:")
    try:
        for name, table in snapshot.tables.items():
            if "/" not in name and name.isidentifier():
                connection.register(name, table)
        yield connection
    finally:
        connection.close()


def prefix_table(snapshot: Snapshot, name: str, cutoff_ns: int) -> pa.Table:
    if name not in snapshot.tables or "/" in name or not name.isidentifier():
        raise WorkbenchError("missing_stream", "Canonical stream unavailable")
    table = snapshot.table(name)
    if "release_ns" not in table.column_names:
        raise WorkbenchError("invalid_stream", "Stream has no release timestamp")
    with snapshot_query(snapshot) as connection:
        # The identifier is checked against the verified table registry, never user SQL.
        return connection.execute(f'SELECT * FROM "{name}" WHERE release_ns <= ? ORDER BY release_ns, source_row_id', [cutoff_ns]).to_arrow_table()
