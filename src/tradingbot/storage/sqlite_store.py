"""SQLite implementation of CandleStore (zero external dependencies)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tradingbot.models import AlertEvent, Candle, TimeRange
from tradingbot.storage.base import CandleStore

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time_ms INTEGER NOT NULL,
    close_time_ms INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    is_closed INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (provider, symbol, timeframe, open_time_ms)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS sync_ranges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    checked_at_ms INTEGER NOT NULL,
    UNIQUE (provider, symbol, timeframe, start_ms, end_ms)
);

CREATE INDEX IF NOT EXISTS idx_sync_ranges_lookup
    ON sync_ranges (provider, symbol, timeframe, start_ms, end_ms);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    anchor_open_time_ms INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions (provider, symbol, timeframe, anchor_open_time_ms DESC, id DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    anchor_open_time_ms INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    consecutive_hits INTEGER NOT NULL,
    rule_name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alerts_asset
    ON alerts (asset_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_unack
    ON alerts (acknowledged, id ASC);
"""


class SQLiteCandleStore(CandleStore):
    """Thread-safe local SQLite store for raw OHLCV candles, sync metadata, predictions, and alerts."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        wal_mode: bool = True,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._db_path_str = str(db_path)
        self._lock = threading.RLock()

        if self._db_path_str != ":memory:":
            Path(self._db_path_str).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self._db_path_str,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

        with self._lock:
            self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)};")
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if wal_mode and self._db_path_str != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.executescript(_SCHEMA_SQL)
            self._ensure_schema_migrations()
            self._conn.commit()

    def _ensure_schema_migrations(self) -> None:
        """Apply safe additive column migrations for databases created under v0.1.0."""
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "asset_id" not in cols:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN asset_id TEXT NOT NULL DEFAULT ''"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_asset "
            "ON predictions (asset_id, created_at_ms DESC, id DESC)"
        )

    def __enter__(self) -> SQLiteCandleStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_candle(row: sqlite3.Row) -> Candle:
        return Candle(
            provider=row["provider"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            open_time_ms=row["open_time_ms"],
            close_time_ms=row["close_time_ms"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            is_closed=bool(row["is_closed"]),
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> AlertEvent:
        return AlertEvent(
            alert_id=int(row["id"]),
            asset_id=row["asset_id"],
            provider=row["provider"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            anchor_open_time_ms=int(row["anchor_open_time_ms"]),
            created_at_ms=int(row["created_at_ms"]),
            action=row["action"],
            confidence=float(row["confidence"]),
            consecutive_hits=int(row["consecutive_hits"]),
            rule_name=row["rule_name"],
            payload_json=row["payload_json"],
            acknowledged=bool(row["acknowledged"]),
        )

    def upsert_candles(self, candles: Sequence[Candle]) -> int:
        if not candles:
            return 0

        rows = [
            (
                c.provider,
                c.symbol,
                c.timeframe,
                c.open_time_ms,
                c.close_time_ms,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                1 if c.is_closed else 0,
            )
            for c in candles
        ]

        sql = """
        INSERT INTO candles (
            provider, symbol, timeframe, open_time_ms, close_time_ms,
            open, high, low, close, volume, is_closed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, symbol, timeframe, open_time_ms)
        DO UPDATE SET
            close_time_ms = excluded.close_time_ms,
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            is_closed = excluded.is_closed
        """

        with self._lock:
            with self._conn:
                self._conn.executemany(sql, rows)
        return len(rows)

    def get_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        closed_only: bool = False,
    ) -> list[Candle]:
        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]

        if start_ms is not None:
            clauses.append("open_time_ms >= ?")
            params.append(start_ms)
        if end_ms is not None:
            clauses.append("open_time_ms < ?")
            params.append(end_ms)
        if closed_only:
            clauses.append("is_closed = 1")

        sql = (
            "SELECT provider, symbol, timeframe, open_time_ms, close_time_ms, "
            "open, high, low, close, volume, is_closed "
            f"FROM candles WHERE {' AND '.join(clauses)} "
            "ORDER BY open_time_ms ASC"
        )

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_candle(r) for r in rows]

    def get_latest_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        closed_only: bool = False,
    ) -> list[Candle]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(f"limit must be a non-negative integer, got {limit!r}")
        if limit == 0:
            return []

        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]
        if closed_only:
            clauses.append("is_closed = 1")
        params.append(limit)

        sql = (
            "SELECT provider, symbol, timeframe, open_time_ms, close_time_ms, "
            "open, high, low, close, volume, is_closed "
            f"FROM candles WHERE {' AND '.join(clauses)} "
            "ORDER BY open_time_ms DESC LIMIT ?"
        )

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        rows.reverse()
        return [self._row_to_candle(r) for r in rows]

    def get_open_times(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        closed_only: bool = True,
    ) -> list[int]:
        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]

        if start_ms is not None:
            clauses.append("open_time_ms >= ?")
            params.append(start_ms)
        if end_ms is not None:
            clauses.append("open_time_ms < ?")
            params.append(end_ms)
        if closed_only:
            clauses.append("is_closed = 1")

        sql = (
            f"SELECT open_time_ms FROM candles WHERE {' AND '.join(clauses)} "
            "ORDER BY open_time_ms ASC"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [int(r["open_time_ms"]) for r in rows]

    def get_bounds(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        closed_only: bool = False,
    ) -> tuple[int, int] | None:
        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]
        if closed_only:
            clauses.append("is_closed = 1")

        sql = (
            "SELECT MIN(open_time_ms) AS min_t, MAX(open_time_ms) AS max_t "
            f"FROM candles WHERE {' AND '.join(clauses)}"
        )
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        if row is None or row["min_t"] is None:
            return None
        return (int(row["min_t"]), int(row["max_t"]))

    def count_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        closed_only: bool = False,
    ) -> int:
        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]
        if closed_only:
            clauses.append("is_closed = 1")

        sql = f"SELECT COUNT(*) AS cnt FROM candles WHERE {' AND '.join(clauses)}"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["cnt"]) if row is not None else 0

    def delete_candles(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> int:
        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]
        if start_ms is not None:
            clauses.append("open_time_ms >= ?")
            params.append(start_ms)
        if end_ms is not None:
            clauses.append("open_time_ms < ?")
            params.append(end_ms)

        sql = f"DELETE FROM candles WHERE {' AND '.join(clauses)}"
        with self._lock:
            with self._conn:
                cur = self._conn.execute(sql, params)
                return int(cur.rowcount)

    def record_sync_range(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        time_range: TimeRange,
        checked_at_ms: int,
    ) -> None:
        if time_range.is_empty:
            return

        sql = """
        INSERT INTO sync_ranges (provider, symbol, timeframe, start_ms, end_ms, checked_at_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, symbol, timeframe, start_ms, end_ms)
        DO UPDATE SET checked_at_ms = excluded.checked_at_ms
        """
        with self._lock:
            with self._conn:
                self._conn.execute(
                    sql,
                    (
                        provider,
                        symbol,
                        timeframe,
                        time_range.start_ms,
                        time_range.end_ms,
                        checked_at_ms,
                    ),
                )

    def get_sync_ranges(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[TimeRange]:
        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]

        # Half-open overlap check: [row.start_ms, row.end_ms) overlaps [start_ms, end_ms)
        if start_ms is not None:
            clauses.append("end_ms > ?")
            params.append(start_ms)
        if end_ms is not None:
            clauses.append("start_ms < ?")
            params.append(end_ms)

        sql = (
            "SELECT start_ms, end_ms FROM sync_ranges "
            f"WHERE {' AND '.join(clauses)} ORDER BY start_ms ASC, end_ms ASC"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [TimeRange(int(r["start_ms"]), int(r["end_ms"])) for r in rows]

    def save_prediction(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        anchor_open_time_ms: int,
        created_at_ms: int,
        model_name: str,
        payload_json: str,
        *,
        asset_id: str = "",
    ) -> int:
        sql = """
        INSERT INTO predictions (
            asset_id, provider, symbol, timeframe, anchor_open_time_ms,
            created_at_ms, model_name, payload_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    sql,
                    (
                        asset_id,
                        provider,
                        symbol,
                        timeframe,
                        anchor_open_time_ms,
                        created_at_ms,
                        model_name,
                        payload_json,
                    ),
                )
                assert cur.lastrowid is not None
                return int(cur.lastrowid)

    def get_predictions(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        asset_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(f"limit must be a non-negative integer, got {limit!r}")
        if limit == 0:
            return []

        clauses = ["provider = ?", "symbol = ?", "timeframe = ?"]
        params: list[Any] = [provider, symbol, timeframe]
        if asset_id is not None:
            clauses.append("asset_id = ?")
            params.append(asset_id)
        params.append(limit)

        sql = (
            "SELECT id, asset_id, provider, symbol, timeframe, anchor_open_time_ms, "
            "created_at_ms, model_name, payload_json "
            f"FROM predictions WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at_ms DESC, id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def save_alert(self, alert: AlertEvent) -> int:
        sql = """
        INSERT INTO alerts (
            asset_id, provider, symbol, timeframe, anchor_open_time_ms,
            created_at_ms, action, confidence, consecutive_hits,
            rule_name, payload_json, acknowledged
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    sql,
                    (
                        alert.asset_id,
                        alert.provider,
                        alert.symbol,
                        alert.timeframe,
                        alert.anchor_open_time_ms,
                        alert.created_at_ms,
                        alert.action,
                        alert.confidence,
                        alert.consecutive_hits,
                        alert.rule_name,
                        alert.payload_json,
                        1 if alert.acknowledged else 0,
                    ),
                )
                assert cur.lastrowid is not None
                return int(cur.lastrowid)

    def get_alerts(
        self,
        *,
        asset_id: str | None = None,
        since_id: int | None = None,
        unacknowledged_only: bool = False,
        limit: int = 100,
    ) -> list[AlertEvent]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(f"limit must be a non-negative integer, got {limit!r}")
        if since_id is not None and (isinstance(since_id, bool) or not isinstance(since_id, int) or since_id < 0):
            raise ValueError(f"since_id must be a non-negative integer, got {since_id!r}")
        if limit == 0:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        if asset_id is not None:
            clauses.append("asset_id = ?")
            params.append(asset_id)
        if since_id is not None:
            clauses.append("id > ?")
            params.append(since_id)
        if unacknowledged_only:
            clauses.append("acknowledged = 0")

        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        # Cursor pagination (since_id) returns oldest-unread-first so external consumers
        # advance their cursor reliably without skipping alerts.
        order_sql = "ORDER BY id ASC" if since_id is not None else "ORDER BY id DESC"
        params.append(limit)

        sql = (
            "SELECT id, asset_id, provider, symbol, timeframe, anchor_open_time_ms, "
            "created_at_ms, action, confidence, consecutive_hits, rule_name, "
            f"payload_json, acknowledged FROM alerts {where_sql}{order_sql} LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> bool:
        if isinstance(alert_id, bool) or not isinstance(alert_id, int) or alert_id < 1:
            raise ValueError(f"alert_id must be a positive integer, got {alert_id!r}")
        sql = "UPDATE alerts SET acknowledged = 1 WHERE id = ? AND acknowledged = 0"
        with self._lock:
            with self._conn:
                cur = self._conn.execute(sql, (alert_id,))
                return int(cur.rowcount) > 0

    def alert_exists(self, alert_id: int) -> bool:
        if isinstance(alert_id, bool) or not isinstance(alert_id, int) or alert_id < 1:
            raise ValueError(f"alert_id must be a positive integer, got {alert_id!r}")
        sql = "SELECT 1 FROM alerts WHERE id = ? LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, (alert_id,)).fetchone()
        return row is not None

    def count_alerts(
        self,
        *,
        asset_id: str | None = None,
        unacknowledged_only: bool = False,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if asset_id is not None:
            clauses.append("asset_id = ?")
            params.append(asset_id)
        if unacknowledged_only:
            clauses.append("acknowledged = 0")
        where_sql = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = f"SELECT COUNT(*) AS cnt FROM alerts {where_sql}"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["cnt"]) if row is not None else 0
