from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import psycopg
from psycopg.rows import dict_row


@dataclass
class PgConnectionInfo:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    def to_conn_str(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


def ensure_pgvector_enabled(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()


def get_connection(info: PgConnectionInfo) -> psycopg.Connection:
    conn = psycopg.connect(info.to_conn_str(), row_factory=dict_row)
    return conn
