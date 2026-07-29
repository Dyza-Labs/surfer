import os
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

load_dotenv()
DEFAULT_MODEL = os.getenv("SUBAGENT_MODEL")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL")

if DEFAULT_MODEL is None:
    raise RuntimeError("SUBAGENT_MODEL does not exist in .env")
if ORCHESTRATOR_MODEL is None:
    raise RuntimeError("ORCHESTRATOR_MODEL does not exist in .env")

# Config settings vary by model
supervisor_model = ChatOpenRouter(model=ORCHESTRATOR_MODEL,
                                  reasoning={"exclude": True},
                                  temperature=0, max_tokens=1024)
subagent_model = ChatOpenRouter(
    model=DEFAULT_MODEL,
    reasoning={"enabled": False},
    model_kwargs={"parallel_tool_calls": False},
    temperature=0,
)

# POSTGRES_URI=postgresql://surfer:surfer@localhost:5432/surfer if using docker-compose.yml
# (adjust to match your docker-compose.yml or Neon connection string)
POSTGRES_URI = os.getenv("POSTGRES_URI")


def _check_connection(conn: Connection[DictRow]) -> None:
    # Inlines ConnectionPool.check_connection: its stub is pinned to Connection[TupleRow]
    # and rejects DictRow, even though the real check is row-factory-agnostic.
    if conn.autocommit:
        conn.execute("")
    else:
        conn.autocommit = True
        try:
            conn.execute("")
        finally:
            conn.autocommit = False


def get_checkpointer() -> BaseCheckpointSaver:
    """Postgres-backed persistence when POSTGRES_URI is set, else an in-memory fallback.

    A pool is built and kept open for the process lifetime, since
    PostgresSaver.from_conn_string() is a context manager that closes on exit."""
    if POSTGRES_URI:
        # check=_check_connection discards stale connections instead of handing them back --
        # hosted Postgres (Neon/Supabase) closes idle connections, and a chat app's
        # connections sit idle between turns longer than a typical web request would.
        pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
            POSTGRES_URI, open=True, min_size=1, max_size=10,
            connection_class=Connection[DictRow],
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            check=_check_connection,
        )
        saver = PostgresSaver(pool)
        saver.setup()  # Idempotent; creates tables/indexes on first run
        return saver
    return InMemorySaver()
