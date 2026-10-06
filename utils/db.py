"""
db.py
-----
Database connection layer. Works with BOTH:

  1. A local MySQL/MariaDB server (e.g. XAMPP) for development.
  2. A cloud MySQL or PostgreSQL database (e.g. PlanetScale, Railway,
     Supabase, Neon, Render) for production deployment (e.g. on Vercel).

Configuration (either one works):

  A) Set a single DATABASE_URL - the standard convention most cloud DB
     hosts give you directly, e.g.:
       postgresql+psycopg2://user:pass@host:5432/dbname
       mysql+pymysql://user:pass@host:3306/dbname
     If a host gives you a plain "postgres://..." or "postgresql://..."
     URL (no driver suffix), this module upgrades it to
     "postgresql+psycopg2://..." automatically.

  B) Or set individual DB_HOST / DB_PORT / DB_USER / DB_PASSWORD /
     DB_NAME / DB_DIALECT vars - this is what a fresh XAMPP install
     needs (and is exactly what happens with no configuration at all,
     since these all have XAMPP-matching defaults).

DATABASE_URL takes priority if both are set. is_postgres() / is_mysql()
let the rest of the app write small dialect-aware branches (e.g. how to
read back an inserted row's ID) without hardcoding one database engine
everywhere.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed - env vars still work if set another way

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

DB_DIALECT = os.environ.get("DB_DIALECT", "mysql").strip().lower()  # "mysql" or "postgres"
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = os.environ.get("DB_PORT", "3306" if DB_DIALECT == "mysql" else "5432")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "aura_db")

_engine = None
_dialect_cache = None


def _normalize_database_url(url: str) -> str:
    """Upgrades bare postgres://... / mysql://... URLs to the driver-specific form."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    elif url.startswith("mysql://"):
        url = "mysql+pymysql://" + url[len("mysql://"):]
    return url


def _build_url():
    if DATABASE_URL:
        return _normalize_database_url(DATABASE_URL)

    password_part = f":{DB_PASSWORD}" if DB_PASSWORD else ""
    if DB_DIALECT.startswith("postgres"):
        return f"postgresql+psycopg2://{DB_USER}{password_part}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return f"mysql+pymysql://{DB_USER}{password_part}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"


def get_engine():
    """Lazily creates a single shared SQLAlchemy engine for the app."""
    global _engine
    if _engine is None:
        _engine = create_engine(_build_url(), pool_pre_ping=True, pool_recycle=280)
    return _engine


def dialect_name():
    """Returns 'postgresql' or 'mysql' based on the active connection."""
    global _dialect_cache
    if _dialect_cache is None:
        _dialect_cache = make_url(_build_url()).get_backend_name()
    return _dialect_cache


def is_postgres():
    return dialect_name().startswith("postgres")


def is_mysql():
    return dialect_name() == "mysql"


def test_connection():
    """Returns (ok: bool, error_message: str | None). Never raises."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:
        return False, str(exc)
