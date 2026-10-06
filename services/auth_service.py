"""
auth_service.py
-----------------
User account logic: signup, lookup, password verification. Backed by
the `users` table (see schema.sql). Passwords are never stored in plain
text - werkzeug's generate_password_hash/check_password_hash (already a
Flask dependency) handles hashing and verification.
"""

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from werkzeug.security import generate_password_hash, check_password_hash

from utils.db import get_engine


class EmailAlreadyExists(Exception):
    pass


class DatabaseUnavailable(Exception):
    """Raised when the app can't reach MySQL - e.g. XAMPP isn't running."""
    pass


def _wrap_db_errors(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OperationalError as exc:
            raise DatabaseUnavailable(
                "Can't reach the database. Make sure XAMPP's MySQL service is running."
            ) from exc
    return wrapped


@_wrap_db_errors
def create_user(full_name, email, password):
    engine = get_engine()
    password_hash = generate_password_hash(password)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO users (full_name, email, password_hash) "
                    "VALUES (:full_name, :email, :password_hash)"
                ),
                {"full_name": full_name.strip(), "email": email.strip().lower(), "password_hash": password_hash},
            )
    except IntegrityError:
        raise EmailAlreadyExists(f"An account with the email {email} already exists.")


@_wrap_db_errors
def get_user_by_email(email):
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": email.strip().lower()},
        ).mappings().first()
    return dict(row) if row else None


@_wrap_db_errors
def get_user_by_id(user_id):
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE id = :id"),
            {"id": user_id},
        ).mappings().first()
    return dict(row) if row else None


def verify_password(user_row, password):
    return check_password_hash(user_row["password_hash"], password)


@_wrap_db_errors
def email_exists(email):
    return get_user_by_email(email) is not None
