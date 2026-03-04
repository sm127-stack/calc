from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash

import model


APP_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(APP_DIR, "app.db")

DATABASE_URL = os.environ.get("DATABASE_URL")

def is_postgres() -> bool:
    return bool(DATABASE_URL)

def exec_sql(con, sql, params=()):
    """Execute SQL and return a cursor (works for sqlite + postgres)."""
    if is_postgres():
        cur = con.cursor()
        cur.execute(sql, params)
        return cur
    else:
        return con.execute(sql, params)

def fetchone(cur):
    return cur.fetchone()

def fetchall(cur):
    return cur.fetchall()


def db():
    if DATABASE_URL:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        con = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return con
    else:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        return con


def init_db() -> None:
    with db() as con:
        if is_postgres():
            exec_sql(con, """
                CREATE TABLE IF NOT EXISTS users (
                  id SERIAL PRIMARY KEY,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
            """)
            exec_sql(con, """
                CREATE TABLE IF NOT EXISTS predictions (
                  id SERIAL PRIMARY KEY,
                  user_id INTEGER NOT NULL,
                  ts TEXT NOT NULL,
                  f0 DOUBLE PRECISION NOT NULL,
                  f1 DOUBLE PRECISION NOT NULL,
                  f2 DOUBLE PRECISION NOT NULL,
                  f3 DOUBLE PRECISION NOT NULL,
                  f4 INTEGER NOT NULL,
                  y  DOUBLE PRECISION NOT NULL
                )
            """)
        else:
            exec_sql(con, """
                CREATE TABLE IF NOT EXISTS users (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
            """)
            exec_sql(con, """
                CREATE TABLE IF NOT EXISTS predictions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  ts TEXT NOT NULL,
                  f0 REAL NOT NULL,
                  f1 REAL NOT NULL,
                  f2 REAL NOT NULL,
                  f3 REAL NOT NULL,
                  f4 INTEGER NOT NULL,
                  y REAL NOT NULL
                )
            """)


def current_user_id() -> int | None:
    email = session.get("user_email")
    if not email:
        return None
    with db() as con:
        cur = exec_sql(con, "SELECT id FROM users WHERE email = %s" if is_postgres() else "SELECT id FROM users WHERE email = ?", (email,))
        row = fetchone(cur)
        return int(row["id"]) if row else None


def get_saved_rows(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    with db() as con:
        rows = con.execute(
            "SELECT ts,f0,f1,f2,f3,f4,y FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows[::-1]:  # oldest->newest
        out.append({
            "ts": r["ts"],
            model.FEATURES[0]: r["f0"],
            model.FEATURES[1]: r["f1"],
            model.FEATURES[2]: r["f2"],
            model.FEATURES[3]: r["f3"],
            model.FEATURES[4]: int(r["f4"]),
            model.TARGET: r["y"],
        })
    return out


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

with app.app_context():
    init_db()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/signup")
def signup():
    email = (request.form.get("email") or "").strip().lower()
    psw = request.form.get("psw") or ""
    if not email or not psw:
        flash("Missing email or password.")
        return redirect(url_for("home"))

    with db() as con:
        try:
            con.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?,?,?)",
                (email, generate_password_hash(psw), datetime.utcnow().isoformat(timespec="seconds")),
            )
        except sqlite3.IntegrityError:
            flash("That user already exists. Try logging in.")
            return redirect(url_for("home"))

    flash("Signup successful. You can now log in.")
    return redirect(url_for("home"))


@app.post("/login")
def login():
    email = (request.form.get("email") or "").strip().lower()
    psw = request.form.get("psw") or ""
    with db() as con:
        sql = "SELECT email,password_hash FROM users WHERE email=%s" if is_postgres() else "SELECT email,password_hash FROM users WHERE email=?"
        cur = exec_sql(con, sql, (email,))
        row = fetchone(cur)

    if not row or not check_password_hash(row["password_hash"], psw):
        flash("Invalid username or password.")
        return redirect(url_for("home"))

    session["user_email"] = email
    return redirect(url_for("member"))


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.get("/guest")
def guest():
    defaults = model.MU.tolist()
    defaults[-1] = int(round(defaults[-1]))
    return render_template(
        "calculator.html",
        title="Guest Calculator",
        mode="guest",
        features=model.FEATURES,
        target=model.TARGET,
        defaults=defaults,
        saved_rows=[],
    )


@app.get("/member")
def member():
    uid = current_user_id()
    if uid is None:
        flash("Please log in to access the member calculator.")
        return redirect(url_for("home"))

    defaults = model.MU.tolist()
    defaults[-1] = int(round(defaults[-1]))
    rows = get_saved_rows(uid, limit=10)
    return render_template(
        "calculator.html",
        title="Member Calculator",
        mode="member",
        features=model.FEATURES,
        target=model.TARGET,
        defaults=defaults,
        saved_rows=rows,
    )


@app.post("/api/guest/predict")
def api_guest_predict():
    payload = request.get_json(force=True, silent=True) or {}
    x = payload.get("x")
    if not isinstance(x, list) or len(x) != len(model.FEATURES):
        return jsonify({"error": "Expected JSON: {x:[5 numbers]}"}), 400
    try:
        y = model.predict(x)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"y": y})


@app.post("/api/member/predict")
def api_member_predict():
    if current_user_id() is None:
        return jsonify({"error": "Not logged in"}), 401
    return api_guest_predict()


@app.post("/api/member/save")
def api_member_save():
    uid = current_user_id()
    if uid is None:
        return jsonify({"error": "Not logged in"}), 401

    payload = request.get_json(force=True, silent=True) or {}
    x = payload.get("x")
    y = payload.get("y")
    if not isinstance(x, list) or len(x) != len(model.FEATURES):
        return jsonify({"error": "Expected JSON: {x:[5 numbers], y:number}"}), 400
    try:
        y = float(y)
        f0, f1, f2, f3, f4 = float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(round(float(x[4])))
    except Exception:
        return jsonify({"error": "Inputs must be numeric"}), 400

    ts = datetime.utcnow().isoformat(timespec="seconds")
    with db() as con:
        con.execute(
            "INSERT INTO predictions (user_id,ts,f0,f1,f2,f3,f4,y) VALUES (?,?,?,?,?,?,?,?)",
            (uid, ts, f0, f1, f2, f3, f4, y),
        )

    rows = get_saved_rows(uid, limit=10)
    return jsonify({"ok": True, "rows": rows})


if __name__ == "__main__":
    # Run: python app.py
    app.run(host="127.0.0.1", port=3000, debug=True)
