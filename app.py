from __future__ import annotations

# standard library imports
import os
from datetime import datetime
from typing import Any, Dict, List

# database driver for postgresql
import psycopg2
from psycopg2.extras import RealDictCursor

# flask utilities used by web app
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
    Response,
)

# utilities for safely hashing and checking user passwords
from werkzeug.security import generate_password_hash, check_password_hash

# local module containing trained linear regression model
import model

# read database connection string from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

# creates abd returns new postgresql connection
def db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# creates required tables if they dont already exist
def init_db() -> None:
    with db() as con:
        with con.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    ts TEXT NOT NULL,
                    f0 DOUBLE PRECISION NOT NULL,
                    f1 DOUBLE PRECISION NOT NULL,
                    f2 DOUBLE PRECISION NOT NULL,
                    f3 DOUBLE PRECISION NOT NULL,
                    f4 INTEGER NOT NULL,
                    y DOUBLE PRECISION NOT NULL
                )
            """)
        con.commit()

# looks up currently logged in users database id
def current_user_id() -> int | None:
    username = session.get("user_name")
    if not username:
        return None

    with db() as con:
        with con.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            return int(row["id"]) if row else None

# fetches latest saved predictions for given user
def get_saved_rows(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    with db() as con:
        with con.cursor() as cur:
            cur.execute(
                """
                SELECT ts, f0, f1, f2, f3, f4, y
                FROM predictions
                WHERE user_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    
    for r in rows[::-1]:     # reverse so oldest of retrieved rows appears first in ui
        out.append(
            {
                "ts": r["ts"],
                model.FEATURES[0]: r["f0"],
                model.FEATURES[1]: r["f1"],
                model.FEATURES[2]: r["f2"],
                model.FEATURES[3]: r["f3"],
                model.FEATURES[4]: int(r["f4"]),
                model.TARGET: r["y"],
            }
        )
    return out

# creates flask app instance
app = Flask(__name__)

# secret key needed for sessions and flash messages
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ensures tables exist as soon as app starts
init_db()

# shows landing page containing login/signup options
@app.get("/")
def home():
    return render_template("index.html")

# creates new user account
@app.post("/signup")
def signup():
    username = (request.form.get("username") or "").strip()
    psw = request.form.get("psw") or ""

    if not username or not psw:
        flash("Missing username or password.")
        return redirect(url_for("home"))

    with db() as con:
        with con.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            existing = cur.fetchone()
            if existing:
                flash("That username already exists. Try logging in.")
                return redirect(url_for("home"))

            cur.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (
                    username,
                    generate_password_hash(psw),
                    datetime.utcnow().isoformat(timespec="seconds"),
                ),
            )
        con.commit()

    session["user_name"] = username
    return redirect(url_for("member"))

# logs existing user into site
@app.post("/login")
def login():
    username = (request.form.get("username") or "").strip()
    psw = request.form.get("psw") or ""

    with db() as con:
        with con.cursor() as cur:
            cur.execute(
                "SELECT username, password_hash FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()

    if not row or not check_password_hash(row["password_hash"], psw):
        flash("Invalid username or password.")
        return redirect(url_for("home"))

    session["user_name"] = username
    return redirect(url_for("member"))

# logs user out by clearing session
@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# shows guest version of calculator
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
        ranges=model.RANGES,
        ylim=model.Y_LIM,
    )

# shows logged in member calculator page
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
        ranges=model.RANGES,
        ylim=model.Y_LIM,
    )

# api endpoint for prediction
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

# member only wrapper around same prediction endpoint
@app.post("/api/member/predict")
def api_member_predict():
    if current_user_id() is None:
        return jsonify({"error": "Not logged in"}), 401
    return api_guest_predict()

# saves prediction to database for logged in user
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
        f0 = float(x[0])
        f1 = float(x[1])
        f2 = float(x[2])
        f3 = float(x[3])
        f4 = int(round(float(x[4])))
    except Exception:
        return jsonify({"error": "Inputs must be numeric"}), 400

    ts = datetime.utcnow().isoformat(timespec="seconds")

    with db() as con:
        with con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (user_id, ts, f0, f1, f2, f3, f4, y)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (uid, ts, f0, f1, f2, f3, f4, y),
            )
        con.commit()

    rows = get_saved_rows(uid, limit=10)
    return jsonify({"ok": True, "rows": rows})

# deletes all saved predictions for logged in user
@app.post("/api/member/clear")
def api_member_clear():
    uid = current_user_id()
    if uid is None:
        return jsonify({"error": "Not logged in"}), 401

    with db() as con:
        with con.cursor() as cur:
            cur.execute("DELETE FROM predictions WHERE user_id = %s", (uid,))
        con.commit()

    return jsonify({"ok": True})

# exports all saved predictions for current user as csv download
@app.get("/api/member/export")
def api_member_export():
    uid = current_user_id()
    if uid is None:
        return jsonify({"error": "Not logged in"}), 401

    with db() as con:
        with con.cursor() as cur:
            cur.execute(
                """
                SELECT ts, f0, f1, f2, f3, f4, y
                FROM predictions
                WHERE user_id = %s
                ORDER BY id DESC
                """,
                (uid,),
            )
            rows = cur.fetchall()

    header = ["timestamp"] + model.FEATURES + [model.TARGET]
    lines = [",".join(header)]

    for r in rows[::-1]:     # reverse so csv exported oldest to newest
        values = [
            r["ts"],
            str(r["f0"]),
            str(r["f1"]),
            str(r["f2"]),
            str(r["f3"]),
            str(int(r["f4"])),
            str(r["y"]),
        ]
        lines.append(",".join(values))

    csv_text = "\n".join(lines) + "\n"

    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


if __name__ == "__main__":
    app.run()
