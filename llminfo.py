#!/usr/bin/env python3
"""llminfo - OpenRouter model price/feature board, grouped and compared.

Tracks pricing (and feature snapshots) of selected OpenRouter models, grouped
into user-defined groups, with per-endpoint prices. The displayed price is
the cheapest endpoint across all vendors; each endpoint is also badged with
its vendor's training status from the provider policy table (informational
only — it does not steer selection).

Data sources (all public, no API key needed):
  GET https://openrouter.ai/api/v1/models
  GET https://openrouter.ai/api/v1/models/<id>/endpoints

Refresh policy:
  - description + feature snapshot: mirrored once when a model is added
    (re-mirror button per model), NOT on every refresh.
  - pricing + endpoint list: background thread every REFRESH_HOURS,
    a global "refresh now", a per-model button, and a lazy refresh on page
    view when data is older than STALE_AFTER seconds.
  - every pricing refresh appends one row per model to price_history, so the
    compare view can chart movement over time.

Run:  ./run.sh  ->  http://127.0.0.1:[REDACTED:loopback-port]/  (nginx: https://llminfo.inbox.li/)
Auth: none of its own. nginx puts it behind the SSO gate; see
      [REDACTED:server-path]
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time

import requests
from flask import (Flask, g, jsonify, redirect, render_template, request,
                   url_for)
from markupsafe import Markup
from werkzeug.middleware.proxy_fix import ProxyFix

# ----------------------------------------------------------------- config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=None):
    """Minimal .env loader so settings work however the app is launched."""
    path = path or os.path.join(BASE_DIR, ".env")
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

HOST = os.environ.get("LLMINFO_HOST", "127.0.0.1")
PORT = int(os.environ.get("LLMINFO_PORT", "[REDACTED:loopback-port]"))
DB_PATH = os.environ.get("LLMINFO_DB", os.path.join(BASE_DIR, "llminfo.db"))
REFRESH_HOURS = float(os.environ.get("LLMINFO_REFRESH_HOURS", "6"))
STALE_AFTER = int(os.environ.get("LLMINFO_STALE_AFTER", "3600"))

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_EP_URL = "https://openrouter.ai/api/v1/models/{mid}/endpoints"
# EU in-region routing (OpenRouter Business): same shape as the global
# catalogue, but only the 69-ish models servable from EU/EEA+UK+CH
# endpoints. App code talks to eu.openrouter.ai only through this constant.
OPENROUTER_EU_MODELS_URL = "https://eu.openrouter.ai/api/v1/models"
EU_BASE_URL = "https://eu.openrouter.ai/api/v1"
HTTP_TIMEOUT = 25
# Largest upstream body we will buffer (OpenRouter catalogue payloads are
# far smaller; anything bigger is a misbehaving response, not data).
HTTP_MAX_BODY = 5 * 1024 * 1024
HEADERS = {
    "User-Agent": "llminfo/1.0 (+https://llminfo.inbox.li)",
    "HTTP-Referer": "https://llminfo.inbox.li/",
    "X-Title": "llminfo",
}

# ----------------------------------------------------------------- hardening ---
# Second layer behind the nginx SSO gate (see llminfo.inbox.li.conf): nginx
# injects X-Llminfo-Gate with this same value on every proxied request, and
# /admin/* refuses anything without it. That keeps a gate misconfiguration
# or a direct 127.0.0.1:[REDACTED:loopback-port] hit from silently opening full write access.
# Fail closed: with the env var unset, every /admin/* answers 403 until the
# operator sets it (same value in both places) and restarts.
GATE_SECRET = os.environ.get("LLMINFO_GATE_SECRET", "")

# OpenRouter model ids are vendor/model with an optional :variant suffix.
# Anything else (/, ?, #, %, ..) is rejected before it reaches a URL.
OPENROUTER_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+"
                              r"(?::[A-Za-z0-9_.\-]+)?$")
OID_MAX = 160
NAME_MAX = 80
LABEL_MAX = 120
NOTE_MAX = 500
KEY_MAX = 80
MAX_GROUPS = 64
MAX_MODELS_PER_GROUP = 200

# Blended $/1M mix used for ranking: 75% input / 25% output tokens.
BLEND_IN_W = 0.75
BLEND_OUT_W = 0.25

# --- training policy defaults ------------------------------------------------
# provider slug fragment -> (trains?, note). Badge-only: the displayed price
# is always the cheapest endpoint; the policy only decides which badge
# (no-train / trains / unknown) the chosen endpoint's vendor carries.
# These are public, documented provider commitments for API (not consumer
# chat) traffic; anything not listed is "unknown". The table is editable in
# the UI (provider overrides) and stored in the DB.
DEFAULT_PROVIDER_POLICY = {
    "openai": ("no", "OpenAI API: data not used for training by default."),
    "mistral": ("no", "Mistral La Plateforme API: no training on API data."),
    "google/vertex": ("no", "Google Vertex AI: no training on customer data."),
    # "Google" (bare) is the Vertex route; "Google AI Studio" is the
    # free-tier route whose terms allow improvement use -> unknown.
    "google ai studio": ("unknown", "AI Studio free-tier terms allow "
                         "improvement use; treat as may-train."),
    "xai": ("unknown", "No clear no-training commitment found."),
    "moonshotai": ("unknown", "No clear no-training commitment found."),
    "deepseek": ("unknown", "DeepSeek platform terms are ambiguous on "
                 "training; treat as may-train."),
    "alibaba": ("unknown", "No clear no-training commitment found."),
    "qwen": ("unknown", "No clear no-training commitment found."),
    "xiaomi": ("unknown", "No clear no-training commitment found."),
    "tencent": ("unknown", "No clear no-training commitment found."),
    "meta": ("unknown", "Meta API terms ambiguous; treat as may-train."),
}

log = logging.getLogger("llminfo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)

_refresh_lock = threading.Lock()
_refresh_running = False

# --------------------------------------------------------------------- db ---
SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  position INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  group_id INTEGER REFERENCES groups(id) ON DELETE CASCADE,
  openrouter_id TEXT NOT NULL UNIQUE,
  label TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  pref_provider TEXT NOT NULL DEFAULT '',
  position INTEGER NOT NULL DEFAULT 0,
  description TEXT NOT NULL DEFAULT '',
  features_json TEXT NOT NULL DEFAULT '{}',
  mirrored_at INTEGER NOT NULL DEFAULT 0,
  added_at INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS endpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_row_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
  provider_name TEXT NOT NULL DEFAULT '',
  tag TEXT NOT NULL DEFAULT '',
  quantization TEXT NOT NULL DEFAULT '',
  context_length INTEGER NOT NULL DEFAULT 0,
  pricing_json TEXT NOT NULL DEFAULT '{}',
  fetched_at INTEGER NOT NULL DEFAULT 0,
  UNIQUE(model_row_id, provider_name, tag)
);
CREATE TABLE IF NOT EXISTS price_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_row_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
  ts INTEGER NOT NULL,
  prompt REAL NOT NULL DEFAULT 0,
  completion REAL NOT NULL DEFAULT 0,
  provider_name TEXT NOT NULL DEFAULT '',
  trains TEXT NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_hist_model_ts ON price_history(model_row_id, ts);
CREATE TABLE IF NOT EXISTS provider_policy (
  provider_key TEXT PRIMARY KEY,
  trains TEXT NOT NULL DEFAULT 'unknown',
  note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS eu_models (
  openrouter_id TEXT PRIMARY KEY,
  name TEXT NOT NULL DEFAULT '',
  pricing_json TEXT NOT NULL DEFAULT '{}',
  context_length INTEGER NOT NULL DEFAULT 0,
  description TEXT NOT NULL DEFAULT '',
  gprompt REAL NOT NULL DEFAULT 0,
  gcompletion REAL NOT NULL DEFAULT 0,
  fetched_at INTEGER NOT NULL DEFAULT 0
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    for key, (trains, note) in DEFAULT_PROVIDER_POLICY.items():
        db.execute(
            "INSERT OR IGNORE INTO provider_policy(provider_key, trains, note)"
            " VALUES (?,?,?)", (key, trains, note))
    db.commit()
    db.close()


def seed_if_empty():
    """The three requested groups with their initial model lists."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    if db.execute("SELECT COUNT(*) c FROM groups").fetchone()["c"]:
        db.close()
        return
    groups = [
        ("Frontier / Subsidised", [
            ("meta/muse-spark-1.3-contributor", "Muse Spark 1.3 Contributor", "", ""),
            ("mistralai/mistral-large-4-0", "Mistral Large 4", "Mistral EU route"),
            ("x-ai/grok-4.7", "Grok 4.7", "", ""),
            ("moonshotai/kimi-k3.1-pro", "Kimi K3.1 Pro", "", ""),
            ("google/gemini-4-argon", "Gemini 4 Argon", "", ""),
            ("openai/gpt-6.1-terra", "GPT 6.1 Terra", "", ""),
        ]),
        ("Cheaper", [
            ("deepseek/deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash 1017", "", ""),
            ("google/gemini-3-flash-preview", "Gemini 3", "", ""),
            ("meta/muse-spark-open", "Muse Spark Open", "", ""),
            ("google/gemini-3.8-flash", "Gemini 3.8 Flash", "", ""),
            ("qwen/qwen3.8-27b", "Qwen 3.8 27B", "", ""),
            ("xiaomi/mimo-v2.6-pro", "MiMo V2.6 Pro", "", ""),
        ]),
        ("Bulk / Translate", [
            ("google/gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "", ""),
            ("qwen/qwen3.8-flash", "Qwen 3.8 Flash", "", ""),
            ("xiaomi/mimo-v2.6-flash", "MiMo V2.6 Flash", "", ""),
            ("tencent/hy3", "Hunyuan HY3", "", ""),
        ]),
    ]
    now = int(time.time())
    for gi, (gname, models) in enumerate(groups):
        gid = db.execute(
            "INSERT INTO groups(name, position) VALUES (?,?)",
            (gname, gi)).lastrowid
        for mi, (oid, label, note) in enumerate(models):
            db.execute(
                "INSERT OR IGNORE INTO models(group_id, openrouter_id, label,"
                " note, position, added_at) VALUES (?,?,?,?,?,?)",
                (gid, oid, label, note, mi, now))
    db.commit()
    db.close()
    log.info("seeded default groups/models")


# ------------------------------------------------------- openrouter fetch ---
def or_get(url):
    # No redirects: the host is pinned to openrouter.ai by construction, and
    # a redirect is the only way a caller-influenced path could leave it.
    # Stream with a hard cap so a misbehaving response cannot balloon memory.
    with requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT,
                      allow_redirects=False, stream=True) as r:
        if r.is_redirect:
            raise ValueError(f"unexpected redirect for {url}")
        r.raise_for_status()
        chunks, total = [], 0
        for chunk in r.iter_content(65536):
            total += len(chunk)
            if total > HTTP_MAX_BODY:
                raise ValueError(f"response too large for {url}")
            chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def valid_openrouter_id(oid: str) -> bool:
    return bool(oid) and len(oid) <= OID_MAX and bool(OPENROUTER_ID_RE.match(oid))


def parse_group_id(raw) -> int:
    """Form group id or raise ValueError (400 at the caller, never a 500)."""
    try:
        return int((raw or "0") or 0)
    except (TypeError, ValueError):
        raise ValueError("bad group_id")


@app.before_request
def _admin_gate():
    """Second auth layer + CSRF brake for the mutating admin surface.

    Gate: nginx injects the shared secret on the gated location; anything
    without it (direct port hit, gate removed) gets 403. CSRF: plain HTML
    forms carry no custom headers, so cross-site form POSTs cannot smuggle
    the gate header — and as a second net, POSTs must carry this host as
    Origin/Referer (requests with neither, e.g. curl, pass through like
    before). Same pattern as the abuse-llm operator console.
    """
    if not request.path.startswith("/admin/"):
        return None
    import hmac
    if not GATE_SECRET or not hmac.compare_digest(
            request.headers.get("X-Llminfo-Gate", ""), GATE_SECRET):
        return "forbidden", 403
    if request.method == "POST":
        host = (request.host or "").split(":")[0].lower()
        for hdr in ("Origin", "Referer"):
            val = request.headers.get(hdr, "")
            if not val:
                continue
            try:
                from urllib.parse import urlparse
                oh = (urlparse(val).hostname or "").lower()
            except Exception:
                return "bad origin", 403
            if oh != host:
                return "bad origin", 403
    return None


def fetch_models_index():
    data = or_get(OPENROUTER_MODELS_URL)
    return {m["id"]: m for m in data.get("data", [])}


def fetch_endpoints(mid):
    data = or_get(OPENROUTER_EP_URL.format(mid=mid))
    d = data.get("data", data)
    return d.get("endpoints", [])


def refresh_eu_snapshot(db):
    """Mirror the EU in-region catalogue (one API call, no per-model crawl).

    Stores each EU model's top pricing plus the matching global top pricing
    so the EU tab can show the in-region premium/discount without live
    fetches at view time. Raises on fetch failure; caller decides whether
    to keep serving the previous snapshot.
    """
    eu_data = or_get(OPENROUTER_EU_MODELS_URL)
    try:
        global_index = fetch_models_index()
    except Exception as e:
        log.warning("EU refresh: global index fetch failed (%s); "
                    "deltas will be blank", e)
        global_index = {}
    now = int(time.time())
    count = 0
    for m in eu_data.get("data", []):
        mid = m.get("id", "")
        if not valid_openrouter_id(mid):
            continue
        g = global_index.get(mid) or {}
        gp = (g.get("pricing") or {})
        db.execute(
            "INSERT OR REPLACE INTO eu_models(openrouter_id, name,"
            " pricing_json, context_length, description, gprompt,"
            " gcompletion, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
            (mid, m.get("name", "") or mid,
             json.dumps(m.get("pricing", {}) or {}),
             m.get("context_length", 0) or 0,
             m.get("description", "") or "",
             float(gp.get("prompt") or 0),
             float(gp.get("completion") or 0), now))
        count += 1
    db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('last_refresh_eu',?)",
               (str(now),))
    db.commit()
    log.info("EU snapshot: %d models", count)
    return count


def last_eu_refresh_ts(db):
    r = db.execute("SELECT v FROM meta WHERE k='last_refresh_eu'").fetchone()
    try:
        return int(r["v"]) if r else 0
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------- policy/prices
def load_policy(db):
    return {r["provider_key"]: (r["trains"], r["note"])
            for r in db.execute("SELECT * FROM provider_policy")}


def trains_for(provider_name, policy):
    """'no' only on an explicit no-training commitment, else fallback."""
    pname = (provider_name or "").lower()
    for key, (trains, _note) in policy.items():
        if key in pname:
            return trains
    return "unknown"


def blended(prompt, completion):
    try:
        return BLEND_IN_W * float(prompt or 0) + BLEND_OUT_W * float(completion or 0)
    except (TypeError, ValueError):
        return 0.0


def per_m(x):
    """$/token -> $/1M tokens."""
    try:
        return float(x or 0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def endpoint_sort_key(ep):
    p = ep.get("pricing", {}) or {}
    return blended(p.get("prompt"), p.get("completion"))


def choose_display(endpoints, policy):
    """Pick the displayed endpoint: plain cheapest, no policy steering.

    The training policy is informational only — the chosen endpoint's
    vendor status is returned alongside so the UI can badge it.
    """
    if not endpoints:
        return None, "unknown"
    ep = min(endpoints, key=endpoint_sort_key)
    return ep, trains_for(ep.get("provider_name", ""), policy)


def mirror_model(db, row, index=None):
    """One-time description/feature snapshot (also called by re-mirror)."""
    mid = row["openrouter_id"]
    index = index if index is not None else fetch_models_index()
    m = index.get(mid)
    if not m:
        # tolerate :variant suffixes (e.g. model:batch) by base-id match
        base = mid.split(":")[0]
        for _id, cand in index.items():
            if _id == base or _id.split(":")[0] == base \
                    or (cand.get("canonical_slug", "") or "").split(":")[0] == base:
                m = cand
                break
    if not m:
        raise ValueError(f"model {mid!r} not found on OpenRouter")
    arch = m.get("architecture", {}) or {}
    features = {
        "name": m.get("name", ""),
        "context_length": m.get("context_length"),
        "modality": arch.get("modality", ""),
        "input_modalities": arch.get("input_modalities", []),
        "output_modalities": arch.get("output_modalities", []),
        "tokenizer": arch.get("tokenizer", ""),
        "supported_parameters": m.get("supported_parameters", []),
        "reasoning": m.get("reasoning", {}),
        "per_request_limits": m.get("per_request_limits"),
        "top_pricing": m.get("pricing", {}),
    }
    now = int(time.time())
    db.execute("UPDATE models SET description=?, features_json=?,"
               " mirrored_at=? WHERE id=?",
               (m.get("description", "") or "", json.dumps(features), now,
                row["id"]))
    db.commit()
    return features


def refresh_model_endpoints(db, row, policy):
    """Re-fetch endpoints + record chosen-price history. No description touch."""
    mid = row["openrouter_id"]
    eps = fetch_endpoints(mid)
    now = int(time.time())
    db.execute("DELETE FROM endpoints WHERE model_row_id=?", (row["id"],))
    for ep in eps:
        db.execute(
            "INSERT INTO endpoints(model_row_id, provider_name, tag,"
            " quantization, context_length, pricing_json, fetched_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (row["id"], ep.get("provider_name", "") or "",
             ep.get("tag", "") or "", ep.get("quantization", "") or "",
             ep.get("context_length", 0) or 0, json.dumps(ep.get("pricing", {}) or {}),
             now))
    chosen, trains = choose_display(eps, policy)
    if chosen is not None:
        p = chosen.get("pricing", {}) or {}
        db.execute(
            "INSERT INTO price_history(model_row_id, ts, prompt, completion,"
            " provider_name, trains) VALUES (?,?,?,?,?,?)",
            (row["id"], now, float(p.get("prompt") or 0),
             float(p.get("completion") or 0),
             chosen.get("provider_name", "") or "", trains))
        db.execute(
            "DELETE FROM price_history WHERE model_row_id=? AND id NOT IN"
            " (SELECT id FROM price_history WHERE model_row_id=? "
            "  ORDER BY ts DESC LIMIT 200)",
            (row["id"], row["id"]))
    db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('last_refresh',?)",
               (str(now),))
    db.commit()
    return chosen, trains, len(eps)


def refresh_all(blocking=False):
    """Refresh every model. Returns immediately unless blocking=True."""
    global _refresh_running
    if blocking:
        return _do_refresh_all()
    with _refresh_lock:
        if _refresh_running:
            return "already running"
        _refresh_running = True
    t = threading.Thread(target=_do_refresh_all, daemon=True)
    t.start()
    return "started"


def _do_refresh_all():
    global _refresh_running
    try:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        policy = load_policy(db)
        rows = db.execute("SELECT * FROM models ORDER BY id").fetchall()
        ok, fail = [], []
        for row in rows:
            try:
                refresh_model_endpoints(db, row, policy)
                ok.append(row["openrouter_id"])
            except Exception as e:  # keep going; one bad model must not stop the rest
                log.warning("refresh %s failed: %s", row["openrouter_id"], e)
                fail.append(row["openrouter_id"])
            time.sleep(0.3)  # be polite to the public API
        try:
            n_eu = refresh_eu_snapshot(db)
        except Exception as e:
            log.warning("EU snapshot refresh failed: %s", e)
            n_eu = -1
        db.close()
        log.info("refresh all: %d ok, %d failed (%s), EU snapshot %s",
                 len(ok), len(fail), fail, n_eu)
        return f"{len(ok)} ok, {len(fail)} failed"
    finally:
        with _refresh_lock:
            _refresh_running = False


def refresh_eu(blocking=False):
    """Refresh just the EU catalogue snapshot (single fast API call)."""
    if blocking:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        try:
            return refresh_eu_snapshot(db)
        finally:
            db.close()
    with _refresh_lock:
        if _refresh_running:
            return "already running"
    t = threading.Thread(target=_refresh_eu_bg, daemon=True)
    t.start()
    return "started"


def _refresh_eu_bg():
    global _refresh_running
    with _refresh_lock:
        _refresh_running = True
    try:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        try:
            refresh_eu_snapshot(db)
        except Exception as e:
            log.warning("EU snapshot refresh failed: %s", e)
        db.close()
    finally:
        with _refresh_lock:
            _refresh_running = False


def last_refresh_ts(db):
    r = db.execute("SELECT v FROM meta WHERE k='last_refresh'").fetchone()
    try:
        return int(r["v"]) if r else 0
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------------ views ---
@app.template_global()
def fmt_ts(ts):
    try:
        ts = int(ts or 0)
    except (TypeError, ValueError):
        return "never"
    if not ts:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


@app.template_global()
def fmt_money(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v == 0:
        return "$0"
    if v >= 10:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:,.3f}"
    return f"${v:,.4f}"


@app.template_global()
def spark(history, w=120, h=28):
    """Inline SVG sparkline of blended $/1M over the stored history."""
    pts = [blended(p["prompt"], p["completion"]) * 1_000_000 for p in history]
    pts = [p for p in pts if p > 0]
    if len(pts) < 2:
        return Markup("")
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    step = w / (len(pts) - 1)
    coords = " ".join(
        f"{i * step:.1f},{h - 2 - (v - lo) / span * (h - 4):.1f}"
        for i, v in enumerate(pts))
    return Markup(
        f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}"'
        f' aria-hidden="true"><polyline points="{coords}"/></svg>')
def model_view(row, db, policy):
    eps = [dict(
        provider_name=e["provider_name"], tag=e["tag"],
        quantization=e["quantization"], context_length=e["context_length"],
        pricing=json.loads(e["pricing_json"] or "{}"),
        fetched_at=e["fetched_at"],
        trains=trains_for(e["provider_name"], policy),
    ) for e in db.execute(
        "SELECT * FROM endpoints WHERE model_row_id=?", (row["id"],))]
    eps.sort(key=endpoint_sort_key)
    chosen, trains = choose_display(
        [{"provider_name": e["provider_name"], "pricing": e["pricing"]}
         for e in eps], policy)
    # map back to the full row for display
    chosen_full = None
    if chosen is not None:
        for e in eps:
            if (e["provider_name"] == chosen["provider_name"]
                    and e["pricing"] == chosen["pricing"]):
                chosen_full = e
                break
    hist = [dict(ts=h["ts"], prompt=h["prompt"], completion=h["completion"],
                 provider_name=h["provider_name"], trains=h["trains"])
            for h in db.execute(
                "SELECT * FROM price_history WHERE model_row_id=?"
                " ORDER BY ts DESC LIMIT 30", (row["id"],))]
    hist.reverse()
    feats = json.loads(row["features_json"] or "{}")
    return dict(row=row, endpoints=eps, chosen=chosen_full, trains=trains,
                history=hist, features=feats)


@app.route("/")
def index():
    db = get_db()
    tab = (request.args.get("tab") or "board").strip().lower()
    if tab not in ("board", "eu"):
        tab = "board"
    policy = load_policy(db)
    # lazy refresh on visit when stale
    if (time.time() - last_refresh_ts(db) > STALE_AFTER
            and not _refresh_running):
        refresh_all()
        refreshing = True
    else:
        refreshing = _refresh_running
    groups = []
    for gr in db.execute("SELECT * FROM groups ORDER BY position, id"):
        mrows = db.execute(
            "SELECT * FROM models WHERE group_id=? ORDER BY position, id",
            (gr["id"],)).fetchall()
        models = [model_view(r, db, policy) for r in mrows]
        # compare ranking within group by chosen blended price
        ranked = sorted(
            [m for m in models if m["chosen"]],
            key=lambda m: endpoint_sort_key({"pricing": m["chosen"]["pricing"]}))
        max_blend = (endpoint_sort_key({"pricing": ranked[-1]["chosen"]["pricing"]})
                     if ranked else 0)
        groups.append(dict(group=gr, models=models, ranked=ranked,
                           max_blend=max_blend))
    providers = sorted({e["provider_name"]
                        for grp in groups for m in grp["models"]
                        for e in m["endpoints"] if e["provider_name"]})
    tracked = {r["openrouter_id"] for grp in groups for m in grp["models"]
               for r in [m["row"]]}
    tracked_base = {t.split(":")[0] for t in tracked}

    eu_models, eu_stale = [], False
    eu_count = db.execute("SELECT COUNT(*) c FROM eu_models").fetchone()["c"]
    if tab == "eu" or not eu_count:
        if not eu_count or (time.time() - last_eu_refresh_ts(db) > STALE_AFTER
                            and not _refresh_running):
            # first visit with an empty snapshot blocks so the tab is not
            # blank; later stales refresh in the background instead
            if not eu_count:
                try:
                    refresh_eu_snapshot(db)
                    eu_count = db.execute(
                        "SELECT COUNT(*) c FROM eu_models").fetchone()["c"]
                except Exception as e:
                    log.warning("EU snapshot on-demand fetch failed: %s", e)
            else:
                refresh_eu()
                refreshing = True
        sort = (request.args.get("sort") or "price").strip().lower()
        if sort not in ("price", "name", "ctx"):
            sort = "price"
        for r in db.execute("SELECT * FROM eu_models"):
            p = json.loads(r["pricing_json"] or "{}")
            b = blended(p.get("prompt"), p.get("completion")) * 1_000_000
            gb = blended(r["gprompt"], r["gcompletion"]) * 1_000_000
            delta = ((b - gb) / gb * 100) if gb > 0 else None
            base = (r["openrouter_id"] or "").split(":")[0]
            eu_models.append(dict(
                row=r, pricing=p, blend=b, global_blend=gb, delta=delta,
                tracked=(r["openrouter_id"] in tracked
                         or base in tracked_base)))
        if sort == "name":
            eu_models.sort(key=lambda m: (m["row"]["name"] or "").lower())
        elif sort == "ctx":
            eu_models.sort(key=lambda m: -(m["row"]["context_length"] or 0))
        else:
            eu_models.sort(key=lambda m: m["blend"])
        eu_stale = (time.time() - last_eu_refresh_ts(db) > STALE_AFTER)
    return render_template(
        "index.html", groups=groups, policy=policy,
        providers=providers, refreshing=refreshing,
        last_refresh=last_refresh_ts(db), now=int(time.time()),
        per_m=per_m, blended=blended,
        tab=tab, eu_models=eu_models, eu_count=eu_count,
        eu_sort=(request.args.get("sort") or "price"),
        eu_updated=last_eu_refresh_ts(db), eu_stale=eu_stale,
        eu_base_url=EU_BASE_URL)


@app.route("/api/state")
def api_state():
    db = get_db()
    policy = load_policy(db)
    out = []
    for gr in db.execute("SELECT * FROM groups ORDER BY position, id"):
        gmodels = []
        for r in db.execute(
                "SELECT * FROM models WHERE group_id=? ORDER BY position, id",
                (gr["id"],)):
            mv = model_view(r, db, policy)
            gmodels.append({
                "openrouter_id": r["openrouter_id"], "label": r["label"],
                "note": r["note"], "chosen": mv["chosen"], "trains": mv["trains"],
                "endpoints": mv["endpoints"],
                "features": mv["features"],
                "description": r["description"],
                "mirrored_at": r["mirrored_at"],
            })
        out.append({"name": gr["name"], "models": gmodels})
    return jsonify({"groups": out, "last_refresh": last_refresh_ts(db)})


# --------------------------------------------------------------- admin -------
@app.route("/admin/refresh", methods=["POST"])
def admin_refresh():
    refresh_all()
    return redirect(url_for("index"))


@app.route("/admin/eu/refresh", methods=["POST"])
def admin_eu_refresh():
    try:
        db = get_db()
        refresh_eu_snapshot(db)
    except Exception as e:
        log.warning("manual EU refresh failed: %s", e)
        return f"EU refresh failed: {e}", 502
    return redirect(url_for("index", tab="eu"))


@app.route("/admin/model/<int:mid>/refresh", methods=["POST"])
def admin_model_refresh(mid):
    db = get_db()
    row = db.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if not row:
        return "no such model", 404
    try:
        refresh_model_endpoints(db, row, load_policy(db))
    except Exception as e:
        log.warning("manual refresh %s failed: %s", row["openrouter_id"], e)
        return f"refresh failed: {e}", 502
    return redirect(url_for("index"))


@app.route("/admin/model/<int:mid>/remirror", methods=["POST"])
def admin_model_remirror(mid):
    db = get_db()
    row = db.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if not row:
        return "no such model", 404
    try:
        mirror_model(db, row)
        refresh_model_endpoints(db, db.execute(
            "SELECT * FROM models WHERE id=?", (mid,)).fetchone(),
            load_policy(db))
    except Exception as e:
        log.warning("remirror %s failed: %s", row["openrouter_id"], e)
        return f"re-mirror failed: {e}", 502
    return redirect(url_for("index"))


@app.route("/admin/model/add", methods=["POST"])
def admin_model_add():
    db = get_db()
    oid = (request.form.get("openrouter_id") or "").strip()
    if not valid_openrouter_id(oid):
        return "bad openrouter_id", 400
    try:
        gid = parse_group_id(request.form.get("group_id"))
    except ValueError:
        return "bad group_id", 400
    label = (request.form.get("label") or "").strip()[:LABEL_MAX]
    if not gid:
        return "openrouter_id and group_id required", 400
    if not db.execute("SELECT id FROM groups WHERE id=?", (gid,)).fetchone():
        return "no such group", 404
    now = int(time.time())
    pos = db.execute("SELECT COUNT(*) c FROM models WHERE group_id=?",
                     (gid,)).fetchone()["c"]
    if pos >= MAX_MODELS_PER_GROUP:
        return "group is full", 400
    try:
        cur = db.execute(
            "INSERT INTO models(group_id, openrouter_id, label, position,"
            " added_at) VALUES (?,?,?,?,?)", (gid, oid, label or oid, pos, now))
        db.commit()
        row = db.execute("SELECT * FROM models WHERE id=?",
                         (cur.lastrowid,)).fetchone()
        # mirror description + endpoints exactly once, at add time
        mirror_model(db, row)
        refresh_model_endpoints(db, db.execute(
            "SELECT * FROM models WHERE id=?", (cur.lastrowid,)).fetchone(),
            load_policy(db))
    except Exception as e:
        db.rollback()
        log.warning("add %s failed: %s", oid, e)
        return f"add failed: {e}", 502
    return redirect(url_for("index"))


@app.route("/admin/model/<int:mid>/delete", methods=["POST"])
def admin_model_delete(mid):
    db = get_db()
    db.execute("DELETE FROM models WHERE id=?", (mid,))
    db.commit()
    return redirect(url_for("index"))


@app.route("/admin/model/<int:mid>/move", methods=["POST"])
def admin_model_move(mid):
    db = get_db()
    row = db.execute("SELECT * FROM models WHERE id=?", (mid,)).fetchone()
    if not row:
        return "no such model", 404
    try:
        gid = parse_group_id(request.form.get("group_id"))
    except ValueError:
        return "bad group_id", 400
    direction = request.form.get("direction", "")
    if gid and gid != row["group_id"]:
        if not db.execute("SELECT id FROM groups WHERE id=?",
                          (gid,)).fetchone():
            return "no such group", 404
        pos = db.execute("SELECT COUNT(*) c FROM models WHERE group_id=?",
                         (gid,)).fetchone()["c"]
        db.execute("UPDATE models SET group_id=?, position=? WHERE id=?",
                   (gid, pos, mid))
    elif direction in ("up", "down"):
        sibs = db.execute("SELECT id FROM models WHERE group_id=?"
                          " ORDER BY position, id",
                          (row["group_id"],)).fetchall()
        ids = [s["id"] for s in sibs]
        i = ids.index(mid)
        j = i - 1 if direction == "up" else i + 1
        if 0 <= j < len(ids):
            ids[i], ids[j] = ids[j], ids[i]
            for p, _id in enumerate(ids):
                db.execute("UPDATE models SET position=? WHERE id=?", (p, _id))
    db.commit()
    return redirect(url_for("index"))


@app.route("/admin/model/<int:mid>/edit", methods=["POST"])
def admin_model_edit(mid):
    db = get_db()
    db.execute("UPDATE models SET label=?, note=? WHERE id=?",
               ((request.form.get("label") or "").strip()[:LABEL_MAX],
                (request.form.get("note") or "").strip()[:NOTE_MAX],
                mid))
    db.commit()
    return redirect(url_for("index"))


@app.route("/admin/group/add", methods=["POST"])
def admin_group_add():
    db = get_db()
    name = (request.form.get("name") or "").strip()[:NAME_MAX]
    if not name:
        return "name required", 400
    if db.execute("SELECT COUNT(*) c FROM groups").fetchone()["c"] >= MAX_GROUPS:
        return "too many groups", 400
    pos = db.execute("SELECT COUNT(*) c FROM groups").fetchone()["c"]
    db.execute("INSERT INTO groups(name, position) VALUES (?,?)", (name, pos))
    db.commit()
    return redirect(url_for("index"))


@app.route("/admin/group/<int:gid>/rename", methods=["POST"])
def admin_group_rename(gid):
    db = get_db()
    name = (request.form.get("name") or "").strip()[:NAME_MAX]
    if name:
        db.execute("UPDATE groups SET name=? WHERE id=?", (name, gid))
        db.commit()
    return redirect(url_for("index"))


@app.route("/admin/group/<int:gid>/delete", methods=["POST"])
def admin_group_delete(gid):
    db = get_db()
    db.execute("DELETE FROM models WHERE group_id=?", (gid,))
    db.execute("DELETE FROM groups WHERE id=?", (gid,))
    db.commit()
    return redirect(url_for("index"))


@app.route("/admin/policy", methods=["POST"])
def admin_policy():
    db = get_db()
    key = (request.form.get("provider_key") or "").strip().lower()[:KEY_MAX]
    trains = (request.form.get("trains") or "unknown").strip()
    note = (request.form.get("note") or "").strip()[:NOTE_MAX]
    if not key or trains not in ("no", "unknown", "yes"):
        return "bad policy", 400
    db.execute("INSERT OR REPLACE INTO provider_policy(provider_key, trains,"
               " note) VALUES (?,?,?)", (key, trains, note))
    db.commit()
    return redirect(url_for("index"))


def main():
    init_db()
    with app.app_context():
        seed_if_empty()

    def loop():
        while True:
            time.sleep(REFRESH_HOURS * 3600)
            _do_refresh_all()
    threading.Thread(target=loop, daemon=True).start()

    with app.app_context():
        db = get_db()
        have = db.execute("SELECT COUNT(*) c FROM endpoints").fetchone()["c"]
    if not have:
        # first boot: fill pricing in background so startup stays fast
        refresh_all()
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()
