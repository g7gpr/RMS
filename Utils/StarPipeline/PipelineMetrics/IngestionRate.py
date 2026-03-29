import time
import json
from pathlib import Path
from .db import runQuery, runQueryScalar, runQueryRows

STATE_FILE = Path("/tmp/ingest_rate_state.json")
REQUIRED_KEYS = {"initial_count", "initial_time", "last_count", "stall_count"}
STALL_THRESHOLD = 5   # number of iterations with no new frames

def getCalstarCount():
    return runQuery("SELECT COUNT(*) FROM public.calstar_files;")[0][0]

def loadState():
    if not STATE_FILE.exists():
        return None

    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        return None

    # Validate required keys
    if not REQUIRED_KEYS.issubset(state.keys()):
        return None

    return state

def saveState(state):
    STATE_FILE.write_text(json.dumps(state))

def activeStationCount(hours=48):
    sql = f"""
        WITH latest AS (
            SELECT MAX(jd_mid) AS jd_max
            FROM frame
        ),
        cutoff AS (
            SELECT jd_max - ({hours}/24.0 * 1e6) AS jd_cut
            FROM latest
        )
        SELECT COUNT(DISTINCT session.station_name)
        FROM frame
        JOIN session ON frame.session_name = session.session_name
        JOIN cutoff ON frame.jd_mid >= cutoff.jd_cut;
    """

    return runQueryScalar(sql)

def calstarsPerDay(window_days=7):
    # --- Find earliest ingestion ---
    sql_min = "SELECT MIN(ingestion_time) FROM calstar_files;"
    min_ing = runQueryScalar(sql_min) / 1e6

    if not min_ing:
        return 0.0

    # --- Compute actual DB history length ---
    now = time.time()
    history_days = (now - min_ing) / (24 * 3600)

    # --- Clamp window to available history ---
    effective_window = min(window_days, history_days)

    if effective_window <= 0:
        return 0.0

    # --- Rolling count over clamped window ---
    sql = f"""
        SELECT COUNT(*)
        FROM calstar_files
        WHERE ingestion_time >= extract(epoch FROM now()) - ({effective_window} * 24 * 3600);
    """
    count = runQueryScalar(sql)

    return count / effective_window if count else 0.0


def ingestionStalled():
    nowCount = getCalstarCount()
    nowTime = time.time()

    state = loadState()

    if state is None:
        state = {
            "last_count": nowCount,
            "stall_count": 0
        }
        saveState(state)
        return False

    # Stall detection
    if nowCount == state["last_count"]:
        state["stall_count"] += 1
    else:
        state["stall_count"] = 0

    stalled = state["stall_count"] >= STALL_THRESHOLD

    state["last_count"] = nowCount
    saveState(state)

    return stalled



