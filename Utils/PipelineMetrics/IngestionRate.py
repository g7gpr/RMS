import time
import json
from pathlib import Path
from .db import runQuery

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

def calstarsPerDay():
    nowCount = getCalstarCount()
    nowTime = time.time()

    state = loadState()

    if state is None:
        # First run — initialise baseline
        state = {
            "initial_count": nowCount,
            "initial_time": nowTime,
            "last_count": nowCount,
            "stall_count": 0
        }
        saveState(state)
        return 0.0, False

    # --- Cumulative FPS ---
    initialCount = state["initial_count"]
    initialTime = state["initial_time"]

    deltaCalstars = nowCount - initialCount
    deltaTime_days = (nowTime - initialTime) / (24 * 3600)

    cpd = deltaCalstars / deltaTime_days if deltaTime_days > 0 else 0.0

    # --- Stall detection ---
    if nowCount == state["last_count"]:
        state["stall_count"] += 1
    else:
        state["stall_count"] = 0

    stalled = state["stall_count"] >= STALL_THRESHOLD

    # Update last_count
    state["last_count"] = nowCount
    saveState(state)

    return cpd, stalled
