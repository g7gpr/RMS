import time
import json
from pathlib import Path
from .db import runQuery

STATE_FILE = Path("/tmp/ingest_rate_state.json")

def getFrameCount():
    return runQuery("SELECT COUNT(*) FROM public.calstar_files;")[0][0]

def loadState():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return None
    return None

def saveState(initialCount, initialTime):
    STATE_FILE.write_text(json.dumps({
        "initial_count": initialCount,
        "initial_time": initialTime
    }))

def framesPerSecond():
    nowCount = getFrameCount()
    nowTime = time.time()

    state = loadState()

    if state is None:
        # First run — initialise baseline
        saveState(nowCount, nowTime)
        return 0.0

    initialCount = state["initial_count"]
    initialTime = state["initial_time"]

    deltaFrames = nowCount - initialCount
    deltaTime = nowTime - initialTime

    if deltaTime <= 0:
        return 0.0

    return deltaFrames / deltaTime
