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

def saveState(count, timestamp):
    STATE_FILE.write_text(json.dumps({
        "count": count,
        "timestamp": timestamp
    }))

def framesPerSecond():
    nowCount = getFrameCount()
    nowTime = time.time()

    state = loadState()
    if state is None:
        saveState(nowCount, nowTime)
        return 0.0

    prevCount = state["count"]
    prevTime = state["timestamp"]

    deltaFrames = nowCount - prevCount
    deltaTime = nowTime - prevTime

    fps = deltaFrames / deltaTime if deltaTime > 0 else 0.0

    saveState(nowCount, nowTime)
    return fps
