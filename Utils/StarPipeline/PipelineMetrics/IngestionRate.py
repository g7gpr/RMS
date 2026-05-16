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




def ingestionStalled():
    """
    Ingestion is stalled if no jobs have been marked 'done'
    in the last 10 minutes.
    """
    from Utils.StarPipeline.PipelineMetrics.db import getConn
    conn = getConn()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*)
            FROM ingest_work
            WHERE status = 'done'
              AND updated_at >= now() - interval '10 minutes';
        """)
        recent_done = cur.fetchone()[0]

    conn.close()

    return recent_done == 0




