from .db import runQuery

def latestSessions(limit=10):
    sql = f"""
        SELECT remote_filename, updated_at, claimed_by, status
        FROM ingest_work
        WHERE claimed_by IS NOT NULL and status='done'
        ORDER BY updated_at DESC
        LIMIT {limit};
    """
    return runQuery(sql)

