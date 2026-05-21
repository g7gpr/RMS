from .db import runQuery

def latestSessions(limit=10):
    sql = f"""
        SELECT remote_filename, updated_at
        FROM ingest_work
        ORDER BY updated_at DESC
        LIMIT {limit};
    """
    return runQuery(sql)

