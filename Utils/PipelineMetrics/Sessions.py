from .db import runQuery

def latestSessions(limit=10):
    return runQuery(f"""
        SELECT session_id, session_name, station_id
        FROM public.session
        ORDER BY session_id DESC
        LIMIT {limit};
    """)
