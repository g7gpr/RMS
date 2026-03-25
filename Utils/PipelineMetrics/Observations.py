from .db import runQuery

def observationCounts(limit=5):
    return runQuery(f"""
        SELECT ss.session_id,
               COUNT(*) AS observations
        FROM public.observation o
        JOIN public.frame f ON o.frame_name = f.frame_name
        JOIN public.session ss ON f.session_name = ss.session_name
        GROUP BY ss.session_id
        ORDER BY ss.session_id DESC
        LIMIT {limit};""")

def totalObservations():
    return runQuery("SELECT COUNT(*) FROM public.observation;")[0][0]
