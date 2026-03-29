from .db import runQuery

def frameCounts():
    return runQuery("""
        SELECT s.name AS station_name,
               COUNT(f.frame_name) AS frames
        FROM public.frame f
        JOIN public.session ss ON f.session_name = ss.session_name
        JOIN public.station s ON ss.station_name = s.station_name
        GROUP BY s.name
        ORDER BY s.name;
    """)
