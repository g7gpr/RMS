import psycopg

def getConn():
    return psycopg.connect(
        dbname="star_data",
        user="ingest_user",
        host="192.168.1.190"
    )

def runQuery(sql):
    conn = getConn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()
