import psycopg

def getConn():
    return psycopg.connect(dbname="star_data", user="ingest_user", host="192.168.1.212")

def runQuery(sql):
    conn = getConn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()

def runQueryRows(sql, params=None):
    conn = getConn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()   # always list of tuples
    finally:
        conn.close()


def runQueryScalar(sql, params=None):
    conn = getConn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None   # always scalar or None
    finally:
        conn.close()


def runCommand(sql, params=None):
    conn = getConn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()
