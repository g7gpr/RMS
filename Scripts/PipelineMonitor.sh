#!/bin/bash

# --- Defaults ---
DB="star_data"
USER="ingest_user"
HOST="localhost"   # override with --host

# --- Parse parameters ---
while [[ "$1" =~ ^-- ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="$1"

# --- Color helpers ---
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
RESET=$(tput sgr0)

# --- Run SQL ---
run() {
    psql -h "$HOST" -U "$USER" -d "$DB" -t -c "$1"
}

# --- Latest sessions ---
latest_sessions() {
    echo "${BLUE}=== Latest Sessions ===${RESET}"
    run "
        SELECT session_id, session_name, station_id
        FROM session
        ORDER BY session_id DESC
        LIMIT 10;
    "
}

# --- Frame counts ---
frame_counts() {
    echo "${BLUE}=== Frame Counts ===${RESET}"
    run "
        SELECT s.name AS station_name,
               COUNT(f.frame_name) AS frames
        FROM frame f
        JOIN session ss ON f.session_name = ss.session_name
        JOIN station s ON ss.station_id = s.station_id
        GROUP BY s.name
        ORDER BY s.name;
    "
}

# --- Observation counts ---
obs_counts() {
    echo "${BLUE}=== Observation Counts ===${RESET}"
    run "
SELECT ss.session_id,
       COUNT(*) AS observations
FROM observation o
JOIN frame f ON o.frame_name = f.frame_name
JOIN session ss ON f.session_name = ss.session_name
GROUP BY ss.session_id
ORDER BY ss.session_id DESC;
    "
}

# --- Latest session detail ---
latest_detail() {
    echo "${BLUE}=== Latest Session Detail ===${RESET}"
    run "
        SELECT *
        FROM session
        ORDER BY session_id DESC
        LIMIT 1;
    "
}

# --- Anomaly detection ---
anomalies() {
    echo "${RED}=== Anomalies (should be empty) ===${RESET}"
    run "
        SELECT session_id, session_name
        FROM session
        WHERE pixel_scale_h IS NULL
           OR pixel_scale_v IS NULL
           OR lat IS NULL
           OR lon IS NULL
           OR elevation IS NULL;
    "
}

# --- Ingestion rate (frames/sec) ---
ingestion_rate() {
    echo "${GREEN}=== Ingestion Rate (frames/sec) ===${RESET}"

    NOW=$(run "SELECT COUNT(*) FROM calstar_files;" | tr -d '[:space:]')

    if [[ -f /tmp/ingest_prev ]]; then
        PREV=$(cat /tmp/ingest_prev)
    else
        PREV=$NOW
    fi

    RATE=$(( NOW - PREV ))

    echo "Frames ingested in last interval: $RATE"

    echo "$NOW" > /tmp/ingest_prev
}

total_obs() {
    echo "${BLUE}=== Total Observations ===${RESET}"
    run "
        SELECT COUNT(*) AS total_observations
        FROM observation;
    "
}


# --- Dashboard ---
dashboard() {
    clear
    latest_sessions
    frame_counts
    #obs_counts
    #total_obs
    latest_detail
    anomalies
    ingestion_rate
}

# --- Dispatch ---
case "$CMD" in
    latest) latest_sessions ;;
    frames) frame_counts ;;
    obs) obs_counts ;;
    detail) latest_detail ;;
    anomalies) anomalies ;;
    rate) ingestion_rate ;;
    dashboard|"") dashboard ;;
    *)
        echo "Usage: $0 [--host HOST] {latest|frames|obs|detail|anomalies|rate|dashboard}"
        ;;
esac
