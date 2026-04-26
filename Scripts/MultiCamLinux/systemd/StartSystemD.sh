#!/bin/bash
set -euo pipefail

STATION_USER="$1"
STATION_ID_UPPER="${STATION_USER^^}"

# Update RMS for this station
/home/${STATION_USER}/source/RMS/Scripts/RMS_Update.sh

# Activate venv
source /home/${STATION_USER}/vRMS/bin/activate

# Run capture
exec python -m RMS.StartCapture \
    -c /srv/rms/Stations/${STATION_ID_UPPER}/station.conf
