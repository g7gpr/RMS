#!/bin/bash
set -e

USER_HOME="$(getent passwd "$USER" | cut -d: -f6)"

source "$USER_HOME/vRMS/bin/activate"
cd "$USER_HOME/source/RMS"
git pull

python -m Utils.StarPipeline.Ingest \
    analysis@gmn.uwo.ca:/home/stationID/files/processed \
    192.168.217.212 \
    --write_db \
    --calstars_data_dir="/mnt/rms/cache/RMS_data/CALSTARS/" \
    -t24
