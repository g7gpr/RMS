#!/bin/bash
set -e

# Activate venv
source /home/star/vRMS/bin/activate

# Move to RMS source directory
cd /home/star/source/RMS

# Update code
git pull

# Launch StarPipeline ingestion
python -m Utils.StarPipeline.Ingest \
    analysis@gmn.uwo.ca:/home/stationID/files/processed \
    192.168.217.212 \
    --write_db \
    --calstars_data_dir="/mnt/rms/cache/RMS_data/CALSTARS/" \
    -t4
