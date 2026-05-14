#!/bin/bash
set -euo pipefail

# Activate venv
source /home/star/vRMS/bin/activate

# Run the module
python -m Utils.StarPipeline.PopulateWorkQueue analysis@gmn.uwo.ca:/home/stationID/files/processed 192.168.217.212

# exit
exit 0
