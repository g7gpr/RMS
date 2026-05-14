#!/bin/bash
set -euo pipefail

# Activate venv
source /home/star/vRMS/bin/activate

# Run the module
python -m Utils.StarPipeline.PopulateWorkQueue gmn@192.168.217.241:/home/stationID/files 192.168.217.212

# exit
exit 0
