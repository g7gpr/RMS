#!/bin/bash
set -euo pipefail

source /home/gmn/vRMS/bin/activate
/home/gmn/source/RMS/Scripts/RMS_Update.sh
exec python -m Utils.MultiCamUploader
