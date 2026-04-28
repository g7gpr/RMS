#!/bin/bash
set -euo pipefail

source /home/rms/vRMS/bin/activate
/home/rms/source/RMS/Scripts/RMS_Update.sh
exec python -m Utils.MultiCamUploader
