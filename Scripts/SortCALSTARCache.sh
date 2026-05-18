#!/bin/bash
set -e

USER_HOME="$(getent passwd "$USER" | cut -d: -f6)"

# Activate venv
source "$USER_HOME/vRMS/bin/activate"

# Run the module
python -m Utils.StarPipeline.SortCALSTARCache /srv/rms/RMS_data/CALSTARS
