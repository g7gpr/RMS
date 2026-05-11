#!/bin/bash

# Activate venv
source /home/star/vRMS/bin/activate

# Run the module
python -m Utils.StarPipeline.SortCALSTARCache /srv/rms/RMS_data/CALSTARS
