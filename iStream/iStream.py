#!/usr/bin/python
import os
import subprocess
import datetime
import logging
from RMS.CaptureDuration import captureDuration
from RMS.Logger import initLogging

def rmsExternal(captured_night_dir, archived_night_dir, config):
    initLogging(config, 'iStream_')
    log = logging.getLogger("logger")
    log.info('iStream external script started')

    # create lock file to avoid RMS rebooting the system
    lockfile = os.path.join(config.data_dir, config.reboot_lock_file)
    with open(lockfile, 'w') as fp:
        pass

    # Compute the capture duration from now
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)

    timenow = datetime.datetime.utcnow()
    remaining_seconds = 0

    # Compute how long to wait before capture
    if start_time != True:
        waitingtime = start_time - timenow
        remaining_seconds = int(waitingtime.total_seconds())		

    # Run the Istrastream shell script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iStream.sh")
    log.info('Calling {}'.format(script_path))

    command = [
            script_path,
            config.stationID,
            captured_night_dir,
            archived_night_dir,
            '{:.6f}'.format(config.latitude),
            '{:.6f}'.format(config.longitude),
            '{:.1f}'.format(config.elevation),
            str(config.width),
            str(config.height),
            str(remaining_seconds)
            ]

    proc = subprocess.Popen(command,stdout=subprocess.PIPE)
   
    # Read iStream script output and append to log file
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        log.info(line.rstrip().decode("utf-8"))

    exit_code = proc.wait()
    log.info('Exit status: {}'.format(exit_code))
    log.info('iStream external script finished')

    # relase lock file so RMS is authorized to reboot, if needed
    os.remove(lockfile)
