#!/usr/bin/env python3
"""
Provision one or more RMS stations on this host.

Automatic mode (requires root, actually runs commands):

    sudo ./provision.py \
        --stations au0004,au0006,au0007 \
        --location "-32.179499,115.859859,30"

Manual mode (no root required, prints commands one by one and waits for confirmation):

    ./provision.py \
        --stations au0004,au0006,au0007 \
        --location "-32.179499,115.859859,30" \
        --manual
"""

import argparse
import os
import pwd
import grp
import subprocess
import sys
from typing import Optional


MANUAL_MODE = False  # set from args
UNIT_PATH = "/etc/systemd/system/rms@.service"


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def logMessage(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def fail(message: str) -> None:
    logMessage("FAIL", message)
    sys.exit(1)


def formatCommand(command: list[str]) -> str:
    return " ".join(command)


def runCommand(command: list[str], as_user: Optional[str] = None, require_root: bool = False) -> None:
    """
    Automatic mode:
        - optionally wraps with sudo
        - executes the command
    Manual mode:
        - prints the command exactly as it should be run (with sudo if needed)
        - waits for operator confirmation
        - does NOT execute anything
    """
    global MANUAL_MODE

    if as_user is not None:
        display_cmd = ["sudo", "-u", as_user] + command
        user_label = as_user
    else:
        display_cmd = ["sudo"] + command if require_root else command
        user_label = "root" if require_root else "current-user"

    if MANUAL_MODE:
        print()
        print(f"STEP: Run this command as {user_label}:")
        print(formatCommand(display_cmd))
        input("Press ENTER once you have executed this command...")
        return

    prefix = "CMD"
    logMessage(prefix, f"as {user_label}: {formatCommand(display_cmd)}")

    try:
        subprocess.run(display_cmd, check=True)
    except subprocess.CalledProcessError as e:
        fail(f"Command failed: {formatCommand(display_cmd)} (exit code {e.returncode})")


def requireRoot() -> None:
    if MANUAL_MODE:
        return
    if os.geteuid() != 0:
        fail("This script must be run as root in automatic mode. Use --manual to print commands instead.")


def validatePathExists(path: str, station_user: str) -> None:
    if MANUAL_MODE and pathBelongsToStationUser(path, station_user):
        logMessage("SKIP", f"Cannot validate directory {path} in manual mode due to permissions.")
        return

    if not os.path.isdir(path):
        fail(f"Directory does not exist: {path}")

    logMessage("CHECK", f"Directory exists at {path}... OK")


def validateFileExists(path: str, station_user: str) -> None:
    if MANUAL_MODE and pathBelongsToStationUser(path, station_user):
        logMessage("SKIP", f"Cannot validate file {path} in manual mode due to permissions.")
        return

    if not os.path.isfile(path):
        fail(f"File does not exist: {path}")

    logMessage("CHECK", f"File exists at {path}... OK")


def pathBelongsToStationUser(path: str, station_user: str) -> bool:
    """
    Returns True if the path is inside a directory owned by the station user.
    In manual mode, these paths cannot be validated due to permissions.
    """
    station_id_upper = station_user.upper()

    return (
        path.startswith(f"/home/{station_user}/") or
        path.startswith(f"/srv/rms/Stations/{station_id_upper}/") or
        path.startswith(f"/srv/rms/RMS_data/{station_id_upper}/")
    )



def validateOwnership(path: str, station_user: str, expected_user: str, expected_group: str) -> None:
    if MANUAL_MODE and pathBelongsToStationUser(path, station_user):
        logMessage("SKIP", f"Cannot validate ownership of {path} in manual mode.")
        return

    st = os.stat(path)
    uid = pwd.getpwnam(expected_user).pw_uid
    gid = grp.getgrnam(expected_group).gr_gid

    if st.st_uid != uid:
        fail(f"Owner UID mismatch for {path}: expected {uid}, got {st.st_uid}")
    if st.st_gid != gid:
        fail(f"Group GID mismatch for {path}: expected {gid}, got {st.st_gid}")

    logMessage("CHECK", f"Ownership of {path} matches {expected_user}:{expected_group}... OK")


# ------------------------------------------------------------
# Provisioning steps
# ------------------------------------------------------------

def stepCreateUser(station_user: str) -> None:
    """Create the station user if missing."""
    try:
        pwd.getpwnam(station_user)
        logMessage("OK", f"User '{station_user}' already exists.")
        return
    except KeyError:
        pass

    logMessage("CREATE", f"Creating user '{station_user}'.")
    runCommand(["useradd", "-m", "-s", "/bin/bash", station_user], require_root=True)

    try:
        pwd.getpwnam(station_user)
        logMessage("CHECK", f"User '{station_user}' exists... OK")
    except KeyError:
        fail(f"User '{station_user}' does not exist after creation step.")


def stepGetUserHome(station_user: str) -> str:
    try:
        home_dir = pwd.getpwnam(station_user).pw_dir
    except KeyError:
        fail(f"User '{station_user}' does not exist.")

    if not os.path.isdir(home_dir):
        fail(f"Home directory for '{station_user}' does not exist: {home_dir}")

    logMessage("CHECK", f"Home directory exists at {home_dir}... OK")
    return home_dir


def stepPrepareSourceDir(home_dir: str, station_user: str) -> str:
    source_root = os.path.join(home_dir, "source")

    runCommand(["mkdir", "-p", source_root], require_root=True)
    runCommand(["chown", f"{station_user}:{station_user}", source_root], require_root=True)

    validatePathExists(source_root, station_user)
    validateOwnership(source_root, station_user, station_user, station_user)

    return source_root


def stepCloneRms(source_root: str, station_user: str) -> str:
    code_dir = os.path.join(source_root, "RMS")
    git_dir = os.path.join(code_dir, ".git")

    if os.path.isdir(git_dir) and not MANUAL_MODE:
        logMessage("OK", f"RMS repo already exists at {code_dir}. Pulling latest changes.")
        runCommand(["git", "-C", code_dir, "pull", "--ff-only"], as_user=station_user)
    elif os.path.isdir(git_dir) and MANUAL_MODE:
        logMessage("OK", f"RMS repo already exists at {code_dir}.")
    else:
        logMessage("CREATE", f"Cloning RMS into {code_dir}.")
        runCommand(["git", "clone", "https://github.com/g7gpr/RMS.git", code_dir],
                   as_user=station_user)

    validatePathExists(code_dir, station_user)
    return code_dir


def stepCreateVenv(home_dir: str, station_user: str) -> str:
    venv_dir = os.path.join(home_dir, "vRMS")

    if not os.path.isdir(venv_dir):
        logMessage("CREATE", f"Creating virtualenv at {venv_dir}")
        runCommand(["python3", "-m", "venv", "--system-site-packages", venv_dir], as_user=station_user)
    else:
        logMessage("OK", f"Virtualenv already exists: {venv_dir}")

    validatePathExists(venv_dir, station_user)
    return venv_dir


def stepInstallRmsIntoVenv(venv_dir: str, code_dir: str, station_user: str) -> None:
    activate = os.path.join(venv_dir, "bin", "activate")
    requirements = os.path.join(code_dir, "requirements.txt")

    if MANUAL_MODE:
        print()
        print("STEP: Install RMS into venv")
        print(f"sudo -u {station_user} bash <<'EOF'")
        print("set -euo pipefail")
        print(f"source '{activate}'")
        print("python -m pip install --upgrade pip setuptools wheel")
        print("python -m pip install 'numpy<2.0'")
        print(f"python -m pip install -r '{requirements}'")
        print(f"cd '{code_dir}'")
        print("python -m pip install . --no-deps --no-build-isolation")
        print("EOF")
        input("Press ENTER once you have executed this block...")
    else:
        command = f"""
set -euo pipefail
source '{activate}'
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'numpy<2.0'
python -m pip install -r '{requirements}'
cd '{code_dir}'
python -m pip install .
"""
        runCommand(["bash", "-c", command], as_user=station_user)

    validateFileExists(activate, station_user)



def stepCreateConfigDir(station_id: str, station_user: str) -> str:
    config_root = "/srv/rms/Stations"
    config_dir = os.path.join(config_root, station_id)

    runCommand(["mkdir", "-p", config_dir], require_root=True)
    runCommand(["chown", f"{station_user}:{station_user}", config_dir], require_root=True)
    runCommand(["chmod", "700", config_dir], require_root=True)

    validatePathExists(config_dir, station_user)
    return config_dir


def stepCreateConfigFile(station_id: str, station_user: str,
                         lat: Optional[float], lon: Optional[float],
                         elev: Optional[float], location: Optional[str]) -> None:

    config_dir = f"/srv/rms/Stations/{station_id}"
    target_path = os.path.join(config_dir, ".config")

    if os.path.isfile(target_path) and MANUAL_MODE:
        logMessage("OK", f".config already exists (manual mode, not overwriting): {target_path}")
        return

    if os.path.isfile(target_path) and not MANUAL_MODE:
        logMessage("OK", f".config already exists: {target_path}")
        return

    loc_lat = loc_lon = loc_elev = None
    if location:
        try:
            parts = [p.strip() for p in location.split(",")]
            if len(parts) == 3:
                loc_lat, loc_lon, loc_elev = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            fail("Invalid --location format. Expected: LAT,LON,ELEV")

    final_lat = lat if lat is not None else loc_lat or 0.0
    final_lon = lon if lon is not None else loc_lon or 0.0
    final_elev = elev if elev is not None else loc_elev or 0

    url = "https://raw.githubusercontent.com/CroatianMeteorNetwork/RMS/refs/heads/master/.config"
    logMessage("CREATE", f"Downloading upstream .config to {target_path}")
    runCommand(["wget", "-qO", target_path, url], require_root=True)

    validateFileExists(target_path, station_user)

    if MANUAL_MODE:
        logMessage("INFO", "Manual mode: you must edit the .config file yourself.")
        print(f"[System]")
        print(f"stationID: {station_id}")
        print(f"latitude: {final_lat}")
        print(f"longitude: {final_lon}")
        print(f"elevation: {final_elev}")
        print(f"[Capture]")
        print(f"data_dir: /srv/rms/RMS_data/{station_id}")
        runCommand(["chown", f"{station_user}:{station_user}", target_path], require_root=True)
        runCommand(["chmod", "600", target_path], require_root=True)
        return

    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    current_section = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            new_lines.append(line)
            continue

        if stripped.startswith(";") or stripped == "" or ":" not in stripped:
            new_lines.append(line)
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()

        if current_section == "System":
            if key == "stationID":
                new_lines.append(f"stationID: {station_id}\n")
                continue
            if key == "latitude":
                new_lines.append(f"latitude: {final_lat}\n")
                continue
            if key == "longitude":
                new_lines.append(f"longitude: {final_lon}\n")
                continue
            if key == "elevation":
                new_lines.append(f"elevation: {final_elev}\n")
                continue

        if current_section == "Capture":
            if key == "data_dir":
                new_lines.append(f"data_dir: /srv/rms/RMS_data/{station_id}\n")
                continue

        new_lines.append(line)

    with open(target_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    runCommand(["chown", f"{station_user}:{station_user}", target_path], require_root=True)
    runCommand(["chmod", "600", target_path], require_root=True)

    validateFileExists(target_path, station_user)


def stepCreateDataDir(station_id: str, station_user: str) -> None:
    data_root = "/srv/rms/RMS_data"
    data_dir = os.path.join(data_root, station_id)

    runCommand(["mkdir", "-p", data_root], require_root=True)

    try:
        grp.getgrnam("rms-data")
        logMessage("OK", "Group 'rms-data' already exists.")
    except KeyError:
        logMessage("CREATE", "Creating group 'rms-data'.")
        runCommand(["groupadd", "rms-data"], require_root=True)

    runCommand(["mkdir", "-p", data_dir], require_root=True)
    runCommand(["chown", f"{station_user}:rms-data", data_dir], require_root=True)
    runCommand(["chmod", "2775", data_dir], require_root=True)

    validatePathExists(data_dir, station_user)


def stepInstallSystemd(unit_path=None) -> None:
    unit_path = UNIT_PATH if unit_path is None else unit_path

    content = """[Unit]
Description=%i
After=network.target

[Service]
User=%i
SyslogIdentifier=%i
WorkingDirectory=/home/%i/source/RMS/
ExecStart=/home/rms/source/RMS/Scripts/MultiCamLinux/systemd/StartSystemD.sh %i
Restart=always
RestartSec=30
RuntimeMaxSec=48h

[Install]
WantedBy=multi-user.target
"""

    if MANUAL_MODE:
        print()
        print(f"STEP: Create or update systemd unit file {unit_path}")
        print(f"sudo tee {unit_path} > /dev/null <<'EOF'")
        print(content, end="")
        print("EOF")
        print(f"sudo chmod 644 {unit_path}")
        print("sudo systemctl daemon-reload")
        input("Press ENTER once you have executed these commands...")
    else:
        with open(unit_path, "w", encoding="utf-8") as f:
            f.write(content)
        runCommand(["chmod", "644", unit_path], require_root=True)
        runCommand(["systemctl", "daemon-reload"], require_root=True)



def stepEnableSystemd(unit_path, station_user: str) -> None:

    validateFileExists(unit_path, station_user)

    logMessage("OK", f"Enabling rms@{station_user}.service")
    runCommand(["systemctl", "enable", f"rms@{station_user}.service"], require_root=True)


def stepStartStation(station_user: str) -> None:
    service_name = f"rms@{station_user}.service"

    # Validate that the service file exists before attempting to start it
    unit_path = f"/etc/systemd/system/rms@.service"
    validateFileExists(unit_path, station_user)

    if MANUAL_MODE:
        print()
        print(f"STEP: Start systemd service for station {station_user}")
        print(f"sudo systemctl start {service_name}")
        input("Press ENTER once you have executed this command...")
    else:
        logMessage("OK", f"Starting {service_name}")
        runCommand(["systemctl", "start", service_name], require_root=True)


# ------------------------------------------------------------
# Argument parsing and main
# ------------------------------------------------------------

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision RMS stations.")
    parser.add_argument("--stations", required=True,
                        help="Comma-separated list of station IDs (e.g. au0004,au0006)")
    parser.add_argument("--lat", type=float, help="Latitude in degrees")
    parser.add_argument("--lon", type=float, help="Longitude in degrees")
    parser.add_argument("--elev", type=float, help="Elevation in meters")
    parser.add_argument("--location", help="Combined LAT,LON,ELEV")
    parser.add_argument("--manual", action="store_true",
                        help="Do not execute commands; print them one by one and wait for confirmation.")
    return parser.parse_args()


def main() -> None:
    global MANUAL_MODE

    unit_path = UNIT_PATH

    args = parseArgs()
    MANUAL_MODE = args.manual

    if MANUAL_MODE:
        logMessage("INFO", "Running in MANUAL mode: commands will NOT be executed.")
        logMessage("INFO", "You will be prompted to run each command, then press ENTER to continue.")
    else:
        logMessage("INFO", "Running in AUTOMATIC mode: commands will be executed.")
        requireRoot()

    station_list = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    if not station_list:
        fail("No valid station IDs provided.")

    stepInstallSystemd(unit_path=UNIT_PATH)

    for station_id in station_list:
        station_user = station_id.lower()
        logMessage("OK", f"Provisioning station {station_id} (user: {station_user})")

        stepCreateUser(station_user)
        home_dir = stepGetUserHome(station_user)

        source_root = stepPrepareSourceDir(home_dir, station_user)
        code_dir = stepCloneRms(source_root, station_user)

        venv_dir = stepCreateVenv(home_dir, station_user)
        stepInstallRmsIntoVenv(venv_dir, code_dir, station_user)

        stepCreateConfigDir(station_id, station_user)
        stepCreateConfigFile(
            station_id,
            station_user,
            args.lat,
            args.lon,
            args.elev,
            args.location
        )

        stepCreateDataDir(station_id, station_user)
        stepEnableSystemd(unit_path, station_user)
        logMessage("OK", f"Provisioning steps completed for station {station_id}.")

        stepStartStation(station_user)

        if MANUAL_MODE:
            logMessage("OK",
                       "Manual mode complete. All commands have been emitted and checks performed where possible.")
        else:
            logMessage("OK", "All stations provisioned successfully.")

if __name__ == "__main__":
    main()

