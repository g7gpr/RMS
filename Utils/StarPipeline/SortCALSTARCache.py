import datetime
import sys
import os
from datetime import tzinfo

import psycopg

from RMS.Misc import mkdirP
from Utils.StarPipeline.PipelineDB import extractStub
from RMS.Logger import LoggingManager, getLogger
import tarfile
import tempfile
import shutil
from pathlib import Path

postgresql_host = '192.168.217.212'



def createRemoteFileStubList(date_string):

    remote_file_stubs_list = []

    with psycopg.connect(host=postgresql_host,
                         dbname="star_data",
                         user="ingest_user") as conn:
        cur = conn.cursor()
        sql = f"SELECT remote_filename FROM ingest_work WHERE remote_filename LIKE '%{date_string}%' ORDER BY remote_filename"
        cur.execute(sql)

        for (fname,) in cur:

            remote_file_stubs_list.append(extractStub(fname))


    return set(remote_file_stubs_list)



def extractDate(filename):
    parts = filename.split("_")
    if len(parts) < 4:
        return None

    return parts[1]

def ensureDirectory(path):
    if not path.exists():
        path.mkdir(mode=0o775)
    return path

def moveFile(src_path, dest_path):
    shutil.move(str(src_path), str(dest_path))

def refileArchives(log, cache_root):


    cache_root = Path(cache_root)
    for day_dir in Path(cache_root).iterdir():

        if not day_dir.name.isdigit():

            continue

        for file_path in list(day_dir.iterdir()):



            date_path = Path(extractDate(str(file_path.name)))



            if not date_path:
                #print("Skipping invalid filename: " + filename)
                continue

            target_dir = ensureDirectory(cache_root / date_path)
            target_path = target_dir / file_path.name

            if file_path == target_path:
                #print("Already in correct location: " + filename)
                continue


            log.info(f"Moving {file_path} -> {target_path}")
            moveFile(file_path, target_path)

def main():


    # Initialize the logger
    log_manager = LoggingManager()

    # Get the logger handle
    log = getLogger("rmslogger")

    if len(sys.argv) != 2:
        print("Usage: refile_archives.py <CACHE_ROOT>")
        sys.exit(1)


    cache_root = Path(sys.argv[1])

    createDayArchives(log, cache_root)

    refileArchives(log, cache_root)




def buildDayArchive(log, cache_root: Path, day: str, cache_file_list):

    archives_dir = cache_root / "archives"
    archives_dir.mkdir(exist_ok=True)
    file_name = Path(f"{day}.tar.bz2")
    final_path = archives_dir / file_name

    with (tempfile.TemporaryDirectory() as tmpdir):
        tmp_path = Path(tmpdir) / file_name

        try:
            file_count = len(cache_file_list)
            width = len(str(file_count))
            archived_mb = 0

            with tarfile.open(tmp_path, "w:bz2") as tar:

                last_modified_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=48)
                total_to_transfer_mb = 0
                for i, fname in enumerate(cache_file_list):
                    archive_target = cache_root / day / fname
                    total_to_transfer_mb += os.path.getsize(archive_target) / 1024 ** 2
                    mtime = os.path.getmtime(archive_target)
                    last_modified_time = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                    if last_modified_time > last_modified_cutoff:
                        log.info(f"Skipping {day} because {archive_target} was modified at {last_modified_time}")
                        return False

                step = 10
                start_time = datetime.datetime.now(datetime.timezone.utc)
                for i, fname in enumerate(cache_file_list):
                    archive_target = cache_root / day / fname


                    try:
                        tar.add(archive_target, arcname=f"{day}/{fname}")
                        archived_mb += os.path.getsize(archive_target) / 1024 ** 2
                    except Exception:
                        return False

                    processed = i + 1
                    if processed % step == 0 or i == file_count:
                        elapsed = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
                        mb_sec = archived_mb / elapsed
                        remaining_to_transfer_sec = (total_to_transfer_mb - archived_mb) / mb_sec
                        forecast = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=remaining_to_transfer_sec)
                        forecast = forecast.replace(microsecond=0)
                        log.info(f"Processed {processed:0{width}d} out of {file_count} files at {(processed/elapsed):.02f} files/sec {(mb_sec):.02f} MB/sec completion at {forecast}")
                        step *= 2

            log.info(f"Archiving completed - copying to {final_path}")
            archive_size = os.path.getsize(tmp_path)  / 1024 ** 2
            start_time = datetime.datetime.now(datetime.timezone.utc)
            shutil.copy2(tmp_path, final_path)
            elapsed = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
            log.info(f"Copied at {(archive_size / elapsed):.2f} MB/s")

            log.info(f"Validating in final location")
            # Validate tarball
            try:
                with tarfile.open(final_path, "r:bz2") as tar:
                    for _ in tar:
                        pass
                return True
            except Exception:
                os.remove(final_path)
                return False

        except Exception:
            return False



def createDayArchives(log, cache_root: Path):
    cache_root_objects_list = os.listdir(cache_root)
    cache_root_dirs_list = []
    mkdirP(os.path.join(cache_root, "archives"))
    existing_archives_list = os.listdir(os.path.join(cache_root, "archives"))

    archive_date_list = []
    for existing_archive in existing_archives_list:
        archive_date_list.append(existing_archive.split("_")[0])

    for cache_root_object in cache_root_objects_list:
        if cache_root_object in archive_date_list:
            continue
        if os.path.isdir(os.path.join(cache_root, cache_root_object)) and len(
                cache_root_object) == 8 and cache_root_object.isnumeric():
            cache_root_dirs_list.append(cache_root_object)

    cache_root_dirs_list.sort()
    cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=14)
    for cache_day_directory in cache_root_dirs_list:
        cache_day_directory_dt = datetime.datetime.strptime(cache_day_directory, "%Y%m%d").replace(tzinfo=datetime.timezone.utc)
        if cache_day_directory_dt > cutoff_dt:
            log.info(f"Skipping {cache_day_directory}, too new")
            continue

        remote_file_stub_set = set(createRemoteFileStubList(cache_day_directory))
        day_directory_full_path = os.path.join(cache_root, cache_day_directory)
        cache_file_list = os.listdir(day_directory_full_path)
        cache_file_stub_list = []
        for cache_file in cache_file_list:
            if os.path.isfile(os.path.join(day_directory_full_path, cache_file)) and cache_file.endswith(
                    "_raw.tar.bz2"):
                cache_file_stub_list.append(extractStub(cache_file))
        cache_file_stub_set = set(cache_file_stub_list)
        missing_files = remote_file_stub_set - cache_file_stub_set
        target_bz2_file = os.path.join(cache_root, "archives", f"{cache_day_directory}.tar.bz2")
        if os.path.exists(target_bz2_file):
            continue
        if len(missing_files) == 0:
            log.info(f"Ready to archive {cache_day_directory} containing {len(cache_file_list)} files")
            buildDayArchive(log, Path(cache_root), cache_day_directory, cache_file_list)


if __name__ == "__main__":

    main()
