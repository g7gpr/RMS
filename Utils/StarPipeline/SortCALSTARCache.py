import sys
import shutil
from pathlib import Path
import psycopg
import tarfile
from RMS.Misc import mkdirP
from Utils.StarPipeline.PipelineDB import extractStub

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
    if len(parts) != 5:
        return None
    if not filename.endswith(".tar.bz2"):
        return None
    return parts[1]

def ensureDirectory(path):
    if not path.exists():
        path.mkdir(mode=0o775)
    return path

def moveFile(src_path, dest_path):
    shutil.move(str(src_path), str(dest_path))

def refileArchives(cache_root):
    for root, dirs, files in os.walk(cache_root):
        root_path = Path(root)

        for filename in files:
            file_path = root_path / filename

            date_str = extractDate(filename)
            if not date_str:
                #print("Skipping invalid filename: " + filename)
                continue

            target_dir = ensureDirectory(cache_root / date_str)
            target_path = target_dir / filename

            if file_path == target_path:
                #print("Already in correct location: " + filename)
                continue

            print("Moving " + str(file_path) + " -> " + str(target_path))
            moveFile(file_path, target_path)

def main():
    if len(sys.argv) != 2:
        print("Usage: refile_archives.py <CACHE_ROOT>")
        sys.exit(1)



    cache_root = Path(sys.argv[1])

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
        if os.path.isdir(os.path.join(cache_root, cache_root_object)) and len(cache_root_object) == 8 and cache_root_object.isnumeric():
            cache_root_dirs_list.append(cache_root_object)

    cache_root_dirs_list.sort()
    for cache_day_directory in cache_root_dirs_list:
        remote_file_stub_set = set(createRemoteFileStubList(cache_day_directory))
        day_directory_full_path = os.path.join(cache_root,cache_day_directory)
        cache_file_list = os.listdir(day_directory_full_path)
        cache_file_stub_list = []
        for cache_file in cache_file_list:
            if os.path.isfile(os.path.join(day_directory_full_path, cache_file)) and cache_file.endswith("_raw.tar.bz2"):
                cache_file_stub_list.append(extractStub(cache_file))
        cache_file_stub_set = set(cache_file_stub_list)
        missing_files = remote_file_stub_set - cache_file_stub_set
        if len(missing_files) == 0:
            print(f"Ready to archive {cache_day_directory}")
            with tarfile.open(os.path.join(cache_root,"archives",f"{cache_day_directory}.tar.bz2"), "w:bz2") as tar:
                for fname in cache_file_list:
                    try:
                        archive_target = os.path.join(cache_root, cache_day_directory, fname)
                        #tar.add(archive_target, arcname=os.path.join(cache_day_directory,fname))
                    except:
                        print("Some file changed during archiving - this directory may not be ready")
                        os.unlink(archive_target)

    if not cache_root.exists():
        print("Error: " + str(cache_root) + " does not exist")
        sys.exit(1)

    refileArchives(cache_root)

if __name__ == "__main__":
    import os
    main()
