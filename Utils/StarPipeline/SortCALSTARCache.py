import sys
import shutil
from pathlib import Path

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
    if not cache_root.exists():
        print("Error: " + str(cache_root) + " does not exist")
        sys.exit(1)

    refileArchives(cache_root)

if __name__ == "__main__":
    import os
    main()
