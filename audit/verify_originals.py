"""Read-only verification against the preserved archive/file inventory."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    inventory = json.loads(Path("audit/inventory.json").read_text())
    for item in inventory["files"]:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Original changed: {item['path']}")
    count = 0
    for item, root in zip(inventory["archives"], (Path("originals/cnn_models"), Path("originals/medical_imaging"))):
        archive_path = Path(item["path"])
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Archive changed")
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    if archive.read(entry) != (root/entry.filename).read_bytes():
                        raise ValueError(f"Archive mismatch: {entry.filename}")
                    count += 1
    print(f"Verified {count} original files byte-for-byte against both archives")


if __name__ == "__main__":
    main()
