"""Generate a paste-ready Kaggle cell using the installed original ich package."""
from pathlib import Path

bootstrap = '''# Paste this entire file into ONE new Kaggle code cell and run it.
from pathlib import Path
import shutil
import sys

project = Path("/kaggle/working/ich-research")
source = Path("/kaggle/input/datasets/mabdulal/ich-repaired-code")
if not (project / "ich" / "__init__.py").exists():
    shutil.copytree(source, project, dirs_exist_ok=True)
sys.path.insert(0, str(project))

'''
driver = '''

# The only accepted image exclusion is the exact short-payload failure
# reproduced for ID_6431af929. Other failures block the final manifest.
data = Path("/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection")
manifest_path = Path("/kaggle/working/rsna-manifest.csv")
checkpoint_path = Path("/kaggle/working/rsna-manifest-checkpoint")

if manifest_path.exists():
    print("An existing manifest is protected:", manifest_path)
else:
    prepare_resumable(
        data / "stage_2_train.csv",
        data / "stage_2_train",
        manifest_path,
        checkpoint_path,
        seed=42,
        exclude_confirmed_short_payload=True,
        progress_every=1000,
    )
'''
cell = bootstrap + Path("ich/manifest_scan.py").read_text(encoding="utf-8") + driver
compile(cell, "kaggle-resume-manifest.txt", "exec")
Path("kaggle-resume-manifest.txt").write_text(cell, encoding="utf-8")
