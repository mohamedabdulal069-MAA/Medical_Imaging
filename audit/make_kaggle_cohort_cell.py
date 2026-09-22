"""Generate a self-contained patch/test/finalize cell for the original Kaggle upload."""
from pathlib import Path

header = '''# Paste the ENTIRE file into one NEW Kaggle cell and run once.
# Explicit protocol: quarantine ALL copies of label-conflicting HU images;
# retain one image per agreeing HU hash; keep linked patient IDs together.
# Source DICOMs, original labels and the scan checkpoint are never modified.
from pathlib import Path
import shutil
import sys
import subprocess
import importlib

project = Path("/kaggle/working/ich-research")
source = Path("/kaggle/input/datasets/mabdulal/ich-repaired-code")
if not (project / "ich" / "__init__.py").exists():
    shutil.copytree(source, project, dirs_exist_ok=True)
sys.path.insert(0, str(project))
files = {}
'''
body = ""
for name in ("ich/dataset.py", "ich/cohort.py", "tests/test_cohort.py"):
    source = Path(name).read_text(encoding="utf-8")
    assert "'''" not in source
    body += f"\nfiles[{name!r}] = r'''{source}'''\n"
footer = '''
for name, content in files.items():
    target = project / name
    if target.exists() and not target.with_suffix(target.suffix + ".before-cohort").exists():
        shutil.copy2(target, target.with_suffix(target.suffix + ".before-cohort"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

print("Checking leakage and cohort regression tests...", flush=True)
completed = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_cohort.py", "tests/test_integrity.py"],
    cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(completed.stdout, flush=True)
if completed.returncode != 0:
    raise RuntimeError("Tests failed; stop and send the output")

import ich.dataset
importlib.reload(ich.dataset)
import ich.cohort
importlib.reload(ich.cohort)

output = Path("/kaggle/working/rsna-reviewed-cohort-v1")
if output.exists():
    print("Existing cohort folder protected:", output)
    print("Do not delete it. Send its contents/output if a previous attempt failed.")
else:
    print("Using the completed checkpoint. No DICOM rescan...", flush=True)
    manifest = ich.cohort.finalize_checkpoint(
        "/kaggle/working/rsna-manifest-checkpoint",
        "/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv",
        output, quarantine_conflicts=True, seed=42)
    print("Ready for review. Save this cohort folder and patched code before training.")
'''
cell = header + body + footer
compile(cell, "kaggle-finalize-cohort.txt", "exec")
Path("kaggle-finalize-cohort.txt").write_text(cell, encoding="utf-8")
