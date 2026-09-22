"""Build the complete installation/test/first-segment Kaggle cell."""
from pathlib import Path

header = '''# Paste this ENTIRE file into ONE new Kaggle code cell.
# Installs the new runner, runs synthetic tests, then starts 100 real batches.
from pathlib import Path
import sys
import subprocess
import importlib
import warnings
import os

project = Path('/kaggle/working/ich-research')
assert (project / 'ich/cohort.py').is_file(), 'Restore the complete training backup first.'
assert Path('/kaggle/working/rsna-reviewed-cohort-v1/manifest.csv').is_file()
sys.path.insert(0, str(project))
files = {}
'''
body = ''
for name in ('ich/resumable.py', 'tests/test_resumable.py'):
    content = Path(name).read_text(encoding='utf-8')
    assert "'''" not in content
    body += f"\nfiles[{name!r}] = r'''{content}'''\n"
footer = '''
for name, content in files.items():
    target = project / name
    if target.exists() and target.read_text(encoding='utf-8') != content:
        raise RuntimeError(f'Different runner already installed: {target}. Do not overwrite an active run.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')

print('Checking recovery, leakage, PCA fit restrictions and cohort tests...', flush=True)
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '-q', 'tests/test_resumable.py',
     'tests/test_integrity.py', 'tests/test_cohort.py'],
    cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'),
)
print(result.stdout, flush=True)
if result.returncode:
    raise RuntimeError('Tests failed. Training has not started.')

# Only the already-reviewed UID syntax warning is hidden; other warnings/errors remain.
warnings.filterwarnings('ignore', message=r"Invalid value for VR UI: 'ID_.*",
                        category=UserWarning, module=r'pydicom\\.valuerep')
importlib.invalidate_caches()
from ich import resumable
import tensorflow as tf
tf.keras.backend.clear_session()

result = resumable.run(
    '/kaggle/working/rsna-reviewed-cohort-v1/manifest.csv',
    '/kaggle/working/resnet50-full-s42',
    baseline='resnet50', epochs=3, batch_size=16, seed=42,
    weights='imagenet', checkpoint_every=100,
    max_batches=100, max_minutes=45, workers=4,
)
print('Segment status:', result['status'])
print('Run folder:', result['output'])
print('This is the start of the full run, not final experimental results.')
'''
Path('kaggle-start-resumable-training.txt').write_text(header + body + footer, encoding='utf-8')

continuation = '''# Run in the SAME restored environment, one invocation at a time.
import sys
sys.path.insert(0, '/kaggle/working/ich-research')
from ich.resumable import run
result = run(
    '/kaggle/working/rsna-reviewed-cohort-v1/manifest.csv',
    '/kaggle/working/resnet50-full-s42',
    baseline='resnet50', epochs=3, batch_size=16, seed=42,
    weights='imagenet', checkpoint_every=250,
    max_batches=None, max_minutes=45, workers=4,
)
print('Segment status:', result['status'])
'''
Path('kaggle-continue-training.txt').write_text(continuation, encoding='utf-8')

backup = '''# Run ONLY after the training segment has returned.
from pathlib import Path
import os
import zipfile
from datetime import datetime
from IPython.display import FileLink, display
work = Path('/kaggle/working')
run_dir = work / 'resnet50-full-s42'
assert (run_dir / 'current.json').is_file(), 'No committed training checkpoint yet.'
archive = work / ('ich-training-progress-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.zip')
folders = ['resnet50-full-s42', 'ich-research', 'rsna-reviewed-cohort-v1']
print('Packaging model, optimizer, progress, manifest and exact runner code...', flush=True)
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for name in folders:
        folder = work / name
        assert folder.is_dir(), f'Missing {folder}'
        for path in folder.rglob('*'):
            if path.is_file() and not any(p in {'__pycache__', '.pytest_cache', '.venv'} for p in path.parts):
                z.write(path, path.relative_to(work))
os.chdir(work)
print('Download this new training-progress ZIP; the old preparation ZIP has no full-run weights.')
display(FileLink(archive.name))
'''
Path('kaggle-backup-training.txt').write_text(backup, encoding='utf-8')
