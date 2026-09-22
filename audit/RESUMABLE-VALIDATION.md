# Resumable training validation — 2026-09-20

Local full suite: **42 passed**, 1066 upstream Keras/NumPy deprecation warnings,
84.16 seconds. After extending the recovery test to interrupt validation as well
as training: **22 targeted tests passed**, 36 such warnings, 20.96 seconds.
All three generated Kaggle cells passed Python syntax parsing.

Synthetic checks exercise actual optimizer/model save and restore; continuation
starts at the next committed training batch, validation accumulators survive a
pause, and completed weights match uninterrupted training for a tiny model with
no dropout. This does not establish bitwise dropout RNG continuity for CNNs.
Further checks reject altered checkpoint bytes, changed resume configuration,
and changed DICOM HU hashes. A failed model write leaves the previous checkpoint
pointer usable. The reader asserts no test images are opened during training.
Existing patient/group leakage, PCA training-only fit, cohort and pipeline
regression tests remain passing.

No full-cohort training, Kaggle timing, two-GPU training, or clinical performance
has been measured by this local work. The starter is explicitly bounded to
100 real training batches, to be executed by the user in their Kaggle session.
PCA ensemble resumption is unsupported and rejected; the original corrected
PCA baseline and its tests remain unchanged. Epochs reload images with integrity
checks; no cross-epoch decoded-image cache has been introduced.

Resume checks fix dependency versions and package source checksums. Restoring
only the preparation archive is insufficient: a paused experiment also requires
its run folder (including current.json and checkpoint folders) and exact patched
code. All artifacts remain local to a Kaggle session until explicitly persisted.
