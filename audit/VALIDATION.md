# Validation record

Executed locally on 2026-09-08, Windows, Python 3.12, CPU. Exact installed dependency closure is in `requirements-lock.txt` (TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.0, pydicom 3.0.2, scikit-image 0.26.0, pytest 9.1.1).

## Executed checks

| Check | Observed outcome |
|---|---|
| `python -m pytest -q --junitxml=audit/test-results.xml` | **32 passed**, 50.97 seconds, zero skipped/failed tests |
| Package editable build/install | Succeeded; `ich --help` exposes prepare/train/evaluate/predict/explain |
| `python -m compileall -q ich tests` | Succeeded |
| `python audit/verify_originals.py` | All 13 source/artifact files matched both ZIP archives byte for byte |
| HDF5 JSON configuration inspection | Confirmed 512-feature PCA head, 256/128 hidden units and six outputs; no training claim inferred |

## What the tests establish

- Patient sets are disjoint across train/validation/test and split assignments are reproducible after reordering rows.
- Cross-partition patient, study, series, SOP, image and exact-pixel identities are rejected; missing patient IDs are rejected.
- PCA fit IDs must equal the complete training partition, not all data or a held-out/mixed subset. Its mean/components match training-only scikit-learn PCA, transformations do not refit it, repeated fitting is rejected, and NPZ round trips preserve projections/provenance.
- Synthetic signed DICOM pixels use slope/intercept correctly; display polarity does not invert HU; padding is handled; missing HU metadata fails; constant-window inputs remain finite.
- Conflicting/missing labels fail. Multilabel co-occurrence survives evaluation, all-negative images are supported, and undefined sensitivity/AUC are represented as null. Loss weighting rejects held-out labels.
- Each of five preprocessing adapters matches the installed Keras implementation and survives serialization. All five real CNNs and the real two-CNN ensemble produce finite six-probability outputs with frozen backbones. Inception and the ensemble also survive full model save/reload with matching outputs. Tests use random initialization and 75×75 inputs for those forward checks; no pretrained weight download is needed.
- The real three-CNN extractor has 5376 features; the original PCA head has six outputs. Zero-activation Grad-CAM is finite and requires an explicit hemorrhage label.
- Synthetic DICOMs traverse manifest creation, two training epochs, save, reload, shared path/batch inference, and held-out evaluation for both CNN and PCA workflows. Small surrogate networks make these lifecycle tests practical. The save guard asserts optimizer updates occurred before classifier saving. Bundle corruption and changed DICOM pixels are rejected.

The final run emitted 1,030 repetitions of a **third-party Keras/NumPy deprecation warning** while serializing arrays (`__array__` does not accept a `copy` keyword). They were not suppressed. No dataset-exhaustion, input-structure, or PCA divide-by-zero warning remains in the final run. JUnit output is in `test-results.xml`.

## What was not tested or claimed

No real RSNA or other patient dataset was present. No baseline was experimentally trained on medical images; no clinical accuracy, AUC, segmentation quality, improvement over historical scores, or leakage prevalence was estimated. Synthetic test training losses/probabilities are test mechanics, not experimental results. No legacy pickle was deserialized. No ImageNet-weight download, GPU determinism, external cohort, enhanced/multiframe CT, compressed decoder plugin, patient-level confidence interval, or 3D viewer validation was performed.

The workflow in `.github/workflows/tests.yml` is provided for future CI; no remote CI run was performed here.

## Follow-up: checkpointed scan

After Kaggle confirmed the short PixelData payload in `ID_6431af929.dcm`, a separately tested resumable scanner was added. `python -m pytest -q tests/test_manifest_scan.py` completed with **4 passed in 4.61 seconds**. These new tests verify cache reuse and changed-file re-reading, graceful interruption/resume, unexpected-error blocking even when the known exclusion is enabled, and an exact defect signature (including retention of a readable replacement). The generated Kaggle cell and new source/test files also passed compilation. The original 32-test suite was not rerun for this additive module; no existing execution path was modified.

The full RSNA scan has not been executed locally. Kaggle's reported 153710 versus 524288 byte mismatch is recorded as an unreadable source image; it does not establish whether the original cause was file truncation or erroneous metadata. The exclusion is explicit, narrowly matched, and reported in scan artifacts. No performance result is inferred from this correction.

## Follow-up: annotation conflicts and duplicate-linked partitions

The user reported 4073 cross-patient matching-HU groups and 838 groups with conflicting annotations after the complete Kaggle scan. The new explicit cohort policy quarantines all label-conflicting copies and deduplicates concordant images while keeping all transitively linked patient IDs together, using links constructed before exclusions. Source IDs and annotations are preserved in audit outputs.

The updated complete local suite passed: **39 passed, 1030 third-party Keras/NumPy deprecation warnings, 106.40 seconds**. See `test-results-cohort.xml`. New tests cover conservative conflict handling (refusal by default), deterministic representative selection, transitive grouping through excluded images, group leakage rejection, row-order reproducibility, complete-cache finalization without DICOM access, and incomplete-cache rejection. The paste-ready `kaggle-finalize-cohort.txt` compiles successfully and runs 19 targeted checks in Kaggle before finalizing the real cohort. The actual Kaggle cohort output and counts have not yet been observed; no real-data training/evaluation result is claimed.
