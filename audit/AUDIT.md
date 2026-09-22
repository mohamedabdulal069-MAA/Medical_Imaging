# Research software audit

Audit date: 2026-09-08. Scope: both supplied ZIP archives, all nine notebooks (all code cells), both Python scripts, and a non-executing inspection of the legacy HDF5 architecture configuration. No training dataset, split manifest, or trustworthy experiment provenance was supplied. Embedded document instructions were treated as source material. The supplied pickle was retained and hashed, not deserialized or executed.

## Findings and repairs

Line references below point to the extracted code under `audit/source/`, whose `# CELL` markers map back to original notebooks.

| Severity | Evidence | Consequence | Repair |
|---|---|---|---|
| Critical | `rsnainception.py:86` splits sampled long-form subtype rows before grouping images | The same image can be present on both sides; no patient protection | Pivot all six labels first, one image record, patient-level three-way split, mandatory cross-partition identity checks |
| Critical | `rsnavgg.py:72`, `rsnaresnet50.py:80`, `rsnadensenet.py:89`, `rsnaefficientnet.py:76`, `ensemblemodel.py:78`, `pcaensemble.py:89` use image-level `train_test_split` | Slices from a patient can cross train/validation; no untouched test set | Split by verified PatientID and reject overlapping patient/study/series/SOP/image/hash identities |
| Critical | `pcaensemble.py:117` fits PCA on concatenated features before the split at line 126; the earlier split is unused | Validation distribution influences learned representation | Split first; PCA requires exactly the training image IDs; save fit IDs and split digest; transform held-out features only |
| High | All training notebooks read raw `pixel_array`, resize, and use per-image max/min normalization | Stored pixel values are not reliably HU; inconsistent signedness/range and zero division | Decode signed pixels with pydicom, validate CT and slope/intercept, apply rescale once, mask padding, fixed HU windows, preserve numeric range |
| High | CNNs use one common [0,1] preprocessing without architecture-specific adapters; ensemble branches share that tensor | Pretrained input conventions are violated | Serialized per-branch Keras ImageNet adapters; EfficientNet receives [0,255] and keeps its built-in rescaling |
| High | VGG line 94, ResNet line 108, DenseNet line 108, EfficientNet line 102 save before `.fit` | Reloaded classifier heads can be untrained | Save only after nonempty, finite training; completion metadata is last; validation-selected weights restored |
| High | Inception saves after its first 3 epochs, then trains 3 more without replacing that checkpoint; inference uses 224 instead of 299 | Reported in-memory and reloaded models differ; incompatible input shape | One training lifecycle; saved input size shared by loader, evaluation and inference |
| High | VGG line 156 and ResNet line 172 take `argmax` over labels/predictions; `classificationmodel.py:114` does likewise | Coexisting hemorrhages and negative images are misrepresented | Six sigmoid probabilities, six independent thresholds, per-label 2×2 counts; no argmax |
| High | Generic Keras `accuracy`, inconsistent macro/weighted aggregation, `zero_division=1`, unguarded ROC curves | Ambiguous accuracy and inflated/undefined scores | Explicit binary accuracy/exact match, well-defined aggregation, undefined rates/AUC as null, per-label support |
| High | PCA notebook computes weights from flattened labels of all data and passes a two-class dictionary to multilabel Keras training | Wrong weighting semantics and held-out label influence | Default BCE; optional per-label positive weighting from training only |
| Critical | `classificationmodel.py:103–108` resizes a slice to 84×64 and flattens to 5376 pixels for PCA | Matching dimensionality conceals a fundamentally different feature space | Saved ResNet/EfficientNet/Inception extractor in original concatenation order, then saved PCA, then trained head |
| High | Classifier maps six outputs to a seven-class scheme including “no hemorrhage” | Labels shifted/misidentified | One canonical six-label order; negative prediction means no labels exceed their thresholds |
| High | `notebook5af9128ec7.py` explains ImageNet ResNet classes; `segmentation.py` thresholds attribution into pseudo-masks | Heatmaps do not establish hemorrhage localization or segmentation accuracy | Explain trained six-label model and explicitly selected label only; no segmentation claims; zero-safe Grad-CAM |
| High | `classificationmodel.py` stacks unrelated series after resizing, casts signed values to uint16, and ends with undefined `image_3d`; `vtkcode.py:63` casts raw pixels to uint8 and flips rows blindly | Anatomically invalid volume and corrupted display | Retain viewer demos only as archival evidence; corrected inference reads independent CT slices. Valid 3D geometry reconstruction is outside the classification pipeline |
| Medium | Missing labels filled as negatives, conflicting duplicates silently dropped, read errors skipped | Unknown labels become asserted negatives; untracked cohort changes | Complete binary-label checks, duplicate conflict rejection, `any` consistency, fail-visible image ingestion |
| Medium | Notebook state, hard-coded paths, runtime pip commands, no full seed/version/split provenance | Runs cannot be reliably reconstructed | Installable package/CLI, version lock, deterministic seeds/TF ops, source and artifact hashes, persisted manifest/config/history |

These findings establish defects in code, not the numerical amount of leakage in an unavailable dataset. Existing scores must be discarded for model comparison and regenerated using a reviewed protocol.

## Preserved baselines

| Baseline | Preserved architecture | Historical source |
|---|---|---|
| VGG16 | Frozen backbone → GAP → Dense 128 ReLU → Dropout .5 → six sigmoids | `rsnavgg.ipynb` |
| ResNet50 | Same 128/.5 head; the notebook misleadingly prints “ResNet50V2” | `rsnaresnet50.ipynb` |
| DenseNet121 | Same 128/.5 head; Adam learning rate .001 retained | `rsnadensenet.ipynb` |
| EfficientNetB0 | Same 128/.5 head | `rsnaefficientnet.ipynb` |
| InceptionV3 | Frozen backbone → GAP → Dense 1024 ReLU → six sigmoids; 299 input | `rsnainception.ipynb` |
| Two-CNN ensemble | EfficientNetB0 then ResNet50 pooled feature concatenation → Dense 256 → Dropout .5 → six sigmoids | `ensemblemodel.ipynb` |
| PCA ensemble | ResNet50/EfficientNetB0/InceptionV3 pooled features (5376 total) → training-only PCA → Dense 256/BN/Dropout .4 → Dense 128/BN/Dropout .3 → six sigmoids | `pcaensemble.ipynb` |

The legacy HDF5's JSON configuration confirms a 512-input dense PCA head (256 and 128 hidden units, six sigmoid outputs). See `legacy-model-config.json`. Configuration inspection proves neither training quality nor correspondence to the supplied PCA pickle. Neither artifact is imported into repaired runs. Originals remain byte-identical to extracted archive entries, tracked by `inventory.json`.

## Preprocessing decisions

The new baseline uses three fixed linear windows `(center,width)` = `(40,80)`, `(80,200)`, `(40,380)` in HU, mapped to float32 [0,255] and resized with antialiasing/preserved range. These are declared research defaults, not empirically optimized choices or claims of improved performance. They replace replicated, min/max-normalized raw pixels; therefore comparisons require new runs for every baseline. No data-driven clipping, normalization, or augmentation is fitted outside training.

Single-frame monochrome CT is supported. MONOCHROME1 polarity does not invert physical HU. Padding values are replaced with -1024 HU before windowing. Missing slope/intercept, unknown units, modality LUTs without a verified HU contract, enhanced/per-frame CT and multiframe data are rejected. Compressed DICOMs may need a pydicom decoder plugin; decoder failures are visible. Historical malformed unsigned CT encodings need a documented, dataset-specific investigation; no heuristic bit/sign manipulation is silently applied.

## Remaining study work and limitations

- Establish the completeness and stability of patient identifiers across sites/visits and validate HU handling on representative real DICOMs. Exact HU hashes detect identical pixel arrays, not re-encoded/resampled near-duplicates. Near-duplicate and cross-dataset patient linkage needs additional dataset-specific review.
- Audit image exclusions and class prevalence. Patient splits are not stratified. Patients with many slices contribute more slice loss/metrics; the pipeline makes no patient-balanced sampling claim.
- Choose the protocol using training/validation only. All baselines must share the same manifest; refit every learned transform inside each training fold if cross-validation is added. The PCA API validates row IDs, but callers must truthfully associate feature rows with those IDs; the integrated runner creates both in the same sorted order.
- Train all baselines from valid initialization and evaluate the held-out test set after freezing choices. The tool does not prevent humans from repeatedly consulting test reports. No original metric is a target for optimization.
- Add patient-cluster uncertainty estimates and independent external validation for publication claims; current reports are slice-level point estimates. Calibration, patient-level aggregation, segmentation evaluation and clinical deployment validation are not supplied.
- Grad-CAM supports the image CNNs and spatial branches of the two-CNN model. PCA explanations explicitly fail rather than returning an unrelated backbone's heatmap. The desktop/VTK demos are not maintained as functioning 3D tools.
- Deterministic operations/seeds improve reproducibility but do not guarantee bitwise equality across hardware, drivers or TensorFlow versions. Run metadata and the dependency lock identify the tested environment.

## Primary implementation references

Model-specific preprocessing follows the [Keras Applications API](https://keras.io/api/applications/), including [ResNet](https://keras.io/api/applications/resnet/), [DenseNet](https://keras.io/api/applications/densenet/), [InceptionV3](https://keras.io/api/applications/inceptionv3/) and [EfficientNet](https://keras.io/api/applications/efficientnet/). Adapter tests compare directly with installed Keras functions.

HU conversion follows the rescale metadata semantics described by [pydicom's rescale API](https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.pixels.apply_rescale.html). The supported reader intentionally rejects ambiguous LUT/enhanced cases.

Training-only PCA follows [scikit-learn's data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html). Test evidence is recorded separately in `VALIDATION.md` and `test-results.xml`; none represents clinical performance.
