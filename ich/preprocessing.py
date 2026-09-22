"""Fixed HU windows to float32 RGB-like channels in [0,255]. No learned scaling."""
import numpy as np

WINDOWS = ((40.0, 80.0), (80.0, 200.0), (40.0, 380.0))
PREPROCESS_VERSION = "ct-hu-linear-windows-v1"


def dicom_to_hu(ds):
    if getattr(ds, "Modality", None) != "CT":
        raise ValueError("Only CT is supported")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1 or hasattr(ds, "PerFrameFunctionalGroupsSequence"):
        raise ValueError("Enhanced/multiframe CT requires a dedicated reader")
    if getattr(ds, "PhotometricInterpretation", "") not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Expected monochrome CT")
    if hasattr(ds, "ModalityLUTSequence"):
        raise ValueError("Modality LUT requires verified HU units; unsupported in this reader")
    if not hasattr(ds, "RescaleSlope") or not hasattr(ds, "RescaleIntercept"):
        raise ValueError("Missing HU rescale metadata")
    if str(getattr(ds, "RescaleType", "HU")).upper() != "HU":
        raise ValueError("Rescale units must be HU")
    slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
    if not np.isfinite([slope, intercept]).all() or slope == 0:
        raise ValueError("Invalid rescale parameters")
    # pydicom decodes signedness from PixelRepresentation. Never reinterpret uint8.
    pixels = ds.pixel_array
    if pixels.ndim != 2:
        raise ValueError("Expected one 2D frame")
    hu = pixels.astype(np.float32) * slope + intercept
    if hasattr(ds, "PixelPaddingValue"):
        lo, hi = sorted([float(ds.PixelPaddingValue), float(getattr(ds, "PixelPaddingRangeLimit", ds.PixelPaddingValue))])
        hu[(pixels >= lo) & (pixels <= hi)] = -1024.0
    if not np.isfinite(hu).all():
        raise ValueError("Nonfinite HU")
    # MONOCHROME1 is display polarity, not an instruction to invert physical HU.
    return hu


def window_image(hu, size=(224, 224)):
    from skimage.transform import resize
    hu = np.asarray(hu, dtype=np.float32)
    if hu.ndim != 2 or not np.isfinite(hu).all():
        raise ValueError("Expected finite 2D HU")
    channels = [np.clip((hu - (center-width/2))/width, 0, 1) for center, width in WINDOWS]
    result = resize(np.stack(channels, axis=-1), (*size, 3), preserve_range=True, anti_aliasing=True, mode="edge")
    return (result*255).astype(np.float32)


def load_image(path, size=(224, 224)):
    import pydicom
    return window_image(dicom_to_hu(pydicom.dcmread(path)), tuple(size))
