import numpy as np
import pandas as pd
import pytest
from ich import LABELS
from ich.dataset import split_manifest


@pytest.fixture
def manifest():
    rows = []
    for patient in range(20):
        for image in range(2):
            key = f"{patient}_{image}"
            labels = [int(patient % 2 == 0), 0, 0, 0, 0, int(patient % 2 == 0)]
            rows.append(dict(patient_id=f"p{patient}", study_id=f"study{patient}", series_id=f"series{patient}",
                sop_id=key, image_id=key, pixel_hash=key, path=key, **dict(zip(LABELS, labels))))
    return split_manifest(pd.DataFrame(rows))


@pytest.fixture
def dicom():
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Modality = "CT"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 2, 2, 1
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 1
    ds.RescaleSlope, ds.RescaleIntercept = 2, -1000
    ds.PixelData = np.array([[-100, 0], [500, 1000]], dtype="<i2").tobytes()
    return ds
