"""Write small synthetic DICOM series for tests (needs pydicom and numpy)."""

from pathlib import Path

import numpy as np


def write_series(folder: Path, patient_id: str, seed: int = 0, manufacturer: str = "SIEMENS",
                 model: str = "Skyra", n_slices: int = 4, rows: int = 16, cols: int = 20,
                 spacing=(0.8, 0.8), thickness: float = 2.0, pixels=None):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vol = pixels if pixels is not None else rng.integers(0, 1000, (n_slices, rows, cols)).astype(np.int16)
    study, series = generate_uid(), generate_uid()
    for k in range(vol.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientID = patient_id
        ds.PatientName = "Test^Phantom"
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.Modality = "MR"
        ds.Manufacturer = manufacturer
        ds.ManufacturerModelName = model
        ds.MagneticFieldStrength = 3.0
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[2]
        ds.PixelSpacing = list(spacing)
        ds.SliceThickness = thickness
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-10.0, -12.0, k * thickness]
        ds.InstanceNumber = k + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = vol[k].astype(np.int16).tobytes()
        # write slices out of order: the reader must sort by position
        ds.save_as(str(folder / f"slice_{(k * 7) % vol.shape[0]:03d}.dcm"), enforce_file_format=True)
    return vol
