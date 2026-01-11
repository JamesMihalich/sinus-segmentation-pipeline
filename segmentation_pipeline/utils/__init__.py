"""Shared utility functions."""

from .io import load_npz, save_npz, load_nifti, save_nifti
from .volume_ops import crop_to_nonzero, pad_to_shape, get_bbox_slices
from .metadata import extract_nifti_metadata, generate_nifti_metadata_csv
from .naming import normalize_filename, extract_patient_id

__all__ = [
    # I/O
    "load_npz",
    "save_npz",
    "load_nifti",
    "save_nifti",
    # Volume operations
    "crop_to_nonzero",
    "pad_to_shape",
    "get_bbox_slices",
    # Metadata
    "extract_nifti_metadata",
    "generate_nifti_metadata_csv",
    # Naming
    "normalize_filename",
    "extract_patient_id",
]
