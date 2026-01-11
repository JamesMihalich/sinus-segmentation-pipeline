"""
3D mesh generation from segmentation masks.

Converts binary masks to STL mesh files for 3D visualization.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure

logger = logging.getLogger(__name__)


def create_stl_mesh(
    mask: np.ndarray,
    output_path: Union[str, Path],
    smooth_sigma: Optional[float] = 1.0,
    step_size: int = 1,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """
    Create STL mesh from binary mask.

    Uses marching cubes algorithm to extract surface mesh.

    Args:
        mask: Binary 3D mask.
        output_path: Path for output STL file.
        smooth_sigma: Gaussian smoothing sigma. None to skip smoothing.
        step_size: Step size for marching cubes (higher = faster, lower quality).
        spacing: Voxel spacing for correct mesh scaling.

    Returns:
        Path to saved STL file.

    Requires:
        numpy-stl package: pip install numpy-stl
    """
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        raise ImportError(
            "numpy-stl package required. Install with: pip install numpy-stl"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure binary
    mask = mask.astype(np.float32)

    # Optional smoothing
    if smooth_sigma is not None and smooth_sigma > 0:
        mask = gaussian_filter(mask, sigma=smooth_sigma)

    # Check for content
    if mask.max() == 0:
        logger.warning("Empty mask, cannot create mesh")
        raise ValueError("Mask is empty")

    # Marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        mask,
        level=0.5,
        step_size=step_size,
        spacing=spacing,
    )

    logger.info(f"Generated mesh: {len(verts)} vertices, {len(faces)} faces")

    # Create mesh
    mesh_obj = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))

    for i, face in enumerate(faces):
        for j in range(3):
            mesh_obj.vectors[i][j] = verts[face[j]]

    # Save
    mesh_obj.save(output_path)
    logger.info(f"Saved STL: {output_path}")

    return output_path


def convert_prediction_to_stl(
    npz_path: Union[str, Path],
    output_path: Union[str, Path],
    mask_key: str = "mask",
    smooth_sigma: float = 1.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Optional[Path]:
    """
    Convert prediction NPZ to STL mesh.

    Args:
        npz_path: Path to NPZ file containing mask.
        output_path: Output STL path.
        mask_key: Key for mask array in NPZ.
        smooth_sigma: Smoothing sigma.
        spacing: Voxel spacing.

    Returns:
        Path to STL file, or None if failed.
    """
    npz_path = Path(npz_path)

    try:
        with np.load(npz_path) as data:
            if mask_key in data:
                mask = data[mask_key]
            elif "label" in data:
                mask = data["label"]
            else:
                mask = data[data.files[0]]

        return create_stl_mesh(
            mask,
            output_path,
            smooth_sigma=smooth_sigma,
            spacing=spacing,
        )

    except Exception as e:
        logger.error(f"Error converting {npz_path.name}: {e}")
        return None


def batch_convert_to_stl(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*_prediction.npz",
    smooth_sigma: float = 1.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list:
    """
    Batch convert predictions to STL meshes.

    Args:
        input_dir: Directory with prediction NPZ files.
        output_dir: Output directory for STL files.
        pattern: Glob pattern for input files.
        smooth_sigma: Smoothing sigma.
        spacing: Voxel spacing.

    Returns:
        List of successfully created STL paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob(pattern))
    logger.info(f"Found {len(input_files)} files to convert")

    results = []
    for npz_path in input_files:
        stl_name = npz_path.stem.replace("_prediction", "") + ".stl"
        stl_path = output_dir / stl_name

        result = convert_prediction_to_stl(
            npz_path,
            stl_path,
            smooth_sigma=smooth_sigma,
            spacing=spacing,
        )

        if result:
            results.append(result)

    logger.info(f"Successfully converted {len(results)}/{len(input_files)} files")
    return results
