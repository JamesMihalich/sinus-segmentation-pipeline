"""
Inference engine for bounding box prediction.

Provides prediction on new volumes with preprocessing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from ..data.preprocessing.windowing import apply_ct_window, resize_volume
from ..utils.bbox_utils import denormalize_bbox
from ..utils.io import load_npz_image, save_npz

logger = logging.getLogger(__name__)


class BBoxPredictor:
    """
    Predictor for 3D bounding box localization.

    Handles preprocessing, model inference, and post-processing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cuda",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        window_level: Optional[float] = None,
        window_width: Optional[float] = None,
    ) -> None:
        """
        Initialize predictor.

        Args:
            model: Trained localization model.
            device: Device to run inference on.
            target_shape: Volume size expected by model.
            window_level: CT window level (None to skip windowing).
            window_width: CT window width.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.target_shape = target_shape
        self.window_level = window_level
        self.window_width = window_width

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        **kwargs,
    ) -> "BBoxPredictor":
        """
        Create predictor from saved checkpoint.

        Args:
            model: Model architecture (uninitialized weights OK).
            checkpoint_path: Path to checkpoint file.
            device: Device to load model on.
            **kwargs: Additional arguments for BBoxPredictor.

        Returns:
            Initialized predictor.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return cls(model, device=device, **kwargs)

    def preprocess(
        self,
        volume: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Preprocess volume for inference.

        Args:
            volume: Input volume.

        Returns:
            Tuple of (preprocessed volume, original shape).
        """
        original_shape = volume.shape

        # Apply CT windowing if specified
        if self.window_level is not None and self.window_width is not None:
            volume = apply_ct_window(
                volume,
                self.window_level,
                self.window_width,
            )

        # Resize to target shape
        if volume.shape != self.target_shape:
            volume = resize_volume(volume, self.target_shape)

        # Normalize to [0, 1]
        volume = volume.astype(np.float32)
        if volume.max() > 1:
            volume = volume / 255.0

        return volume, original_shape

    def predict_single(
        self,
        volume: np.ndarray,
        return_normalized: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict bounding box on a single volume.

        Args:
            volume: Input volume (D, H, W).
            return_normalized: If True, also return normalized coordinates.

        Returns:
            Absolute bbox coordinates [z1, y1, x1, z2, y2, x2].
            If return_normalized=True, returns (absolute, normalized) tuple.
        """
        # Preprocess
        preprocessed, original_shape = self.preprocess(volume)

        # Convert to tensor
        input_tensor = (
            torch.from_numpy(preprocessed)
            .unsqueeze(0)  # Batch dimension
            .unsqueeze(0)  # Channel dimension
            .to(self.device)
        )

        # Forward pass
        with torch.no_grad():
            normalized_bbox = self.model(input_tensor)

        # Convert to numpy
        normalized_bbox = normalized_bbox.squeeze().cpu().numpy()

        # Convert to absolute coordinates
        absolute_bbox = denormalize_bbox(normalized_bbox, original_shape)

        if return_normalized:
            return absolute_bbox, normalized_bbox
        return absolute_bbox

    def predict_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Predict on a single NPZ file.

        Args:
            input_path: Path to input NPZ file.
            output_path: Optional output path for results.

        Returns:
            Dictionary with prediction results.
        """
        input_path = Path(input_path)

        # Load volume
        volume = load_npz_image(input_path)

        # Predict
        absolute_bbox, normalized_bbox = self.predict_single(
            volume, return_normalized=True
        )

        result = {
            "input_file": str(input_path),
            "normalized_bbox": normalized_bbox.tolist(),
            "absolute_bbox": absolute_bbox.tolist(),
            "original_shape": list(volume.shape),
        }

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            save_npz(
                output_path,
                predicted_bbox=absolute_bbox,
                normalized_bbox=normalized_bbox,
                original_shape=np.array(volume.shape),
            )
            result["output_file"] = str(output_path)

        return result

    def predict_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.npz",
    ) -> List[Dict]:
        """
        Batch predict on all files in a directory.

        Args:
            input_dir: Input directory.
            output_dir: Output directory for results.
            pattern: Glob pattern for input files.

        Returns:
            List of prediction result dictionaries.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        input_files = sorted(input_dir.glob(pattern))
        logger.info(f"Found {len(input_files)} files to process")

        results = []
        for input_path in input_files:
            output_path = None
            if output_dir:
                output_path = output_dir / f"{input_path.stem}_prediction.npz"

            try:
                result = self.predict_file(input_path, output_path)
                results.append(result)
                logger.info(f"Processed: {input_path.name}")
            except Exception as e:
                logger.error(f"Error processing {input_path.name}: {e}")

        return results


def predict_from_nifti(
    predictor: BBoxPredictor,
    nifti_path: Union[str, Path],
) -> Dict:
    """
    Predict bounding box directly from NIfTI file.

    Args:
        predictor: Initialized BBoxPredictor.
        nifti_path: Path to NIfTI file.

    Returns:
        Prediction result dictionary.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    nifti_path = Path(nifti_path)

    # Load NIfTI
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()

    # Predict
    absolute_bbox, normalized_bbox = predictor.predict_single(
        volume, return_normalized=True
    )

    return {
        "input_file": str(nifti_path),
        "normalized_bbox": normalized_bbox.tolist(),
        "absolute_bbox": absolute_bbox.tolist(),
        "original_shape": list(volume.shape),
    }
