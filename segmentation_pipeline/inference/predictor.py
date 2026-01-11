"""
Inference engine for 3D volume segmentation.

Provides sliding window inference with overlap averaging.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ..utils.io import load_npz_image, save_npz
from .postprocessing import postprocess_prediction

logger = logging.getLogger(__name__)


class VolumePredictor:
    """
    Sliding window predictor for 3D volumes.

    Handles:
    - Patch-based inference with overlap
    - Automatic padding for small volumes
    - Post-processing pipeline
    - Batch inference over directories
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cuda",
        patch_size: Tuple[int, int, int] = (224, 224, 256),
        overlap: float = 0.5,
        threshold: float = 0.5,
        apply_postprocessing: bool = True,
        keep_largest_component: bool = True,
    ) -> None:
        """
        Initialize predictor.

        Args:
            model: Trained segmentation model.
            device: Device to run inference on.
            patch_size: Size of patches for sliding window (D, H, W).
            overlap: Overlap fraction between patches (0.0 to 0.9).
            threshold: Probability threshold for binarization.
            apply_postprocessing: Whether to apply post-processing.
            keep_largest_component: Keep only largest connected component.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.patch_size = patch_size
        self.overlap = overlap
        self.threshold = threshold
        self.apply_postprocessing = apply_postprocessing
        self.keep_largest_component = keep_largest_component

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        **kwargs,
    ) -> "VolumePredictor":
        """
        Create predictor from saved checkpoint.

        Args:
            model: Model architecture (uninitialized weights OK).
            checkpoint_path: Path to checkpoint file.
            device: Device to load model on.
            **kwargs: Additional arguments for VolumePredictor.

        Returns:
            Initialized predictor.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

        return cls(model, device=device, **kwargs)

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """
        Preprocess volume for inference.

        Args:
            volume: Input volume.

        Returns:
            Normalized volume.
        """
        volume = volume.astype(np.float32)

        # Normalize to [0, 1] if needed
        if volume.max() > 1:
            volume = volume / 255.0

        # Z-score standardization
        if volume.std() > 0:
            volume = (volume - volume.mean()) / (volume.std() + 1e-6)

        return volume

    def _pad_volume(
        self,
        volume: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Pad volume to at least patch size.

        Args:
            volume: Input volume.

        Returns:
            Tuple of (padded_volume, padding_amounts).
        """
        D, H, W = volume.shape
        pD, pH, pW = self.patch_size

        pad_d = max(0, pD - D)
        pad_h = max(0, pH - H)
        pad_w = max(0, pW - W)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(
                volume,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant",
            )

        return volume, (pad_d, pad_h, pad_w)

    def _compute_starts(
        self,
        dim_size: int,
        patch_size: int,
        stride: int,
    ) -> List[int]:
        """Compute starting positions for sliding window."""
        starts = list(range(0, dim_size - patch_size + 1, stride))

        # Ensure last patch covers the end
        if len(starts) == 0 or starts[-1] + patch_size < dim_size:
            starts.append(max(0, dim_size - patch_size))

        return starts

    def predict_sliding_window(
        self,
        volume: np.ndarray,
    ) -> np.ndarray:
        """
        Run sliding window inference on a volume.

        Args:
            volume: Preprocessed volume (D, H, W).

        Returns:
            Probability map (D, H, W).
        """
        # Pad if needed
        padded_volume, padding = self._pad_volume(volume)
        D, H, W = padded_volume.shape
        pD, pH, pW = self.patch_size

        # Compute strides
        stride_d = int(pD * (1 - self.overlap))
        stride_h = int(pH * (1 - self.overlap))
        stride_w = int(pW * (1 - self.overlap))

        # Initialize accumulators
        probability_map = torch.zeros((D, H, W), dtype=torch.float32)
        count_map = torch.zeros((D, H, W), dtype=torch.float32)

        # Compute patch positions
        d_starts = self._compute_starts(D, pD, stride_d)
        h_starts = self._compute_starts(H, pH, stride_h)
        w_starts = self._compute_starts(W, pW, stride_w)

        with torch.no_grad():
            for d in d_starts:
                for h in h_starts:
                    for w in w_starts:
                        # Extract patch
                        patch = padded_volume[
                            d : d + pD,
                            h : h + pH,
                            w : w + pW,
                        ]

                        # Convert to tensor
                        input_tensor = (
                            torch.from_numpy(patch)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self.device)
                        )

                        # Forward pass
                        with torch.autocast(
                            device_type="cuda",
                            enabled=torch.cuda.is_available(),
                        ):
                            logits = self.model(input_tensor)
                            probs = torch.sigmoid(logits)

                        # Accumulate
                        probs_cpu = probs.squeeze().cpu()
                        probability_map[d : d + pD, h : h + pH, w : w + pW] += probs_cpu
                        count_map[d : d + pD, h : h + pH, w : w + pW] += 1.0

        # Average overlapping regions
        final_probs = probability_map / count_map

        # Remove padding
        if any(p > 0 for p in padding):
            pad_d, pad_h, pad_w = padding
            final_probs = final_probs[
                : D - pad_d if pad_d > 0 else D,
                : H - pad_h if pad_h > 0 else H,
                : W - pad_w if pad_w > 0 else W,
            ]

        return final_probs.numpy()

    def predict_single(
        self,
        volume: np.ndarray,
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict on a single volume.

        Args:
            volume: Input volume (D, H, W).
            return_probabilities: If True, also return probability map.

        Returns:
            Binary mask, or (mask, probabilities) if return_probabilities=True.
        """
        # Preprocess
        preprocessed = self.preprocess(volume)

        # Run sliding window
        probabilities = self.predict_sliding_window(preprocessed)

        # Post-process
        if self.apply_postprocessing:
            mask = postprocess_prediction(
                probabilities,
                threshold=self.threshold,
                keep_largest=self.keep_largest_component,
            )
        else:
            mask = (probabilities > self.threshold).astype(np.uint8)

        if return_probabilities:
            return mask, probabilities
        return mask

    def predict_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_probabilities: bool = True,
    ) -> Optional[Path]:
        """
        Predict on a single NPZ file.

        Args:
            input_path: Path to input NPZ file.
            output_path: Optional output path. If None, creates *_prediction.npz.
            save_probabilities: Save probability map alongside mask.

        Returns:
            Output path if successful, None otherwise.
        """
        input_path = Path(input_path)

        if output_path is None:
            stem = input_path.stem.replace("_cropped_img", "")
            output_path = input_path.parent / f"{stem}_prediction.npz"
        output_path = Path(output_path)

        try:
            # Load
            volume = load_npz_image(input_path)

            # Predict
            mask, probabilities = self.predict_single(
                volume, return_probabilities=True
            )

            # Save
            if save_probabilities:
                save_npz(
                    output_path,
                    mask=mask,
                    probabilities=probabilities,
                )
            else:
                save_npz(output_path, mask=mask)

            logger.info(f"Saved prediction: {output_path.name}")
            return output_path

        except Exception as e:
            logger.error(f"Error processing {input_path.name}: {e}")
            return None

    def predict_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.npz",
        exclude_pattern: str = "prediction",
        save_probabilities: bool = True,
        show_progress: bool = True,
    ) -> List[Path]:
        """
        Batch predict on all files in a directory.

        Args:
            input_dir: Input directory.
            output_dir: Output directory. If None, saves alongside inputs.
            pattern: Glob pattern for input files.
            exclude_pattern: Skip files containing this string.
            save_probabilities: Save probability maps.
            show_progress: Show progress bar.

        Returns:
            List of successfully processed output paths.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else None

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Find files
        all_files = list(input_dir.glob(pattern))
        input_files = [f for f in all_files if exclude_pattern not in f.name]

        logger.info(f"Found {len(input_files)} files to process")

        results = []
        iterator = input_files
        if show_progress:
            iterator = tqdm(input_files, desc="Inference")

        for input_path in iterator:
            if output_dir:
                stem = input_path.stem.replace("_cropped_img", "")
                output_path = output_dir / f"{stem}_prediction.npz"
            else:
                output_path = None

            result = self.predict_file(
                input_path,
                output_path=output_path,
                save_probabilities=save_probabilities,
            )

            if result:
                results.append(result)

        logger.info(f"Processed {len(results)}/{len(input_files)} files")
        return results
