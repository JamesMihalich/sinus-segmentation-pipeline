"""
Amira file format converter.

Converts Amira binary files (.am) to NIfTI format (.nii).
Supports both label masks (RLE compressed uint8) and volume data (raw int16/uint8).
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# Token types
TOKEN_BYTEDATA = "bytedata"
TOKEN_NEWLINE = "newline"

# Regex patterns
RE_BYTEDATA_KEY = re.compile(rb"^@(\d+)$")
RE_WHITESPACE_SPLITTER = re.compile(rb'(".*")|[ \t\n]')


def _is_bytedata_key(data: bytes) -> bool:
    """Check if bytes match bytedata key pattern (@1, @2, etc.)."""
    return RE_BYTEDATA_KEY.match(data) is not None


def rle_decompress(buf: bytes) -> bytes:
    """
    Decompress Amira HxByteRLE encoded data.

    RLE encoding:
    - Control byte <= 127: Repeat next byte that many times
    - Control byte > 127: Copy next (control - 128) bytes literally
    - Control byte == 0: End of data

    Args:
        buf: RLE compressed bytes.

    Returns:
        Decompressed bytes.
    """
    result = []
    idx = 0
    buflen = len(buf)

    while idx < buflen:
        control_byte = buf[idx]
        idx += 1

        if control_byte == 0:
            break
        elif control_byte <= 127:
            # Repeat mode
            repeats = control_byte
            new_byte = buf[idx : idx + 1]
            idx += 1
            result.append(new_byte * repeats)
        else:
            # Literal mode
            num_bytes = control_byte - 128
            new_bytes = buf[idx : idx + num_bytes]
            idx += num_bytes
            result.append(new_bytes)

    return b"".join(result)


class AmiraTokenizer:
    """
    Tokenizer for Amira binary file format.

    Parses header information and extracts binary data blocks.
    """

    def __init__(self, fileobj) -> None:
        """
        Initialize tokenizer with file object.

        Args:
            fileobj: File object opened in binary mode.
        """
        self.buf = fileobj.read()
        self.defines: Dict[str, any] = {}
        self.data_type = np.uint8
        self.is_rle = False

    def _parse_header_line(self, line: bytes) -> None:
        """Parse a header line and extract definitions."""
        parts = RE_WHITESPACE_SPLITTER.split(line)
        parts = [p for p in parts if p and p.strip()]

        # Parse "define Lattice x y z"
        if len(parts) >= 5 and parts[0] == b"define" and parts[1] == b"Lattice":
            dims = [int(p) for p in parts[2:5]]
            self.defines["Lattice"] = dims

        # Parse "BoundingBox xmin xmax ymin ymax zmin zmax"
        if len(parts) >= 7 and parts[0] == b"BoundingBox":
            bounds = [float(p.replace(b",", b"")) for p in parts[1:7]]
            self.defines["BoundingBox"] = bounds

        # Detect data type from header
        if b"Lattice" in line and b"{" in line:
            if b"short" in line:
                self.data_type = np.int16
                self.is_rle = False
            elif b"byte" in line:
                self.data_type = np.uint8
                # Labels typically use RLE, volumes don't

    def _decode_binary_data(
        self,
        raw_buf: bytes,
        use_rle: bool = False,
    ) -> np.ndarray:
        """
        Decode binary data from buffer.

        Args:
            raw_buf: Raw binary buffer.
            use_rle: Whether to apply RLE decompression.

        Returns:
            Decoded numpy array reshaped to volume dimensions.
        """
        shape = self.defines.get("Lattice")
        if not shape:
            raise ValueError("No 'define Lattice' found in header")

        num_voxels = shape[0] * shape[1] * shape[2]

        # Decompress if needed
        if use_rle:
            raw_buf = rle_decompress(raw_buf)
            arr = np.frombuffer(raw_buf, dtype=np.uint8)
        else:
            # Calculate expected size
            bytes_per_voxel = np.dtype(self.data_type).itemsize
            expected_bytes = num_voxels * bytes_per_voxel

            # Truncate if file has extra data
            if len(raw_buf) > expected_bytes:
                raw_buf = raw_buf[:expected_bytes]

            # Force little endian
            dt = np.dtype(self.data_type).newbyteorder("<")
            arr = np.frombuffer(raw_buf, dtype=dt)

        # Handle size mismatches
        if arr.size != num_voxels:
            logger.warning(
                f"Size mismatch: expected {num_voxels}, got {arr.size}. "
                "Padding/truncating."
            )
            if arr.size > num_voxels:
                arr = arr[:num_voxels]
            else:
                arr = np.pad(arr, (0, num_voxels - arr.size), "constant")

        # Reshape: Amira stores Z, Y, X -> convert to X, Y, Z for NIfTI
        arr = arr.reshape((shape[2], shape[1], shape[0]))
        arr = np.swapaxes(arr, 0, 2)

        return arr

    def parse(self, file_type: str = "auto") -> Optional[np.ndarray]:
        """
        Parse the Amira file and extract volume data.

        Args:
            file_type: Type of file - "label" (RLE uint8), "volume" (raw),
                      or "auto" (detect from header).

        Returns:
            Volume data as numpy array, or None if parsing fails.
        """
        newline = b"\n"
        lineno = 0

        while len(self.buf) > 0:
            try:
                idx = self.buf.index(newline) + 1
            except ValueError:
                idx = len(self.buf)

            this_line = self.buf[:idx]
            rest_buf = self.buf[idx:]
            stripped = this_line.strip()

            # Parse header
            self._parse_header_line(this_line)

            # Check for binary data marker
            if _is_bytedata_key(stripped):
                # Determine if RLE should be used
                if file_type == "label":
                    use_rle = True
                elif file_type == "volume":
                    use_rle = False
                else:
                    # Auto-detect: uint8 labels typically use RLE
                    use_rle = self.data_type == np.uint8

                return self._decode_binary_data(rest_buf, use_rle=use_rle)

            self.buf = rest_buf
            lineno += 1

        return None

    def get_affine(self) -> np.ndarray:
        """
        Construct affine transformation matrix from header info.

        Returns:
            4x4 affine matrix.
        """
        dims = self.defines.get("Lattice", [1, 1, 1])
        bounds = self.defines.get(
            "BoundingBox", [0, dims[0], 0, dims[1], 0, dims[2]]
        )

        # Calculate spacing
        sx = (bounds[1] - bounds[0]) / dims[0] if dims[0] > 0 else 1.0
        sy = (bounds[3] - bounds[2]) / dims[1] if dims[1] > 0 else 1.0
        sz = (bounds[5] - bounds[4]) / dims[2] if dims[2] > 0 else 1.0

        # Origin
        ox, oy, oz = bounds[0], bounds[2], bounds[4]

        return np.array(
            [
                [sx, 0, 0, ox],
                [0, sy, 0, oy],
                [0, 0, sz, oz],
                [0, 0, 0, 1],
            ]
        )


class AmiraConverter:
    """
    Converter for Amira files to NIfTI format.

    Handles both label masks and volume data with automatic type detection.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize converter.

        Args:
            output_dir: Directory for output files. If None, saves alongside input.
        """
        self.output_dir = Path(output_dir) if output_dir else None

    def convert(
        self,
        filepath: Union[str, Path],
        file_type: Literal["label", "volume", "auto"] = "auto",
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Convert a single Amira file to NIfTI.

        Args:
            filepath: Path to input .am file.
            file_type: Type hint - "label" for RLE masks, "volume" for raw data,
                      "auto" to detect from header/filename.
            output_path: Optional explicit output path.

        Returns:
            Path to output NIfTI file, or None if conversion failed.
        """
        filepath = Path(filepath)
        logger.info(f"Converting: {filepath.name}")

        # Auto-detect file type from filename if needed
        if file_type == "auto":
            if "label" in filepath.name.lower():
                file_type = "label"
            else:
                file_type = "volume"

        try:
            with open(filepath, "rb") as f:
                tokenizer = AmiraTokenizer(f)
                data = tokenizer.parse(file_type=file_type)

                if data is None:
                    logger.error(f"Failed to parse {filepath.name}")
                    return None

                # Get affine matrix
                affine = tokenizer.get_affine()

                # Ensure correct dtype
                if file_type == "label":
                    data = data.astype(np.uint8)

                # Log info
                logger.info(f"  Shape: {data.shape}")
                logger.info(f"  Dtype: {data.dtype}")
                logger.info(f"  Range: [{data.min()}, {data.max()}]")

                # Determine output path
                if output_path is None:
                    if self.output_dir:
                        output_path = self.output_dir / filepath.with_suffix(".nii").name
                    else:
                        output_path = filepath.with_suffix(".nii")

                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save NIfTI
                nii = nib.Nifti1Image(data, affine)
                nib.save(nii, output_path)

                logger.info(f"  Saved: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Error converting {filepath.name}: {e}")
            return None

    def convert_directory(
        self,
        input_dir: Union[str, Path],
        pattern: str = "*.am",
        file_type: Literal["label", "volume", "auto"] = "auto",
    ) -> List[Path]:
        """
        Convert all matching Amira files in a directory.

        Args:
            input_dir: Directory containing .am files.
            pattern: Glob pattern for matching files.
            file_type: Type hint for all files.

        Returns:
            List of successfully converted output paths.
        """
        input_dir = Path(input_dir)
        files = sorted(input_dir.glob(pattern))

        logger.info(f"Found {len(files)} files matching '{pattern}'")

        results = []
        for filepath in files:
            output_path = self.convert(filepath, file_type=file_type)
            if output_path:
                results.append(output_path)

        logger.info(
            f"Successfully converted {len(results)}/{len(files)} files"
        )
        return results


def convert_amira_to_nifti(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    file_type: Literal["label", "volume", "auto"] = "auto",
) -> Optional[Path]:
    """
    Convenience function to convert a single Amira file.

    Args:
        input_path: Path to input .am file.
        output_path: Optional output path (defaults to same location with .nii).
        file_type: "label", "volume", or "auto".

    Returns:
        Path to output file, or None if failed.
    """
    converter = AmiraConverter()
    return converter.convert(
        input_path,
        file_type=file_type,
        output_path=Path(output_path) if output_path else None,
    )
