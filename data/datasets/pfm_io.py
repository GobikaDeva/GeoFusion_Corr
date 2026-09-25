"""Minimal .pfm reader/writer for DTU ground-truth depth maps.

DTU's Depths/scan{}_train/depth_map_{:04d}.pfm files are single-channel (grayscale)
PFM, written by the original DTU MVS toolkit. This avoids pulling in a heavier image
library just for this one format.
"""
from __future__ import annotations

import numpy as np


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")

        dims_line = f.readline().decode("utf-8").rstrip()
        while dims_line.startswith("#"):  # skip comments
            dims_line = f.readline().decode("utf-8").rstrip()
        width, height = map(int, dims_line.split())

        scale = float(f.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM stores bottom-to-top
        return data.astype(np.float32) * scale


def write_pfm(path: str, image: np.ndarray, scale: float = 1.0) -> None:
    with open(path, "wb") as f:
        color = image.ndim == 3 and image.shape[2] == 3
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("utf-8"))
        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and np.little_endian):
            scale = -scale
        f.write(f"{scale}\n".encode("utf-8"))
        np.flipud(image).astype(np.float32).tofile(f)
