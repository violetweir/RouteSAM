"""Decode mask/probability PNGs without saturating uint16 consensus maps."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

def load_probability(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        mode,fmt=image.mode,image.format
        array=np.asarray(image)
    if array.dtype==np.uint16 or (fmt=='PNG' and mode in ('I','I;16','I;16B','I;16L')):
        result=array.astype(np.float32)/65535.
    elif np.issubdtype(array.dtype,np.floating):
        result=array.astype(np.float32)
    elif array.dtype==np.bool_:
        result=array.astype(np.float32)
    else:
        result=array.astype(np.float32)/255.
    if result.ndim!=2 or not np.isfinite(result).all():
        raise ValueError(f'Invalid probability map: {path}')
    return np.clip(result,0.,1.)
