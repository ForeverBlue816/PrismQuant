"""PrismQuant: quantizer-aware anchor rotations for low-bit language models."""
from .calibration import fit_rotation
from .hub import download_checkpoint, list_models
from .quantization import dynamic_asym_int4
from .rotations import RotationFactor, WYFactor, compact_wy

__version__ = '0.2.0'
__all__ = ['fit_rotation', 'download_checkpoint', 'list_models', 'load_model',
           'dynamic_asym_int4', 'RotationFactor', 'WYFactor', 'compact_wy']


def load_model(*args, **kwargs):
    """Load a released checkpoint; see :func:`prismquant.model.load_model`."""
    from .model import load_model as load
    return load(*args, **kwargs)
