"""PyTorch thread limits.

Must be imported before any other torch work. PyTorch requires set_num_threads
to be called before eager, JIT or autograd code runs, and set_num_interop_threads
only once and before any inter-op work starts. Applying it as an import side
effect is therefore the reliable option.
"""

import torch

from defector.settings import settings

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return

    torch.set_num_threads(settings.torch_num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already set, or inter-op work has started. Not fatal.
        pass

    _applied = True


def describe() -> dict[str, int | bool]:
    """Runtime thread configuration, for diagnostics and the thesis environment table."""
    return {
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


apply()
