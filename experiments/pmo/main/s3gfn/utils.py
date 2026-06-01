"""Small tensor helpers used by the PMO S3-GFN adapter."""

import numpy as np
import torch


def unique(rows: torch.Tensor) -> torch.Tensor:
    """Return indices of unique rows while preserving their original order."""
    _, indices = np.unique(rows.detach().cpu().numpy(), axis=0, return_index=True)
    return torch.as_tensor(np.sort(indices), dtype=torch.long, device=rows.device)
