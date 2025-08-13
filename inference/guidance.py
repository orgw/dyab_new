from __future__ import annotations
from typing import Dict

import torch


def mix_guided_fields(
    fields_uncond: Dict[str, torch.Tensor],
    fields_cond: Dict[str, torch.Tensor],
    guidance_scale: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """Classifier-free guidance blending for dyAb-style flows.
    Linearly interpolate translation/rotation vector fields and sequence logits.
    """
    lam = guidance_scale
    out = {}
    for k in ("v_tr", "v_rot", "logits"):
        if k in fields_uncond and k in fields_cond and fields_uncond[k] is not None and fields_cond[k] is not None:
            out[k] = lam * fields_cond[k] + (1.0 - lam) * fields_uncond[k]
    return out
