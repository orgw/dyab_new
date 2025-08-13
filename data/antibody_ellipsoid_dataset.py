from __future__ import annotations
from typing import Dict, Any, List

from torch.utils.data import Dataset

from ..conditioning.antibody_ellipsoids import extract_antibody_ellipsoids


class AntibodyEllipsoidDataset(Dataset):
    """
    Thin wrapper that augments dyAb samples with ellipsoid conditioning.
    Items must include at least: {"pdb_path", optional chain ids: heavy_ids, light_ids, antigen_ids}
    """
    def __init__(self, items: List[Dict[str, Any]], require_anarci: bool = False):
        self.items = items
        self.require_anarci = require_anarci

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.items[idx])
        ell = extract_antibody_ellipsoids(
            item["pdb_path"],
            heavy_ids=item.get("heavy_ids"),
            light_ids=item.get("light_ids"),
            antigen_ids=item.get("antigen_ids"),
            require_anarci=self.require_anarci,
        )
        mu, Sigma, n, feat_t, vocab = ell.to_torch()
        item["ellipsoids"] = {"mu": mu, "Sigma": Sigma, "n": n, "feat": feat_t, "vocab": vocab}
        return item
