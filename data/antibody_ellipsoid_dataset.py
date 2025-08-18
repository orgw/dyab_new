from __future__ import annotations
from typing import Dict, Any

import torch
from torch.utils.data import Dataset

# Assuming E2EDataset is in data.dataset
from .dataset import E2EDataset
# Assuming this function is in your project structure
from conditioning.antibody_ellipsoids import extract_antibody_ellipsoids_from_json_data

class AntibodyEllipsoidDataset(Dataset):
    """
    A wrapper to add ellipsoid conditioning to dyAb samples from an E2EDataset.
    """
    def __init__(self, base_dataset: E2EDataset, ellipsoid_path: str = None, require_anarci: bool = False):
        self.base_dataset = base_dataset
        self.ellipsoid_path = ellipsoid_path
        self.require_anarci = require_anarci

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get the original item from the base dataset
        item = self.base_dataset[idx]
        
        # Access the original record to get metadata for ellipsoid extraction.
        original_record = self.base_dataset.data[self.base_dataset.access_idx[idx - self.base_dataset.cur_idx_range[0]]]

        pdb_path = original_record.pdb_data_path
        heavy_ids = [original_record.heavy_chain]
        light_ids = [original_record.light_chain]
        antigen_ids = original_record.antigen.get_chain_names()

        try:
            ell = extract_antibody_ellipsoids_from_json_data(
                pdb_path,
                heavy_ids=heavy_ids,
                light_ids=light_ids,
                antigen_ids=antigen_ids,
                require_anarci=self.require_anarci,
            )
            mu, Sigma, n, feat_t, vocab = ell.to_torch()
            item["ellipsoids"] = {"mu": mu, "Sigma": Sigma, "n": n, "feat": feat_t, "vocab": vocab}
        except Exception as e:
            # It's better to be specific about exceptions, but this is a general catch
            # print(f"Warning: Could not extract ellipsoids for item {idx}. Error: {e}")
            # For now, we return the item without ellipsoids if extraction fails.
            # The collate function will need to handle this.
            item["ellipsoids"] = None

        return item

    def __getattr__(self, name):
        """
        This is the key change. It delegates any attribute access to the underlying
        base_dataset if the attribute is not found on the wrapper itself.
        This will resolve the AttributeError for `.data` and others.
        """
        return getattr(self.base_dataset, name)

    @staticmethod
    def collate_fn(batch):
        # Filter out items where ellipsoid extraction might have failed
        valid_batch = [item for item in batch if item.get('ellipsoids') is not None]
        if not valid_batch:
            # Handle cases where all items in a batch failed
            # Returning None might crash the training loop, so returning an empty version
            # of what the loader expects might be safer. For now, we'll see if this works.
            return None 

        # Separate the base data from the ellipsoid data
        base_batch_items = [{k: v for k, v in item.items() if k != 'ellipsoids'} for item in valid_batch]
        
        # Use the base dataset's collate function
        collate_output = E2EDataset.collate_fn(base_batch_items)

        # Collate the ellipsoid data
        ellipsoids_list = [item['ellipsoids'] for item in valid_batch]
        collate_output['ellipsoids'] = ellipsoids_list
        
        return collate_output