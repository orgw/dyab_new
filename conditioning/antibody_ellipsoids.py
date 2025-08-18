from __future__ import annotations
import warnings
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppressing a common warning from Biopython PDB parsing
warnings.filterwarnings("ignore", "Ignoring unrecognized record 'END'", UserWarning)

try:
    from Bio.PDB import PDBParser
except Exception:  # pragma: no cover
    PDBParser = None  # type: ignore

# --- PyMOL for visualization (optional) ---
try:
    import pymol
    from pymol import cmd
    from pymol.cgo import BEGIN, END, TRIANGLE_STRIP, VERTEX, NORMAL, COLOR
    _HAS_PYMOL = True
except (ImportError, ModuleNotFoundError): # pragma: no cover
    _HAS_PYMOL = False
    # Define dummy classes if PyMOL is not available to avoid runtime errors on type hints
    cmd = None
    BEGIN = END = TRIANGLE_STRIP = VERTEX = NORMAL = COLOR = None


@dataclass
class Ellipsoid:
    """Represents a single ellipsoid fitted to a region of a protein."""
    mu: np.ndarray            # Centroid (3,)
    Sigma: np.ndarray         # Covariance matrix (3, 3)
    n: int                    # Number of residues in the region
    feat: Dict[str, object]   # Metadata (region, cdr, chain, etc.)


@dataclass
class EllipsoidSet:
    """Container for a set of ellipsoids as a batch-friendly dict of arrays."""
    mu: np.ndarray            # (K, 3)
    Sigma: np.ndarray         # (K, 3, 3)
    n: np.ndarray             # (K,)
    feat: Dict[str, np.ndarray]  # str -> (K,) encoded ints or bools
    vocab: Dict[str, Dict[str, int]]  # Categorical vocabularies for features

    def to_torch(self):
        """Converts the numpy arrays to PyTorch tensors."""
        import torch
        mu = torch.as_tensor(self.mu, dtype=torch.float32)
        Sigma = torch.as_tensor(self.Sigma, dtype=torch.float32)
        n = torch.as_tensor(self.n, dtype=torch.float32)
        feat_t = {k: torch.as_tensor(v, dtype=torch.long if v.dtype.kind in "iu" else torch.float32)
                  for k, v in self.feat.items()}
        return mu, Sigma, n, feat_t, self.vocab


def _compute_gaussian(coords: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Fits a Gaussian (mean and covariance) to a set of 3D coordinates."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be a (N, 3) array.")
    mu = coords.mean(axis=0)
    # Add a small epsilon to the diagonal for numerical stability (to ensure invertibility)
    Sigma = np.cov(coords, rowvar=False) + eps * np.eye(3, dtype=coords.dtype)
    return mu, Sigma

# (The core data extraction functions from the previous version are kept here without changes)
# ... extract_antibody_ellipsoids_from_json_data(...)
# ... process_pdb_from_json(...)
# --- For brevity, these functions are collapsed. The full code block below will contain them. ---

def create_cgo_ellipsoid(mu: np.ndarray, Sigma: np.ndarray, color: List[float], scale_factor: float = 1.5) -> List:
    """
    Creates a PyMOL CGO object for a single ellipsoid.
    Adapted from the reference code.
    """
    # Use eigenvalues and eigenvectors to define ellipsoid orientation and radii
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    radii = np.sqrt(np.maximum(eigenvalues, 1e-9)) * scale_factor

    # Generate points on a sphere
    u = np.linspace(0, 2 * np.pi, 48)  # Increase segments for smoother surface
    v = np.linspace(0, np.pi, 24)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and translate the sphere points to form the final ellipsoid
    points_rotated = np.dot(np.stack([x, y, z], axis=-1), eigenvectors.T)
    points_final = points_rotated + mu

    # Build the CGO object with triangle strips for a solid surface
    cgo_obj = [COLOR, *color]
    for i in range(len(u) - 1):
        cgo_obj.extend([BEGIN, TRIANGLE_STRIP])
        for j in range(len(v)):
            for k in [i, i + 1]:
                point = points_final[k, j]
                # Calculate normal vector for smooth shading
                normal = point - mu
                norm_mag = np.linalg.norm(normal)
                if norm_mag > 1e-6:
                    normal /= norm_mag
                cgo_obj.extend([NORMAL, *normal, VERTEX, *point])
        cgo_obj.append(END)
    return cgo_obj


def save_ellipsoids_to_pymol_session(
    ellipsoid_set: EllipsoidSet,
    pdb_path: str,
    heavy_ids: List[str],
    light_ids: List[str],
    antigen_ids: List[str],
    output_pse_path: str,
    transparency: float = 0.5,
):
    """
    Saves the antibody structure and its calculated ellipsoids to a PyMOL session file.
    """
    if not _HAS_PYMOL:
        print("\n[WARN] PyMOL library not found. Skipping visualization (.pse file creation).")
        print("To enable this, please install PyMOL (e.g., `conda install -c schrodinger pymol`).")
        return

    print(f"\n--- Creating PyMOL Session: {output_pse_path} ---")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_pse_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Reinitialize PyMOL to a clean state
    cmd.reinitialize()
    cmd.load(os.path.abspath(pdb_path), "antibody_complex")

    # Basic scene setup
    cmd.bg_color("white")
    cmd.hide("everything", "all")
    cmd.show("cartoon")
    cmd.util.cbag("antibody_complex") # Color by chain rainbow

    # Color specific chains for clarity
    if heavy_ids: cmd.color("lightblue", f"chain {'+'.join(heavy_ids)}")
    if light_ids: cmd.color("palecyan", f"chain {'+'.join(light_ids)}")
    if antigen_ids: cmd.color("lightorange", f"chain {'+'.join(antigen_ids)}")
        
    # Define colors for different ellipsoid categories
    colors = {
        'H_interface': [1.0, 0.2, 0.2],  # Red
        'L_interface': [1.0, 0.5, 0.2],  # Orange
        'H_framework': [0.2, 0.4, 1.0],  # Blue
        'L_framework': [0.4, 0.8, 1.0],  # Cyan
    }

    # Group CGOs by category for better organization in PyMOL
    cgo_groups = {
        'H_Interface_Ellipsoids': [], 'L_Interface_Ellipsoids': [],
        'H_Framework_Ellipsoids': [], 'L_Framework_Ellipsoids': []
    }

    # Unpack features from the EllipsoidSet
    feat = ellipsoid_set.feat
    is_heavy = (feat['chain'] == ellipsoid_set.vocab['chain']['H'])
    is_interface = (feat['interface'] == ellipsoid_set.vocab['interface']['True'])

    # Generate CGO for each ellipsoid
    for i in range(len(ellipsoid_set.mu)):
        if is_heavy[i]:
            category_key = 'H_interface' if is_interface[i] else 'H_framework'
            group_name = 'H_Interface_Ellipsoids' if is_interface[i] else 'H_Framework_Ellipsoids'
        else: # Light chain
            category_key = 'L_interface' if is_interface[i] else 'L_framework'
            group_name = 'L_Interface_Ellipsoids' if is_interface[i] else 'L_Framework_Ellipsoids'
            
        color = colors[category_key]
        cgo_obj = create_cgo_ellipsoid(ellipsoid_set.mu[i], ellipsoid_set.Sigma[i], color)
        cgo_groups[group_name].extend(cgo_obj)

    # Load CGOs into PyMOL
    loaded_groups = []
    for name, cgo_list in cgo_groups.items():
        if cgo_list:
            cmd.load_cgo(cgo_list, name)
            loaded_groups.append(name)
    
    # Group all ellipsoid objects together
    if loaded_groups:
        cmd.group("All_Ellipsoids", ' '.join(loaded_groups))
        cmd.set("cgo_transparency", transparency, "All_Ellipsoids")

    cmd.zoom("all")
    cmd.save(os.path.abspath(output_pse_path))
    print(f"[SUCCESS] Saved PyMOL session to {output_pse_path}")


# --- Full Script (including collapsed functions) ---

def extract_antibody_ellipsoids_from_json_data(
    pdb_path: str,
    heavy_ids: List[str],
    light_ids: List[str],
    antigen_ids: List[str],
    cdr_ranges: Dict[str, Tuple[int, int]],
    interface_thresh: float = 5.0,
) -> EllipsoidSet:
    if PDBParser is None:
        raise ImportError("Biopython is not installed. Please install it via `pip install biopython`.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = next(structure.get_models())
    chains = {ch.id: ch for ch in model.get_chains()}

    antigen_atoms = []
    for cid in antigen_ids:
        if cid in chains:
            antigen_atoms.extend([a for a in chains[cid].get_atoms() if a.element != 'H'])

    ellipsoids: List[Ellipsoid] = []
    vocab = { "region": {"cdr": 0, "loop": 1}, "cdr_type": {"None": 0, "H1": 1, "H2": 2, "H3": 3, "L1": 4, "L2": 5, "L3": 6}, "chain": {"H": 0, "L": 1}, "interface": {"False": 0, "True": 1}}

    def _add_region(coords: np.ndarray, chain_label: str, region: str, cdr_type: str):
        if len(coords) < 3: return
        mu, Sigma = _compute_gaussian(coords)
        at_interface = False
        if antigen_atoms:
            from scipy.spatial.distance import cdist
            ag_coords = np.array([a.coord for a in antigen_atoms], dtype=np.float32)
            if coords.size > 0 and ag_coords.size > 0:
                min_dist = np.min(cdist(coords, ag_coords))
                at_interface = bool(min_dist <= interface_thresh)
        ellipsoids.append(Ellipsoid(mu=mu, Sigma=Sigma, n=len(coords), feat={"region": region, "cdr_type": cdr_type, "chain": chain_label, "interface": at_interface}))

    for chain_label, ids in (("H", heavy_ids), ("L", light_ids)):
        for cid in ids:
            if cid not in chains:
                warnings.warn(f"Chain '{cid}' not found in PDB file '{pdb_path}'. Skipping.")
                continue
            chain = chains[cid]
            cas = np.array([res['CA'].coord for res in chain.get_residues() if 'CA' in res], dtype=np.float32)
            if cas.shape[0] == 0: continue
            used_residue_mask = np.zeros(len(cas), dtype=bool)
            for cdr_name, (start, end) in cdr_ranges.items():
                if cdr_name.startswith(chain_label):
                    if start < len(cas) and end < len(cas):
                        coords = cas[start : end + 1]
                        _add_region(coords, chain_label=chain_label, region="cdr", cdr_type=cdr_name)
                        used_residue_mask[start : end + 1] = True
                    else:
                        warnings.warn(f"CDR '{cdr_name}' range [{start}, {end}] is out of bounds for chain '{cid}' (length {len(cas)}).")
            start_idx = None
            for i, is_used in enumerate(np.append(used_residue_mask, True)):
                if not is_used and start_idx is None: start_idx = i
                elif is_used and start_idx is not None:
                    if i - start_idx >= 5:
                        coords = cas[start_idx:i]
                        _add_region(coords, chain_label=chain_label, region="loop", cdr_type="None")
                    start_idx = None
    if not ellipsoids:
        raise RuntimeError(f"No ellipsoids were extracted for PDB {pdb_path}. Check chain IDs and CDR ranges.")
    K = len(ellipsoids)
    feat_arrays: Dict[str, np.ndarray] = {}
    for key in vocab.keys():
        if key == "interface": vals = [e.feat[key] for e in ellipsoids]; feat_arrays[key] = np.array([1 if v else 0 for v in vals], dtype=np.int64)
        else: vocab_k = vocab[key]; vals = [str(e.feat[key]) for e in ellipsoids]; feat_arrays[key] = np.array([vocab_k[v] for v in vals], dtype=np.int64)
    return EllipsoidSet(mu=np.stack([e.mu for e in ellipsoids]), Sigma=np.stack([e.Sigma for e in ellipsoids]), n=np.array([e.n for e in ellipsoids]), feat=feat_arrays, vocab=vocab)


def process_pdb_from_json(pdb_id: str, json_path: str) -> Optional[Tuple[EllipsoidSet, Dict]]:
    pdb_id_lower = pdb_id.lower()
    with open(json_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['pdb'].lower() == pdb_id_lower:
                print(f"Found entry for PDB ID: {pdb_id}")
                cdr_ranges = {"H1": tuple(data["cdrh1_pos"]), "H2": tuple(data["cdrh2_pos"]), "H3": tuple(data["cdrh3_pos"]), "L1": tuple(data["cdrl1_pos"]), "L2": tuple(data["cdrl2_pos"]), "L3": tuple(data["cdrl3_pos"])}
                ellipsoid_set = extract_antibody_ellipsoids_from_json_data(
                    pdb_path=data["pdb_data_path"], heavy_ids=[data["heavy_chain"]], light_ids=[data["light_chain"]],
                    antigen_ids=data["antigen_chains"], cdr_ranges=cdr_ranges
                )
                return ellipsoid_set, data
    print(f"PDB ID '{pdb_id}' not found in {json_path}")
    return None


# --- Example Usage ---
if __name__ == "__main__":
    sabdab_json_path = "/nfsdata/home/kiwoong.yoo/workspace/dyAb_test/all_data/sabdab_all.json"
    target_pdb_id = "9ei9"
    output_dir = "./pymol_sessions"

    try:
        result = process_pdb_from_json(target_pdb_id, sabdab_json_path)
        if result:
            ellipsoid_set, pdb_data = result
            print(f"Successfully extracted {len(ellipsoid_set.mu)} ellipsoids for {target_pdb_id}.")
            
            # Now, create the PyMOL visualization
            save_ellipsoids_to_pymol_session(
                ellipsoid_set=ellipsoid_set,
                pdb_path=pdb_data["pdb_data_path"],
                heavy_ids=[pdb_data["heavy_chain"]],
                light_ids=[pdb_data["light_chain"]],
                antigen_ids=pdb_data["antigen_chains"],
                output_pse_path=os.path.join(output_dir, f"{target_pdb_id}_ellipsoids.pse"),
                transparency=0.4
            )

    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"An error occurred: {e}")