from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from Bio.PDB import PDBParser, PPBuilder
except Exception:  # pragma: no cover
    PDBParser = PPBuilder = None  # type: ignore

# Try ANARCI for CDR annotation. If unavailable, we fall back to loop-only.
try:
    from anarci import run_anarci  # type: ignore
    _HAS_ANARCI = True
except Exception:  # pragma: no cover
    _HAS_ANARCI = False


@dataclass
class Ellipsoid:
    mu: np.ndarray            # (3,)
    Sigma: np.ndarray         # (3, 3)
    n: int                    # residue count
    feat: Dict[str, object]   # metadata (region, cdr, chain, interface, etc.)


@dataclass
class EllipsoidSet:
    """Container for a set of ellipsoids as a batch-friendly dict of arrays."""
    mu: np.ndarray            # (K, 3)
    Sigma: np.ndarray         # (K, 3, 3)
    n: np.ndarray             # (K,)
    feat: Dict[str, np.ndarray]  # str-> (K,) encoded ints or bools
    vocab: Dict[str, Dict[str, int]]  # categorical vocabularies

    def to_torch(self):
        import torch
        mu = torch.as_tensor(self.mu, dtype=torch.float32)
        Sigma = torch.as_tensor(self.Sigma, dtype=torch.float32)
        n = torch.as_tensor(self.n, dtype=torch.float32)
        feat_t = {k: torch.as_tensor(v, dtype=torch.long if v.dtype.kind in "iu" else torch.float32)
                  for k, v in self.feat.items()}
        return mu, Sigma, n, feat_t, self.vocab


def _compute_gaussian(coords: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mean/cov to coordinates of shape (M, 3)."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (M,3)")
    mu = coords.mean(axis=0)
    X = coords - mu
    Sigma = np.cov(X, rowvar=False)
    Sigma = Sigma + eps * np.eye(3, dtype=Sigma.dtype)
    return mu, Sigma


def _sequence_from_chain(chain) -> str:
    if PPBuilder is None:
        raise ImportError("Biopython not available. Install biopython to parse PDB files.")
    ppb = PPBuilder()
    seq = ""
    for pp in ppb.build_peptides(chain):
        seq += str(pp.get_sequence())
    return seq


def _anarci_cdr_ranges(seq: str, chain_type: str, scheme: str = "imgt") -> Dict[str, Tuple[int, int]]:
    """
    Return IMGT CDR ranges in *sequence index* (0-based, inclusive). Requires ANARCI.
    """
    if not _HAS_ANARCI:
        raise RuntimeError("ANARCI is not installed; cannot robustly annotate CDRs.")
    numbering, _, _ = run_anarci([('X', seq)], scheme=scheme, assign_germline=True)
    if not numbering or numbering[0][0] is None:
        raise RuntimeError("ANARCI failed to number sequence")
    per_res = numbering[0][0]  # list of tuples: ((pos, ins), aa, region)
    ranges = {}
    current, start = None, None
    for i, (_, aa, region) in enumerate(per_res):
        if region != current:
            if current in ('CDR1', 'CDR2', 'CDR3') and start is not None:
                ranges[current] = (start, i-1)
            current = region
            start = i if region in ('CDR1', 'CDR2', 'CDR3') else None
    if current in ('CDR1', 'CDR2', 'CDR3') and start is not None:
        ranges[current] = (start, len(per_res)-1)
    if chain_type == 'H':
        mapping = {'CDR1': 'H1', 'CDR2': 'H2', 'CDR3': 'H3'}
    else:
        mapping = {'CDR1': 'L1', 'CDR2': 'L2', 'CDR3': 'L3'}
    return {mapping[k]: v for k, v in ranges.items() if k in mapping}


def extract_interface_labels(antibody_atoms: List, antigen_atoms: List, thresh: float = 5.0):
    """Return set of residue ids at the interface (any heavy atom within `thresh` Å)."""
    from scipy.spatial import cKDTree
    ag_coords = np.array([atom.coord for atom in antigen_atoms], dtype=np.float32)
    tree = cKDTree(ag_coords) if len(ag_coords) else None
    interface_resids = set()
    if tree is None:
        return interface_resids
    for atom in antibody_atoms:
        d, _ = tree.query(atom.coord, k=1)
        if d <= thresh:
            res = atom.get_parent(); chain = res.get_parent()
            interface_resids.add((chain.id, res.id[1]))  # (chain_id, resseq)
    return interface_resids


def extract_antibody_ellipsoids(
    pdb_path: str,
    heavy_ids: Optional[List[str]] = None,
    light_ids: Optional[List[str]] = None,
    antigen_ids: Optional[List[str]] = None,
    scheme: str = "imgt",
    interface_thresh: float = 5.0,
    require_anarci: bool = False,
) -> EllipsoidSet:
    """
    Build ellipsoids for antibody regions (CDRs + non-CDR loops), with interface/non-interface labels.
    - If ANARCI is installed, uses IMGT CDRs. Otherwise, can run with require_anarci=False to skip CDRs.
    - Ellipsoids are fit to Cα coordinates for each region.
    """
    if PDBParser is None:
        raise ImportError("Biopython not available. Install biopython to parse PDB files.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = next(structure.get_models())

    chains = {ch.id: ch for ch in model.get_chains()}
    chain_ids = list(chains.keys())

    # Heuristics for chain IDs if not provided
    if heavy_ids is None or light_ids is None or antigen_ids is None:
        lengths = {cid: sum(1 for _ in chains[cid].get_residues()) for cid in chain_ids}
        sorted_ids = sorted(lengths, key=lengths.get)
        if heavy_ids is None or light_ids is None:
            ig_like = [cid for cid in sorted_ids if 80 <= lengths[cid] <= 150]
            if len(ig_like) >= 2:
                light_ids = [ig_like[0]] if light_ids is None else light_ids
                heavy_ids = [ig_like[1]] if heavy_ids is None else heavy_ids
            else:
                light_ids = [sorted_ids[0]] if light_ids is None else light_ids
                heavy_ids = [sorted_ids[1]] if heavy_ids is None else heavy_ids
        if antigen_ids is None:
            antigen_ids = [cid for cid in chain_ids if cid not in set((light_ids or []) + (heavy_ids or []))]

    def _atoms_of_chain(cid):
        return [a for a in chains[cid].get_atoms() if a.element != 'H']

    antigen_atoms = []
    for cid in antigen_ids or []:
        antigen_atoms.extend(_atoms_of_chain(cid))

    ellipsoids: List[Ellipsoid] = []
    vocab = {
        "region": {"cdr": 0, "loop": 1, "framework": 2},
        "cdr_type": {"None": 0, "H1": 1, "H2": 2, "H3": 3, "L1": 4, "L2": 5, "L3": 6},
        "chain": {"H": 0, "L": 1},
        "interface": {"False": 0, "True": 1},
    }

    def _add_region(coords: np.ndarray, chain_label: str, region: str, cdr_type: str):
        if len(coords) < 3:
            return
        mu, Sigma = _compute_gaussian(coords)
        # Interface label via min distance to antigen atoms
        at_interface = False
        if len(antigen_atoms) and coords.size:
            from scipy.spatial.distance import cdist
            ag = np.array([a.coord for a in antigen_atoms], dtype=np.float32)
            dmin = np.min(cdist(coords, ag)) if ag.size else np.inf
            at_interface = bool(dmin <= interface_thresh)
        ellipsoids.append(Ellipsoid(
            mu=mu, Sigma=Sigma, n=len(coords),
            feat={"region": region, "cdr_type": cdr_type, "chain": chain_label, "interface": at_interface}
        ))

    # Build per-chain ellipsoids
    for chain_label, ids in (("H", heavy_ids or []), ("L", light_ids or [])):
        for cid in ids:
            chain = chains[cid]
            cas, res_index_map = [], []
            for i, res in enumerate(chain.get_residues()):
                if 'CA' in res:
                    cas.append(res['CA'].coord)
                    res_index_map.append(i)
            cas = np.array(cas, dtype=np.float32)
            if cas.size == 0:
                continue

            cdr_ranges = {}
            if _HAS_ANARCI:
                try:
                    seq = _sequence_from_chain(chain)
                    cdr_ranges = _anarci_cdr_ranges(seq, chain_type=chain_label, scheme=scheme)
                except Exception as e:
                    if require_anarci:
                        raise
                    warnings.warn(f"ANARCI failed ({e}); proceeding without CDR ellipsoids.")

            used = np.zeros(len(res_index_map), dtype=bool)
            # CDR ellipsoids
            for cdr_name, (start, end) in cdr_ranges.items():
                idx = np.arange(start, end + 1)
                idx = idx[(idx >= 0) & (idx < len(res_index_map))]
                if idx.size == 0:
                    continue
                _add_region(cas[idx], chain_label=chain_label, region="cdr", cdr_type=cdr_name)
                used[idx] = True

            # Non-CDR loop chunks (simple contiguous segments of unused residues >=5)
            start = None
            for i, flag in enumerate(used):
                if not flag and start is None:
                    start = i
                if (flag or i == len(used) - 1) and start is not None:
                    end_idx = i if flag else i + 1
                    if end_idx - start >= 5:
                        _add_region(cas[start:end_idx], chain_label=chain_label, region="loop", cdr_type="None")
                    start = None

    if not ellipsoids:
        raise RuntimeError("No ellipsoids extracted; check chain ids or provide ANARCI.")

    # Pack
    K = len(ellipsoids)
    mu = np.stack([e.mu for e in ellipsoids], axis=0)
    Sigma = np.stack([e.Sigma for e in ellipsoids], axis=0)
    n = np.array([e.n for e in ellipsoids], dtype=np.int32)

    feat_arrays: Dict[str, np.ndarray] = {}
    for key in ("region", "cdr_type", "chain", "interface"):
        vocab_k = vocab[key]
        vals = np.array([e.feat[key] for e in ellipsoids], dtype=object)
        if key == "interface":
            enc = np.array([1 if v else 0 for v in vals], dtype=np.int64)
        else:
            enc = np.array([vocab_k[str(v)] for v in vals], dtype=np.int64)
        feat_arrays[key] = enc

    return EllipsoidSet(mu=mu, Sigma=Sigma, n=n, feat=feat_arrays, vocab=vocab)