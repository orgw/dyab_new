#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
This script generates ellipsoid representations for antibody-antigen complexes.

It reads a summary JSON file (like the one produced by data/download.py),
iterates through each PDB entry, calculates the interface and secondary
structures, segments the complex into ellipsoids, and saves the resulting
ellipsoid data to a new JSON file.
"""
import os
import json
import argparse
import io
import multiprocessing
from tqdm import tqdm

# --- Core Scientific Libraries ---
import numpy as np
import networkx as nx
import pydssp
from scipy.spatial.distance import cdist

# --- Imports from dyAb project ---
from data.pdb_utils import AgAbComplex, Protein, Peptide
from utils.logger import print_log

# --- Constants ---
DSSP_SS_MAP = {
    0: 'loop', 1: 'alpha', 2: 'beta', 3: 'beta',
    4: 'alpha', 5: 'alpha', 6: 'loop', 7: 'loop',
}

# #############################################################
# ##### CORE ELLIPSOID GENERATION FUNCTIONS #####
# #############################################################

def get_all_atom_positions(chain_obj):
    """
    Extracts all atom positions for each residue from a Peptide object.
    Returns a list of numpy arrays, where each array holds the coordinates
    of atoms for a single residue.
    """
    if not chain_obj:
        return []

    all_pos = []
    for res in chain_obj:
        # This creates a numpy array of coordinates for each residue
        coords = [atom_coord for atom_name, atom_coord in res]
        all_pos.append(np.array(coords))

    return all_pos

def get_residue_indices(chain_obj):
    """Extracts residue indices (the integer part of the residue ID)."""
    return [res.get_id()[0] for res in chain_obj] if chain_obj else []

def calculate_interface_residues(heavy_obj, light_obj, antigen_obj, distance_cutoff=4.5):
    """
    Identifies epitope and paratope residues based on a distance cutoff.
    """
    if not all([heavy_obj, light_obj, antigen_obj]):
        return set(), set(), set()

    # Extract atom positions for all chains
    h_pos = get_all_atom_positions(heavy_obj)
    l_pos = get_all_atom_positions(light_obj)
    ag_pos = get_all_atom_positions(antigen_obj)

    # Get residue numbers for mapping back
    heavy_res_indices = get_residue_indices(heavy_obj)
    light_res_indices = get_residue_indices(light_obj)
    antigen_res_indices = get_residue_indices(antigen_obj)

    # Create a map to trace a residue index back to its original chain and number
    num_heavy_res = len(heavy_obj)
    ab_res_map = [('H', heavy_res_indices[i]) for i in range(num_heavy_res)] + \
                 [('L', light_res_indices[i]) for i in range(len(light_obj))]
    ab_pos = h_pos + l_pos

    num_ab_res, num_ag_res = len(ab_pos), len(ag_pos)
    res_dist_mat = np.full((num_ab_res, num_ag_res), np.inf)

    # Calculate residue-residue distance matrix
    for i in range(num_ab_res):
        ab_res_atoms = ab_pos[i]
        if ab_res_atoms.shape[0] == 0: continue
        for j in range(num_ag_res):
            ag_res_atoms = ag_pos[j]
            if ag_res_atoms.shape[0] == 0: continue
            min_dist = cdist(ab_res_atoms, ag_res_atoms).min()
            res_dist_mat[i, j] = min_dist
            
    # Find interface pairs below the distance cutoff
    interface_ab_indices, interface_ag_indices = np.where(res_dist_mat < distance_cutoff)

    # Compile sets of unique PDB residue numbers
    paratope_H_res_nums, paratope_L_res_nums = set(), set()
    for idx in set(interface_ab_indices):
        chain_type, res_num = ab_res_map[idx]
        if chain_type == 'H':
            paratope_H_res_nums.add(res_num)
        else:
            paratope_L_res_nums.add(res_num)
            
    epitope_res_nums = {antigen_res_indices[i] for i in set(interface_ag_indices)}

    return paratope_H_res_nums, paratope_L_res_nums, epitope_res_nums

def get_secondary_structure(chain_obj):
    """
    Calculates secondary structure for a given Peptide object using pydssp.assign,
    which works directly with coordinates. This is more robust than using pydssp.DSSP.
    """
    if chain_obj is None or len(chain_obj) == 0:
        return np.array([], dtype=int)

    coords = []
    valid_res_indices = []
    for i, res in enumerate(chain_obj):
        try:
            # DSSP requires N, CA, C, O atoms.
            n_coord = res.get_coord('N')
            ca_coord = res.get_coord('CA')
            c_coord = res.get_coord('C')
            o_coord = res.get_coord('O')
            coords.append([n_coord, ca_coord, c_coord, o_coord])
            valid_res_indices.append(i)
        except KeyError:
            # Residue is missing a backbone atom, skip it for DSSP calculation.
            continue
    
    if not coords:
        # No residues with a complete backbone were found.
        return np.full(len(chain_obj), 6, dtype=int) # Default all to loop

    coords = np.array(coords)
    
    try:
        # Run DSSP on the extracted backbone coordinates
        dssp_results = pydssp.assign(coords, out_type="index")
        
        # Create a full-length array and populate it with DSSP results,
        # ensuring residues that were skipped get a 'loop' assignment.
        full_dssp = np.full(len(chain_obj), 6, dtype=int) # Default all to loop
        for i, res_idx in enumerate(valid_res_indices):
            full_dssp[res_idx] = dssp_results[i]
        return full_dssp
    except Exception as e:
        print_log(f"pydssp.assign failed for chain {chain_obj.get_id()}: {e}", level='WARN')
        return np.full(len(chain_obj), 6, dtype=int) # Default to loop

def get_ca_positions_and_mask(chain_obj):
    """Extracts C-alpha positions and a mask of their existence."""
    ca_pos = []
    mask = []
    if not chain_obj:
        return np.array([]), np.array([], dtype=bool)
    for res in chain_obj:
        if 'CA' in res.coordinate:
            ca_pos.append(res.get_coord('CA'))
            mask.append(True)
        else:
            ca_pos.append([0.0, 0.0, 0.0]) # Placeholder
            mask.append(False)
    return np.array(ca_pos), np.array(mask, dtype=bool)

def segment_and_label_ellipsoids(chain_obj, dssp_indices, chain_type, interface_residues, radius_thresh=5.0, size_thresh=3):
    """
    Segments a chain into ellipsoids based on secondary structure and proximity.
    """
    if chain_obj is None or len(dssp_indices) == 0:
        return []
        
    ca_pos, ca_mask = get_ca_positions_and_mask(chain_obj)
    valid_indices = np.where(ca_mask)[0]
    if len(valid_indices) < 2: return []

    distmat = cdist(ca_pos[valid_indices], ca_pos[valid_indices])
    
    G = nx.Graph()
    for i_idx, i in enumerate(valid_indices):
        for j_idx in range(i_idx + 1, len(valid_indices)):
            j = valid_indices[j_idx]
            # Connect nodes if they have the same SS type and are close enough
            if distmat[i_idx, j_idx] < radius_thresh and dssp_indices[i] == dssp_indices[j]:
                G.add_edge(i, j)

    ellipsoids = []
    residue_indices = get_residue_indices(chain_obj)
    for component_indices in nx.connected_components(G):
        component_indices = list(component_indices)
        if len(component_indices) < size_thresh:
            continue
            
        blob_pos = ca_pos[component_indices]
        covar = np.cov(blob_pos.T) if blob_pos.shape[0] > 1 else np.identity(3) * 1e-9
        covar += 1e-6 * np.identity(3) # Add epsilon for numerical stability
        
        component_dssp = dssp_indices[component_indices]
        # Get the most common SS in the component
        dominant_ss_code = np.bincount(component_dssp).argmax()
        ss_type = DSSP_SS_MAP.get(dominant_ss_code, 'loop')
        
        # Determine if the ellipsoid is part of the interface
        is_interface = any(residue_indices[res_idx] in interface_residues for res_idx in component_indices)
        
        # Assign final label based on chain type and interface status
        if chain_type in ['H', 'L']:
            final_label = 'Paratope' if is_interface else 'Framework'
        elif chain_type == 'Ag':
            final_label = 'Epitope' if is_interface else 'Non-Epitope'
        else:
            final_label = 'Unknown'

        chain_tag = "Antigen" if chain_type == 'Ag' else ('VH' if chain_type == 'H' else 'VL')
        
        ellipsoids.append({
            "pos": blob_pos.mean(0).tolist(),
            "covar": covar.tolist(),
            "type": ss_type,
            "label": final_label,
            "chain_tag": chain_tag,
            "residue_count": len(component_indices)
        })
    return ellipsoids

def process_entry(item, out_dir):
    """
    Main processing function for a single PDB entry.
    """
    pdb_id = item['pdb']
    pdb_path = item['pdb_data_path']
    ellipsoid_path = os.path.join(out_dir, f'{pdb_id}_ellipsoids.json')

    # Skip if already processed
    if os.path.exists(ellipsoid_path):
        return f"Skipped {pdb_id}, already exists."

    try:
        cplx = AgAbComplex.from_pdb(
            pdb_path, item['heavy_chain'], item['light_chain'],
            item['antigen_chains'], skip_validity_check=True)

        heavy_obj = cplx.get_heavy_chain()
        light_obj = cplx.get_light_chain()
        antigen_obj = cplx.get_antigen()

        # Flatten antigen chains into a single Peptide object for easier processing
        ag_residues = []
        for _, chain in antigen_obj:
            ag_residues.extend(chain.residues)
        flat_antigen_chain = Peptide('A', ag_residues) # Use a dummy chain ID

        # 1. Calculate Interface
        paratope_H, paratope_L, epitope = calculate_interface_residues(
            heavy_obj, light_obj, flat_antigen_chain)

        # 2. Get Secondary Structure
        dssp_H = get_secondary_structure(heavy_obj)
        dssp_L = get_secondary_structure(light_obj)
        dssp_Ag = get_secondary_structure(flat_antigen_chain)

        # 3. Segment and Label Ellipsoids
        ellipsoids_H = segment_and_label_ellipsoids(heavy_obj, dssp_H, 'H', paratope_H)
        ellipsoids_L = segment_and_label_ellipsoids(light_obj, dssp_L, 'L', paratope_L)
        ellipsoids_Ag = segment_and_label_ellipsoids(flat_antigen_chain, dssp_Ag, 'Ag', epitope)

        all_ellipsoids = ellipsoids_H + ellipsoids_L + ellipsoids_Ag
        
        # 4. Save to JSON
        with open(ellipsoid_path, 'w') as f:
            json.dump(all_ellipsoids, f, indent=2)
        
        return f"Successfully processed {pdb_id}."

    except Exception as e:
        return f"Failed to process {pdb_id}: {e}"

# ###########################################################
# ##### MAIN EXECUTION BLOCK #####
# ###########################################################

def main(args):
    # Load the summary file
    with open(args.summary_json, 'r') as f:
        items = [json.loads(line) for line in f]
    
    print_log(f"Found {len(items)} total entries in {args.summary_json}")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    print_log(f"Ellipsoid files will be saved to: {os.path.abspath(args.out_dir)}")

    # Create a list of arguments for the process_map
    process_args = [(item, args.out_dir) for item in items]

    # Use multiprocessing to run the processing in parallel
    with multiprocessing.Pool(args.n_cpu) as pool:
        # Wrap the map function with tqdm for a progress bar
        results = list(tqdm(pool.starmap(process_entry, process_args), total=len(items)))

    # Log summary of the run
    success_count = sum(1 for r in results if "Successfully" in r)
    skipped_count = sum(1 for r in results if "Skipped" in r)
    failed_count = sum(1 for r in results if "Failed" in r)
    
    print_log("\n--- Ellipsoid Generation Summary ---")
    print_log(f"Successfully processed: {success_count}")
    print_log(f"Skipped (already exist): {skipped_count}")
    print_log(f"Failed: {failed_count}")
    print_log("------------------------------------")


def parse():
    parser = argparse.ArgumentParser(description='Generate ellipsoid database from a SAbDab summary file.')
    parser.add_argument('--summary_json', type=str, required=True,
                        help='Path to the processed SAbDab JSON summary file (e.g., sabdab_all.json).')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save the generated ellipsoid JSON files.')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='Number of CPU cores to use for parallel processing.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
