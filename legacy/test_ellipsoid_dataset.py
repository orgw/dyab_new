#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import json
import random
import os
import numpy as np
from io import StringIO

from data.ellipsoid_dataset import EllipsoidComplexDataset, ellipsoid_collate_fn

# --- PyMOL (optional, for visualization) ---
try:
    import pymol
    from pymol import cmd
    from pymol.cgo import BEGIN, END, TRIANGLE_STRIP, VERTEX, NORMAL, COLOR
    _HAS_PYMOL = True
except ImportError:
    print("[WARN] PyMOL module not found. Will not be able to generate .pse files.")
    _HAS_PYMOL = False

# --- PyMOL HELPER FUNCTIONS (from reference script) ---

def create_cgo_ellipsoid(center, covar, color):
    ws, V = np.linalg.eigh(covar)
    radii = np.sqrt(np.maximum(ws, 1e-9)) * 1.5
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    points_rotated = np.dot(np.stack([x, y, z], axis=-1), V.T)
    points_final = points_rotated + center
    obj = [COLOR, color[0], color[1], color[2]]
    for i in range(len(u) - 1):
        obj.extend([BEGIN, TRIANGLE_STRIP])
        for j in range(len(v)):
            for k in [i, i + 1]:
                point = points_final[k, j]
                normal = point - center
                nrm = np.linalg.norm(normal)
                if nrm > 1e-6:
                    normal /= nrm
                obj.extend([NORMAL, *normal, VERTEX, *point])
        obj.append(END)
    return obj

def create_pymol_session(final_pdb_path, heavy_chain_id, light_chain_id, antigen_chain_id,
                         all_ellipsoids, save_path, transparency=0.6):
    if not _HAS_PYMOL:
        print(f"[WARN] PyMOL not available; skipping PSE: {save_path}")
        return
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cmd.reinitialize()
    cmd.load(os.path.abspath(final_pdb_path), "complex")
    cmd.bg_color("white")
    cmd.hide("everything", "all")
    cmd.show("cartoon", "all")
    if light_chain_id:  cmd.color("palecyan",  f"complex and chain {light_chain_id}")
    if heavy_chain_id:  cmd.color("lightblue", f"complex and chain {heavy_chain_id}")
    if antigen_chain_id: cmd.color("lightorange", f"complex and chain {antigen_chain_id}")
    
    colors = {
        'Paratope':    [1.0, 0.2, 0.2], # Red
        'Framework':   [0.6, 0.6, 1.0], # Blue
        'Epitope':     [1.0, 0.7, 0.2], # Orange
        'Non-Epitope': [1.0, 0.8, 0.6], # Salmon
    }
    
    cgo_groups = {
        'Paratope_Ellipsoids': [], 'Framework_Ellipsoids': [],
        'Epitope_Ellipsoids': [], 'Non-Epitope_Ellipsoids': []
    }
    
    for blob in all_ellipsoids:
        category = blob.get('label', 'Non-Epitope')
        if category in colors:
            cgo_obj = create_cgo_ellipsoid(blob['pos'], blob['covar'], colors[category])
            cgo_groups[f'{category}_Ellipsoids'].extend(cgo_obj)
            
    loaded = []
    for name, cgo_list in cgo_groups.items():
        if cgo_list:
            cmd.load_cgo(cgo_list, name)
            loaded.append(name)
            
    if loaded:
        cmd.group('All_Ellipsoids', ' '.join(loaded))
    cmd.set("cgo_transparency", float(transparency))
    cmd.zoom("all")
    cmd.save(os.path.abspath(save_path))
    print(f"[OK] PyMOL session saved: {os.path.abspath(save_path)}")


def debug_single_item(dataset, index, debug_dir):
    """
    Processes and prints detailed debug info for a single item from the dataset.
    Also generates a PyMOL session file for visual inspection.
    """
    print(f"\n{'='*20} DEBUGGING ITEM AT INDEX {index} {'='*20}")
    rec = dataset.records[index]
    pdb_id = rec.get("pdb", f"rec_{index}")
    pdb_path = rec.get("pdb_data_path")

    try:
        # Re-run the core logic here to get all intermediate products for debugging
        if not os.path.isabs(pdb_path):
            pdb_path = os.path.join(dataset.pdb_base_path, pdb_path)

        if not os.path.exists(pdb_path):
            print(f"==> RESULT: PDB file not found at '{pdb_path}'. Skipping.")
            return

        with open(pdb_path, "r") as f:
            pdb_str_raw = f.read()
        
        # Use the dataset's internal methods to process the data
        pdb_str = dataset._sanitize_pdb(pdb_str_raw)
        all_ellipsoids = dataset._extract_ellipsoids(pdb_str, rec)

        if not all_ellipsoids:
            print("==> RESULT: Item is EMPTY (no ellipsoids could be extracted).")
            return

        # --- Report counts ---
        context_count = sum(1 for e in all_ellipsoids if e.role in ["Framework", "Non-Epitope"])
        interface_count = sum(1 for e in all_ellipsoids if e.role in ["Paratope", "Epitope"])
        print("==> RESULT: Successfully processed item.")
        print(f"  Context ellipsoids found: {context_count}")
        print(f"  Interface ellipsoids found: {interface_count}")

        # --- Create PyMOL session ---
        if not _HAS_PYMOL:
            print("  Skipping PyMOL session generation (PyMOL not installed).")
            return
            
        protein_obj = dataset._from_pdb_string_safe(pdb_str)
        if not protein_obj:
            print("  Could not parse PDB to find antigen chain for PyMOL.")
            return
        
        _, ag_used = dataset._load_first_valid_antigen(protein_obj, rec.get("antigen_chains", []))

        all_ellipsoids_dict = [{
            "pos": e.mu, "covar": e.cov, "label": e.role,
        } for e in all_ellipsoids]
        
        pse_path = os.path.join(debug_dir, f"{pdb_id}.pse")
        create_pymol_session(
            final_pdb_path=pdb_path,
            heavy_chain_id=rec.get("heavy_chain"),
            light_chain_id=rec.get("light_chain"),
            antigen_chain_id=ag_used,
            all_ellipsoids=all_ellipsoids_dict,
            save_path=pse_path
        )

    except Exception as e:
        print(f"==> RESULT: An ERROR occurred during processing.")
        import traceback
        traceback.print_exc()
    finally:
        print(f"{'='*58}")


def main_debugger(json_path, num_samples=20):
    """
    Tests the data pipeline on a random subset of the dataset with verbose logging.
    """
    print("--- Starting Full-Logic Ellipsoid Dataset Debugger ---")

    try:
        with open(json_path, 'r') as f:
            all_records = [json.loads(line) for line in f if line.strip()]
        print(f"✅ Loaded {len(all_records)} records from '{json_path}'")
    except Exception as e:
        print(f"❌ Error reading records: {e}")
        return

    # Create directory for PyMOL debug files
    debug_dir = "./debug_pse"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"✅ PyMOL debug sessions will be saved to '{os.path.abspath(debug_dir)}'")

    dataset = EllipsoidComplexDataset(records=all_records, pdb_base_path="./")
    
    if len(dataset) < num_samples:
        sample_indices = list(range(len(dataset)))
    else:
        sample_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"✅ Will now process {len(sample_indices)} random samples individually...\n")

    for index in sample_indices:
        debug_single_item(dataset, index, debug_dir)

    print("\n--- Debug Session Complete ---")
    print("Review the logs and check the generated .pse files in the 'debug_pse' directory.")

if __name__ == '__main__':
    dataset_json_path = './all_data/RAbD/train.json'
    
    if not os.path.exists(dataset_json_path):
        print(f"❌ Error: Main dataset index not found at '{dataset_json_path}'.")
    else:
        main_debugger(dataset_json_path, num_samples=20)