import os
import json
import re
import numpy as np

# Assuming these utilities are in the path and configured
from data.pdb_utils import AgAbComplex, Protein
from evaluation.rmsd import compute_rmsd
from evaluation.tm_score import tm_score
from evaluation.lddt import lddt_full
# The dockq utility is no longer needed as a separate function
# from evaluation.dockq import dockq 
from utils.logger import print_log
from configs import DOCKQ_DIR, CACHE_DIR # Make sure these are configured
from utils.time_sign import get_time_sign


def evaluate_and_print_metrics(inputs):
    """
    Calculates metrics for a predicted antibody structure against a reference,
    and prints key sequence information.
    """
    mod_pdb_path, ref_pdb_path, H, L, A, cdr_type, sidechain = inputs
    
    # Load structures with their correct, respective chain IDs
    # Generated PDB always uses A, B, C for H, L, Antigen
    mod_cplx = AgAbComplex.from_pdb(mod_pdb_path, 'A', 'B', ['C'], skip_epitope_cal=True, skip_validity_check=True)
    # Reference PDB uses the chain IDs from the JSON file
    ref_cplx = AgAbComplex.from_pdb(ref_pdb_path, H, L, A, skip_epitope_cal=False, skip_validity_check=False)

    # Ensure the generated model uses the same CDR definitions as the reference
    mod_cplx.cdr_pos = ref_cplx.cdr_pos

    # --- Print Sequence Information ---
    print("\n--- Sequence Information ---")
    
    # Reference Sequences
    print("\n[Reference PDB]")
    print(f"  Heavy Chain ({H}): {ref_cplx.get_heavy_chain().get_seq()}")
    print(f"  Light Chain ({L}): {ref_cplx.get_light_chain().get_seq()}")
    for chain_id, antigen_chain in ref_cplx.antigen:
        print(f"  Antigen Ch. {chain_id} : {antigen_chain.get_seq()}")
    
    # Generated Model Sequences
    print("\n[Generated PDB]")
    mod_heavy_chain = mod_cplx.get_heavy_chain()
    mod_light_chain = mod_cplx.get_light_chain()
    if mod_heavy_chain: print(f"  Heavy Chain (A): {mod_heavy_chain.get_seq()}")
    if mod_light_chain: print(f"  Light Chain (B): {mod_light_chain.get_seq()}")
    for chain_id, antigen_chain in mod_cplx.antigen:
        print(f"  Antigen Ch. {chain_id} : {antigen_chain.get_seq()}")

    # CDR H3 Sequences
    print("\n[CDR H3 Sequence]")
    ref_h3 = ref_cplx.get_cdr('H3')
    mod_h3 = mod_cplx.get_cdr('H3')
    if ref_h3: print(f"  Reference : {ref_h3.get_seq()}")
    if mod_h3: print(f"  Generated : {mod_h3.get_seq()}")
    
    # --- Calculate Metrics ---
    results = {}
    
    # RMSD(CA)
    ref_heavy_chain = ref_cplx.get_heavy_chain()
    ref_light_chain = ref_cplx.get_light_chain()
    gt_coords = {('heavy', r.get_id()): r.get_coord('CA') for r in ref_heavy_chain}
    gt_coords.update({('light', r.get_id()): r.get_coord('CA') for r in ref_light_chain})
    pred_coords = {('heavy', r.get_id()): r.get_coord('CA') for r in mod_heavy_chain}
    pred_coords.update({('light', r.get_id()): r.get_coord('CA') for r in mod_light_chain})
    common_residues = sorted(list(set(gt_coords.keys()) & set(pred_coords.keys())))
    if common_residues:
        gt_x = np.array([gt_coords[res_id] for res_id in common_residues])
        pred_x = np.array([pred_coords[res_id] for res_id in common_residues])
        results['RMSD(CA) aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)
        results['RMSD(CA)'] = compute_rmsd(gt_x, pred_x, aligned=True)
    
    # TMscore & LDDT
    results['TMscore'] = tm_score(mod_cplx.antibody, ref_cplx.antibody)
    results['LDDT'], _ = lddt_full(mod_cplx, ref_cplx)

    # DockQ Calculation (Corrected)
    try:
        prefix = get_time_sign(suffix=ref_cplx.get_id().replace('(', '').replace(')', ''))
        mod_dockq_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_mod.pdb')
        ref_dockq_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_ref.pdb')
        
        mod_cplx.to_pdb(mod_dockq_pdb)
        ref_cplx.to_pdb(ref_dockq_pdb)

        # Use -no_needle to prevent dependency issues, call without old flags
        cmd = f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_dockq_pdb} {ref_dockq_pdb} -no_needle'
        with os.popen(cmd) as p:
            text = p.read()
        
        res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
        if res:
            score = float(res.group(1))
        else:
            print_log(f'DockQ calculation failed. Full output:\n{text}', level='ERROR')
            score = 0.0
        results['DockQ'] = score
        os.remove(mod_dockq_pdb)
        os.remove(ref_dockq_pdb)
    except Exception as e:
        print_log(f'Error in dockq calculation: {e}', level='ERROR')
        results['DockQ'] = 0.0

    return results



def dockq(mod_cplx: AgAbComplex, ref_cplx: AgAbComplex, cdrh3_only=False):
    """
    Calculates the DockQ score between a model and reference complex.

    Args:
        mod_cplx (AgAbComplex): The modeled/predicted Ag-Ab complex.
        ref_cplx (AgAbComplex): The reference/native Ag-Ab complex.
        cdrh3_only (bool, optional): If True, calculates DockQ for the CDRH3 
                                     and antigen interface only. Defaults to False.

    Returns:
        float: The calculated DockQ score.
    """
    H, L = ref_cplx.heavy_chain, ref_cplx.light_chain
    prefix = get_time_sign(suffix=ref_cplx.get_id().replace('(', '').replace(')', ''))
    mod_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_mod.pdb')
    ref_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_ref.pdb')
    
    # The heavy and light chain IDs must be consistent for the DockQ script flags.
    # The model complex uses 'A' for heavy and 'B' for light.
    model_heavy_chain_id = 'A'
    model_light_chain_id = 'B'

    # The reference complex uses its native chain IDs.
    ref_heavy_chain_id = ref_cplx.get_heavy_chain().get_id()
    ref_light_chain_id = ref_cplx.get_light_chain().get_id()

    mod_cplx.to_pdb(mod_pdb)
    ref_cplx.to_pdb(ref_pdb)

    cmd = (f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {ref_pdb} '
           f'-model_chain1 {model_heavy_chain_id} {model_light_chain_id} '
           f'-native_chain1 {ref_heavy_chain_id} {ref_light_chain_id} -no_needle')
    
    with os.popen(cmd) as p:
        text = p.read()
    
    res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
    
    score = 0.0
    if res:
        score = float(res.group(1))
    else:
        # Handle cases where DockQ fails
        print(f"Warning: DockQ calculation failed for {ref_cplx.get_id()}. Output:\n{text}")

    os.remove(mod_pdb)
    os.remove(ref_pdb)
    
    return score


# --- Main Script Logic ---
if __name__ == '__main__':
    # --- 1. Set parameters for your evaluation ---
    pdb_file = "/nfsdata/home/kiwoong.yoo/workspace/BoltzDesign1/outputs/protein_5hi4_5hi4_H3_inpainting_TEST/pdb/5hi4_results_itr1_length119_model_0.pdb"
    ref_pdb_file = "./all_structures/imgt/5hi4.pdb"
    json_info_file = "./pdb_info.json" # Assumes this file exists
    
    pdb_id = os.path.basename(ref_pdb_file).split('.')[0]
    
    heavy_chain_id = None
    light_chain_id = None
    antigen_chain_ids = None

    try:
        with open(json_info_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                # This handles both standard JSON and JSON Lines format
                data_item = json.loads(line.strip().rstrip(','))
                if isinstance(data_item, list): # Full JSON array
                    pdb_info = next((item for item in data_item if item.get("pdb") == pdb_id), None)
                elif data_item.get("pdb") == pdb_id: # JSON Lines
                    pdb_info = data_item
                else:
                    continue
                
                if pdb_info:
                    heavy_chain_id = pdb_info['heavy_chain']
                    light_chain_id = pdb_info['light_chain']
                    antigen_chain_ids = pdb_info['antigen_chains']
                    break
        if not heavy_chain_id:
            print(f"\nERROR: PDB ID '{pdb_id}' not found in {json_info_file}")
            exit()
    except FileNotFoundError:
        print(f"\nERROR: JSON info file not found at: {json_info_file}")
        exit()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\nERROR: Failed to parse JSON or find required keys. Details: {e}")
        exit()

    cdr_type = None
    match = re.search(r'_([HL]\d)_inpainting_TEST', pdb_file)
    if match:
        cdr_type = match.group(1)
    else:
        print(f"\nERROR: Could not determine CDR type from path: {pdb_file}")
        exit()

    sidechain_packing = False

    # --- 2. Run Evaluation ---
    if not all([pdb_file, ref_pdb_file, heavy_chain_id, light_chain_id, antigen_chain_ids, cdr_type]):
        print("\nERROR: One or more evaluation parameters are missing. Aborting.")
    elif not os.path.exists(pdb_file):
        print(f"\nERROR: Model PDB file not found at: {pdb_file}")
    elif not os.path.exists(ref_pdb_file):
        print(f"\nERROR: Reference PDB file not found at: {ref_pdb_file}")
    else:
        metric_inputs = (
            pdb_file, ref_pdb_file, heavy_chain_id, light_chain_id, antigen_chain_ids, cdr_type, sidechain_packing,
        )
        
        results = evaluate_and_print_metrics(metric_inputs)

        print("\n--- Final Evaluation Results ---")
        if results:
            for metric_name, value in results.items():
                print(f"{metric_name}: {value:.4f}")
        else:
            print("Evaluation did not complete successfully.")
