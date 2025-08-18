import os
import json
import re
import numpy as np

# Assuming these utilities are in the path and configured
from data.pdb_utils import AgAbComplex, Protein
from evaluation.rmsd import compute_rmsd
from evaluation.tm_score import tm_score
from evaluation.lddt import lddt_full
from utils.logger import print_log
# Import DockQ functions to use it as a library
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

# Define the contact distance threshold
CONTACT_DIST = 5.0

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
    
    # AAR and CAAR Calculation
    gt_s, pred_s = '', ''
    is_contact = []
    epitope = ref_cplx.get_epitope()
    
    cdr_to_eval = [cdr_type] if isinstance(cdr_type, str) else cdr_type
    for cdr in cdr_to_eval:
        gt_cdr = ref_cplx.get_cdr(cdr)
        mod_cdr = mod_cplx.get_cdr(cdr)

        if gt_cdr and mod_cdr:
            cur_gt_s = gt_cdr.get_seq()
            cur_pred_s = mod_cdr.get_seq()
            gt_s += cur_gt_s
            pred_s += cur_pred_s

            cur_contact = []
            for ab_residue in gt_cdr:
                contact = any(ab_residue.dist_to(ag_residue) < CONTACT_DIST for ag_residue, _, _ in epitope)
                cur_contact.append(int(contact))
            is_contact.extend(cur_contact)

    if len(gt_s) > 0 and len(gt_s) == len(pred_s):
        hit = sum(1 for a, b in zip(gt_s, pred_s) if a == b)
        chit = sum(1 for a, b, c in zip(gt_s, pred_s, is_contact) if a == b and c == 1)
        
        results['AAR'] = hit / len(gt_s)
        results['CAAR'] = chit / (sum(is_contact) + 1e-10)

    # RMSD(CA) - More direct calculation from reference
    gt_x, pred_x = [], []
    for xl, cplx in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
        for chain in [cplx.get_heavy_chain(), cplx.get_light_chain()]:
             if chain:
                for i in range(len(chain)):
                    try:
                        xl.append(chain.get_ca_pos(i))
                    except KeyError: # Skip if CA not found
                        pass
    
    if len(gt_x) == len(pred_x) and len(gt_x) > 0:
        gt_x, pred_x = np.array(gt_x), np.array(pred_x)
        results['RMSD(CA) aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)
        results['RMSD(CA)'] = compute_rmsd(gt_x, pred_x, aligned=True)

    # CDR-specific RMSD
    for cdr in cdr_to_eval:
        gt_cdr, mod_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
        if gt_cdr and mod_cdr and len(gt_cdr) == len(mod_cdr):
            gt_x = np.array([gt_cdr.get_ca_pos(i) for i in range(len(gt_cdr))])
            pred_x = np.array([mod_cdr.get_ca_pos(i) for i in range(len(mod_cdr))])
            results[f'RMSD(CA) CDR{cdr}'] = compute_rmsd(gt_x, pred_x, aligned=True)
            results[f'RMSD(CA) CDR{cdr} aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)

    # TMscore & LDDT
    results['TMscore'] = tm_score(mod_cplx.antibody, ref_cplx.antibody)
    results['LDDT'], _ = lddt_full(mod_cplx, ref_cplx)

    # DockQ Calculation
    try:
        model = load_PDB(mod_pdb_path)
        native = load_PDB(ref_pdb_path)
        
        chain_map = { H: 'A', L: 'B' }
        for ag_chain in A:
            chain_map[ag_chain] = 'C'

        _, dockq_score = run_on_all_native_interfaces(model, native, chain_map=chain_map)
        results['DockQ'] = dockq_score
    except Exception as e:
        print_log(f'Error in DockQ calculation: {e}', level='ERROR')
        results['DockQ'] = 0.0

    return results


# --- Main Script Logic ---
if __name__ == '__main__':
    # --- 1. Set parameters for your evaluation ---
    pdb_file = "/nfsdata/home/kiwoong.yoo/workspace/BoltzDesign1/outputs/protein_5hi4_5hi4_H3_inpainting_SLURM/pdb/5hi4_results_itr1_length119_model_0.pdb"
    ref_pdb_file = "./all_structures/imgt/5hi4.pdb"
    json_info_file = "/nfsdata/home/kiwoong.yoo/workspace/dyAb/all_data/RAbD/test.json"
    
    pdb_id = os.path.basename(ref_pdb_file).split('.')[0]
    
    heavy_chain_id = None
    light_chain_id = None
    antigen_chain_ids = None

    try:
        with open(json_info_file, 'r') as f:
            # This logic handles both a single JSON object/array and JSON Lines format
            content = f.read()
            try:
                # Try loading as a single JSON entity
                data = json.loads(content)
                if isinstance(data, list):
                    pdb_info = next((item for item in data if item.get("pdb") == pdb_id), None)
                else: # Should not happen with consistent format
                    pdb_info = None
            except json.JSONDecodeError:
                # If that fails, parse as JSON Lines
                pdb_info = None
                for line in content.splitlines():
                    if not line.strip(): continue
                    data_item = json.loads(line)
                    if data_item.get("pdb") == pdb_id:
                        pdb_info = data_item
                        break
        
        if pdb_info:
            heavy_chain_id = pdb_info['heavy_chain']
            light_chain_id = pdb_info['light_chain']
            antigen_chain_ids = pdb_info['antigen_chains']
        else:
            print(f"\nERROR: PDB ID '{pdb_id}' not found in {json_info_file}")
            exit()

    except FileNotFoundError:
        print(f"\nERROR: JSON info file not found at: {json_info_file}")
        exit()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\nERROR: Failed to parse JSON or find required keys. Details: {e}")
        exit()

    cdr_type = 'H3'

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
