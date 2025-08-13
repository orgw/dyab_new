import torch
import json
import os

# --- Import Your Project's Modules ---
from data.ellipsoid_dataset import EllipsoidComplexDataset, ellipsoid_collate_fn
from models.ellipsoid_inpainter_model import EllipsoidInpainterModel

def run_smoke_test(json_path, pdb_base_path):
    """
    Performs a smoke test on the EllipsoidInpainterModel.
    """
    print("--- Starting Model Smoke Test ---")

    # 1. Load a single data point
    try:
        with open(json_path, 'r') as f:
            # Load just the first record for a quick test
            first_line = f.readline()
            record = [json.loads(first_line)]
        
        dataset = EllipsoidComplexDataset(records=record, pdb_base_path=pdb_base_path)
        
        # Use the collate function to prepare a batch of size 1
        data_point = ellipsoid_collate_fn([dataset[0]])
        
        print(f"✅ Successfully loaded and processed one data point from '{json_path}'")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Initialize the model
    try:
        # These dimensions should match your model's configuration
        token_dim = dataset.token_dim 
        hidden_dim = 128  # Example hidden dimension
        num_heads = 4
        num_layers = 3
        
        model = EllipsoidInpainterModel(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        model.eval() # Set to evaluation mode
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return

    # 3. Run a single forward pass
    try:
        print("\n--- Running Forward Pass ---")
        context_tokens = data_point["context_tokens"]
        context_mask = data_point["context_mask"]

        # Run the model
        predictions = model(context_tokens, context_mask)
        print("✅ Forward pass completed without errors.")
        
        # 4. Print output shapes to verify
        print("\n--- Output Tensor Shapes ---")
        for name, tensor in predictions.items():
            print(f"  - {name}: {tensor.shape}")
            
    except Exception as e:
        print(f"❌ An error occurred during the forward pass:")
        import traceback
        traceback.print_exc()
        return
        
    print("\n--- Smoke Test Passed! ---")
    print("The model successfully processed a data point and produced output.")

if __name__ == '__main__':
    # --- Configuration ---
    # 1. Path to your dataset JSON file
    DATASET_JSON_PATH = './all_data/RAbD/train.json'
    # 2. Base path for the PDB files
    PDB_BASE_DIRECTORY = "./"

    if not os.path.exists(DATASET_JSON_PATH):
        print(f"❌ Error: Dataset index not found at '{DATASET_JSON_PATH}'.")
    else:
        run_smoke_test(
            json_path=DATASET_JSON_PATH,
            pdb_base_path=PDB_BASE_DIRECTORY
        )