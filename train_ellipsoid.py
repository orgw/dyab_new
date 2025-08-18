import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataset import E2EDataset
from data.antibody_ellipsoid_dataset import AntibodyEllipsoidDataset
from utils.io import read_json, save_json, read_csv
from utils.logger import create_logger
from models.dyAb_itf import dyAbITF 

def main(args):
    # your existing main function
    # with the one-line addition
    
    # SETUP
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = create_logger(name=__name__,
                           path=os.path.join(args.save_dir, 'log.txt'))
    logger.info(args)

    # DATASETS
    train_set = E2EDataset(args.train_set, cdr=args.cdr, paratope=args.paratope,
                           flexible=args.flexible,
                           seq_warmup=args.seq_warmup,
                           struct_only=args.struct_only,
                           backbone_only=args.backbone_only,
                           no_pred_edge_dist=args.no_pred_edge_dist,
                           no_memory=args.no_memory)
    valid_set = E2EDataset(args.valid_set, cdr=args.cdr, paratope=args.paratope,
                           struct_only=args.struct_only,
                           backbone_only=args.backbone_only,
                           no_pred_edge_dist=args.no_pred_edge_dist,
                           no_memory=args.no_memory)
    test_set = E2EDataset(args.test_set, cdr=args.cdr, paratope=args.paratope,
                           struct_only=args.struct_only,
                           backbone_only=args.backbone_only,
                           no_pred_edge_dist=args.no_pred_edge_dist,
                           no_memory=args.no_memory)

    # WRAP DATASETS
    train_set = AntibodyEllipsoidDataset(train_set)
    valid_set = AntibodyEllipsoidDataset(valid_set)
    test_set = AntibodyEllipsoidDataset(test_set)
    
    collate_fn = train_set.collate_fn # MODIFIED: Use the collate_fn from the wrapper

    # DATALOADERS
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    args.test_data = test_set.data
    logger.info(f'step per epoch: {len(train_loader)}')
    
    # FIX: Explicitly set the model type
    args.model_type = "dyMEAN"
    
    # MODEL
    if args.model_type == 'dyMEAN':
        model = dyAbITF(args)
    # the rest of your model selection logic
    else:
        # The original source of the error
        raise NotImplementedError(f'model {args.model_type} not implemented')
        
    # CHECKPOINT CALLBACK
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=args.save_topk,
        mode='min'
    )

    # TRAINER
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        strategy=args.strategy
    )
    
    # TRAINING
    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ALL YOUR ARGUMENTS...
    
    args = parser.parse_args()
    main(args)