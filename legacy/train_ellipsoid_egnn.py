#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import time
import json
import argparse
import torch
import os.path as osp
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

from data.ellipsoid_dataset_gnn import EllipsoidComplexDataset, ellipsoid_collate_fn
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as plog

from models.flow_matching_itf import EllipsoidFlowMatchingITF
from models.callbacks_gnn import SetupCallback, BestCheckpointCallback, EpochEndCallback, VisualizationCallback

from torch import distributed as dist

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def parse():
    parser = argparse.ArgumentParser(description='Training Ellipsoid EGNN Flow Matching Model')
    # ... (other arguments are the same)
    parser.add_argument('--task', type=str, default='ellipsoid_flow_matching')
    parser.add_argument('--ex_name', type=str, default='flow_matching_egnn_v2_with_viz')
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--wandb_offline', type=int, default=1)
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, required=True)
    parser.add_argument('--pdb_base_path', type=str, default='./')
    parser.add_argument('--cache_path', type=str, default=None, help='Directory to store pre-processed data.')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_topk', type=int, default=5)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument("--local_rank", type=int, default=-1)
    
    parser.add_argument('--token_dim', type=int, default=17)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    
    parser.add_argument('--guidance_dropout_rate', type=float, default=0.1, help='Dropout rate for context embedding for CFG.')
    parser.add_argument('--test', action='store_true', help='Run a quick test on a small subset of data.')
    
    # --- NEW ARGUMENT FOR VISUALIZATION ---
    parser.add_argument('--viz_every_n_epochs', type=int, default=10, help='Frequency of generating ellipsoid visualizations during validation.')

    return parser.parse_args()


def main(args):
    config = args.__dict__
    
    local_rank, _ = get_dist_info()
    if (len(args.gpus) > 1 and int(local_rank) == 0) or len(args.gpus) <= 1:
        print_log(args)

    with open(args.train_set, 'r') as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    with open(args.valid_set, 'r') as f:
        valid_records = [json.loads(line) for line in f if line.strip()]
        
    if args.test:
        print_log("--- RUNNING IN TEST MODE ---")
        train_records = train_records[:32]
        valid_records = valid_records[:32]
        args.max_epochs = 50
        args.cache_path = None
        print_log("Caching is disabled for test mode.")

    train_cache_path = osp.join(args.cache_path, 'train') if args.cache_path else None
    valid_cache_path = osp.join(args.cache_path, 'valid') if args.cache_path else None

    train_set = EllipsoidComplexDataset(records=train_records, pdb_base_path=args.pdb_base_path, cache_path=train_cache_path)
    valid_set = EllipsoidComplexDataset(records=valid_records, pdb_base_path=args.pdb_base_path, cache_path=valid_cache_path)

    if args.local_rank in [0, -1]:
        print_log(f'Training set size: {len(train_set)}')
        print_log(f'Validation set size: {len(valid_set)}')

    if len(args.gpus) > 1:
        args.batch_size = int(args.batch_size / len(args.gpus))
        if local_rank == 0: print_log(f'Batch size on a single GPU: {args.batch_size}')
    config['local_rank'] = args.local_rank

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle, pin_memory=True, collate_fn=ellipsoid_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=ellipsoid_collate_fn)
    
    model_itf = EllipsoidFlowMatchingITF(**vars(args))
    
    save_dir = osp.join(args.save_dir, args.ex_name)
    ckpt_dir = osp.join(save_dir, 'checkpoints')
    
    setup_callback = SetupCallback(prefix=args.task, setup_time=time.strftime('%Y%m%d_%H%M%S', time.localtime()), save_dir=save_dir, ckpt_dir=ckpt_dir, args=args, argv_content=sys.argv + ["gpus: {}".format(torch.cuda.device_count())])
    ckpt_callback = BestCheckpointCallback(monitor='valid_loss', filename='{epoch:02d}_{valid_loss:.3f}', mode='min', save_last=True, save_top_k=args.save_topk, dirpath=ckpt_dir, verbose=True)
    epochend_callback = EpochEndCallback()
    
    # --- NEW: Add Visualization Callback ---
    viz_callback = VisualizationCallback(viz_every_n_epochs=args.viz_every_n_epochs)
    callbacks = [setup_callback, ckpt_callback, epochend_callback, viz_callback]
    
    logger = plog.WandbLogger(project='dyAb-ellipsoid-flow', name=args.ex_name, save_dir=save_dir, config=config, offline=bool(args.wandb_offline))
    
    trainer = Trainer(
        accelerator=args.accelerator, 
        strategy=args.strategy, 
        callbacks=callbacks, 
        max_epochs=args.max_epochs, 
        devices=args.gpus, 
        gradient_clip_val=args.gradient_clip_val, 
        gradient_clip_algorithm='norm', 
        logger=logger
    )
    
    trainer.fit(model=model_itf, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    args = parse()
    main(args)