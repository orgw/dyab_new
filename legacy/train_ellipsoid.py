#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.ellipsoid_dataset import EllipsoidComplexDataset, collate_ellipsoid
from models.ellipsoid_itf import EllipsoidInpaintingITF

def parse_args():
    p = argparse.ArgumentParser("Train Step-1 Ellipsoid Inpainter (CFM)")
    # accept multiple aliases -> single dest
    p.add_argument('-t','--train_json','--train_jsonl', dest='train_path', type=str, required=True)
    p.add_argument('-v','--valid_json','--valid_jsonl', dest='valid_path', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max_epochs', type=int, default=50)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--acc', type=str, default=('gpu' if torch.cuda.is_available() else 'cpu'))
    return p.parse_args()

def main():
    args = parse_args()
    train_set = EllipsoidComplexDataset(path=args.train_path)
    valid_set = EllipsoidComplexDataset(path=args.valid_path)

    label_dims = train_set.label_dims
    token_dim = train_set.token_dim

    model = EllipsoidInpaintingITF(
        token_dim=token_dim, hidden_dim=args.hidden_dim, num_heads=args.heads, num_layers=args.layers,
        label_dims=label_dims, lr=args.lr
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_ellipsoid, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                              collate_fn=collate_ellipsoid, pin_memory=True)

    trainer = pl.Trainer(
        accelerator=args.acc, devices=args.devices, max_epochs=args.max_epochs,
        precision=('bf16-mixed' if torch.cuda.is_available() else 32),
        gradient_clip_val=1.0, log_every_n_steps=10
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == "__main__":
    main()