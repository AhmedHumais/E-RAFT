from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from loader.loader_mvsec_gnn import MVSEC20hz_outdoor_day1
from model.eraftv2 import ERAFT
import pytorch_lightning as pl

from loader.utils import dsec_collate_fn
from loader.loader_dsec_gnn import Sequence

# from torch.utils.tensorboard import SummaryWriter

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

class EraftTrainer(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = ERAFT(args.n_graph_feat)
        self.args = args
        # self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # optimizer = self.optimizers()
        # # scheduler = self.lr_schedulers()
        # optimizer.zero_grad()
        # self.model.train()

        graph_data, gt = batch

        flow = gt[:,0]
        valid = gt[:,1]
        
        # flow = torch.unsqueeze(flow, dim=0)      # for batch stuff    
        # valid = torch.unsqueeze(valid, dim=0)    # for batch stuff
        _, flow_predictions = self.model(graph_data, flow, iters=self.args.iters)            

        loss, metrics = self.sequence_loss(flow_predictions, flow, valid, self.args.gamma)
        # self.manual_backward(loss)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        
        # optimizer.step()
        # scheduler.step()
        metrics = {f'{key}_train': value for key, value in metrics.items()}
        self.log_dict(metrics, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # self.model.eval()
        graph_data, gt = batch

        flow = gt[:,0]
        valid = gt[:,1]

        # with torch.no_grad():
        _, flow_predictions = self.model(graph_data, flow, iters=self.args.iters)            

        loss, metrics = self.sequence_loss(flow_predictions, flow, valid, self.args.gamma)
        metrics = {f'{key}_val': value for key, value in metrics.items()}
        self.log_dict(metrics, prog_bar=True)
        
        return loss
   
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)

        return optimizer
   
    def sequence_loss(self, flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(flow_preds)    
        flow_loss = 0.0
        
        # flow_gt = flow_gt[:,:,2:258, 1:-1]       # because of padding issue of output of nn
        # valid = valid[:,1,2:258, 1:-1]           # because of padding issue of output of nn
        valid = valid[:,1]           # because of padding issue of output of nn

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--n_graph_feat', type=int, default=4, help='number of input graph features')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    data_path = [Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_01_a'),
                 Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_02_a'),
                 Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_02_c'),
                 Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_05_a')]
    seq_path = [Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_01_a'),
                Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_02_a'),
                Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_02_c'),
                Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_05_a')]
    
    val_data_path = [Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_11_a')]
    val_seq_path = [Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_11_a')]
                     
    data_set = Sequence(flow_paths=seq_path, event_paths=data_path)
    train_loader = DataLoader(data_set, batch_size=1, collate_fn=dsec_collate_fn, num_workers=16)
    val_set = Sequence(flow_paths=val_seq_path, event_paths=val_data_path)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=dsec_collate_fn, num_workers=8)

    trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=args.num_steps, 
        devices=2,
        # gradient_clip_val=args.clip, 
        strategy="ddp", 
        # limit_train_batches=512, 
        default_root_dir="checkpoints", 
        logger=pl.loggers.CSVLogger('checkpoints'), 
        profiler="advanced"
        )
    
    trainer.fit(EraftTrainer(args), train_loader, val_loader) 