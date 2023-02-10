import torch
import numpy as np

from pathlib import Path
import os
import h5py

from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.transforms import Cartesian
from tqdm.auto import tqdm
from model import eraftv2

def make_graph(ev_arr, gt, beta=0.5e4):
    ts_sample = ev_arr[:, 3] - ev_arr[0, 3]
    ts_sample = torch.tensor(ts_sample*beta).float().reshape(-1, 1)

    coords = torch.tensor(ev_arr[:, :2]).float()
    pos = torch.hstack((ts_sample, coords))

    edge_index = knn_graph(pos, k=32)

    pol = torch.tensor(ev_arr[:, 2]).float().reshape(-1, 1)
    #feature = pol
    feature = torch.hstack((pos, pol))

    graph = Data(x=feature, edge_index=edge_index, pos=pos, y = torch.tensor(gt))
    graph = Cartesian()(graph)

    return graph

class MVSECDataset:

    def __init__(
            self,
            graphs_per_pred = 5
            ):
        self.root = Path('/media/ahmed/drive1/flow-data/mvsec20/MVSECGraphDatset/')
        self.graphs_per_pred = graphs_per_pred

        if len(os.listdir(self.root / 'raw')) > len(os.listdir(self.root / 'processed')):
            print('processing')
            self.process()

    def process(self):
        for h5_file in tqdm(os.listdir(self.root / 'raw'), desc='processing'):
            events = h5py.File(self.root / 'raw' / h5_file)['myDataset']
            events = np.stack([events[col].astype(np.float64) for col in ['x', 'y', 'ts', 'p']]).T
            flow = np.load(self.root / 'gt' / (h5_file[:-3] + '.npy'))

            knots = np.linspace(events[0, 2], events[-1, 2], num=self.graphs_per_pred+1)
            knot_idx = np.searchsorted(events[:, 2], knots)
            for idx, i in enumerate(range(self.graphs_per_pred)):
                outdir = self.root / 'processed' / (h5_file[:-3] + f'_{idx+1}.pt')
                torch.save(make_graph(events[knot_idx[i]:knot_idx[i+1]], gt=flow), outdir)

    def __getitem__(self, idx):
        out_graphs = []
        for i in range(1, self.graphs_per_pred+1):
            out_graphs.append(torch.load(self.root / 'processed' / f'{idx:06}_{i}.pt'))
        return out_graphs
    
    def __len__(self):
        return len(os.listdir(self.root / 'raw'))
    
def collate_fn(list_of_list_of_graphs):
    batch_size = len(list_of_list_of_graphs)
    graphs_per_pred = len(list_of_list_of_graphs[0])
    make_batch = lambda list_of_graphs: Batch.from_data_list(list_of_graphs)
    return [
        make_batch([list_of_list_of_graphs[i][j] for i in range(batch_size)]) 
        for j in range(graphs_per_pred)
        ]

dset = MVSECDataset()
loader = torch.utils.data.DataLoader(dset, batch_size=1, collate_fn=collate_fn)
# print(next(iter(loader)))


data = next(iter(loader))
gpu = torch.device('cuda:' + str(0))
model = eraftv2.ERAFT(4)
model = model.to(gpu)

for it in data:
    it = it.cuda()
    
flow, flow_list = model(data)