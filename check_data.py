

from pathlib import Path
from loader.loader_dsec_gnn import Sequence
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import numpy as np
from utils.visualization import visualize_optical_flow
from loader.utils import dsec_collate_fn

if __name__ == '__main__':
    data_path = [Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_01_a'),
                 Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_02_a'),
                 Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_02_c')]
    seq_path = [Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_01_a'),
                Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_02_a'),
                Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_02_c')]

    data_set = Sequence(flow_paths=seq_path, event_paths=data_path)
    loader = DataLoader(data_set, batch_size=1, collate_fn=dsec_collate_fn)
    rt, val = next(iter(loader))
    print(rt, '\n\n\n\n\n\n',val.shape)
    #print(rt, val.shape)
    flow_res = val[:,0]
    flow = flow_res.squeeze().numpy()
    img, _ = visualize_optical_flow(flow)

    plt.imshow(img)
    plt.show()


