

from pathlib import Path
from loader.loader_dsec_gnn import Sequence
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import numpy as np
from utils.visualization import visualize_optical_flow

if __name__ == '__main__':
    data_path = Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_events/zurich_city_01_a')
    seq_path = Path('/media/kucarst3-dlws/HDD3/humais/data/dsec/train_optical_flow/zurich_city_01_a')

    data_set = Sequence(seq_path=seq_path, event_path=data_path)
    loader = DataLoader(data_set, batch_size=1)
    data = next(iter(loader))
    flow_res = data['gt_flow']
    event_frame = data['ref_events']
    event_frame = event_frame.numpy().squeeze()
    ax = plt.figure().add_subplot(projection='3d')
    vals = np.argwhere(event_frame[0,:,:] != 0).T
    ax.scatter(*vals, c = event_frame[event_frame[0,:,:]!=0])
    plt.show()
    flow = flow_res.squeeze().numpy().transpose(2,0,1)
    img, _ = visualize_optical_flow(flow)

    # plt.imshow(img)
    # plt.show()
