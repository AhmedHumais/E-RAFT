import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, voxel_grid, max_pool_x
from typing import Callable, List, Optional, Tuple, Union

#https://github.com/uzh-rpg/aegnn/blob/master/aegnn/models/layer/max_pool.py
class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: List[int], size: int):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"


class MaxPooling(torch.nn.Module):

    def __init__(self, size: List[int], transform: Callable[[Data, ], Data] = None):
        super(MaxPooling, self).__init__()
        self.voxel_size = list(size)
        self.transform = transform

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None, return_data_obj: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        assert edge_index is not None, "edge_index must not be None"

        cluster = voxel_grid(pos[:, 1:3], batch=batch, size=self.voxel_size)  # cluster on spatial dimension
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"
    
    
class MaxPooling2(torch.nn.Module):

    def __init__(self, stride: int, transform: Callable[[Data, ], Data] = None):
        super(MaxPooling2, self).__init__()
        self.voxel_size = [stride+1, stride+1]
        self.scale = stride
        self.transform = transform

    def forward(self, data: Data, return_data_obj: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:

        cluster = voxel_grid(data.pos[:, 1:3], batch=data.batch, size=self.voxel_size)  # cluster on spatial dimension
        # data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        data.pos[:,1:3] = data.pos[:,1:3] // self.scale
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"