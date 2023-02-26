
import torch.nn as nn
from torch_geometric.nn import SplineConv
from torch_geometric.transforms import Cartesian
from .maxpooling import MaxPooling, MaxPoolingX, MaxPooling2
from torch_geometric.nn.norm import BatchNorm
# TODO: find the difference between batchnorm and instance norm and if we need it for context and feature networks

class GraphEncoder(nn.Module):

    def __init__(
            self, output_dim=256, 
            n_feature=4
            ) -> None:
        super().__init__()

        pseudo = Cartesian(norm=True, cat=False)

        self.conv1 = SplineConv(n_feature, 32, dim=3, kernel_size=2)
        self.norm1 = BatchNorm(32)

        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=2)
        self.norm2 = BatchNorm(64)
        self.pool2 = MaxPooling2(2, transform=pseudo)

        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=2)
        self.norm3 = BatchNorm(64)
        self.pool3 = MaxPooling2(2, transform=pseudo)
        
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=2)
        self.norm4 = BatchNorm(64)
        self.pool4 = MaxPooling2(2, transform=pseudo)

        self.conv5 = SplineConv(64, 128, dim=3, kernel_size=2)
        self.norm5 = BatchNorm(128)

        self.conv6 = SplineConv(128, output_dim, dim=3, kernel_size=2)
        self.norm6 = BatchNorm(output_dim)
        

    def forward(self, in_data):
        
        out = []
        is_list = isinstance(in_data, tuple) or isinstance(in_data, list)
        if is_list:
            
            for data in in_data:
                data.x = nn.functional.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                data.x = self.norm1(data.x)

                data.x = nn.functional.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                data.x = self.norm2(data.x)
                data = self.pool2(data, return_data_obj=True)

                data.x = nn.functional.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
                data.x = self.norm3(data.x)
                data = self.pool3(data, return_data_obj=True)


                data.x = nn.functional.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
                data.x = self.norm4(data.x)
                data = self.pool4(data, return_data_obj=True)

                data.x = nn.functional.elu(self.conv5(data.x, data.edge_index, data.edge_attr))
                data.x = self.norm5(data.x)

                data.x = nn.functional.elu(self.conv6(data.x, data.edge_index, data.edge_attr))
                data.x = self.norm6(data.x)
                out.append(data)
                # print('run')
        else:
            data = in_data
            data.x = nn.functional.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1(data.x)

            data.x = nn.functional.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm2(data.x)
            data = self.pool2(data, return_data_obj=True)

            data.x = nn.functional.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm3(data.x)
            data = self.pool3(data, return_data_obj=True)

            data.x = nn.functional.elu(self.conv4(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm4(data.x)
            data = self.pool4(data, return_data_obj=True)

            data.x = nn.functional.elu(self.conv5(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm5(data.x)

            data.x = nn.functional.elu(self.conv6(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm6(data.x)
            out = data            
        # print(out)
        return out