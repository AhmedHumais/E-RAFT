import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid
import numpy as np
try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrGraph:
    def __init__(self, graphs_list, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.num_corr_volumes = len(graphs_list)-1
        self.radius = radius
        self.corr_gph_pyramids = []

        corr_pyramid = []
        # all pairs correlation
        for j in range(self.num_corr_volumes):
            fmap1 = CorrGraph.graph2fmap(graphs_list[j])
            fmap2 = CorrGraph.graph2fmap(graphs_list[j+1])

            # for batch stuff we add batch dimension, may need to change later for batch training and modify graph2fmap function
            fmap1 = torch.unsqueeze(fmap1, dim=0)
            fmap2 = torch.unsqueeze(fmap2, dim=0)
                       
            corr = CorrGraph.corr(fmap1, fmap2)
            
            batch, h1, w1, dim, h2, w2 = corr.shape
            corr = corr.reshape(batch*h1*w1, dim, h2, w2)
            
            corr_pyramid.append(corr)
            for i in range(self.num_levels-1):
                corr = F.avg_pool2d(corr, 2, stride=2)
                corr_pyramid.append(corr)
            
            self.corr_gph_pyramids.append(corr_pyramid)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_list = []
        for j in range(self.num_corr_volumes):
            out_pyramid = []
            for i in range(self.num_levels):
                corr = (self.corr_gph_pyramids[j])[i]
                dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
                dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
                delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

                centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
                delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
                coords_lvl = centroid_lvl + delta_lvl

                corr = bilinear_sampler(corr, coords_lvl)
                corr = corr.view(batch, h1, w1, -1)
                out_pyramid.append(corr)

            out = torch.cat(out_pyramid, dim=-1)
            
            out_list.append(out.permute(0, 3, 1, 2).contiguous().float())
        
        return torch.stack(out_list, dim=0)
    
    @staticmethod
    def graph2fmap(graph):
        fmap = np.zeros((graph.x.shape[1], graph.y.shape[1], graph.y.shape[2]))
        for i in range(graph.x.shape[0]):
            fmap[:,int(graph.pos[i,2]), int(graph.pos[i,1])] = graph.x[i,:]

        return torch.Tensor(fmap)            

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())