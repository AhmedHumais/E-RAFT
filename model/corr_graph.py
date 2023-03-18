import torch
import torch.nn.functional as F
from model.utils import bilinear_sampler, coords_grid
import numpy as np
# try:
#     import alt_cuda_corr
# except:
#     # alt_cuda_corr is not compiled
#     pass

# TODO: find a better way to deal with batch stuff in graph2fmap

class CorrGraph:
    def __init__(self,imsz, graphs_list, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.num_corr_volumes = len(graphs_list)-1
        self.radius = radius
        self.corr_gph_pyramids = []

        corr_pyramid = []
        # all pairs correlation
        for j in range(self.num_corr_volumes):
            fmap1 = CorrGraph.graph2fmap(imsz, graphs_list[j])
            fmap2 = CorrGraph.graph2fmap(imsz, graphs_list[j+1])
                       
            corr = CorrGraph.corr(fmap1, fmap2)
            for k in range(self.num_corr_volumes-j-1):
                fmap = CorrGraph.graph2fmap(imsz, graphs_list[j+k+2])
                corr = corr + CorrGraph.corr(fmap1, fmap)
            
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

            out_list.append(torch.cat(out_pyramid, dim=-1))
            
        out = torch.cat(out_list, dim=-1)
        
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    @staticmethod
    def graph2fmap(im_size,graph_batch):
        fmap = np.zeros((graph_batch.num_graphs, graph_batch.x.shape[1], im_size[0], im_size[1]))
        fmap = torch.Tensor(fmap).cuda()
        for i in range(graph_batch.x.shape[0]):
            if graph_batch.pos[i,2] >= im_size[0] or graph_batch.pos[i,1] >= im_size[1] or graph_batch.pos[i,2] < 0 or graph_batch.pos[i,1] < 0:
                continue
            else:
                fmap[graph_batch.batch[i],:,int(graph_batch.pos[i,2]), int(graph_batch.pos[i,1])] = graph_batch.x[i,:]

        return fmap       

    @staticmethod
    def corr(fmap1, fmap2):
        
        # print(fmap1.shape)
        # print(fmap2.shape)
        
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())
