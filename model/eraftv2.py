import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .encoder import GraphEncoder
from .corr_graph import CorrGraph
from .corr import CorrBlock
from model.utils import coords_grid, upflow8
from argparse import Namespace
from utils.image_utils import ImagePadder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args



class ERAFT(nn.Module):
    def __init__(self, n_first_channels):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()
        self.args = args
        self.image_padder = ImagePadder(min_size=32)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        args.corr_volumes = 1

        # feature network, context network, and update block
        self.fnet = GraphEncoder(output_dim= 256, n_feature= n_first_channels)        
        self.cnet = GraphEncoder(output_dim= hdim+cdim, n_feature= n_first_channels)
        # self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
        #                             n_first_channels=n_first_channels)
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=0,
        #                             n_first_channels=n_first_channels)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N = img.shape[0]
        H = img.shape[-2]
        W = img.shape[-1]
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, graph_list, img, iters=12, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """
        # Pad Image (for flawless up&downsampling)
        
        # maybe make contiguous
        # for graph in graph_list:
        #     graph = graph.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        im_ht = img.shape[-2] // 8
        im_wd = img.shape[-1] // 8
        cinp = graph_list[0].clone()

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fembedding_list = self.fnet(graph_list)
        
        # print(fembedding_list)
        corr_fn = CorrGraph( [im_ht,im_wd], fembedding_list, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet_graph = self.cnet(cinp)
            cnet = CorrGraph.graph2fmap([im_ht,im_wd], cnet_graph)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(img)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.image_padder.unpad(flow_up))

        return coords1 - coords0, flow_predictions
