import numpy
import os
import pandas
from PIL import Image
import random
import torch
from itertools import chain
import h5py
import json

from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.transforms import Cartesian
from utils.dsec_utils import VoxelGrid

"""@TODO: gt should be only in first graph to save GPU memeory space"""
def make_graph(ev_arr, gt=None, beta=0.5e4, k=16):
    """
    ev_arr should be an n_events x 4 size array such that
    ev_arr[:, 0] is x
    ev_arr[:, 1] is y
    ev_arr[:, 2] is p
    ev_arr[:, 3] is t
    """
    # ts_sample = ev_arr[:, 3] - ev_arr[0, 3]     why is it here? should be outside graph making
    ts_sample = ev_arr[:,3]
    ts_sample = torch.tensor(ts_sample*beta).float().reshape(-1, 1)

    coords = torch.tensor(ev_arr[:, :2]).float()
    pos = torch.hstack((ts_sample, coords))

    edge_index = knn_graph(pos, k=k)

    pol = torch.tensor(ev_arr[:, 2]).float().reshape(-1, 1)
    #feature = pol
    feature = torch.hstack((pos, pol))
    y = torch.tensor(gt)[None, :] if not gt is None else gt
    graph = Data(x=feature, edge_index=edge_index, pos=pos, y = y)
    graph = Cartesian()(graph)

    return graph

def make_graph_from_voxel(grid: torch.Tensor, gt=None, k=8):
    mask = torch.nonzero(grid, as_tuple=True)

    if mask[0].size()[0] <= 100:
        print('no valid events found jumping to random item')
        return None

    t = torch.tensor(mask[0].numpy()).float().reshape(-1, 1)
    y = torch.tensor(mask[1].numpy()).float().reshape(-1, 1)
    x = torch.tensor(mask[2].numpy()).float().reshape(-1, 1)
    val = torch.tensor(grid[mask].numpy()).float().reshape(-1, 1)
    feature = torch.hstack((x,y,t,val))
    pos = torch.hstack((t,x,y))
    # edge_index = knn_graph(pos, k=k)
    edge_index = radius_graph(pos, r=7, max_num_neighbors=16,flow='source_to_target')
    y = torch.tensor(gt)[None, :] if not gt is None else gt

    # graph = Data(x=feature, edge_index=edge_index, pos=pos, y=y)
    graph = Data(x=val, edge_index=edge_index, pos=pos, y=y)
    graph = Cartesian()(graph)
    return graph 
class EventSequence(object):
    def __init__(self, dataframe, params, features=None, timestamp_multiplier=None, convert_to_relative=False):
        if isinstance(dataframe, pandas.DataFrame):
            self.feature_names = dataframe.columns.values
            self.features = dataframe.to_numpy()
        else:
            self.feature_names = numpy.array(['ts', 'x', 'y', 'p'], dtype=object)
            if features is None:
                self.features = numpy.zeros([1, 4])
            else:
                self.features = features
        self.image_height = params['height']
        self.image_width = params['width']
        if not self.is_sorted():
            self.sort_by_timestamp()
        if timestamp_multiplier is not None:
            self.features[:,0] *= timestamp_multiplier
        if convert_to_relative:
            self.absolute_time_to_relative()

    def get_sequence_only(self):
        return self.features

    def __len__(self):
        return len(self.features)

    def __add__(self, sequence):
        event_sequence = EventSequence(dataframe=None,
                                       features=numpy.concatenate([self.features, sequence.features]),
                                       params={'height': self.image_height,
                                               'width': self.image_width})
        return event_sequence

    def is_sorted(self):
        return numpy.all(self.features[:-1, 0] <= self.features[1:, 0])

    def sort_by_timestamp(self):
        if len(self.features[:, 0]) > 0:
            sort_indices = numpy.argsort(self.features[:, 0])
            self.features = self.features[sort_indices]

    def absolute_time_to_relative(self):
        """Transforms absolute time to time relative to the first event."""
        start_ts = self.features[:,0].min()
        assert(start_ts == self.features[0,0])
        self.features[:,0] -= start_ts


def get_image(image_path):
    try:
        im = Image.open(image_path)
        # print(image_path)
        return numpy.array(im)
    except OSError:
        raise


def get_events(event_path):
    # It's possible that there is no event file! (camera standing still)
    try:
        f = pandas.read_hdf(event_path, "myDataset")
        return f[['ts', 'x', 'y', 'p']]
    except OSError:
        print("No file " + event_path)
        print("Creating an array of zeros!")
        return 0


def get_ts(path, i, type='int'):
    try:
        f = open(path, "r")
        if type == 'int':
            return int(f.readlines()[i])
        elif type == 'double' or type == 'float':
            return float(f.readlines()[i])
    except OSError:
        raise


def get_batchsize(path_dataset):
    filepath = os.path.join(path_dataset, "cam0", "timestamps.txt")
    try:
        f = open(filepath, "r")
        return len(f.readlines())
    except OSError:
        raise


def get_batch(path_dataset, i):
    return 0


def dataset_paths(dataset_name, path_dataset, subset_number=None):
    cameras = {'cam0': {}, 'cam1': {}, 'cam2': {}, 'cam3': {}}
    if subset_number is not None:
        dataset_name = dataset_name + "_" + str(subset_number)
    paths = {'dataset_folder': os.path.join(path_dataset, dataset_name)}

    # For every camera, define its path
    for camera in cameras:
        cameras[camera]['image_folder'] = os.path.join(paths['dataset_folder'], camera, 'image_raw')
        cameras[camera]['event_folder'] = os.path.join(paths['dataset_folder'], camera, 'events')
        cameras[camera]['disparity_folder'] = os.path.join(paths['dataset_folder'], camera, 'disparity_image')
        cameras[camera]['depth_folder'] = os.path.join(paths['dataset_folder'], camera, 'depthmap')
    cameras["timestamp_file"] = os.path.join(paths['dataset_folder'], 'cam0', 'timestamps.txt')
    cameras["image_type"] = ".png"
    cameras["event_type"] = ".h5"
    cameras["disparity_type"] = ".png"
    cameras["depth_type"] = ".tiff"
    cameras["indexing_type"] = "%0.6i"
    paths.update(cameras)
    return paths


def get_indices(path_dataset, dataset, filter, shuffle=False):
    samples = []
    for dataset_name in dataset:
        for subset in dataset[dataset_name]:
            # Get all the dataframe paths
            paths = dataset_paths(dataset_name, path_dataset, subset)

            # import timestamps
            ts = numpy.loadtxt(paths["timestamp_file"])

            # frames = []
            # For every timestamp, import according data
            for idx in eval(filter[dataset_name][str(subset)]):
                frame = {}
                frame['dataset_name'] = dataset_name
                frame['subset_number'] = subset
                frame['index'] = idx
                frame['timestamp'] = ts[idx]
                samples.append(frame)
    # shuffle dataset
    if shuffle:
        random.shuffle(samples)
    return samples


def get_flow_h5(flow_path):
    scaling_factor = 0.05 # seconds/frame
    f = h5py.File(flow_path, 'r')
    height, width = int(f['header']['height']), int(f['header']['width'])
    assert(len(f['x']) == height*width)
    assert(len(f['y']) == height*width)
    x = numpy.array(f['x']).reshape([height,width])*scaling_factor
    y = numpy.array(f['y']).reshape([height,width])*scaling_factor
    return numpy.stack([x,y])


def get_flow_npy(flow_path):
    # Array 2,height, width
    # No scaling needed.
    return numpy.load(flow_path, allow_pickle=True)


def get_pose(pose_path, index):
    pose = pandas.read_csv(pose_path, delimiter=',').loc[index].to_numpy()
    # Convert Timestamp to int (as all the other timestamps)
    pose[0] = int(pose[0])
    return pose


def load_config(path, datasets):
    config = {}
    for dataset_name in datasets:
        config[dataset_name] = {}
        for subset in datasets[dataset_name]:
            name = "{}_{}".format(dataset_name, subset)
            try:
                config[dataset_name][subset] = json.load(open(os.path.join(path, name, "config.json")))
            except:
                print("Could not find config file for dataset" + dataset_name + "_" + str(subset) +
                      ". Please check if the file 'config.json' is existing in the dataset-scene directory")
                raise
    return config


def mvsec_collate_fn(list_of_list_of_graphs):
    batch_size = len(list_of_list_of_graphs)
    graphs_per_pred = len(list_of_list_of_graphs[0])
    make_batch = lambda list_of_graphs: Batch.from_data_list(list_of_graphs)
    return [
        make_batch([list_of_list_of_graphs[i][j] for i in range(batch_size)]) 
        for j in range(graphs_per_pred)
        ]

def dsec_collate_fn(ref_tgt_valid_list):
    ref_tgt_list = []
    valid = []

    for rt, v in ref_tgt_valid_list:
        ref_tgt_list.append(rt)
        valid.append(v)

    batch_size = len(ref_tgt_valid_list)
    make_batch = lambda list_of_graphs: Batch.from_data_list(list_of_graphs)
    graph_batch = [
        make_batch([ref_tgt_list[i][j] for i in range(batch_size)]) 
        for j in range(2)
        ]
    valid_batch = torch.stack(valid)
    return graph_batch, valid_batch