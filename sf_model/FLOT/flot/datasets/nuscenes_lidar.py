import os
import glob, pickle
import numpy as np
from .generic import SceneFlowDataset
from .augmentation import DataAugmentation

class lidarNuScenes(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode, dataset_name = "nuscenes"):
        super(lidarNuScenes, self).__init__(nb_points)
        self.dataset_name = dataset_name
        self.mode = mode
        self.root_dir = root_dir
        self.filenames = self.get_file_list()
        self.transform = DataAugmentation(prune_threshold = 5)
        self.anchors = np.load("./nuscenes/nuscene_lidar_sf/n008-2018-08-30-15-31-50-0400__LIDAR_TOP__1535657710848953/anchors.npy", allow_pickle=True).astype(np.float32)

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        """
        # subfolders = [entry.path for entry in os.scandir(self.root_dir) if entry.is_dir()]
        with open('./kitti-od/data_odometry_velodyne/list_lidarnuscenes.pkl', 'rb') as file:
            subfolders = pickle.load(file)

        return list(subfolders)

    def load_sequence(self, idx):
        # Load data
        sequence = []  # [Point cloud 1, Point cloud 2]
        # for fname in ["pc1.npy", "pc3.npy"]:
        #     pc = np.load(os.path.join(self.filenames[idx], fname), allow_pickle=True).astype(np.float32)
        #     pc = pc[:, [2, 1, 0]]
        #     pc[:, 2:3] = pc[:, 2:3]*(-1.0)+2.05
        #     sequence.append(pc[:, [1,2,0]])
        
        sequence.append(np.load(os.path.join("../../../",self.filenames[idx], "pc1.npy"), allow_pickle=True).astype(np.float32))
        global_params = np.load(os.path.join("../../../",self.filenames[idx], "global_params.npy"), allow_pickle=True).astype(np.float32)
        perbox_params = np.load(os.path.join("../../../",self.filenames[idx], "perbox_params.npy"), allow_pickle=True).astype(np.float32)
        pc2_loaded = self.transform.augment(sequence[0], global_params, perbox_params, self.anchors)
        sequence.append(pc2_loaded.numpy())
        sequence[0], sequence[1] = sequence[0][:, [2, 1, 0]], sequence[1][:, [2, 1, 0]]
        sequence[0][:, 2:3], sequence[1][:, 2:3] = sequence[0][:, 2:3]*(-1.0)+2.05, sequence[1][:, 2:3]*(-1.0)+2.05
        sequence[0], sequence[1] = sequence[0][:, [1,2,0]], sequence[1][:, [1,2,0]]

        near_mask = np.logical_and.reduce((
        sequence[0][:, 0] > -25.0, sequence[1][:, 1] > -25.0,
        sequence[0][:, 0] < 25.0, sequence[1][:, 1] < 25.0,
        sequence[0][:, 1] > 0.80, sequence[1][:, 1] > 0.80,
        sequence[0][:, 2] < 35.0, sequence[1][:, 2] < 35.0,
        sequence[0][:, 2] > -35.0, sequence[1][:, 2] > -35.0))
        sequence[0] = sequence[0][near_mask]
        sequence[1] = sequence[1][near_mask]

        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            sequence[1] - sequence[0],
        ]  # [Occlusion mask, flow]
        return sequence, ground_truth


