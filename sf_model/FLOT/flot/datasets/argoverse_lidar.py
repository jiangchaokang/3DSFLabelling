import os
import glob, pickle
import numpy as np
from .generic import SceneFlowDataset
from .augmentation import DataAugmentation


class lidarArgoverse(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode, dataset_name = "argoverse"):
        super(lidarArgoverse, self).__init__(nb_points)
        self.dataset_name = dataset_name
        self.mode = mode
        self.root_dir = root_dir
        self.filenames = self.get_file_list()
        self.transform = DataAugmentation(prune_threshold = 10)
        self.anchors = np.load("./argoverse/Ablation_slope/315986786560354000/anchors.npy", allow_pickle=True).astype(np.float32)

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        subfolders = []
        for root, dirs, files in os.walk(self.root_dir):
            for dir in dirs:
                subfolder_path = os.path.join(root, dir)
                subfolders.append(subfolder_path)
                if len(subfolders)>18000:
                    return subfolders
        return subfolders

    def load_sequence(self, idx):
        sequence = []  # [Point cloud 1, Point cloud 2]
        pc3 = np.load(os.path.join(self.filenames[idx], "pc3.npy"), allow_pickle=True).astype(np.float32)
        sequence.append(np.load(os.path.join(self.filenames[idx], "pc1.npy"), allow_pickle=True).astype(np.float32))
        global_params = np.load(os.path.join(self.filenames[idx], "global_params.npy"), allow_pickle=True).astype(np.float32)
        perbox_params = np.load(os.path.join(self.filenames[idx], "perbox_params.npy"), allow_pickle=True).astype(np.float32)
        pc2_loaded = self.transform.augment(sequence[0], global_params, perbox_params, self.anchors)
        if pc2_loaded is not None:
            sequence.append(pc2_loaded.numpy())
        else:
            sequence.append(pc3)
        # sequence.append(pc2_loaded.numpy())
        # sequence[0], sequence[1] = sequence[0][:, [2, 1, 0]], sequence[1][:, [2, 1, 0]]
        
        sequence[0], sequence[1] = sequence[0][:, [0, 2, 1]], sequence[1][:, [0, 2, 1]]
        near_mask = np.logical_and.reduce((
            sequence[0][:, 1] < 35.0, sequence[1][:, 1] < 35.0,
            sequence[0][:, 1] > -35.0, sequence[1][:, 1] > -35.0,
            sequence[0][:, 0] < 35.0, sequence[1][:, 0] < 35.0,
            sequence[0][:, 0] > -35.0, sequence[1][:, 0] > -35.0))
        sequence[0] = sequence[0][near_mask]
        sequence[1] = sequence[1][near_mask]
        flow = sequence[1] - sequence[0]

        indices = np.arange(sequence[0].shape[0])
        np.random.shuffle(indices)
        sequence[0] = sequence[0][indices]
        flow = flow[indices]
        np.random.shuffle(indices)
        sequence[1] = sequence[1][indices]

        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            flow,
        ]  # [Occlusion mask, flow]
        return sequence, ground_truth


