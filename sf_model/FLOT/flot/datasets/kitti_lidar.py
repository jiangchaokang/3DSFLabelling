import os, pickle, pdb
import glob
import numpy as np
from .generic import SceneFlowDataset
from .augmentation import DataAugmentation

class lidarKITTI(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, mode):
        """
        Parameters
        ----------
        root_dir : str
            Path to root directory containing the datasets.
        nb_points : int
            Maximum number of points in point clouds.
        mode : str
            'train': training dataset.
            
            'val': validation dataset.
            
            'test': test dataset

        """

        super(lidarKITTI, self).__init__(nb_points)

        self.mode = mode
        self.root_dir = root_dir
        self.filenames = self.get_file_list()
        self.transform = DataAugmentation()
        self.anchors_stereo = np.load('./kitti-od/data_odometry_velodyne/sf_kitti_lidar2/00_pair_0/anchors_stereo.npy', allow_pickle=True).astype(np.float32)

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        
        """
        # subfolders = [entry.path for entry in os.scandir(self.root_dir) if entry.is_dir()]
        with open('./kitti-od/data_odometry_velodyne/list_lidarkitti.pkl', 'rb') as file:
            subfolders = pickle.load(file)

        return list(subfolders)

    def load_sequence(self, idx):
        """
        Load a sequence of point clouds.

        Parameters
        ----------
        idx : int
            Index of the sequence to load.

        Returns
        -------
        sequence : list(np.array, np.array)
            List [pc1, pc2] of point clouds between which to estimate scene 
            flow. pc1 has size n x 3 and pc2 has size m x 3.
            
        ground_truth : list(np.array, np.array)
            List [mask, flow]. mask has size n x 1 and pc1 has size n x 3. 
            flow is the ground truth scene flow between pc1 and pc2. mask is 
            binary with zeros indicating where the flow is not valid/occluded.

        """

        # Load data
        
        sequence = []  # [Point cloud 1, Point cloud 2]
        # for fname in ["pc1.npy", "pc3.npy"]:
        #     pc = np.load(os.path.join(self.filenames[idx], fname), allow_pickle=True).astype(np.float32)
        #     # pc[..., 0] *= -1
        #     # pc[..., -1] *= -1
        #     pc = pc[:, [2, 0, 1]][:, [1, 0, 2]]
        #     sequence.append(pc)
        # is_ground = np.logical_and(sequence[1][:,1] < -1.4, sequence[0][:,1] < -1.4)
        # not_ground = np.logical_not(is_ground)

        # sequence[1] = sequence[1][not_ground]
        # sequence[0] = sequence[0][not_ground]
        sequence.append(np.load(os.path.join("../../",self.filenames[idx], "pc1.npy"), allow_pickle=True).astype(np.float32))
        global_params = np.load(os.path.join("../../",self.filenames[idx], "global_params.npy"), allow_pickle=True).astype(np.float32)
        perbox_params = np.load(os.path.join("../../",self.filenames[idx], "perbox_params.npy"), allow_pickle=True).astype(np.float32)
        pc2_loaded = self.transform.augment(sequence[0], global_params, perbox_params, self.anchors_stereo)
        # pdb.set_trace()
        sequence.append(pc2_loaded.numpy())
        sequence[0], sequence[1] = sequence[0][:, [2, 0, 1]][:, [1, 0, 2]], sequence[1][:, [2, 0, 1]][:, [1, 0, 2]]
        is_ground = np.logical_and(sequence[1][:,1] < -1.4, sequence[0][:,1] < -1.4)
        not_ground = np.logical_not(is_ground)
        sequence[1] = sequence[1][not_ground]
        sequence[0] = sequence[0][not_ground]
        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            sequence[1] - sequence[0],
        ]  # [Occlusion mask, flow]
        return sequence, ground_truth


