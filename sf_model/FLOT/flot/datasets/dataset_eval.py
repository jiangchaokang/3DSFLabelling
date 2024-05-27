import os
import glob, random
import numpy as np
from .generic import SceneFlowDataset


class lidarEval(SceneFlowDataset):
    def __init__(self, root_dir, nb_points, dataset_name = "nuscenes"):
        super(lidarEval, self).__init__(nb_points)
        self.num_points = nb_points
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.filenames = self.get_file_list()

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        """
        npz_files = glob.glob(os.path.join(self.root_dir, '*.npz'))

        return list(npz_files)

    def load_sequence(self, idx):
        # Load data
        data = np.load(self.filenames[idx], allow_pickle=True)
        pc1 = data['pc1'].astype('float32')
        pc2 = data['pc2'].astype('float32')
        flow = data['flow'].astype('float32')
        mask1_flow = data['mask1_tracks_flow']
        mask2_flow = data['mask2_tracks_flow']
        # n1 = len(pc1)
        # n2 = len(pc2)

        # full_mask1 = np.arange(n1)
        # full_mask2 = np.arange(n2)
        # mask1_noflow = np.setdiff1d(full_mask1, mask1_flow, assume_unique=True)
        # mask2_noflow = np.setdiff1d(full_mask2, mask2_flow, assume_unique=True)

        # num_points = self.num_points
        # nonrigid_rate = 0.8
        # rigid_rate = 0.2
        # if n1 >= num_points:
        #     if int(num_points * nonrigid_rate) > len(mask1_flow):
        #         num_points1_flow = len(mask1_flow)
        #         num_points1_noflow = num_points - num_points1_flow
        #     else:
        #         num_points1_flow = int(num_points * nonrigid_rate)
        #         num_points1_noflow = int(num_points * rigid_rate) + 1
        #     sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
        #     try:  # ANCHOR: nuscenes has some cases without nonrigid flows.
        #         sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
        #     except:
        #         sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
        #     sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))

        # else:
        #     sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, num_points - n1, replace=True)), axis=-1)
        # pc1_ = pc1[sample_idx1, :]
        # flow_ = flow[sample_idx1, :]

        # pc1 = pc1_.astype('float32')
        # flow = flow_.astype('float32')
        # pc2 = pc1 + flow

        pc1, pc2, flow = self.genflow(pc1, pc2, mask1_flow, mask2_flow, flow)
        
        pc3 = pc1 + flow
        # pc2 = pc1 + flow
        mask = pc1[:,2] < 35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,1] < 35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,0] < 35.0
        pc1, pc3 = pc1[mask], pc3[mask]

        mask2 = pc2[:,2] < 35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,1] < 35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,0] < 35.0
        pc2= pc2[mask2]

        mask = pc1[:,2] > -35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,1] > -35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,0] > -35.0
        pc1, pc3 = pc1[mask], pc3[mask]

        mask2 = pc2[:,2] > -35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,1] > -35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,0] > -35.0
        pc2= pc2[mask2]

        num_pc = min(pc1.shape[0], pc2.shape[0])
        pc2 = pc2[:, [1,2,0]]  # [:, [1,2,0]]
        pc1, pc3 = pc1[:, [1,2,0]], pc3[:, [1,2,0]]
        flow = pc3 - pc1

        indices = np.arange(num_pc)
        np.random.shuffle(indices)
        pc1 = pc1[indices]
        flow = flow[indices]
        random.shuffle(indices)
        pc2 = pc2[indices]
        pc3 = pc3[indices]

        sequence = [pc1, pc3]  # [Point cloud 1, Point cloud 2]
        
        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            flow,
        ]  # [Occlusion mask, flow]
        return sequence, ground_truth
    
    def genflow(self, pc1, pc2, mask1_flow, mask2_flow, flow):
        n1 = len(pc1)
        n2 = len(pc2)

        full_mask1 = np.arange(n1)
        full_mask2 = np.arange(n2)
        mask1_noflow = np.setdiff1d(full_mask1, mask1_flow, assume_unique=True)
        mask2_noflow = np.setdiff1d(full_mask2, mask2_flow, assume_unique=True)

        
        nonrigid_rate = 0.8
        rigid_rate = 0.2
        num_points =self.num_points
        if n1 >= num_points:
            if int(num_points * nonrigid_rate) > len(mask1_flow):
                num_points1_flow = len(mask1_flow)
                num_points1_noflow = num_points - num_points1_flow
            else:
                num_points1_flow = int(num_points * nonrigid_rate)
                num_points1_noflow = int(num_points * rigid_rate) + 1

            try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
            except:
                sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
            sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
            sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))

            pc1_ = pc1[sample_idx1, :]
            flow_ = flow[sample_idx1, :]

            pc1 = pc1_.astype('float32')
            flow = flow_.astype('float32')

        if n2 >= num_points:
            if int(num_points * nonrigid_rate) > len(mask2_flow):
                num_points2_flow = len(mask2_flow)
                num_points2_noflow = num_points - num_points2_flow
            else:
                num_points2_flow = int(num_points * nonrigid_rate)
                num_points2_noflow = int(num_points * rigid_rate) + 1
                
            try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
            except:
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=True)
            sample_idx2_flow = np.random.choice(mask2_flow, num_points2_flow, replace=False)
            sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))

            pc2_ = pc2[sample_idx2, :]
            pc2 = pc2_.astype('float32')
        return pc1, pc2, flow
    


class lidarKITTIEval(SceneFlowDataset):
    def __init__(self, root_dir, nb_points):
        super(lidarKITTIEval, self).__init__(nb_points)
        self.num_points = nb_points
        self.root_dir = root_dir
        self.filenames = self.get_file_list()

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        """
        npz_files = glob.glob(os.path.join(self.root_dir, "*.npz"))
        npz_files = [os.path.abspath(file) for file in npz_files]
        # res_paths = []
        # for file in npz_files:
        #     res_paths.append(file)
        if len(npz_files) < 2:
            raise ValueError("The number of file paths is less than 2, please make sure there are at least two file paths.")

        return list(npz_files)

    def load_sequence(self, idx):
        # Load data
        try:
            # pc1 = np.load(path, allow_pickle=True)['pc1'].astype(np.float32)
            # pc3 = pc1 + np.load(path, allow_pickle=True)['flow'].astype(np.float32)
            data = np.load(self.filenames[idx], allow_pickle=True)
            pc1 = data["pc1"][data['pc1_cam_mask']].astype(np.float32)
            pc2 = pc1 + data["flow"] # data["pc2"].astype(np.float32)
            # flow = data['flow'].astype('float32')

            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)
            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]
            mask = pc1[:,2] < 35.0
            pc1, pc2 = pc1[mask], pc2[mask]
        except:
            print("Encountered file loading error:\n", self.filenames[idx])
        flow = pc2 - pc1
        indices = np.arange(pc1.shape[0])
        np.random.shuffle(indices)
        pc1 = pc1[indices]
        flow = flow[indices]
        np.random.shuffle(indices)
        pc2 = pc2[indices]
        sequence = [pc1, pc2]  # [Point cloud 1, Point cloud 2]

        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
            flow,
        ]  # [Occlusion mask, flow]
        
        return sequence, ground_truth
    



class lidarWaymoEval(SceneFlowDataset):
    def __init__(self, root_dir, nb_points):
        super(lidarWaymoEval, self).__init__(nb_points)
        self.num_points = nb_points
        self.root_dir = root_dir
        self.filenames = self.get_file_list()

    def __len__(self):

        return len(self.filenames)

    def get_file_list(self):
        """
        Find and filter out paths to all examples in the dataset. 
        """
        npz_files = glob.glob(os.path.join(self.root_dir, '*.npz'))

        return list(npz_files)

    def load_sequence(self, idx):
        # Load data
        data = np.load(self.filenames[idx], allow_pickle=True)
        pc1 = data['pc1'].astype('float32')
        pc2 = data['pc2'].astype('float32')
        flow = data['flow'].astype('float32')
        pc3 = pc1 + flow
        # pc2 = pc1 + flow
        mask = pc1[:,2] < 35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,1] < 35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,0] < 35.0
        pc1, pc3 = pc1[mask], pc3[mask]

        mask2 = pc2[:,2] < 35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,1] < 35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,0] < 35.0
        pc2= pc2[mask2]

        mask = pc1[:,2] > -35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,1] > -35.0
        pc1, pc3 = pc1[mask], pc3[mask]
        mask = pc1[:,0] > -35.0
        pc1, pc3 = pc1[mask], pc3[mask]

        mask2 = pc2[:,2] > -35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,1] > -35.0
        pc2 = pc2[mask2]
        mask2 = pc2[:,0] > -35.0
        pc2= pc2[mask2]

        num_pc = min(pc1.shape[0], pc2.shape[0])
        pc2 = pc2[:, [0, 2, 1]]
        pc1, pc3 = pc1[:, [0, 2, 1]], pc3[:, [0, 2, 1]]
        flow = pc3 - pc1

        indices = np.arange(num_pc)
        np.random.shuffle(indices)
        pc1 = pc1[indices]
        flow = flow[indices]
        random.shuffle(indices)
        pc2 = pc2[indices]
        pc3 = pc3[indices]
        # pc1, pc2, pc3 = pc1[:num_pc,:], pc2[:num_pc,:], pc3[:num_pc,:]
        sequence = [pc1, pc3]  # [Point cloud 1, Point cloud 2]

        ground_truth = [
            np.ones_like(sequence[0][:, 0:1]),
           flow,
        ]  # [Occlusion mask, flow]
        return sequence, ground_truth