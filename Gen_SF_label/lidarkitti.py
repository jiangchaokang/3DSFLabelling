import os
import torch
import logging

import numpy as np
import torch.utils.data as data

projection_matrix = np.array([[7.25995070e+02, 6.04164924e+02, -9.75088151e+00, 4.48572800e+01],
                              [-5.84187278e+00, 1.69760552e+02, -7.22248117e+02, 2.16379100e-01],
                              [ 7.40252700e-03, 9.99963100e-01, -4.35161400e-03, 2.74588400e-03]])
camera_size = (375, 1242)

def normal_frame(points):
    # x = -points[...,0:1]
    # y = points[...,2:3]
    # z = points[...,1:2]
    # return np.concatenate((x,y,z), axis = -1)
    x = points[...,0:1]
    y = points[...,1:2]
    z = points[...,2:3]
    return np.concatenate((y,x,z), axis = -1)
    return points

# def normal_frame_nusc(points):
#     x = points[...,1:2]+1.8
#     y = points[...,0:1]
#     z = -points[...,2:3]

def normal_frame_nusc(points):
    x = points[...,1:2]+0.2
    y = points[...,0:1]
    z = -points[...,2:3]

    return np.concatenate((z,y,x), axis = -1)

def rot_normal_frame(R):
    """
    Converts a rotation matrix R into normal coordinates
    :param R: 3x3 numpy array
    :return: 3x3 numpy array
    """
    return np.array([[R[0,0], -R[0,2], -R[0,1]], [-R[2,0], R[2,2], R[2,1]], [-R[1,0], R[1,2], R[1,1]]])

def collate_fn(data):
    output = [[to_tensor(data[j][i]) for j in range(len(data))] for i in range(len(data[0]))]
    return tuple(output)

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, str):
        return x
    else:
        raise ValueError("Can not convert to torch tensor {}".format(x))

class MELidarDataset(data.Dataset):
    def __init__(self, phase, data_filename, config):

        self.files = []
        self.root = config['data']['root']
        self.config = config
        self.num_points = config['misc']['num_points']
        self.remove_ground = config['data']['remove_ground']
        if ('crop' in config['data']):
            self.crop = config['data']['crop']
        else:
            self.crop = 'full'
        self.dataset = config['data']['dataset']
        self.only_near_points = config['data']['only_near_points']
        self.filter_normals = config['data']['filter_normals']
        self.phase = phase

        self.randng = np.random.RandomState()
        self.device = torch.device('cuda' if (torch.cuda.is_available() and config['misc']['use_gpu']) else 'cpu')

        self.augment_data = config['data']['augment_data']

        logging.info("Loading the subset {} from {}".format(phase, self.root))

        # subset_names = open(self.DATA_FILES[phase]).read().split()
        subset_names = open(data_filename).read().split()

        for name in subset_names:
            self.files.append(name)

    def __getitem__(self, idx):
        file = os.path.join(self.root, self.files[idx])
        file_name = file.replace(os.sep, '/').split('/')[-1]

        # Load the data
        data = np.load(file)
        pc_1 = data['pc1']
        pc_2 = data['pc2']

        if 'normvector1' in data:
            pc1_normals = data['normvector1']
        else:
            pc1_normals = np.zeros_like(pc_1)
        
        if 'normvector2' in data:
            pc2_normals = data['normvector2']
        else:
            pc2_normals = np.zeros_like(pc_2)
        # print('pc1_normals',pc1_normals,pc1_normals.shape)
        # print('pc2_normals',pc2_normals,pc2_normals.shape)
        if 'pc1_cam_mask' in data:
            pc1_cam_mask = data['pc1_cam_mask']
        elif 'front_mask_s' in data:
            pc1_cam_mask = data['front_mask_s']
        else:
            pc1_cam_mask = np.ones(pc_1[:,0].shape, dtype=np.bool_)

        if 'pc2_cam_mask' in data:
            pc2_cam_mask = data['pc2_cam_mask']
        elif 'front_mask_t' in data:
            pc2_cam_mask = data['front_mask_t']
        else:
            pc2_cam_mask = np.ones(pc_2[:,0].shape, dtype=np.bool_)
        # pc1_cam_mask = np.ones(pc_1[:,0].shape, dtype=np.bool)
        # pc2_cam_mask = np.ones(pc_2[:,0].shape, dtype=np.bool)

        if 'pose_s' in data:
            pose_1 = data['pose_s']
        else:
            pose_1 = np.eye(4)

        if 'pose_t' in data:
            pose_2 = data['pose_t']
        else:
            pose_2 = np.eye(4)

        if 'sem_label_s' in data:
            s_labels_1 = data['sem_label_s']
        else:
            s_labels_1 = np.zeros(pc_1.shape[0])

        if 'sem_label_t' in data:
            s_labels_2 = data['sem_label_t']
        else:
            s_labels_2 = np.zeros(pc_2.shape[0])

        if 'mot_label_s' in data:
            m_labels_1 = data['mot_label_s']
        elif 'inst_pc1' in data:
            m_labels_1 = data['inst_pc1']
        else:
            m_labels_1 = np.zeros(pc_1.shape[0])

        if 'mot_label_t' in data:
            m_labels_2 = data['mot_label_t']
        elif 'inst_pc2' in data:
            m_labels_2 = data['inst_pc2']
        else:
            m_labels_2 = np.zeros(pc_2.shape[0])

        if 'flow' in data:
            flow = data['flow']
        else:
            flow = np.zeros((np.sum(pc1_cam_mask), 3), dtype=pc_1.dtype)

        labels_1 = np.logical_and(m_labels_1, s_labels_1!=254)
        labels_2 = np.logical_and(m_labels_2, s_labels_2!=254)

        if self.dataset == 'NuScenes_ME':
            pc_1 = normal_frame_nusc(pc_1)
            pc_2 = normal_frame_nusc(pc_2)
            pc_1[:,2]-=1.9
            pc_2[:,2]-=1.9
            pc1_normals = normal_frame_nusc(pc1_normals)
            pc2_normals = normal_frame_nusc(pc2_normals)
            flow = normal_frame_nusc(flow)
        else:
            pc_1 = normal_frame(pc_1)
            pc_2 = normal_frame(pc_2)
            pc1_normals = normal_frame(pc1_normals)
            pc2_normals = normal_frame(pc2_normals)
            flow = normal_frame(flow)


        # Remove the ground and far away points
        # In stereoKITTI the direct correspondences are provided therefore we remove,
        # if either of the points fullfills the condition (as in hplflownet, flot, ...)

        if self.remove_ground:
            # is_not_ground_s = (pc_1[:, 2] > -1.4)
            # is_not_ground_t = (pc_2[:, 2] > -1.4)
            is_not_ground_s = ~data['ground1_mask']
            is_not_ground_t = ~data['ground2_mask']

            pc_1 = pc_1[is_not_ground_s, :]
            pc1_normals = pc1_normals[is_not_ground_s]
            labels_1 = labels_1[is_not_ground_s]
            flow = flow[is_not_ground_s[pc1_cam_mask], :]
            pc1_cam_mask = pc1_cam_mask[is_not_ground_s]

            pc_2 = pc_2[is_not_ground_t, :]
            pc2_normals = pc2_normals[is_not_ground_t]
            labels_2 = labels_2[is_not_ground_t]
            pc2_cam_mask = pc2_cam_mask[is_not_ground_t]

        if self.filter_normals:
            horizontal_normals_s = np.abs(pc1_normals[:, -1]) < .85
            horizontal_normals_t = np.abs(pc2_normals[:, -1]) < .85

            pc_1 = pc_1[horizontal_normals_s]
            pc1_normals = pc1_normals[horizontal_normals_s]
            labels_1 = labels_1[horizontal_normals_s]
            flow = flow[horizontal_normals_s[pc1_cam_mask], :]
            pc1_cam_mask = pc1_cam_mask[horizontal_normals_s]

            pc_2 = pc_2[horizontal_normals_t]
            pc2_normals = pc2_normals[horizontal_normals_t]
            labels_2 = labels_2[horizontal_normals_t]
            pc2_cam_mask = pc2_cam_mask[horizontal_normals_t]

        if self.only_near_points:
            is_near_s = (np.amax(np.abs(pc_1), axis=1) < 52)  # 35
            is_near_t = (np.amax(np.abs(pc_2), axis=1) < 52)  # 35

            pc_1 = pc_1[is_near_s, :]
            pc1_normals = pc1_normals[is_near_s]
            labels_1 = labels_1[is_near_s]
            flow = flow[is_near_s[pc1_cam_mask], :]
            pc1_cam_mask = pc1_cam_mask[is_near_s]

            pc_2 = pc_2[is_near_t, :]
            pc2_normals = pc2_normals[is_near_t]
            labels_2 = labels_2[is_near_t]
            pc2_cam_mask = pc2_cam_mask[is_near_t]

        if self.crop=='front':
            is_front_s = pc_1[:, 1]>=(np.abs(pc_1[:, 0])-8)
            is_front_t = pc_2[:, 1]>=(np.abs(pc_2[:, 0])-8)

            pc_1 = pc_1[is_front_s, :]
            pc1_normals = pc1_normals[is_front_s]
            labels_1 = labels_1[is_front_s]
            flow = flow[is_front_s[pc1_cam_mask], :]
            pc1_cam_mask = pc1_cam_mask[is_front_s]

            pc_2 = pc_2[is_front_t, :]
            pc2_normals = pc2_normals[is_front_t]
            labels_2 = labels_2[is_front_t]
            pc2_cam_mask = pc2_cam_mask[is_front_t]

        elif self.crop=='camera':
            pc_1 = pc_1[pc1_cam_mask, :]
            pc1_normals = pc1_normals[pc1_cam_mask]
            labels_1 = labels_1[pc1_cam_mask]
            pc1_cam_mask = pc1_cam_mask[pc1_cam_mask]

            pc_2 = pc_2[pc2_cam_mask, :]
            pc2_normals = pc2_normals[pc2_cam_mask]
            labels_2 = labels_2[pc2_cam_mask]
            pc2_cam_mask = pc2_cam_mask[pc2_cam_mask]

        # Augment the point cloud by randomly rotating and translating them (recompute the ego-motion if augmention is applied!)
        if self.augment_data and self.phase != 'test':
            T_1 = np.eye(4)
            T_2 = np.eye(4)

            T_1[0:3, 3] = (np.random.rand(3) - 0.5) * 0.5
            T_2[0:3, 3] = (np.random.rand(3) - 0.5) * 0.5

            T_1[1, 3] = (np.random.rand(1) - 0.5) * 0.1
            T_2[1, 3] = (np.random.rand(1) - 0.5) * 0.1

            pc_1 = (np.matmul(T_1[0:3, 0:3], pc_1.transpose()) + T_1[0:3, 3:4]).transpose()
            pc_2 = (np.matmul(T_2[0:3, 0:3], pc_2.transpose()) + T_2[0:3, 3:4]).transpose()

            pose_1 = np.matmul(pose_1, np.linalg.inv(T_1))
            pose_2 = np.matmul(pose_2, np.linalg.inv(T_2))

            rel_trans = np.linalg.inv(pose_2) @ pose_1

            R_ego = rel_trans[0:3, 0:3]
            t_ego = rel_trans[0:3, 3:4]
        else:
            # Compute relative pose that transform the point from the source point cloud to the target
            # rel_trans = np.linalg.inv(pose_2) @ pose_1
            # rel_trans = self.cal_pose0to1(pose_1, pose_2)
            rel_trans = data['ego_motion']
            R_ego = rel_trans[0:3, 0:3]
            t_ego = rel_trans[0:3, 3:4]

        # Sample n points for evaluation before the voxelization
        # If less than desired points are available just consider the maximum
        # if pc_1.shape[0] > self.num_points:
        #     idx_1 = np.random.choice(pc_1.shape[0], self.num_points, replace=False)
        # else:
        #     idx_1 = np.random.choice(pc_1.shape[0], pc_1.shape[0], replace=False)

        # if pc_2.shape[0] > self.num_points:
        #     idx_2 = np.random.choice(pc_2.shape[0], self.num_points, replace=False)
        # else:
        #     idx_2 = np.random.choice(pc_2.shape[0], pc_2.shape[0], replace=False)

        idx_1 = np.arange(pc_1.shape[0])
        idx_2 = np.arange(pc_2.shape[0])

        pc_1_eval = pc_1[idx_1, :]
        pc1_normals_eval = pc1_normals[idx_1]
        flow_idx = np.cumsum(pc1_cam_mask)-1
        flow_idx = flow_idx[idx_1[pc1_cam_mask[idx_1]]]
        assert np.all(flow_idx>=0)
        flow_eval = flow[flow_idx]
        labels_1_eval = labels_1[idx_1]
        pc1_cam_mask = pc1_cam_mask[idx_1]

        pc_2_eval = pc_2[idx_2, :]
        pc2_normals_eval = pc2_normals[idx_2]
        labels_2_eval = labels_2[idx_2]
        pc2_cam_mask = pc2_cam_mask[idx_2]

        pc_1_eval = pc_1_eval.astype(np.float32)
        pc_2_eval = pc_2_eval.astype(np.float32)
        pc1_normals_eval = pc1_normals_eval.astype(np.float32)
        pc2_normals_eval = pc2_normals_eval.astype(np.float32)
        flow_eval = flow_eval.astype(np.float32)
        labels_1_eval = labels_1_eval.astype(np.float32)
        labels_2_eval = labels_2_eval.astype(np.float32)

        # R_ego = np.transpose(rot_normal_frame(R_ego)).astype(np.float32)
        # t_ego = normal_frame(t_ego.reshape(3)).astype(np.float32)

        return (pc_1_eval, pc_2_eval, pc1_normals_eval, pc2_normals_eval, pc1_cam_mask, pc2_cam_mask, labels_1_eval, labels_2_eval, R_ego, t_ego, flow_eval, file)

    def __len__(self):
        return len(self.files)

    def reset_seed(self, seed=41):
        logging.info('Resetting the data loader seed to {}'.format(seed))
        self.randng.seed(seed)

    def cal_pose0to1(self, pose0, pose1):
        """
        Note(Qingwen 2023-12-05 11:09):
        Don't know why but it needed set the pose to float64 to calculate the inverse 
        otherwise it will be not expected result....
        """
        pose1_inv = np.eye(4, dtype=np.float64)
        pose1_inv[:3, :3] = pose1[:3, :3].T
        pose1_inv[:3, 3] = (pose1[:3, :3].T * -pose1[:3, 3]).sum(axis=1)
        pose_0to1 = np.matmul(pose1_inv, pose0.astype(np.float64))
        return pose_0to1.astype(np.float32)

class StereoKITTI_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'test': './configs/kittisf_files.txt'
    }

class SemanticKITTI_ME(MELidarDataset):
    # 3D Match dataset all files
    DATA_FILES = {
        'test': './configs/semantic_kitti_files.txt'
    }

class LidarKITTI_ME(MELidarDataset):
    DATA_FILES = {
        'test': './configs/LidarKITTI_files/kitti_file_name_1.txt',
    }

class NuScenes_ME(MELidarDataset):
    # 3D Match dataset all files  ./configs/nuscenes_files.txt  argoverse_files   waymo_file
    DATA_FILES = {
        'test': './configs/nuscenes_files.txt'
    }

# Map the datasets to string names
ALL_DATASETS = [StereoKITTI_ME, SemanticKITTI_ME, LidarKITTI_ME, NuScenes_ME]

dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, data_filename, neighborhood_limits=None, shuffle_dataset=None):
    """
    Defines the data loader based on the parameters specified in the config file
    Args:
        config (dict): dictionary of the arguments
        phase (str): phase for which the data loader should be initialized in [train,val,test]
        shuffle_dataset (bool): shuffle the dataset or not
    Returns:
        loader (torch data loader): data loader that handles loading the data to the model
    """

    assert config['misc']['run_mode'] in ['train', 'val', 'test']

    if shuffle_dataset is None:
        shuffle_dataset = config['misc']['run_mode'] != 'test'

    # Select the defined dataset
    Dataset = dataset_str_mapping[config['data']['dataset']]

    dset = Dataset(phase, data_filename, config=config)

    drop_last = False if config['misc']['run_mode'] == 'test' else True

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config[phase]['batch_size'],
        shuffle=shuffle_dataset,
        num_workers=config[phase]['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=drop_last
    )

    return loader