import pdb
import torch
import numpy as np
from pytorch3d import transforms
import gensf_utils
import random
from pytorch3d.transforms import Transform3d

def flip_point_cloud(pc, image_h, image_w, f, cx, cy, flip_mode):
    assert flip_mode in ['lr', 'ud']
    pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y

    if flip_mode == 'lr':
        image_x = image_w - 1 - image_x
    else:
        image_y = image_h - 1 - image_y

    pc_x = (image_x - cx) * depth / f
    pc_y = (image_y - cy) * depth / f
    pc = np.concatenate([pc_x[:, None], pc_y[:, None], depth[:, None]], axis=-1)

    return pc


def flip_scene_flow(pc1, flow_3d, image_h, image_w, f, cx, cy, flip_mode):
    new_pc1 = flip_point_cloud(pc1, image_h, image_w, f, cx, cy, flip_mode)
    new_pc1_warp = flip_point_cloud(pc1 + flow_3d[:, :3], image_h, image_w, f, cx, cy, flip_mode)
    return np.concatenate([new_pc1_warp - new_pc1, flow_3d[:, 3:]], axis=-1)


def random_flip_pc(image_h, image_w, pc1, pc2, flow_3d, f, cx, cy, flip_mode):
    assert flow_3d.shape[1] <= 4
    assert flip_mode in ['lr', 'ud']

    if np.random.rand() < 0.5:  # do nothing
        return pc1, pc2, flow_3d

    # flip point clouds
    new_pc1 = flip_point_cloud(pc1, image_h, image_w, f, cx, cy, flip_mode)
    new_pc2 = flip_point_cloud(pc2, image_h, image_w, f, cx, cy, flip_mode)

    # flip scene flow
    new_flow_3d = flip_scene_flow(pc1, flow_3d, image_h, image_w, f, cx, cy, flip_mode)

    return new_pc1, new_pc2, new_flow_3d



def rodrigues_rotation_matrix(axis, theta):
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    return torch.tensor([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def rotate_aug(R, theta_range_deg, flag_ego):
    axis = torch.randint(3, (1,))
    if flag_ego:
        if axis == 1:
            theta_range_deg += 2
        else:
            theta_range_deg -= 2

        if axis == 0:
            axis = torch.tensor([1., 0., 0.])
        elif axis == 1:
            axis = torch.tensor([0., 1., 0.])
        elif axis == 2:
            axis = torch.tensor([0., 0., 1.])

    else:
        axis = torch.tensor([0., 1., 0.]) ## Only Z Axis
    theta_range_rad = torch.deg2rad(torch.tensor(theta_range_deg))
    theta = (torch.rand(1) - 0.5) * 2 * theta_range_rad

    R_rot = rodrigues_rotation_matrix(axis, theta)
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if R_rot.dim() == 2:
        R_rot = R_rot.unsqueeze(0)

    R_new = torch.einsum('bij,bjk->bik', R, R_rot)
    return R_new

def translate_aug(T, translate_range):
    T_noise = (torch.rand_like(T) - 0.5) * 2 * translate_range
    T_noise[:,1] = random.uniform(-0.05, 0.05)
    T_new = T + T_noise
    return T_new
def trans_warp(transform_temp, T_new):
    matrix = transform_temp.get_matrix()
    R = matrix[:, :3, :3]
    T = matrix[:, :3, 3]

    T_new = T + T_new.squeeze(0)

    new_matrix = torch.eye(4)
    new_matrix[:3, :3] = R.squeeze(0)
    new_matrix[:3, 3] = T_new
    
    transform_temp = Transform3d(matrix=new_matrix)
    return transform_temp

def ego_augment(ego_transform, theta_range_deg, translate_range):
    matrix = ego_transform.get_matrix()
    R = matrix[:, :3, :3]
    T = matrix[:, :3, 3]
    R_new = rotate_aug(R, theta_range_deg, True)
    T_new = translate_aug(T, translate_range)
    new_matrix = torch.eye(4)
    new_matrix[:3, :3] = R_new.squeeze(0)
    new_matrix[:3, 3] = T_new.squeeze(0)
    ego_transform_new = Transform3d(matrix=new_matrix)

    return ego_transform_new

def augment_v2(R_apply, t_apply, theta_range_deg, translate_range):
    R, T = R_apply, t_apply

    R_new_list = []
    T_new_list = []

    for i in range(R.shape[0]):
        R_new = rotate_aug(R[i].unsqueeze(0), theta_range_deg, False)
        T_new = translate_aug(T[i].unsqueeze(0), translate_range)

        R_new_list.append(R_new)
        T_new_list.append(T_new)

    R_new_tensor = torch.cat(R_new_list, dim=0)
    T_new_tensor = torch.cat(T_new_list, dim=0)
    return R_new_tensor, T_new_tensor

def flow_inference(pc1, global_params, perbox_params, anchors, prune_threshold):

    filter = []

    ego_transform_old = gensf_utils.global_params2Rt(global_params.unsqueeze(0))
    boxes, box_transform = gensf_utils.perbox_params2boxesRt(perbox_params.unsqueeze(0), anchors)
    ego_transform = ego_augment(ego_transform_old, 3, 0.2)  # 3, 0.2  # 1, 0.1 # 5 0.5
    # ego_transform = ego_transform_old
                                
    box_transform_comp = transforms.Transform3d(matrix=ego_transform.get_matrix().repeat_interleave(len(anchors), dim=0)).compose(box_transform)

    filter.append(boxes[:, 0] > random.uniform(0.7, 0.9))  #
    filter.append(gensf_utils.num_points_in_box(pc1, boxes)>prune_threshold)  
    deltas = torch.norm(perbox_params[:, -2:], dim=-1)
    filter.append(deltas>0)
    
    filter = torch.all(torch.stack(filter, dim=1), dim=1)
    boxes = boxes[filter] 
    box_transform_comp = box_transform_comp.cpu()
    filter = filter.cpu()
    
    box_transform_comp = box_transform_comp[filter]
    if len(boxes) == 0:
        # print("len(boxes) == 0")
        return None

    bprt = torch.cat([boxes, box_transform_comp.get_matrix()[:,:3,:3].reshape(-1,9), box_transform_comp.get_matrix()[:,3,:3]], axis=-1)
    bprt = gensf_utils.tighten_boxes(bprt, pc1)
    bprt = gensf_utils.nms(bprt.cpu(), confidence_threshold=0.80)
    if bprt == None:
        print("bprt == None")
        return None
    else:
        segmentation = gensf_utils.box_segment(pc1, bprt)

        motion_parameters = {'ego_transform': ego_transform, 'boxes':bprt[:,:8],
                         'box_transform': gensf_utils.get_rigid_transform(bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20])}

        R_ego, t_ego = ego_transform.get_matrix()[:, :3, :3], ego_transform.get_matrix()[:, 3, :3]
        R_apply, t_apply = bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20]
        R_ego, R_apply, t_ego, t_apply = R_ego.cpu(), R_apply.cpu(), t_ego.cpu(), t_apply.cpu()
        R_apply, t_apply = augment_v2(R_apply, t_apply, 5, 0.5) # 1, 0.1  # 3, 0.3  # 5, 0.5
        R_combined, t_combined = torch.cat([R_ego, R_apply], dim = 0), torch.cat([t_ego, t_apply], dim = 0)
        final_transform = gensf_utils.get_rigid_transform(R_combined, t_combined)
        
        pc1, segmentation = pc1.cpu(), segmentation.cpu()
        transformed_pts = final_transform[segmentation].transform_points(pc1.unsqueeze(1)).squeeze(1)
        

    return transformed_pts

class DataAugmentation:
    def __init__(self, rotation_range_degrees=[-1.2, 1.2], translation_range_meters=[-0.3, 0.3],prune_threshold = 80):
        self.rotation_range = [np.radians(x) for x in rotation_range_degrees]
        self.translation_range = translation_range_meters
        self.prune_threshold = prune_threshold

    def numpy_to_tensor(self, numpy_array):
        return torch.from_numpy(numpy_array)

    def augment(self, pc1, global_params, perbox_params, anchors):
        # Convert numpy arrays to tensors if necessary
        if isinstance(anchors, np.ndarray):
            anchors = self.numpy_to_tensor(anchors)
            # anchors = anchors.cuda()
        if isinstance(pc1, np.ndarray):
            pc1 = self.numpy_to_tensor(pc1)
            # pc1 = pc1.cuda()
        if isinstance(global_params, np.ndarray):
            global_params = self.numpy_to_tensor(global_params) # np.expand_dims(, axis=0)
            # global_params= global_params.cuda()
        if isinstance(perbox_params, np.ndarray):
            perbox_params = self.numpy_to_tensor(perbox_params)
            # perbox_params= perbox_params.cuda()
        # pdb.set_trace()
        pc2 = flow_inference(pc1, global_params, perbox_params, anchors, self.prune_threshold)
        return pc2


if __name__ == '__main__':
    import os
    # Path to the directory containing 3D Scene Flow Example data
    filenames = './3DSFLabelling_Training_Data_for_3D_SceneFlow/Argoverse_Scene_Flow_Example'
    
    # Initialize the data augmentation transform with a pruning threshold
    transform = DataAugmentation(prune_threshold=10)

    # Load the first point cloud
    pc1 = np.load(os.path.join(filenames, "pc1.npy"), allow_pickle=True).astype(np.float32)
    # Load global transformation parameters
    global_params = np.load(os.path.join(filenames, "global_params.npy"), allow_pickle=True).astype(np.float32)
    # Load per-box transformation parameters
    perbox_params = np.load(os.path.join(filenames, "perbox_params.npy"), allow_pickle=True).astype(np.float32)
    anchors = np.load(os.path.join(filenames, "anchors.npy"), allow_pickle=True).astype(np.float32)
    
    # Perform data augmentation on the loaded point cloud
    pc2_loaded = transform.augment(pc1, global_params, perbox_params, anchors)
    
    # Prepare a list to hold the sequence of point clouds
    sequence = [pc1]
    
    # Append the augmented or original second point cloud to the sequence
    if pc2_loaded is not None:
        sequence.append(pc2_loaded.numpy())
    else:
        # If augmentation is not successful, load the original second point cloud
        pc2 = np.load(os.path.join(filenames, "pc3.npy"), allow_pickle=True).astype(np.float32)
        sequence.append(pc2)

    # Calculate the ground truth scene flow between the two point clouds
    SF_ground_truth = sequence[1] - sequence[0]
