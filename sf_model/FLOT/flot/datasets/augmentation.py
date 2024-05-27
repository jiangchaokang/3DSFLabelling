import pdb
import torch
import numpy as np
from pytorch3d import transforms
from ..datasets import rsf_utils

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


def joint_augmentation_pc(pc1, pc2, flow_3d, f, cx, cy, image_h, image_w):
    # FlyingThings3D
    enabled = True
    random_horizontal_flip = True
    random_vertical_flip = True

    if random_horizontal_flip:
        pc1, pc2, flow_3d = random_flip_pc(
            image_h, image_w, pc1, pc2, flow_3d, f, cx, cy, flip_mode='lr'
        )

    if random_vertical_flip:
        pc1, pc2, flow_3d = random_flip_pc(
            image_h, image_w, pc1, pc2, flow_3d, f, cx, cy, flip_mode='ud'
        )

    return pc1, pc2, flow_3d, f, cx, cy


import random
from pytorch3d import transforms
from pytorch3d.transforms import Transform3d
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
    # 提取旋转和平移部分
    R = matrix[:, :3, :3]
    T = matrix[:, :3, 3]

    T_new = T + T_new.squeeze(0)

    new_matrix = torch.eye(4)
    new_matrix[:3, :3] = R.squeeze(0)
    new_matrix[:3, 3] = T_new
    
    transform_temp = Transform3d(matrix=new_matrix)
    return transform_temp

# def augment(ego_transform, theta_range_deg, translate_range):
#     matrix = ego_transform.get_matrix()
#     R = matrix[:, :3, :3]
#     T = matrix[:, :3, 3]

#     transform_matrices = []

#     for i in range(R.shape[0]):
#         R_new = rotate_aug(R[i].unsqueeze(0), theta_range_deg)
#         T_new = translate_aug(T[i].unsqueeze(0), translate_range)
        
#         ego_transform_temp = Transform3d()
#         ego_transform_temp = ego_transform_temp.rotate(R_new.squeeze())
#         ego_transform_temp = trans_warp(ego_transform_temp,T_new)

#         transform_matrices.append(ego_transform_temp.get_matrix().unsqueeze(0))

#     # 将所有的3D变换矩阵合并成一个张量
#     transform_matrices_tensor = torch.cat(transform_matrices, dim=0)

#     # pdb.set_trace()
#     ego_transform_new = Transform3d(matrix=transform_matrices_tensor.squeeze(1))

#     return ego_transform_new

def ego_augment(ego_transform, theta_range_deg, translate_range):
    # 从 ego_transform 中提取旋转和平移部分
    matrix = ego_transform.get_matrix()
    R = matrix[:, :3, :3]
    T = matrix[:, :3, 3]

    # 对旋转和平移进行增强
    R_new = rotate_aug(R, theta_range_deg, True)
    T_new = translate_aug(T, translate_range)

    # 将增强后的旋转和平移重新组合成一个新的 Transform3d 对象
    new_matrix = torch.eye(4)
    new_matrix[:3, :3] = R_new.squeeze(0)
    new_matrix[:3, 3] = T_new.squeeze(0)

    # 创建新的 Transform3d 对象
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

    filter = []  # 初始化一个空列表，用于存储后续的过滤条件

    ego_transform_old = rsf_utils.global_params2Rt(global_params.unsqueeze(0))  # 将全局参数转换为齐次变换矩阵，形状为(1, 4, 4)，然后存储到ego_transform中
    boxes, box_transform = rsf_utils.perbox_params2boxesRt(perbox_params.unsqueeze(0), anchors)  # 将每个anchor的参数转换为box，并计算其相应的齐次变换矩阵，分别存储在boxes和box_transform中
    ego_transform = ego_augment(ego_transform_old, 3, 0.2)  # 3, 0.2  # 1, 0.1 # 5 0.5
    # ego_transform = ego_transform_old
                                
    box_transform_comp = transforms.Transform3d(matrix=ego_transform.get_matrix().repeat_interleave(len(anchors), dim=0)).compose(box_transform)  # 将ego_transform的齐次变换矩阵按照anchor数进行扩充，并与box_transform进行复合运算，将结果存储到box_transform_comp中

    filter.append(boxes[:, 0] > 0.0)  #  random.uniform(0.7, 0.9) 将boxes的置信度大于confidence_threshold的box加入到过滤条件中
    filter.append(rsf_utils.num_points_in_box(pc1, boxes)>prune_threshold)  # 将点云中位于box内的点数大于prune_threshold的加入到过滤条件中

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
    # 这段代码是根据筛选后的边界框与相应的变换矩阵生成最终的包围盒表示结果bprt。
    bprt = torch.cat([boxes, box_transform_comp.get_matrix()[:,:3,:3].reshape(-1,9), box_transform_comp.get_matrix()[:,3,:3]], axis=-1)
    bprt = rsf_utils.tighten_boxes(bprt, pc1)
    bprt = rsf_utils.nms(bprt.cpu(), confidence_threshold=0.80)
    if bprt == None:
        # print("bprt == None")
        return None
    else:
        segmentation = rsf_utils.box_segment(pc1, bprt)

        motion_parameters = {'ego_transform': ego_transform, 'boxes':bprt[:,:8],
                         'box_transform': rsf_utils.get_rigid_transform(bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20])}

        R_ego, t_ego = ego_transform.get_matrix()[:, :3, :3], ego_transform.get_matrix()[:, 3, :3]
        R_apply, t_apply = bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20]
        R_ego, R_apply, t_ego, t_apply = R_ego.cpu(), R_apply.cpu(), t_ego.cpu(), t_apply.cpu()
        R_apply, t_apply = augment_v2(R_apply, t_apply, 5, 0.5) # 1, 0.1  # 3, 0.3  # 5, 0.5
        R_combined, t_combined = torch.cat([R_ego, R_apply], dim = 0), torch.cat([t_ego, t_apply], dim = 0)
        final_transform = rsf_utils.get_rigid_transform(R_combined, t_combined)
        
        pc1, segmentation = pc1.cpu(), segmentation.cpu()
        transformed_pts = final_transform[segmentation].transform_points(pc1.unsqueeze(1)).squeeze(1)
        

    return transformed_pts


# def flow_inference(pc1, global_params, perbox_params, anchors,prune_threshold):

#     filter = []  # 初始化一个空列表，用于存储后续的过滤条件

#     ego_transform = rsf_utils.global_params2Rt(global_params.unsqueeze(0))  # 将全局参数转换为齐次变换矩阵，形状为(1, 4, 4)，然后存储到ego_transform中
#     boxes, box_transform = rsf_utils.perbox_params2boxesRt(perbox_params.unsqueeze(0), anchors)  # 将每个anchor的参数转换为box，并计算其相应的齐次变换矩阵，分别存储在boxes和box_transform中
#     box_transform_comp = transforms.Transform3d(
#         matrix=ego_transform.get_matrix().repeat_interleave(len(anchors), dim=0)).compose(box_transform)  # 将ego_transform的齐次变换矩阵按照anchor数进行扩充，并与box_transform进行复合运算，将结果存储到box_transform_comp中

#     filter.append(boxes[:, 0]>0.80)  # 将boxes的置信度大于confidence_threshold的box加入到过滤条件中
#     filter.append(rsf_utils.num_points_in_box(pc1, boxes)>prune_threshold)  # 将点云中位于box内的点数大于prune_threshold的加入到过滤条件中

#     deltas = torch.norm(perbox_params[:, -2:], dim=-1)
#     filter.append(deltas>0)
    
#     filter = torch.all(torch.stack(filter, dim=1), dim=1)
#     boxes = boxes[filter] 
#     box_transform_comp = box_transform_comp.cpu()
#     filter = filter.cpu()
    
#     box_transform_comp = box_transform_comp[filter]
#     if len(boxes) == 0:
#         print("len(boxes) == 0")
#         return None

#     bprt = torch.cat([boxes, box_transform_comp.get_matrix()[:,:3,:3].reshape(-1,9), box_transform_comp.get_matrix()[:,3,:3]], axis=-1)
#     bprt = rsf_utils.tighten_boxes(bprt, pc1)
#     # pdb.set_trace()
#     bprt = rsf_utils.nms(bprt.cpu(), confidence_threshold=0.80)
#     if bprt == None:
#         print("bprt == None")
#         return None
#     else:
#         segmentation = rsf_utils.box_segment(pc1, bprt)

#         motion_parameters = {'ego_transform': ego_transform, 'boxes':bprt[:,:8],
#                          'box_transform': rsf_utils.get_rigid_transform(bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20])}

#         R_ego, t_ego = ego_transform.get_matrix()[:, :3, :3], ego_transform.get_matrix()[:, 3, :3]
#         R_apply, t_apply = bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20]
#         R_ego, R_apply, t_ego, t_apply = R_ego.cpu(), R_apply.cpu(), t_ego.cpu(), t_apply.cpu()
#         R_combined, t_combined = torch.cat([R_ego, R_apply], dim = 0), torch.cat([t_ego, t_apply], dim = 0)
#         final_transform = rsf_utils.get_rigid_transform(R_combined, t_combined)
#         pc1, segmentation = pc1.cpu(), segmentation.cpu()
#         transformed_pts = final_transform[segmentation].transform_points(pc1.unsqueeze(1)).squeeze(1)
#         # pdb.set_trace()

#     return transformed_pts

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

# class DataAugmentation:
#     def __init__(self, rotation_range_degrees=[-1.2, 1.2], translation_range_meters=[-0.3, 0.3], prune_threshold=80):
#         """
#         :param rotation_range_degrees: Rotation range in degrees
#         :param translation_range_meters: Translation range in meters
#         """
#         # Set the ranges for rotation (converted to radians) and translation
#         self.rotation_range = [np.radians(x) for x in rotation_range_degrees]
#         self.translation_range = translation_range_meters
#         self.prune_threshold = prune_threshold
#     def numpy_to_tensor(self, numpy_array):
#         """
#         :param numpy_array: Input numpy array
#         :return: Converted tensor
#         """
#         return torch.from_numpy(numpy_array)

#     def augment(self, pc1, global_params, perbox_params, anchors):
#         """
#         :param pc1: nx3 tensor 
#         :param global_params: bx12 tensor for ego motion
#         :param perbox_params: (36, 15) tensor
#         pc1_loaded: torch.Size([n, 3]) / global_params: torch.Size([12])
#         perbox_params: torch.Size([36, 15]) anchors: torch.Size([36, 7])
#         global_params: tensor([ 1.1115,  0.0102, -0.0181, -0.0104,  1.0115,  0.0179,  0.0172, -0.0181,
#          0.9115, -0.0042, -0.0310,  0.0165])
#         """
        
#         # Convert numpy arrays to tensors if necessary
#         if isinstance(anchors, np.ndarray):
#             anchors = self.numpy_to_tensor(anchors)
#             # anchors = anchors.cuda()
#         if isinstance(pc1, np.ndarray):
#             pc1 = self.numpy_to_tensor(pc1)
#             # pc1 = pc1.cuda()
#         if isinstance(global_params, np.ndarray):
#             global_params = self.numpy_to_tensor(global_params) # np.expand_dims(, axis=0)
#             # global_params= global_params.cuda()
#         if isinstance(perbox_params, np.ndarray):
#             perbox_params = self.numpy_to_tensor(perbox_params)
#             # perbox_params= perbox_params.cuda()

#         # Add noise to global params
#         global_rotation_noise = (torch.rand((9)) * (self.rotation_range[1] - self.rotation_range[0]) + self.rotation_range[0]) # .cuda()
#         global_translation_noise = (torch.rand((3)) * (self.translation_range[1] - self.translation_range[0]) + self.translation_range[0]) # .cuda()
#         global_params_noisy = global_params.clone()
#         global_params_noisy[:9] += global_rotation_noise
#         global_params_noisy[9:] += global_translation_noise
#         perbox_params_noisy = perbox_params.clone()
#         for i in range(perbox_params.size(0)):
#             perbox_rotation_noise = (torch.rand((4)) * (self.rotation_range[1] - self.rotation_range[0]) + self.rotation_range[0]) #.cuda()
#             perbox_translation_noise = (torch.rand((2)) * (self.translation_range[1] - self.translation_range[0]) + self.translation_range[0]) #.cuda()
#             perbox_params_noisy[i, 9:13] += perbox_rotation_noise
#             perbox_params_noisy[i, 13:] += perbox_translation_noise
        
#         pc2 = flow_inference(pc1, global_params_noisy, perbox_params_noisy, anchors, self.prune_threshold)
#         # import pdb
#         # pdb.set_trace()
#         return pc2
