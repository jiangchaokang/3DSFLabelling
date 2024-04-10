import numpy as np
import torch
from pytorch3d import transforms
from pytorch3d.structures import Pointclouds, list_to_padded
from pytorch3d.ops import ball_query
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon

def symmetric_orthogonalization(x, d=3):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, d, d)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :-1, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r

#from pytorch3d
def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, eps=eps)

def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5
    return torch.acos(torch.clamp(phi_cos, min=-1, max=1))

def angle2rot_2d(angle):
    """
    :param angle: N tensor of angles in radians
    :return: Nx2x2 tensor of rotation matrices
    """
    return torch.stack([torch.stack([torch.cos(angle), -torch.sin(angle)]),
                 torch.stack([torch.sin(angle), torch.cos(angle)])]).permute(2, 0, 1)

def get_rigid_transform(R, t):
    """
    can also do no batch dimension
    :param points: bxnx3 tensor
    :param R: bx3x3 tensor
    :param t: bx3 tensor
    :return: bxnx3 tensor
    """
    if len(t.shape)==1:
        t = t.unsqueeze(0)
    transform = transforms.Transform3d(device='cuda').rotate(R).translate(t)
    return transform

def get_box_rigid_transform(boxes, R, t):
    """
    :param points: bxnx3 tensor
    :param boxes: bx8 tensor
    :param R: bx3x3 tensor
    :param t: bx3 tensor
    :return: bxnx3 tensor
    """
    offsets = boxes[:, 1:4]
    offset_n = transforms.Translate(-offsets)
    offset_p = transforms.Translate(offsets)
    return offset_n.compose(get_rigid_transform(R, t), offset_p)

# def pointwise_rigid_transform(points, R, t):
#     """
#     :param points: nx3 tensor
#     :param R: nx3x3 tensor
#     :param t: nx3 tensor
#     :return: nx3 tensor
#     """
#     points = points.unsqueeze(-1)
#     rotated = torch.einsum('ijk,ikl->ijl', R, points)
#     return rotated[...,0]+t
#     # return torch.matmul(points.unsqueeze(1), R).squeeze(1)+t

def parameters2boxes(box_parameters, anchors):
    """
    :param box_parameters: bxkx9 tensor
    :param anchors: kx7 tensor
    :return: bxkx8 tensor
    """
    confidences = torch.sigmoid(box_parameters[:, :, :1])
    positions = box_parameters[:, :, 1:4] + anchors[:, :3].unsqueeze(0)
    # positions = 9*(torch.sigmoid(box_parameters[:, :, 1:4])-.5) + anchors[:, :3].unsqueeze(0)
    dimensions = torch.exp(box_parameters[:, :, 4:7]) * anchors[:, 3:6].unsqueeze(0)
    headings = torch.atan2(-box_parameters[:, :, 7:8], box_parameters[:, :, 8:9] + 1e-3)+anchors[:,6:7].unsqueeze(0)
    return torch.cat((confidences, positions, dimensions, headings), dim=-1)

def box_coordinate(points, boxes, inverse=False):
    """
    :param points: bxnx3 tensor
    :param boxes: bx8 tensor
    :return: bxnx3 tensor
    """
    centers = boxes[:,1:4]
    headings = boxes[:,7]
    forward = torch.stack([-torch.sin(headings), torch.cos(headings), torch.zeros_like(headings)], dim = -1)
    right = torch.stack([torch.cos(headings), torch.sin(headings), torch.zeros_like(headings)], dim = -1)
    up = torch.tensor([[0,0,1]], device=points.device, dtype=torch.float32).repeat(boxes.shape[0], 1)
    rot_matrix = torch.stack([right, forward, up], dim = -2)
    transform = transforms.Transform3d(device='cuda').translate(-centers).rotate(rot_matrix.transpose(-1,-2))
    if inverse:
        transform = transform.inverse()
    output = transform.transform_points(points)
    return output

def inside_box(points, boxes, padded=True):
    """
    :param points: batch size n point cloud
    :param boxes: nx8 tensor
    :return: binary mask in padded format if padded=True, else in packed format
    """
    box_coord = points.update_padded(box_coordinate(points.points_padded(), boxes))
    box_shape = torch.stack([-boxes[:,4:7]/2, boxes[:,4:7]/2], axis=1)
    # idx = torch.all(box_coord.inside_box(box_shape), dim=1) #python 3.6
    idx = box_coord.inside_box(box_shape)
    if padded:
        idx = idx.view(len(points.points_list()), -1)
    return idx

def sigmoid_weights(coords, boxes, slope=8):
    """
    :param coords: bxnx3 tensor
    :param boxes: bx8 box parameters
    :param slope: slope of sigmoid
    :return: bxn tensor
    """
    wlh_r = boxes[:, 4:7].unsqueeze(1)/2
    weights = torch.sigmoid(slope*(coords+wlh_r))-torch.sigmoid(slope*(coords-wlh_r))
    return torch.prod(weights, dim=-1)

def box_weights(points, boxes, crop_threshold = 1e-6, slope = 8, normalize_weights=False, normals = None):
    """
    :param points: batch size n point cloud
    :param boxes: (nxk)x8 tensor
    :return: new cropped pointcloud with max num points n_new, (nxk)xn_new weights for each point
    """
    num_boxes = (int)(boxes.shape[0]/len(points))
    points_padded = torch.repeat_interleave(points.points_padded(), num_boxes, dim=0)
    boxes_flat = boxes#.view(-1, 8)
    box_coord = box_coordinate(points_padded, boxes_flat)
    box_weights = sigmoid_weights(box_coord, boxes_flat, slope)#.view(boxes.shape[0], boxes.shape[1], -1)
    idx = [torch.nonzero(box_weights[i][:points.num_points_per_cloud()[i//num_boxes]]>crop_threshold).squeeze(-1) for i in range(box_weights.shape[0])]
    not_empty = [len(i)>0 for i in idx]
    new_points = [points_padded[i][idx[i]] for i in range(box_weights.shape[0])]
    # if not np.any(not_empty):
    #     if normals is not None:
    #         return None, None, None, None, None
    #     else:
    #         return None, None, None, None
    new_points = Pointclouds(new_points)
    new_weights = [box_weights[i][idx[i]] for i in range(box_weights.shape[0])]
    new_weights = list_to_padded(new_weights)
    if normalize_weights:
        new_weights = normalize(new_weights, dim=1)
    if normals is not None:
        normals_repeat = torch.repeat_interleave(normals, num_boxes, dim=0)
        new_normals = [normals_repeat[i][idx[i]] for i in range(box_weights.shape[0])]
        new_normals = list_to_padded(new_normals)
        return new_points, new_weights, box_weights, not_empty, new_normals
    return new_points, new_weights, box_weights, not_empty

def normalize(input, dim = None, eps = 1e-7):
    if dim is None:
        return input/(torch.sum(input)+eps)
    return input / (torch.sum(input, dim=dim, keepdim=True) + eps)

def perbox_params2boxesRt(perbox_params, anchors):
    box_params, R, t = perbox_params[..., :9], perbox_params[..., 9:13], perbox_params[..., 13:15]
    boxes = parameters2boxes(box_params, anchors)
    R = symmetric_orthogonalization(R.view(-1, 4), 2).view(R.shape[0], R.shape[1], 2, 2)
    R, t = rotation_2dto3d(R), translation_2dto3d(t)
    transform = get_box_rigid_transform(boxes.view(-1, 8), R.view(-1, 3, 3), t.view(-1, 3))
    return boxes.view(-1, 8), transform

def get_reverse_boxesRt(params, boxes, transform):
    c, R, t = torch.sigmoid(params[..., 0:1]).view(-1, 1), params[..., 1:5], params[..., 5:]
    transformed_boxes = transform_boxes(boxes, transform)
    transformed_boxes = torch.cat([c, transformed_boxes[:, 1:]], dim=-1)
    R = symmetric_orthogonalization(R.view(-1, 4), 2).view(R.shape[0], R.shape[1], 2, 2)
    R, t = rotation_2dto3d(R), translation_2dto3d(t)
    transform = get_box_rigid_transform(transformed_boxes, R.view(-1, 3, 3), t.view(-1, 3))
    return transformed_boxes, transform

def global_params2Rt(global_params):
    R_ego, t_ego = global_params[..., :9], global_params[..., 9:]
    R_ego = symmetric_orthogonalization(R_ego)
    return get_rigid_transform(R_ego, t_ego)

def global_params2d2Rt(global_params):
    R_ego, t_ego = global_params[..., :4], global_params[..., 4:]
    R_ego = symmetric_orthogonalization(R_ego.view(-1, 4), 2)
    R_ego, t_ego = rotation_2dto3d(R_ego), translation_2dto3d(t_ego)
    return get_rigid_transform(R_ego, t_ego)

def transform_boxes(box_parameters, transform):
    """
    :param box_parameters: bx8
    :param R: bx3x3
    :param t:transformed_headings
    :return:
    """
    positions = box_parameters[:,1:4]
    headings = box_parameters[:,7:8]
    transformed_positions = transform.transform_points(positions.unsqueeze(1)).squeeze(1)
    transformed_headings = headings+transforms.matrix_to_euler_angles(transform.get_matrix()[:,:3,:3], 'ZYX')[:, :1]
    return torch.cat((box_parameters[:,:1], transformed_positions, box_parameters[:,4:7], transformed_headings), dim=-1)

# def get_z_rotation(R):
#     """
#     :param R: 3x3 or bx3x3 tensor
#     :return: 2x2 or bx2x2 tensor
#     """
#     angle_2d = transforms.matrix_to_euler_angles(R, 'ZYX')[:, 0]
#     # matrix_2d= torch.stack([torch.stack([torch.cos(angle_2d), -torch.sin(angle_2d)]),
#     #              torch.stack([torch.sin(angle_2d), torch.cos(angle_2d)])]).permute(2, 0, 1)
#     matrix_2d = angle2rot_2d(angle_2d)
#     return matrix_2d

def rotation_2dto3d(R):
    R = torch.cat([R, torch.zeros_like(R[...,:1])], axis=-1)
    last_row = torch.zeros_like(R[...,:1,:])
    last_row[...,2] = 1
    R = torch.cat([R, last_row], axis=-2)
    return R

def translation_2dto3d(t):
    t = torch.cat([t, torch.zeros_like(t[...,:1])], axis=-1)
    return t

def normal_frame(points):
    x = -points[...,0:1]
    y = points[...,2:3]
    z = points[...,1:2]
    return torch.cat((x,y,z), dim = -1)


### inference utils ###
def box2corners(box_params):
    """
    :param box_params: bx8 tensor
    :return: bx8x3 tensor
    """
    center = box_params[:,1:4]
    forward = torch.stack([-torch.sin(box_params[:,7]), torch.cos(box_params[:,7]), torch.zeros(box_params.shape[0], device='cuda', dtype=torch.float32)], dim=-1) * box_params[:,5:6] / 2
    right = torch.stack([torch.cos(box_params[:,7]), torch.sin(box_params[:,7]), torch.zeros(box_params.shape[0], device='cuda', dtype=torch.float32)], dim=-1) * box_params[:,4:5] / 2
    up = torch.stack([torch.zeros(box_params.shape[0], device='cuda', dtype=torch.float32), torch.zeros(box_params.shape[0], device='cuda', dtype=torch.float32), box_params[:,6] / 2], dim=-1)
    p1 = center - forward - right - up
    p2 = center + forward - right - up
    p3 = center + forward + right - up
    p4 = center - forward + right - up
    p5 = center - forward - right + up
    p6 = center + forward - right + up
    p7 = center + forward + right + up
    p8 = center - forward + right + up
    corners = torch.stack((p1, p2, p3, p4, p5, p6, p7, p8), dim=1)
    return corners

def tighten_boxes(boxes, points):
    """
    :param boxes: bx8+ tensor
    :param points: nx3 tensor
    :return: bx8+ tensor
    """
    points_r = points.repeat(boxes.shape[0], 1, 1)
    box_coord = box_coordinate(points_r, boxes)
    box_coord_pc = Pointclouds(box_coord)
    box_shape = torch.stack([-boxes[:,4:7]/2-.25, boxes[:,4:7]/2+.25], dim=1)
    idx = box_coord_pc.inside_box(box_shape)
    idx = idx.view(points_r.shape[0], -1)
    inside_box_pc = Pointclouds([p[i] for p, i in zip(box_coord, idx)])
    tightened_boxes = inside_box_pc.get_bounding_boxes()
    tightened_centers = torch.mean(tightened_boxes, dim=-1)
    # tightened_centers[:, -1]=0 #fix this to just be the og
    tightened_centers = box_coordinate(tightened_centers.unsqueeze(1), boxes, inverse=True).squeeze(1)
    tightened_shape = tightened_boxes[:,:,1]-tightened_boxes[:,:,0]+.4
    output = torch.cat([boxes[:,:1], tightened_centers, tightened_shape, boxes[:,7:]], dim=-1)
    return output

def cycle_consistency(boxes, transforms):
    """
    :param boxes: bx8+ tensor
    :param transforms: b transforms
    :return: b errors
    """
    box_copy = torch.clone(boxes)
    box_copy[:,4:7] = 1
    corners = box2corners(box_copy)
    transformed_corners = transforms.transform_points(corners)
    return torch.mean(torch.norm(corners-transformed_corners, dim=-1), dim=-1)

def nms(box_params, confidence_threshold=.8, iou_threshold=.15, return_index = False):
    """
    :param box_params: kx8+ tensor
    :param iou_threshold:
    :return: k'x8+ tensor subset of boxes
    """
    box_params_copy = torch.clone(box_params).detach().cpu().numpy()
    corners = box2corners(box_params)[:,:4,:2].detach().cpu().numpy()
    polygons = [Polygon(c) for c in corners]
    num_boxes = len(polygons)
    iou = np.zeros((num_boxes, num_boxes))
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i,j] = polygons[i].intersection(polygons[j]).area/polygons[i].union(polygons[j]).area
    output = []
    tops = []
    while np.any(box_params_copy[:,0]>confidence_threshold):
        top = np.argmax(box_params_copy[:,0])
        tops.append(top)
        output.append(box_params[top])
        box_params_copy[iou[top]>iou_threshold,0]=0
    if len(output)>0:
        if return_index:
            return torch.stack(output, dim=0), tops
        return torch.stack(output, dim=0)
    else:
        # print('no detected objects')
        return None

def init_nms(box_params, iou_threshold=.15):
    """
    :param box_params: kx8+ tensor
    :param iou_threshold:
    :return: k'x8+ tensor subset of boxes
    """
    box_params_copy = torch.clone(box_params).detach().cpu().numpy()
    corners = box2corners(box_params)[:,:4,:2].detach().cpu().numpy()
    polygons = [Polygon(c) for c in corners]
    num_boxes = len(polygons)
    iou = np.zeros((num_boxes, num_boxes))
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i,j] = polygons[i].intersection(polygons[j]).area/polygons[i].union(polygons[j]).area
    idx = []
    while np.any(box_params_copy[:,0]!=-1):
        top = np.argmax(np.abs(box_params_copy[:,0]-.5)-np.Inf*(box_params_copy[:,0]==-1))
        idx.append(top)
        box_params_copy[iou[top]>iou_threshold,0]=-1
    return idx

def num_points_in_box(points, box_params):
    """
    :param points: nx3 tensor
    :param box_params: kx8+ tensor
    :param threshold:
    :return: k'x8+ tensor subset of boxes
    """
    memberships = inside_box(Pointclouds([points]*box_params.shape[0]), box_params)
    return torch.sum(memberships, dim=1)

def box_segment(points, box_params):
    """
    :param points: nx3 tensor
    :param box_params: kx8+ tensor
    :return: n integer tensor indicating which box the point belongs to
    """
    box_params = box_params[box_params[:,0].argsort(descending=True)]
    memberships = inside_box(Pointclouds([points]*box_params.shape[0]), box_params)
    point_memberships = torch.argmax(torch.cat((torch.zeros_like(memberships[:1,]), memberships), dim=0).float(), dim=0) #zero label corresponds to not in any box
    return point_memberships

def cc_in_box(points, box_params, seg_threshold=.005):
    """
    :param points: nx3 tensor
    :param box_params: kx8+ tensor
    :return: n integer tensor indicating which moving object the point belongs to (after connected component grouping)
    """
    cc = graph_segmentation(points, threshold=seg_threshold)
    box_memberships = box_segment(points, box_params)
    output = torch.zeros_like(box_memberships)
    for i in np.arange(torch.max(box_memberships).item(), 0, -1):
        segments = torch.unique(cc[box_memberships==i])
        output[np.isin(cc.detach().cpu().numpy(), segments.detach().cpu().numpy())] = i
    return output

def flow_segmentation(pc, sf, threshold = 100, scale = 100, min_size = 500):
    """
    :param pc: nx3 array
    :param sf: nx3 array
    :param threshold:
    :param scale:
    :param min_size:
    :return: n array of integer labels
    """
    graph_feat = np.concatenate((pc,scale*sf), axis = 1)
    labels = graph_segmentation(graph_feat, threshold)

    # sets largest connected component as background and gives it label 0
    unique, counts = np.unique(labels, return_counts=True)
    background_label = unique[np.argmax(counts)]
    labels[labels==background_label]=-1

    # relabels components with size < min_size
    toosmall_labels = unique[np.nonzero(counts<min_size)]
    labels[np.isin(labels,toosmall_labels)] = -1
    unique, inverse = np.unique(labels, return_inverse=True)
    labels = inverse
    return labels

def graph_segmentation(graph_feat, threshold=.005):
    """
    :param graph_feat: nxd tensor
    :param threshold:
    :return: n integer tensor of cc labels
    """
    adjacency_matrix = graph_connectivity_mem(graph_feat, graph_feat, threshold)
    graph = csr_matrix(adjacency_matrix)
    return torch.tensor(connected_components(csgraph=graph, directed=False, return_labels=True)[1], device=graph_feat.device, dtype=torch.float32)

def pairwise_distance(pc1, pc2):
    """
    supports batching as well
    :param pc1: nxd tensor
    :param pc2: nxd tensor
    :return: nxn tensor
    """
    inner = -2 * torch.matmul(pc1, torch.transpose(pc2, -1, -2))
    x1x1 = torch.sum(pc1 ** 2, dim=-1, keepdims=True)
    x2x2 = torch.sum(pc2 ** 2, dim=-1, keepdims=True)
    pairwise_distance = x1x1 + inner + torch.transpose(x2x2, -1, -2)
    return pairwise_distance

def graph_connectivity_mem(pc1, pc2, threshold):
    step=2500
    output = []
    for i in range(0, len(pc1), step):
        next_step = min(i+step, len(pc1))
        inner = -2 * torch.matmul(pc1[i:next_step], torch.transpose(pc2, -1, -2))
        x1x1 = torch.sum(pc1[i:next_step] ** 2, dim=-1, keepdims=True)
        x2x2 = torch.sum(pc2 ** 2, dim=-1, keepdims=True)
        pairwise_distance = x1x1 + inner + torch.transpose(x2x2, -1, -2)
        output.append((pairwise_distance<threshold).detach().cpu().numpy())
    return np.concatenate(output, axis=0)

def iou(pred, labels):
    if torch.sum(labels)==0:
        print('Empty scene')
        return (torch.sum(pred)==0).float()
    true_negatives = (pred+labels)==0
    true_positives = (pred+labels)==2
    return torch.sum(true_positives)/torch.sum(true_negatives)

def precision_at_one(pred, target):
    """
    Computes the precision and recall of the binary fg/bg segmentation
    Args:
    pred (torch.Tensor): predicted foreground labels
    target (torch.Tensor): : gt foreground labels
    Returns
    precision_f (float): foreground precision
    precision_b (float): background precision
    recall_f (float): foreground recall
    recall_b (float): background recall
    """
    if torch.sum(target)==0:
        print('no moving objects')

    precision_f = (pred[target == 1] == 1).float().sum() / ((pred == 1).float().sum() + 1e-6)
    precision_b = (pred[target == 0] == 0).float().sum() / ((pred == 0).float().sum() + 1e-6)

    recall_f = (pred[target == 1] == 1).float().sum() / ((target == 1).float().sum() + 1e-6)
    recall_b = (pred[target == 0] == 0).float().sum() / ((target == 0).float().sum() + 1e-6)

    accuracy = (pred==target).sum() / len(pred)

    tp = (pred[target == 1] == 1).float().sum()
    fp = (pred[target == 0] == 1).float().sum()
    fn = (pred[target == 1] == 0).float().sum()
    tn = (pred[target == 0] == 0).float().sum()

    return precision_f, precision_b, recall_f, recall_b, accuracy, tp, fp, fn, tn

def compute_epe(est_flow, gt_flow, sem_label=None, eval_stats=False, mask=None):
    """
    Compute 3d end-point-error
    Args:
        st_flow (torch.Tensor): estimated flow vectors [n,3]
        gt_flow  (torch.Tensor): ground truth flow vectors [n,3]
        eval_stats (bool): compute the evaluation stats as defined in FlowNet3D
        mask (torch.Tensor): boolean mask used for filtering the epe [n]
    Returns:
        epe (float): mean EPE for current batch
        epe_bckg (float): mean EPE for the background points
        epe_forg (float): mean EPE for the foreground points
        acc3d_strict (float): inlier ratio according to strict thresh (error smaller than 5cm or 5%)
        acc3d_relax (float): inlier ratio according to relaxed thresh (error smaller than 10cm or 10%)
        outlier (float): ratio of outliers (error larger than 30cm or 10%)
    """

    metrics = {}
    error = est_flow - gt_flow

    # If mask if provided mask out the flow
    if mask is not None:
        error = error[mask > 0.5]
        gt_flow = gt_flow[mask > 0.5, :]

    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()
    median_epe = epe_per_point.median()

    metrics['epe'] = epe.item()
    metrics['median_epe'] = median_epe.item()

    if sem_label is not None:
        # Extract epe for background and foreground separately (background = class 0)
        bckg_mask = (sem_label == 0)
        forg_mask = (sem_label == 1)

        bckg_epe = epe_per_point[bckg_mask]
        forg_epe = epe_per_point[forg_mask]

        metrics['bckg_epe'] = bckg_epe.mean().item()
        metrics['bckg_epe_median'] = bckg_epe.median().item()

        if torch.sum(forg_mask) > 0:
            metrics['forg_epe_median'] = forg_epe.median().item()
            metrics['forg_epe'] = forg_epe.mean().item()

    if eval_stats:
        gt_f_magnitude = torch.norm(gt_flow, dim=-1)
        gt_f_magnitude_np = np.linalg.norm(gt_flow.cpu(), axis=-1)
        relative_err = epe_per_point / (gt_f_magnitude + 1e-4)
        acc3d_strict = (
            (torch.logical_or(epe_per_point < 0.05, relative_err < 0.05)).type(torch.float).mean()
        )
        acc3d_relax = (
            (torch.logical_or(epe_per_point < 0.1, relative_err < 0.1)).type(torch.float).mean()
        )
        outlier = (torch.logical_or(epe_per_point > 0.3, relative_err > 0.1)).type(torch.float).mean()

        metrics['acc3d_s'] = acc3d_strict.item()
        metrics['acc3d_r'] = acc3d_relax.item()
        metrics['outlier'] = outlier.item()

    metrics['n'] = len(gt_flow)

    return metrics