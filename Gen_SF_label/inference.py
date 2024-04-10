import torch
import pdb
import rsf_utils
from pytorch3d import transforms
import numpy as np

def flow_inference(pc1, global_params, perbox_params, anchors, config, cc = True, cycle = False):
    """
    :param pc1: nx3 tensor
    :param R_ego: 3x3 tensor
    :param t_ego: 3 tensor
    :param boxes: kx8 tensor
    :param R: kx3x3 tensor
    :param t: kx3 tensor
    :return: predicted sf: nx3 tensor
    """

    filter = []

    ego_transform = rsf_utils.global_params2Rt(global_params.unsqueeze(0))
    boxes, box_transform = rsf_utils.perbox_params2boxesRt(perbox_params.unsqueeze(0), anchors)
    box_transform_comp = transforms.Transform3d(
        matrix=ego_transform.get_matrix().repeat_interleave(len(anchors), dim=0)).compose(box_transform)

    filter.append(boxes[:, 0]>config['confidence_threshold'])

    if cycle:
        boxes_2, box_transform_2 = rsf_utils.get_reverse_boxesRt(perbox_params[:, 15:].unsqueeze(0), boxes, box_transform_comp)
        filter.append(boxes_2[:, 0]>config['confidence_threshold'])
        ego_inverse = transforms.Transform3d(matrix = ego_transform.inverse().get_matrix().repeat_interleave(len(anchors), dim=0))
        cycle_error = rsf_utils.cycle_consistency(boxes, box_transform_comp.compose(box_transform_2).compose(ego_inverse))
        filter.append(cycle_error<config['cycle_threshold'])

    filter.append(rsf_utils.num_points_in_box(pc1, boxes)>config['prune_threshold'])

    deltas = torch.norm(perbox_params[:, -2:], dim=-1)
    filter.append(deltas>config['delta_threshold'])

    filter = torch.all(torch.stack(filter, dim=1), dim=1)
    boxes, box_transform_comp = boxes[filter], box_transform_comp[filter]

    if len(boxes) == 0:
        return no_detection_return(ego_transform, pc1)

    bprt = torch.cat([boxes, box_transform_comp.get_matrix()[:,:3,:3].reshape(-1,9), box_transform_comp.get_matrix()[:,3,:3]], axis=-1)

    bprt = rsf_utils.tighten_boxes(bprt, pc1)

    bprt = rsf_utils.nms(bprt, confidence_threshold=config['confidence_threshold'])
    if bprt == None:
        return no_detection_return(ego_transform, pc1)
    else:
        if cc:
            segmentation = rsf_utils.cc_in_box(pc1, bprt, seg_threshold=config['seg_threshold'])
        else:
            segmentation = rsf_utils.box_segment(pc1, bprt)

        motion_parameters = {'ego_transform': ego_transform, 'boxes':bprt[:,:8],
                         'box_transform': rsf_utils.get_rigid_transform(bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20])}

        R_ego, t_ego = ego_transform.get_matrix()[:, :3, :3], ego_transform.get_matrix()[:, 3, :3]
        R_apply, t_apply = bprt[:, 8:17].reshape(-1, 3, 3), bprt[:, 17:20]
        R_combined, t_combined = torch.cat([R_ego, R_apply], dim = 0), torch.cat([t_ego, t_apply], dim = 0)
        final_transform = rsf_utils.get_rigid_transform(R_combined, t_combined)
        transformed_pts = final_transform[segmentation].transform_points(pc1.unsqueeze(1)).squeeze(1)

        R_ego_b = R_ego.unsqueeze(0)
        t_ego_b = t_ego.unsqueeze(0)
        ego_transform_pts = ego_transform.transform_points(pc1)
        R_ego_np = R_ego.cpu().numpy()
        t_ego_np = t_ego.reshape(1, 3, 1).cpu().numpy()
        pose = np.concatenate([R_ego_np, t_ego_np], axis=2)
        last_row = np.array([[[0, 0, 0, 1]]])
        pose_np = np.concatenate([pose, last_row], axis=1)
    return transformed_pts - pc1, segmentation, motion_parameters, pose_np, ego_transform_pts

def no_detection_return(ego_transform, pc1):
    motion_parameters = {'ego_transform': ego_transform, 'boxes': None, 'box_transform': None}
    transformed_pts = ego_transform.transform_points(pc1)
    segmentation = torch.zeros_like(pc1[:, 0])
    R_ego, t_ego = ego_transform.get_matrix()[:, :3, :3], ego_transform.get_matrix()[:, 3, :3]
    R_ego_np = R_ego.cpu().numpy()
    t_ego_np = t_ego.reshape(1, 3, 1).cpu().numpy()
    pose = np.concatenate([R_ego_np, t_ego_np], axis=2)
    last_row = np.array([[[0, 0, 0, 1]]])
    pose_np = np.concatenate([pose, last_row], axis=1)
    return transformed_pts - pc1, segmentation, motion_parameters, pose_np, transformed_pts