import numpy as np
import time
import torch
import torch.optim as optim
import yaml
from lidarkitti import make_data_loader
import rsf_utils
import open3d as o3d
from rsf_loss import RSFLossv2, RSFLossCycle
from inference import flow_inference
from pytorch3d.structures import Pointclouds, list_to_padded
from pytorch3d.ops import estimate_pointcloud_normals, iterative_closest_point
from pytorch3d.ops.knn import knn_points
from pytorch3d import transforms
import sys,os,pdb
import os.path as osp
import argparse
import pickle
from collections import defaultdict

def create_3d_box(box):
    # Create a unit cube.
    mesh_box = o3d.geometry.TriangleMesh.create_box()
    # Convert the box to numpy array and ensure it's float64.
    box = np.array(box, dtype=np.float64)
    # Scale the cube according to the dimensions of the 3D bounding box.
    for i, scale in enumerate(box[3:6]):
        # Scale each axis independently
        transform = np.identity(4)
        transform[i, i] *= scale
        mesh_box.transform(transform)
    # Move the cube to the center of the 3D bounding box.
    mesh_box.translate(box[:3])
    # Rotate the cube according to the rotation angle of the 3D bounding box.
    R = mesh_box.get_rotation_matrix_from_xyz((0, 0, box[6]))
    mesh_box.rotate(R, center=mesh_box.get_center())
    return mesh_box
def save_bounding_boxes_as_mesh(anchors, filename):
    mesh_boxes = []
    # Convert tensor to numpy array if necessary
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()
    for anchor in anchors:
        mesh_box = create_3d_box(anchor)
        mesh_boxes.append(mesh_box)
    # Merge all the mesh boxes.
    mesh = mesh_boxes[0]
    for mesh_box in mesh_boxes[1:]:
        mesh += mesh_box
    # Save as a .ply file.
    o3d.io.write_triangle_mesh(filename, mesh)

class SF_Optimizer:
    def __init__(self, anchors, config, pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, filename, init_perbox=None, init_global=None, use_gt_ego=False, icp_init=False):
        self.anchors = anchors
        self.num_boxes = anchors.shape[0]
        self.config = config
        self.batch_size = len(pc1)
        self.filename = filename

        pc1_opt, pc2_opt = [torch.clone(p) for p in pc1], [torch.clone(p) for p in pc2]
        pc1_normals_opt, pc2_normals_opt = [torch.clone(p) for p in pc1_normals], [torch.clone(p) for p in pc2_normals]

        self.pc1, self.pc2 = Pointclouds(pc1).to(device='cuda'), Pointclouds(pc2).to(device='cuda')
        self.pc1_normals, self.pc2_normals = list_to_padded(pc1_normals).to(device='cuda'), list_to_padded(pc2_normals).to(device='cuda')
        self.pc1_opt, self.pc2_opt = Pointclouds(pc1_opt).to(device='cuda'), Pointclouds(pc2_opt).to(device='cuda')
        self.pc1_normals_opt, self.pc2_normals_opt = list_to_padded(pc1_normals_opt).to(device='cuda'), list_to_padded(pc2_normals_opt).to(device='cuda')

        self.mask1, self.mask2 = mask1, mask2
        self.seg1, self.seg2 = seg1, seg2
        self.gt_R_ego, self.gt_t_ego = torch.stack(R_ego).transpose(-1, -2).to('cuda'), torch.stack(t_ego).to('cuda')
        self.gt_ego_transform = rsf_utils.get_rigid_transform(self.gt_R_ego, self.gt_t_ego)
        self.sf = sf
        self.predicted_flow, self.segmentation, self.motion_parameters, self.output_pose_np, self.output_ego_transform_pts = None, None, None, None, None
        self.updated = True

        if init_perbox is None:
            if config['cycle']:
                perbox_params = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.1, 0, 0, .9, 0, 0, 0, 1.1, 0, 0, .9, 0, 0]), (self.batch_size, self.num_boxes, 1))
            else:
                perbox_params = np.tile(np.array([0,0,0,0,0,0,0,0,0,1.1,0,0,.9,0,0]), (self.batch_size, self.num_boxes, 1))
            self.perbox_params = torch.tensor(perbox_params, requires_grad=True, device='cuda', dtype=torch.float32)
        else:
            self.perbox_params = init_perbox
        if init_global is None:
            if use_gt_ego:
                self.global_params = torch.cat([torch.stack(R_ego).transpose(-1, -2).to('cuda').reshape(len(R_ego), -1), torch.stack(t_ego).to('cuda')], dim=-1)
            elif icp_init:
                icp_output = iterative_closest_point(self.pc1_opt, self.pc2_opt)
                R_icp, t_icp, scale_icp = icp_output[3]
                self.global_params = torch.tensor(np.concatenate([R_icp.detach().cpu().numpy().reshape(R_icp.shape[0], -1),
                                        t_icp.detach().cpu().numpy()], axis=-1), requires_grad=True, device='cuda', dtype=torch.float32)
            else:
                self.global_params = torch.tensor([[1.1,0,0,0,1,0,0,0,.9,0,0,0]]*self.batch_size, requires_grad=True, device='cuda', dtype=torch.float32)
        else:
            self.global_params = init_global
        if use_gt_ego:
            self.opt = optim.AdamW([self.perbox_params], lr=config['lr'], weight_decay=1e-3)
        else:
            self.opt = optim.AdamW([self.global_params, self.perbox_params], lr=config['lr'], weight_decay=1e-3)

        if config['cycle']:
            self.loss_function = RSFLossCycle(anchors, config)
        else:
            self.loss_function = RSFLossv2(anchors, config)

    def poly_lr_scheduler(self, optimizer, base_lr, iter, max_iter, power=0.98, min_lr=2e-3):
        """
        Polynomial Learning Rate Adjustment Strategy
        """
        lr = max(base_lr * (1 - float(iter) / max_iter) ** power, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def Samplepc(self, pc_1, pc_2, num_points):
        if self.num_pc1 > num_points:
            indices1 = torch.multinomial(torch.ones(self.num_pc1), num_points, replacement=False)
        else:
            indices1 = torch.multinomial(torch.ones(self.num_pc1), self.num_pc1, replacement=False)
        if self.num_pc2 > num_points:
            indices2 = torch.multinomial(torch.ones(self.num_pc2), num_points, replacement=False)
        else:
            indices2 = torch.multinomial(torch.ones(self.num_pc2), self.num_pc2, replacement=False)
        return indices1, indices2
        

    def optimize(self, epochs):
        for j in range(epochs):
            lr = self.poly_lr_scheduler(self.opt, self.config['lr'], iter=j, max_iter=epochs)
            self.opt.zero_grad()
            # if j < epochs/2.0:
            #     indices1, indices2 = self.Samplepc(self.pc1_opt, self.pc2_opt, 4096)
            #     loss = self.loss_function(self.pc1_opt[indices1], self.pc2_opt[indices2], self.pc1_normals_opt[indices1], self.pc2_normals_opt[indices2], self.global_params, self.perbox_params)
            # else:
            loss = self.loss_function(self.pc1_opt, self.pc2_opt, self.pc1_normals_opt, self.pc2_normals_opt, self.global_params, self.perbox_params, j)
            if self.config['print_loss']:
                print(loss['total_loss'].item())
            loss['total_loss'].backward()
            self.opt.step()
        print("final learning rate:", lr)
        self.updated = True

    def predict(self):
        if self.updated:
            output_flow, output_seg, output_params, output_pose_np, output_ego_transform_pts = [], [], [], [], []
            with torch.no_grad():
                for vis_idx in range(self.batch_size):
                    predicted_flow, segmentation, motion_parameters, pose_np, ego_transform_pts = flow_inference(self.pc1.points_list()[vis_idx], self.global_params[vis_idx], self.perbox_params[vis_idx], self.anchors, self.config, cc=False, cycle=self.config['cycle'])
                    output_flow.append(predicted_flow)
                    output_seg.append(segmentation)
                    output_params.append(motion_parameters)
                    output_pose_np.append(pose_np)
                    output_ego_transform_pts.append(ego_transform_pts)
            self.predicted_flow, self.segmentation, self.motion_parameters, self.output_pose_np, self.output_ego_transform_pts = output_flow, output_seg, output_params, output_pose_np, output_ego_transform_pts
            self.updated = False
        return self.predicted_flow, self.segmentation, self.motion_parameters, self.output_pose_np, self.output_ego_transform_pts

    def evaluate_flow(self):
        errors = defaultdict(list)
        predicted_flow_batch, segmentation_batch, motion_parameters_batch, output_pose_np, output_ego_transform_pts = self.predict()
        with torch.no_grad():
            for vis_idx, predicted_flow in enumerate(predicted_flow_batch):
                pc3 =  self.pc1.points_list()[vis_idx] + predicted_flow[self.mask1[vis_idx]]
                
                # save_path = self.filename[vis_idx].replace("nuscene_lidar", "nuscene_lidar_sf")
                save_path = self.filename[vis_idx].replace("processed_kitti_lidar", "kitti_sf_label")
                
                os.makedirs(save_path[:-4], exist_ok=True)
                np.save(osp.join(save_path[:-4], 'pc1.npy'), self.pc1.points_list()[vis_idx].cpu().numpy())
                np.save(osp.join(save_path[:-4], 'pc2.npy'), self.pc2.points_list()[vis_idx].cpu().numpy())
                np.save(osp.join(save_path[:-4], 'pc3.npy'), pc3.cpu().numpy())
                np.save(osp.join(save_path[:-4], 'perbox_params.npy'), self.perbox_params[vis_idx].cpu().numpy())
                np.save(osp.join(save_path[:-4], 'global_params.npy'), self.global_params[vis_idx].cpu().numpy())
                np.save(osp.join(save_path[:-4], 'anchors.npy'), self.anchors.cpu().numpy())
                np.save(osp.join(save_path[:-4], 'segmentation.npy'), self.segmentation[vis_idx].cpu().numpy())
                # with open(osp.join(save_path[:-4], 'motion_parameters.pkl'), "wb") as file:
                #     pickle.dump(self.motion_parameters[vis_idx].cpu(), file)
                np.save(osp.join(save_path[:-4], 'pose.npy'), output_pose_np[vis_idx])
                np.save(osp.join(save_path[:-4], 'pc1_ego.npy'), output_ego_transform_pts[vis_idx].cpu().numpy())
                np.save(osp.join(save_path[:-4], 'pc1_normals.npy'), self.pc1_normals_opt[vis_idx].cpu().numpy())
                np.save(osp.join(save_path[:-4], 'pc2_normals.npy'), self.pc2_normals_opt[vis_idx].cpu().numpy())

                data_raw = np.load(self.filename[vis_idx])
                pc1_cam_mask = data_raw["pc1_cam_mask"]
                pc2_cam_mask = data_raw["pc2_cam_mask"]
                ground1_mask = data_raw["ground1_mask"]
                ground2_mask = data_raw["ground2_mask"]
                np.save(osp.join(save_path[:-4], 'pc1_cam_mask.npy'), pc1_cam_mask)
                np.save(osp.join(save_path[:-4], 'pc2_cam_mask.npy'), pc2_cam_mask)
                np.save(osp.join(save_path[:-4], 'ground1_mask.npy'), ground1_mask)
                np.save(osp.join(save_path[:-4], 'ground2_mask.npy'), ground2_mask)

                # np.savez(save_path, pc1=self.pc1.points_list()[vis_idx].cpu().numpy(), pc2=self.pc2.points_list()[vis_idx].cpu().numpy(), pc3 = pc3.cpu().numpy())
                # gt_sf = self.sf[vis_idx].to('cuda')
                # metrics = rsf_utils.compute_epe(predicted_flow[self.mask1[vis_idx]], gt_sf, eval_stats=True)
                # for k, v in metrics.items():
                #     errors[k].append(v)
        return errors

    def evaluate_segmentation(self):
        errors = defaultdict(list)
        predicted_flow_batch, segmentation_batch, motion_parameters_batch, output_pose_np, output_ego_transform_pts = self.predict() 
        with torch.no_grad():
            for vis_idx, segmentation in enumerate(segmentation_batch):
                gt_seg1 = self.seg1[vis_idx].to('cuda')
                precision_f, precision_b, recall_f, recall_b, accuracy, tp, fp, fn, tn = rsf_utils.precision_at_one(segmentation > 0, gt_seg1)
                errors['precision_f'].append(precision_f.item())
                errors['precision_b'].append(precision_b.item())
                errors['recall_f'].append(recall_f.item())
                errors['recall_b'].append(recall_b.item())
                errors['accuracy'].append(accuracy.item())
                errors['contains_moving'].append(torch.sum(gt_seg1).item()>0)
                errors['tp'].append(tp.item())
                errors['fp'].append(fp.item())
                errors['fn'].append(fn.item())
                errors['tn'].append(tn.item())
        return errors

    def evaluate_ego(self):
        ego_transform = rsf_utils.global_params2Rt(self.global_params)
        R_ego, t_ego = ego_transform.get_matrix()[:,:3,:3], ego_transform.get_matrix()[:,3,:3]
        rot_error = torch.abs(torch.rad2deg(rsf_utils.so3_relative_angle(R_ego, self.gt_R_ego)))
        trans_error = torch.linalg.norm(t_ego - self.gt_t_ego, dim=-1)
        return {'R_ego_error':rot_error.tolist(), 't_ego_error':trans_error.tolist(), 'contains_moving':[torch.sum(s).item()>0 for s in self.seg1]}

    def evaluate_chamfer(self):
        warped_pc_batch = []
        predicted_flow_batch, segmentation_batch, motion_parameters_batch,_ ,_ = self.predict()
        with torch.no_grad():
            for vis_idx, predicted_flow in enumerate(predicted_flow_batch):
                pc1_eval = self.pc1.points_list()[vis_idx]
                warped_pc = pc1_eval + predicted_flow
                warped_pc_batch.append(warped_pc)
            warped_pc_batch = Pointclouds(warped_pc_batch)
            warped_normals = estimate_pointcloud_normals(warped_pc_batch, neighborhood_size=self.config['k_normals'])
            cat1 = torch.cat((warped_pc_batch.points_padded(), warped_normals), dim=-1)
            cat2 = torch.cat((self.pc2.points_padded(), self.pc2_normals), dim=-1)
            knn1 = knn_points(cat1, cat2, warped_pc_batch.num_points_per_cloud(), self.pc2.num_points_per_cloud())[0].squeeze(-1)
            knn2 = knn_points(cat2, cat1, self.pc2.num_points_per_cloud(), warped_pc_batch.num_points_per_cloud())[0].squeeze(-1)
            knn1 = [k[torch.nonzero(self.mask1[i])] for i, k in enumerate(knn1)]
            knn2 = [k[torch.nonzero(self.mask2[i])] for i, k in enumerate(knn2)]
            cd = [torch.mean(torch.cat(k, dim=0)).item() for k in zip(knn1, knn2)]
        return cd


def optimize(cfg):
    dataset_map = {'StereoKITTI_ME': 'stereo', 'SemanticKITTI_ME': 'semantic', 'LidarKITTI_ME': 'lidar', 'NuScenes_ME': 'nuscenes', 'Argoverse_ME': 'nuscenes', 'Waymo_ME': 'nuscenes', 'KITTIo_ME': 'lidar'}
    dataset = dataset_map[cfg['data']['dataset']]
    hyperparameters = cfg['hyperparameters']

    ##### GENERATE ANCHORS ##### box_depth, box_scale  = 4, 1.25
    max_depth = 33
    min_depth = 2
    box_depth = hyperparameters['box_depth'] #6
    box_width = box_depth
    z_center = -1.3
    box_scale = hyperparameters['box_scale'] #1.25
    anchor_width = 1.6*box_scale
    anchor_length = 3.9*box_scale
    anchor_height = 2.0*box_scale

    anchors = []
    if dataset == 'stereo':
        for i, depth in enumerate(np.arange(min_depth, max_depth, box_depth)):
            row = torch.cat([torch.tensor([[x_coord, depth+box_depth/2, z_center, anchor_width, anchor_length, anchor_height, 0]])#, [x_coord, depth+box_depth/2, z_center, anchor_width, anchor_length, anchor_height, np.pi/2]])
                             for x_coord in np.arange(-1*i*box_width, (i+1)*box_width, 2*box_width)], dim = 0)
            anchors.append(row)

        anchors = torch.cat(anchors, dim=0)

    elif dataset == 'lidar':
        anchor_x = torch.arange(-34, 34, 4, dtype=torch.float32)
        anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
        anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y), dim=-1)
        offsets = torch.tensor([[0,3],[0,0]]).repeat(anchors_xy.shape[0]//2, 1)
        if anchors_xy.shape[0] % 2 != 0:
            offsets = torch.cat((offsets, torch.tensor([[0,3]])), dim=0)
        anchors_xy += offsets.unsqueeze(1)
        anchors_xy = anchors_xy.view(-1,2)
        anchors_xy -= torch.mean(anchors_xy, dim=0, keepdim=True)
        anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])]*anchors_xy.shape[0], dim=0)), dim=1)

    elif dataset == 'semantic':
        anchor_x = torch.arange(-34, 34, 4, dtype=torch.float32)
        anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
        anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y), dim=-1)
        offsets = torch.tensor([[0,3],[0,0]]).repeat(anchors_xy.shape[0]//2, 1)
        if anchors_xy.shape[0] % 2 != 0:
            offsets = torch.cat((offsets, torch.tensor([[0,3]])), dim=0)
        anchors_xy += offsets.unsqueeze(1)
        anchors_xy = anchors_xy.view(-1,2)
        anchors_xy -= torch.mean(anchors_xy, dim=0, keepdim=True)
        anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])]*anchors_xy.shape[0], dim=0)), dim=1)

    elif dataset == 'nuscenes':
        anchor_x = torch.arange(-34, 34, 3, dtype=torch.float32)
        anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
        anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y), dim=-1)
        offsets = torch.tensor([[0,3],[0,0]]).repeat(anchors_xy.shape[0]//2, 1)
        if anchors_xy.shape[0] % 2 != 0:
            offsets = torch.cat((offsets, torch.tensor([[0,3]])), dim=0)
        anchors_xy += offsets.unsqueeze(1)
        anchors_xy = anchors_xy.view(-1,2)
        anchors_xy -= torch.mean(anchors_xy, dim=0, keepdim=True)
        anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])]*anchors_xy.shape[0], dim=0)), dim=1)
    
    # save_bounding_boxes_as_mesh(anchors, "./sceneflow_eval_dataset/kitti-od/data_odometry_velodyne/visualization/anchors.ply")
    anchors = anchors.float().to(device='cuda')

    data = make_data_loader(cfg, phase='test')

    errors = defaultdict(list)

    for i, batch in enumerate(data):
        epe_init =9.9
        epe_num = 0
        pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, filename = batch
        # pc1[0], pc2[0] = pc1[0][:,[0,2,1]], pc2[0][:,[0,2,1]]
        optimizer = SF_Optimizer(anchors, hyperparameters, pc1, pc2, pc1_normals, pc2_normals, mask1, mask2, seg1, seg2, R_ego, t_ego, sf, filename)
        optimizer.optimize(hyperparameters['epochs'])
        metrics_t = optimizer.evaluate_flow()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    #     if hyperparameters['evaluate_train']:
    #         for j in range(hyperparameters['epochs']*8):
    #             optimizer.optimize(2)
    #             if dataset != 'semantic':
    #                 metrics_t = optimizer.evaluate_flow()
    #                 epe_list = sum(metrics_t['epe'])
    #                 epe_agv = float(epe_list)/len(metrics_t['epe'])
    #                 if epe_agv < epe_init:
    #                     epe_init = epe_agv
    #                     epe_num = epe_num + 1
    #                     metrics = metrics_t
    #                     print('jjjj:', j, 'epe_num', epe_num,'epe_init::', epe_init, 'epe_list:',epe_list, '^^^',str(i) + ' EPE: ' + str(metrics_t['epe']))


    #     if dataset != 'semantic':
    #         if not hyperparameters['evaluate_train']:
    #             if i<2:
    #                 print('*****************&& No training evaluation process &&*******************')
    #             metrics = optimizer.evaluate_flow()
    #         #  metrics = optimizer.evaluate_flow()
    #         print(str(i) + ' EPE: ' + str(metrics['epe']))
    #         for k, v in metrics.items():
    #             errors[k]+=v
    #     elif dataset == 'semantic':
    #         metrics = optimizer.evaluate_ego()
    #         print(str(i) + ' ego:' + str(metrics))
    #         for k, v in metrics.items():
    #             errors[k]+=v
    #     metrics = optimizer.evaluate_segmentation()
    #     print(str(i) + ' segmentation:' + str(metrics))
    #     for k, v in metrics.items():
    #         errors[k]+=v

    #     if cfg['misc']['visualize']:
    #         # optimizer.visualize()
    #         break
    #     else:
    #         file = open(args.error_filename + '.pkl', 'wb')
    #         pickle.dump(errors, file)
    #     weights = np.array(errors['n'])
    #     total = np.sum(weights)
    #     if dataset != 'semantic':
    #         for k, v in errors.items():
    #             if k =='epe':
    #                 output = np.sum(np.array(v) * weights) / total
    #                 print('~~~~~Average endpoint error display during training', '\n', k + ' : ' + str(output))
    #     torch.cuda.empty_cache()
    # weights = np.array(errors['n'])
    # total = np.sum(weights)
    # for k, v in errors.items():
    #     output = np.sum(np.array(v) * weights) / total
    #     print(k + ' : ' + str(output))
    # piou = np.sum(errors['tp']) / (np.sum(errors['tp']) + np.sum(errors['fp']) + np.sum(errors['fn']))
    # print('IOU : ' + str(piou))
    # niou = np.sum(errors['tn']) / (np.sum(errors['tn']) + np.sum(errors['fp']) + np.sum(errors['fn']))
    # miou = .5*(piou+niou)
    # print('mIOU : ' + str(miou))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--error_filename', type=str, default='errors_file')
    parser.add_argument('--cfg', type=str, default='configs/stereo_cfg.yaml')
    args = parser.parse_args()

    with open(args.cfg) as file:
        cfg = yaml.safe_load(file)
    print('=======================================================================','\n')
    print('configuration file:','\n')
    for key, value in cfg.items():
        print(key,': ' , value, '\n')
    print('=======================================================================','\n')

    optimize(cfg)
