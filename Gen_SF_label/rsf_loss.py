import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d import transforms
import rsf_utils

class RSFLossv2:
    def __init__(self, anchors, config):
        self.anchors = anchors
        self.anchor_rots = rsf_utils.angle2rot_2d(anchors[:, 6])
        self.num_boxes = len(anchors)
        self.bgb_coeff = config['background_boost_coeff']
        self.sigmoid_slope = config['sigmoid_slope']
        self.epsilon = config['epsilon']
        self.heading_loss_coeff = config['heading_loss_coeff']
        self.angle_loss_coeff = config['angle_loss_coeff']
        self.mass_loss_coeff = config['mass_loss_coeff']
        self.dim_loss_coeff = config['dim_loss_coeff']

    def __call__(self, pc1, pc2, pc1_normals, pc2_normals, global_params, perbox_params, current_epoch):
        """
        :param pc1: Pointcloud object
        :param pc2: Pointcloud object
        :param global_params: bx12 tensor for ego motion (9 rotation + 3 translation)
        :param perbox_params: bxkx15 tensor (1 confidence + 8 box params + 4 rotation + 2 translation)
        :return: loss
        """

        pc1_padded, pc2_padded = pc1.points_padded(), pc2.points_padded()
        ego_transform = rsf_utils.global_params2Rt(global_params)
        boxes, box_transform = rsf_utils.perbox_params2boxesRt(perbox_params, self.anchors)
        box_transform_comp = transforms.Transform3d(matrix=ego_transform.get_matrix().detach().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)

        pc1_ego = ego_transform.transform_points(pc1_padded)
        pc1_normals_ego = transforms.Rotate(R=ego_transform.get_matrix()[:, :3, :3], device='cuda').transform_points(
            pc1_normals)
        bg_nnd_1 = knn_points(torch.cat((pc1_ego, pc1_normals_ego), dim=-1), torch.cat((pc2_padded, pc2_normals), dim=-1), pc1.num_points_per_cloud(), pc2.num_points_per_cloud())[0].squeeze(-1)
        bg_nnd_1 = torch.repeat_interleave(bg_nnd_1, self.num_boxes, dim=0)

        box_pc1, box_weights_1, weights_1, not_empty_1, box_pc1_normals = rsf_utils.box_weights(pc1, boxes, slope=self.sigmoid_slope, normals=pc1_normals)
        box_pc1_t = box_transform_comp[not_empty_1].transform_points(box_pc1.points_padded()[not_empty_1])
        box_pc1_normals = transforms.Rotate(R=box_transform_comp[not_empty_1].get_matrix()[:, :3, :3],
                                            device='cuda').transform_points(box_pc1_normals[not_empty_1])
        fg_nnd_1 = knn_points(torch.cat((box_pc1_t, box_pc1_normals), dim=-1), torch.repeat_interleave(torch.cat((pc2_padded, pc2_normals), dim=-1), self.num_boxes, dim=0)[not_empty_1],
                              box_pc1.num_points_per_cloud()[not_empty_1], pc2.num_points_per_cloud().repeat_interleave(self.num_boxes)[not_empty_1])[0].squeeze(-1)
        fg_nnd_1 = fg_nnd_1 + self.epsilon  # .005
        bg_nnd_1 = bg_nnd_1[not_empty_1]

        bg_nnd_var = torch.mean(torch.var(bg_nnd_1, dim=1),  dim=-1, keepdim=True)[0]
        if current_epoch>600 and current_epoch%10==0 and bg_nnd_var < 0.04 and self.sigmoid_slope<10:
            self.sigmoid_slope = self.sigmoid_slope + 0.02
        if current_epoch==460 and bg_nnd_var > 0.21 and self.sigmoid_slope>5.5:
            self.sigmoid_slope = self.sigmoid_slope - 1.5
        if current_epoch>650 and current_epoch%20==0 and bg_nnd_var > 2.5 and self.sigmoid_slope>2.5:
                self.sigmoid_slope = self.sigmoid_slope -0.06
        if current_epoch>800 and bg_nnd_var < 0.15 and bg_nnd_var > 0.02 and self.sigmoid_slope>4 and current_epoch%20==0:
            self.sigmoid_slope = self.sigmoid_slope -0.05
        # if current_epoch%1000==0 or current_epoch==100:
        #     print(current_epoch,'bg_nnd_1:',torch.round(torch.var(bg_nnd_1, dim=1)*1000)/1000,bg_nnd_var, bg_nnd_1.shape, 'self.sigmoid_slope:', self.sigmoid_slope)

        normalized_box_weights_1, normalized_weights_1 = rsf_utils.normalize(box_weights_1[not_empty_1], dim=-1), rsf_utils.normalize(weights_1[not_empty_1], dim=-1)
        fg_mean_1 = torch.sum(normalized_box_weights_1 * fg_nnd_1, dim=1, keepdim=True)
        bg_mean_1 = torch.sum(normalized_weights_1 * bg_nnd_1, dim=1, keepdim=True)

        confidence = boxes[:, :1][not_empty_1]
        foreground_loss = torch.sum(confidence * fg_mean_1) / global_params.shape[0]
        background_loss = torch.sum((1 - confidence) * bg_mean_1) / global_params.shape[0]
        background_boost = torch.sum((1 - confidence.detach()) * bg_mean_1) / global_params.shape[0]

        avg_sf = ego_transform.inverse().transform_points(box_transform_comp.transform_points(boxes[..., 1:4].unsqueeze(1)).view(-1, self.num_boxes, 3)) - boxes[..., 1:4].view(-1, self.num_boxes, 3)
        aligned_heading = torch.einsum('bij,abjk->abik', self.anchor_rots,perbox_params[:, :, 7:9].unsqueeze(-1)).squeeze(-1)
        heading_loss = torch.mean(torch.sum(torch.sum((aligned_heading - avg_sf[..., :2].detach()) ** 2, dim=-1), dim=-1))

        R_angle_loss = torch.sum(torch.square(transforms.matrix_to_euler_angles(box_transform.get_matrix()[:, :3, :3], 'ZYX')[..., 0]))/global_params.shape[0]

        # box regularization
        box_mass_1 = torch.sum(weights_1, dim=1)
        mass_loss = -torch.sum(box_mass_1) / global_params.shape[0]  # .02
        dim_regularization2 = torch.sum(perbox_params[:, :, 4:7] * perbox_params[:, :, 4:7]) / global_params.shape[0]

        total_loss = foreground_loss + background_loss + background_boost * self.bgb_coeff + mass_loss * self.mass_loss_coeff + dim_regularization2 * self.dim_loss_coeff \
                     + heading_loss * self.heading_loss_coeff + R_angle_loss * self.angle_loss_coeff
        loss = {'total_loss': total_loss,
                'foreground_loss': foreground_loss,
                'background_loss': background_loss,
                'background_boost': background_boost,
                'mass_loss': mass_loss,
                'dim_regularization': dim_regularization2,
                'heading_loss': heading_loss,
                'R_angle_loss': R_angle_loss}
        return loss

class RSFLossCycle:
    def __init__(self, anchors, config):
        self.anchors = anchors
        self.anchor_rots = rsf_utils.angle2rot_2d(anchors[:, 6])
        self.num_boxes = len(anchors)
        self.bgb_coeff = config['background_boost_coeff']
        self.sigmoid_slope = config['sigmoid_slope']
        self.epsilon = config['epsilon']
        self.heading_loss_coeff = config['heading_loss_coeff']
        self.angle_loss_coeff = config['angle_loss_coeff']
        self.mass_loss_coeff = config['mass_loss_coeff']
        self.dim_loss_coeff = config['dim_loss_coeff']

    def __call__(self, pc1, pc2, pc1_normals, pc2_normals, global_params, perbox_params):
        """
        :param pc1: Pointcloud object
        :param pc2: Pointcloud object
        :param global_params: bx12 tensor for ego motion (9 rotation + 3 translation)
        :param perbox_params: bxkx15 tensor (1 confidence + 8 box params + 4 rotation + 2 translation)
        :return: loss
        """

        pc1_padded, pc2_padded = pc1.points_padded(), pc2.points_padded()
        ego_transform = rsf_utils.global_params2Rt(global_params)
        boxes, box_transform = rsf_utils.perbox_params2boxesRt(perbox_params, self.anchors)
        box_transform_comp = transforms.Transform3d(matrix=ego_transform.get_matrix().detach().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)

        pc1_ego = ego_transform.transform_points(pc1_padded)
        pc1_normals_ego = transforms.Rotate(R=ego_transform.get_matrix()[:, :3, :3], device='cuda').transform_points(pc1_normals)
        bg_nnd_1 = knn_points(torch.cat((pc1_ego, pc1_normals_ego), dim=-1), torch.cat((pc2_padded, pc2_normals), dim=-1), pc1.num_points_per_cloud(), pc2.num_points_per_cloud())[0].squeeze(-1)
        bg_nnd_1 = torch.repeat_interleave(bg_nnd_1, self.num_boxes, dim=0)

        box_pc1, box_weights_1, weights_1, not_empty_1, box_pc1_normals = rsf_utils.box_weights(pc1, boxes, slope=self.sigmoid_slope, normals=pc1_normals)
        box_pc1_t = box_transform_comp[not_empty_1].transform_points(box_pc1.points_padded()[not_empty_1])
        box_pc1_normals = transforms.Rotate(R=box_transform_comp[not_empty_1].get_matrix()[:, :3, :3],
                                            device='cuda').transform_points(box_pc1_normals[not_empty_1])
        fg_nnd_1 = knn_points(torch.cat((box_pc1_t, box_pc1_normals), dim=-1), torch.repeat_interleave(torch.cat((pc2_padded, pc2_normals), dim=-1), self.num_boxes, dim=0)[not_empty_1],
                              box_pc1.num_points_per_cloud()[not_empty_1], pc2.num_points_per_cloud().repeat_interleave(self.num_boxes)[not_empty_1])[0].squeeze(-1)
        fg_nnd_1 = fg_nnd_1 + self.epsilon
        bg_nnd_1 = bg_nnd_1[not_empty_1]

        normalized_box_weights_1, normalized_weights_1 = rsf_utils.normalize(box_weights_1[not_empty_1], dim=-1), rsf_utils.normalize(weights_1[not_empty_1], dim=-1)
        fg_mean_1 = torch.sum(normalized_box_weights_1 * fg_nnd_1, dim=1, keepdim=True)
        bg_mean_1 = torch.sum(normalized_weights_1 * bg_nnd_1, dim=1, keepdim=True)

        confidence = boxes[:, :1][not_empty_1]
        foreground_loss = torch.sum(confidence * fg_mean_1) / global_params.shape[0]
        background_loss = torch.sum((1 - confidence) * bg_mean_1) / global_params.shape[0]

        # cycle consistency
        boxes_2, box_transform_2 = rsf_utils.get_reverse_boxesRt(perbox_params[:,:, 15:], boxes, transforms.Transform3d(matrix = box_transform_comp.get_matrix().detach()))
        bg_nnd_2 = knn_points(torch.cat((pc2_padded, pc2_normals), dim=-1), torch.cat((pc1_ego, pc1_normals_ego), dim=-1), pc2.num_points_per_cloud(), pc1.num_points_per_cloud())[0].squeeze(-1)
        bg_nnd_2 = torch.repeat_interleave(bg_nnd_2, self.num_boxes, dim=0)
        box_pc2, box_weights_2, weights_2, not_empty_2, box_pc2_normals = rsf_utils.box_weights(pc2, boxes_2, slope=self.sigmoid_slope, normals=pc2_normals)
        box_pc2_t = box_transform_2[not_empty_2].transform_points(box_pc2.points_padded()[not_empty_2])
        box_pc2_normals = transforms.Rotate(R=box_transform_2[not_empty_2].get_matrix()[:, :3, :3],
                                            device='cuda').transform_points(box_pc2_normals[not_empty_2])
        fg_nnd_2 = knn_points(torch.cat((box_pc2_t, box_pc2_normals), dim=-1), torch.repeat_interleave(torch.cat((pc1_ego, pc1_normals_ego), dim=-1), self.num_boxes, dim=0)[not_empty_2],
                              box_pc2.num_points_per_cloud()[not_empty_2], pc1.num_points_per_cloud().repeat_interleave(self.num_boxes)[not_empty_2])[0].squeeze(-1)
        fg_nnd_2 = fg_nnd_2 + self.epsilon
        bg_nnd_2 = bg_nnd_2[not_empty_2]

        normalized_box_weights_2, normalized_weights_2 = rsf_utils.normalize(box_weights_2[not_empty_2], dim=-1), rsf_utils.normalize(weights_2[not_empty_2], dim=-1)
        fg_mean_2 = torch.sum(normalized_box_weights_2 * fg_nnd_2, dim=1, keepdim=True)
        bg_mean_2 = torch.sum(normalized_weights_2 * bg_nnd_2, dim=1, keepdim=True)

        confidence_2 = boxes_2[:, :1][not_empty_2]
        foreground_loss_2 = torch.sum(confidence_2 * fg_mean_2) / global_params.shape[0]  # +.1*torch.mean(fg_var)
        background_loss_2 = torch.sum((1 - confidence_2) * bg_mean_2) / global_params.shape[0]  # +.1*torch.mean(bg_var)


        avg_sf = ego_transform.inverse().transform_points(box_transform_comp.transform_points(boxes[..., 1:4].unsqueeze(1)).view(-1, self.num_boxes, 3)) - boxes[..., 1:4].view(-1, self.num_boxes, 3)
        aligned_heading = torch.einsum('bij,abjk->abik', self.anchor_rots,perbox_params[:, :, 7:9].unsqueeze(-1)).squeeze(-1)
        heading_loss = torch.mean(torch.sum(torch.sum((aligned_heading - avg_sf[..., :2].detach()) ** 2, dim=-1), dim=-1))  # 10000

        R_angle_loss = torch.sum(torch.square(transforms.matrix_to_euler_angles(box_transform.get_matrix()[:, :3, :3], 'ZYX')[..., 0]))/global_params.shape[0]

        # box regularization
        box_mass_1 = torch.sum(weights_1, dim=1)
        mass_loss = -torch.sum(box_mass_1) / global_params.shape[0]  # .02
        dim_regularization2 = torch.sum(perbox_params[:, :, 4:7] * perbox_params[:, :, 4:7]) / global_params.shape[0]

        total_loss = foreground_loss + background_loss + foreground_loss_2 + background_loss_2 + mass_loss * self.mass_loss_coeff + dim_regularization2 * self.dim_loss_coeff \
                     + heading_loss * self.heading_loss_coeff + R_angle_loss * self.angle_loss_coeff
        loss = {'total_loss': total_loss,
                'foreground_loss': foreground_loss,
                'background_loss': background_loss,
                'mass_loss': mass_loss,
                'dim_regularization': dim_regularization2,
                'heading_loss': heading_loss,
                'R_angle_loss': R_angle_loss}
        return loss

