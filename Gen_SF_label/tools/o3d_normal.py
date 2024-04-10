import numpy as np
import open3d as o3d
import os
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

# ------------------------    stereo_kitti    ------------------------  
# folder_path = './sceneflow_eval_dataset/stereo_kitti'
# out_path = './sceneflow_eval_dataset/stereo_kitti/withNormal/'

# ------------------------    lidar_kitti2    ------------------------  
# folder_path = './sceneflow_eval_dataset/lidar_kitti2'
# out_path = './sceneflow_eval_dataset/lidar_kitti2/withNormal/'

# ------------------------    nuscenes_kitti    ------------------------  
# folder_path = './sceneflow_eval_dataset/nuscenes/nuscenes/val'
# out_path = './sceneflow_eval_dataset/nuscenes/nuscenes/withNormal/'

# ------------------------    waymo_flow_gt    ------------------------  
# folder_path = '../../dataset/sceneflow_eval_dataset/waymo_flow_gt'
# out_path = '../../dataset/sceneflow_eval_dataset/waymo_flow_gt/withNormal/'


folder_path = '../../dataset/sceneflow_eval_dataset/argoverse/merge_argoverse/'
out_path = '../../dataset/sceneflow_eval_dataset/argoverse/merge_argoverse/withNormal/'

if not os.path.exists(out_path):
    os.makedirs(out_path)
#     print(f'{out_path} 目录已创建')
# else:
#     print(f'{out_path} 目录已存在')


for file_name in os.listdir(folder_path):
    if file_name.endswith('.npz'):
        # 读取点云数据
        data = np.load(os.path.join(folder_path, file_name))

        new_data = {}
        for key in data.keys():
            if key!='pc1' or key!='pc2':
                array = data[key]
                new_data[key] = array
        

        points1 = data['pc1']
        points2 = data['pc2']

        # 将点云转换成open3d格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points1)
        # 计算法向量
        pcd.estimate_normals()
        points_with_normals = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        new_data['pc1'] = points_with_normals
        new_data['pc1_normals'] = normals

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        # 计算法向量
        pcd2.estimate_normals()

        points_with_normals2 = np.asarray(pcd2.points)
        normals2 = np.asarray(pcd2.normals)
        new_data['pc2'] = points_with_normals2
        new_data['pc2_normals'] = normals2
        np.savez(os.path.join(out_path, file_name), **new_data)
        print(os.path.join(out_path, file_name),"completed")