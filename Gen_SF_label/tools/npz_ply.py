import numpy as np
import open3d as o3d

# 定义文件路径
file_paths = ['/mnt/cfs/algorithm/chaokang.jiang/3DSFLabeling/Driving_datasets/nuScenes/n015-2018-07-18-11-18-34+0800__LIDAR_TOP__1531884527500329.npz']

for file_path in file_paths:
    # 读取npz文件
    data = np.load(file_path)
    point_cloud1 = data['pc1']
    point_cloud2 = data['pc2']
    flow = point_cloud1 # + data['flow']

    # 合并两帧点云数据
    merged_point_cloud = np.vstack((point_cloud1, flow))

    # 创建索引数组
    index1 = np.arange(point_cloud1.shape[0])
    index2 = np.arange(flow.shape[0]) + point_cloud1.shape[0]

    # 将两个索引数组合并成一个
    merged_index = np.concatenate((index1.reshape(-1, 1), index1.reshape(-1, 1)), axis=1)

    ## colors
    pink = np.array([255, 192, 203], dtype=np.float32) / 255.0
    skyblue = np.array([135, 206, 235], dtype=np.float32) / 255.0
    limegreen = np.array([50, 205, 50], dtype=np.float32) / 255.0
    colors = np.zeros((point_cloud1.shape[0], 3))
    colors[0] = pink

    # 将点云转换为Open3D格式
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
    pcd1.colors = o3d.utility.Vector3dVector(colors)
    colors = np.zeros((point_cloud2.shape[0], 3))
    colors[1] = skyblue

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_cloud2)
    pcd2.colors = o3d.utility.Vector3dVector(colors)
    print(index1.shape,merged_index.shape)
    colors = np.zeros((point_cloud1.shape[0], 3))
    colors[2] = limegreen
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(merged_point_cloud),
        lines=o3d.utility.Vector2iVector(merged_index),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)


    # 将点云和场景流可视化
    ##  o3d.visualization.draw_geometries([pcd1, pcd2, ls])

    file_path = file_path.replace("flow_gt", "flow_gt_ply")
     # 将点云和场景流保存为ply格式
    o3d.io.write_point_cloud(file_path.replace('.npz', '_pcd1.ply'), pcd1)
    o3d.io.write_point_cloud(file_path.replace('.npz', '_pcd2.ply'), pcd2)
    o3d.io.write_line_set(file_path.replace('.npz', '_flow.ply'), ls)