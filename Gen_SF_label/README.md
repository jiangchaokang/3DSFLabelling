
cd /mnt/cfs/algorithm/chaokang.jiang/rsf-Optimizing/GMSF/
pip install pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple shapely pypng numba numpy==1.19.5
cd /mnt/cfs/algorithm/chaokang.jiang/3DSFLabeling/Gen_SF_label

python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error

##  argoverse_cfg     ----flow-jck-sleep6--------

CUDA_VISIBLE_DEVICES=0 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_0.txt  > ./log_vis/argoverse_log/file0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_1.txt  > ./log_vis/argoverse_log/file1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_2.txt  > ./log_vis/argoverse_log/file2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_3.txt  > ./log_vis/argoverse_log/file3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_4.txt  > ./log_vis/argoverse_log/file4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_5.txt  > ./log_vis/argoverse_log/file5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_6.txt  > ./log_vis/argoverse_log/file6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_7.txt  > ./log_vis/argoverse_log/file7.txt 2>&1 &

### ----flow-jck-sleep10--------
CUDA_VISIBLE_DEVICES=0 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_8.txt  > ./log_vis/argoverse_log/file8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_9.txt  > ./log_vis/argoverse_log/file9.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_10.txt  > ./log_vis/argoverse_log/file10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_11.txt  > ./log_vis/argoverse_log/file11.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_12.txt  > ./log_vis/argoverse_log/file12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_13.txt  > ./log_vis/argoverse_log/file13.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_14.txt  > ./log_vis/argoverse_log/file14.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_15.txt  > ./log_vis/argoverse_log/file15.txt 2>&1 &

### sleep29-jck-master-0

CUDA_VISIBLE_DEVICES=0 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_0.txt > ./log_vis/lidarKITTI/file0.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_1.txt > ./log_vis/lidarKITTI/file1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_2.txt > ./log_vis/lidarKITTI/file2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_3.txt > ./log_vis/lidarKITTI/file3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_4.txt > ./log_vis/lidarKITTI/file4.txt 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_5.txt > ./log_vis/lidarKITTI/file5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_6.txt > ./log_vis/lidarKITTI/file6.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_7.txt > ./log_vis/lidarKITTI/file7.txt 2>&1 &

### sleep2-jck-master-0

CUDA_VISIBLE_DEVICES=0 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_8.txt > ./log_vis/lidarKITTI/file8.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_9.txt > ./log_vis/lidarKITTI/file9.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_10.txt > ./log_vis/lidarKITTI/file10.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_11.txt > ./log_vis/lidarKITTI/file11.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_12.txt > ./log_vis/lidarKITTI/file12.txt 2>&1 &

# ----------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=3 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_13.txt > ./log_vis/lidarKITTI/file13.txt 2>&1 &
# ----------------------------------------------------------------------------------------------------------------------------

cd /mnt/cfs/algorithm/chaokang.jiang/3DSFLabeling/Gen_SF_label
CUDA_VISIBLE_DEVICES=6 nohup python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_14.txt > ./log_vis/lidarKITTI/file14.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python3.8 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error --data_filename ./configs/LidarKITTI_files/kitti_file_name_15.txt > ./log_vis/lidarKITTI/file15.txt 2>&1 &


## NuScenes

cd /mnt/cfs/algorithm/chaokang.jiang/3DSFLabeling/Gen_SF_label

python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error

### flow-jck-sleep0
CUDA_VISIBLE_DEVICES=0 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_0.txt > ./log_vis/nuScenes_log/file0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_1.txt > ./log_vis/nuScenes_log/file1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_2.txt > ./log_vis/nuScenes_log/file2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_3.txt > ./log_vis/nuScenes_log/file3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_4.txt > ./log_vis/nuScenes_log/file4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_5.txt > ./log_vis/nuScenes_log/file5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_6.txt > ./log_vis/nuScenes_log/file6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_7.txt > ./log_vis/nuScenes_log/file7.txt 2>&1 &

### flow-jck-sleep1
CUDA_VISIBLE_DEVICES=0 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_8.txt > ./log_vis/nuScenes_log/file8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_9.txt > ./log_vis/nuScenes_log/file9.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_10.txt > ./log_vis/nuScenes_log/file10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_11.txt > ./log_vis/nuScenes_log/file11.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_12.txt > ./log_vis/nuScenes_log/file12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_13.txt > ./log_vis/nuScenes_log/file13.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_14.txt > ./log_vis/nuScenes_log/file14.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 optimizer_sf_label.py --cfg configs/nuscenes_cfg.yaml --error_filename ./log_vis/nuScenes_log/error --data_filename ./configs/nuScenes_files/nuScenes_file_name_15.txt > ./log_vis/nuScenes_log/file15.txt 2>&1 &







##  semantic_kitti
CUDA_VISIBLE_DEVICES=0 nohup python3 without_learning2.py --cfg configs/semantic_cfg.yaml --error_filename ./vis/semantic_kitti/error > ./vis/semantic_kitti/epoch1_75kb4.txt 2>&1 &


# data 

waymo_open: pc1, pc2, sem_label_s, sem_label_t, inst_label_s, inst_label_t, mot_label_s, mot_label_t, pose_s, pose_t
semantic_kitti: pc1, pc2, pose_s, pose_t
argoverse: pc1, pc2, mask1_tracks_flow, mask2_tracks_flow
