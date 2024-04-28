
cd /mnt/cfs/algorithm/chaokang.jiang/rsf-Optimizing/GMSF/
pip install pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple shapely pypng numba numpy==1.19.5
cd ./Gen_SF_label

python3 optimizer_sf_label.py --cfg configs/lidar_cfg.yaml --error_filename ./log_vis/lidarKITTI/error

##  argoverse_cfg 

CUDA_VISIBLE_DEVICES=0 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_0.txt  > ./log_vis/argoverse_log/file0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 optimizer_sf_label.py --cfg configs/argoverse_cfg.yaml --error_filename ./log_vis/argoverse_log/error --data_filename ./configs/argoverse_files/argoverse_file_name_1.txt  > 
