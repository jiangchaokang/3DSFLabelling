import shutil
import os

def merge_folders(source_folders, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for folder in source_folders:
        for file_name in os.listdir(folder):
            abs_path = os.path.join(folder, file_name)
            if os.path.isfile(abs_path):
                shutil.copy(abs_path, destination_folder)

def save_file_names(file_names, file_path):
    with open(file_path, "w") as f:
        for file_name in file_names:
            f.write(file_name + "\n")

def list_subfolders(path):
    subfolders = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders

            
folders_path = "../../dataset/sceneflow_eval_dataset/argoverse/argoverse/val/"
source_folders = list_subfolders(folders_path)
print("How many subfiles in total:",len(source_folders))
destination_folder = "../../dataset/sceneflow_eval_dataset/argoverse/merge_argoverse/"
merge_folders(source_folders, destination_folder)
print("Complete the merger.")


file_names = os.listdir(destination_folder)
file_path = "../../dataset/sceneflow_eval_dataset/argoverse/merge_argoverse/argoverse_names.txt"
save_file_names(file_names, file_path)
print("Complete the write.")



## -------------------------------------------------------------------------------
# import os
# import shutil

# # 定义要合并的文件夹路径和新文件夹路径
# folder_path = "sceneflow_eval_dataset/waymo_open/flow_gt_raw/"
# new_folder_path = "sceneflow_eval_dataset/waymo_open/flow_gt/"

# # 新文件名计数器
# count = 0

# # 创建一个txt文件，用于保存新文件名
# with open("new_file_names.txt", "w") as f:
#     # 遍历文件夹下所有子文件夹
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             # 获取文件后缀名
#             file_ext = os.path.splitext(file)[1]
#             # 生成新文件名
#             new_file_name = f"waymo_flow_{count}{file_ext}"
#             # 拼接新文件路径
#             new_file_path = os.path.join(new_folder_path, new_file_name)
#             # 拼接原文件路径
#             file_path = os.path.join(root, file)
#             # 复制文件到新文件夹并重命名
#             if not os.path.isfile(file_path):
#                 print(f'Error: {file_path} does not exist.')
#                 exit()
#             if not os.path.exists(os.path.dirname(new_file_path)):
#                 os.makedirs(os.path.dirname(new_file_path))
#             # os.rename(file_path, new_file_path)
#             try:
#                 shutil.move(file_path, new_file_path)
#             except FileNotFoundError as e:
#                 print(f'Error: {e}')
#                 exit()
#             except Exception as e:
#                 print(f'Error: {e}')
#                 exit()

#             print(f'{file_path} has been moved to {new_file_path}.')
#             # 将新文件名写入txt文件
#             f.write(new_file_name + "\n")
#             # 计数器加1
#             count += 1
#             print(':count:',count,'\n','new_file_path::',new_file_path)