import os
import glob

def list_files(path):
    files = []
    for file in glob.glob(os.path.join(path, "*")):
        if os.path.isfile(file):
            files.append(os.path.basename(file))
    return files

def find_same_files(paths):
    files = []
    for path in paths:
        files += list_files(path)
    duplicates = []
    for file in set(files):
        if files.count(file) > 1:
            duplicates.append(file)
    return duplicates

def list_subfolders(path):
    subfolders = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders

path = "./sceneflow_eval_dataset/argoverse/argoverse/val/"
subfolders = list_subfolders(path)
print('subfolders:','\n',subfolders)

# paths = ["/path/to/folder1", "/path/to/folder2", "/path/to/folder3"]
duplicates = find_same_files(subfolders)
print('duplicates:','\n', duplicates)