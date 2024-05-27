# ViDAR: Visual Point Cloud Forecasting

## Table of Contents

1. [Installation](#installation)
2. [Optimization](#train-and-evaluate)



## Installation <a name="installation"></a>
### Installing system software packages

For convenience, we list the steps below:
```bash
sudo apt-get update
sudo dpkg --set-selections < ./requirements/installed_packages.txt
sudo apt-get dselect-upgrade
```

### Create a Python virtual environment (optional)
```bash
python3 -m venv new_env
source new_env/bin/activate
```

### Install Python packages
```bash
pip3 install -r ./requirements/python_packages.txt
```

## Optimization <a name="train-and-evaluate"></a>

```bash
nohup python3 optimizer_sf_label.py \
  --cfg configs/argoverse_cfg.yaml \
  --error_filename ./log_vis/argoverse_log/error \
  --data_filename ./configs/argoverse_files/argoverse_file_name_0.txt \
  > ./log_vis/argoverse_log/file0.txt 2>&1 &
```

- `nohup` ensures that the script continues running after the terminal is closed or disconnected.
- `python3` executes the Python script `optimizer_sf_label.py`.
- `--cfg`, `--error_filename`, and `--data_filename` pass the configuration file path, error log file path, and data file path, respectively.
- `>` and `2>&1` redirect the standard output and standard error to the specified log file.
- `&` runs the command in the background.

