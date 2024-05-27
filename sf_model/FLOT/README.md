
2. Install the repository:
```bash
cd sf_model
$ pip install -e ./FLOT
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple shapely pypng numba numpy==1.19.5
```

[pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl](https://huggingface.co/lilpotat/pytorch3d/raw/main/pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl)
```
pip install pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
cd sf_model/FLOT/flot/scripts
```
 You can edit flot's code on the fly and import function and classes of flot in other project as well.

* To uninstall this package, run:
```bash
$ pip uninstall flot
```

* **Quickest test.** Type:
```bash
$ cd /path/to/flot/scripts/
$ python val_test.py
```

For help on how to use this script, type:  
```bash
$ cd /path/to/flot/scripts/
$ python val_test.py --help
```
 
### Evalating

To evaluate this pretrained model on **Argoverse Scene Flow Test Dataset**, type:
```bash
$ cd /path/to/flot/scripts/
$ python val_test.py --dataset argoverse_lidar --test --nb_points 8192 --path2ckpt /path/to/sf_model/checkpoints/flot_argoverse_epe0.043.tar
```
To evaluate this pretrained model on **lidarKITTI Scene Flow Test Dataset**, type:
```bash
$ cd /path/to/flot/scripts/
$ python val_test.py --dataset kitti_lidar2 --test --nb_points 8192 --path2ckpt /path/to/sf_model/checkpoints/flot_lidarKITTI_epe0.018.tar
```

To evaluate this pretrained model on **nuScenes Scene Flow Test Dataset**, type:
```bash
$ cd /path/to/flot/scripts/
$ python val_test.py --dataset nuscenes_lidar --test --nb_points 8192 --path2ckpt /path/to/sf_model/checkpoints/flot_nuScenes_epe0.061.tar
```

To evaluate this pretrained model on **Waymo Scene Flow Test Dataset**, type:
```bash
$ cd /path/to/flot/scripts/
$ python val_test.py --dataset waymo_lidar --test --nb_points 8192 --path2ckpt /path/to/sf_model/checkpoints/flot_nuScenes_epe0.061.tar
```



### Training
By default, the model and tensorboard's logs are stored in `/path/to/flot/experiments`. A script to train a flot model is available in `/path/to/flot/train.py`. For help on how to use this script, please type:
```bash
$ cd /path/to/flot/scripts/
$ python train.py --help
```

1. **KITTI datasets.** Use 3DSFLabelling to generate 3D scene flow ground truth on the KITTI odometry dataset, and evaluate it on the lidarKITTI testset, type:
```bash
$ cd /path/to/flot/
$ nohup python train.py --nb_iter 1 --dataset kitti_lidar2 --dataset_path ./kitti-od/kitti_sf_label/ --path2val ./sceneflow_eval_dataset/lidar_kitti2 --nb_points 8192 --path2ckpt sf_model/checkpoints/flot_lidarKITTI_epe0.018.tar --batch_size 2 --nb_epochs 60 > ../log/kitti2_iter1_aug_log.txt 2>&1 &
```

2. **Argoverse datasets.** Use 3DSFLabelling to generate 3D scene flow ground truth on the KITTI odometry dataset, and evaluate it on the lidarKITTI testset, type:
```bash
$ cd /path/to/flot/
$ nohup python train.py --nb_iter 1 --dataset argoverse_lidar --dataset_path ./argoverse/argoverse_SF_label --nb_points 8192 --path2ckpt sf_model/checkpoints/flot_argoverse_epe0.043.tar --batch_size 8 --nb_epochs 60 > ../log/argo_iter1_aug_log.txt 2>&1 &
```

3. **Argoverse datasets.** Use 3DSFLabelling to generate 3D scene flow ground truth on the KITTI odometry dataset, and evaluate it on the lidarKITTI testset, type:
```bash
$ cd /path/to/flot/
$ nohup python train.py --nb_iter 1 --dataset argoverse_lidar --dataset_path ./argoverse/argoverse_SF_label --nb_points 8192 --path2ckpt sf_model/checkpoints/flot_argoverse_epe0.043.tar --batch_size 8 --nb_epochs 60 > ../log/argo_iter1_aug_log.txt 2>&1 &
```

### Using your own dataset

It is possible to train FLOT on you own dataset by creating a new dataloader that inherits from `flot.datasets.generic.SceneFlowDataset`.

Your new dataloader's class then needs to implement the function `load_sequence(self, idx)` that loads the `idx` example of the dataset. Please refer to the documentation of `flot.datasets.generic.SceneFlowDataset.load_sequence` for more information.

Examples of dataloaders are available in the directory `datasets`, see, e.g., `flot.datasets.flyingthing3D_hplflownet`.

Once your new dataloader is implemented, it can be used in the script `train.py` for training or in the `val_test.py` for evaluation by importing this new dataset in the function `my_main`.

### Using FLOT

Import FLOT by typing
```python
from flot.models import FLOT
```

FLOT's constructor accepts one argument: `nb_iter`, which is the number of unrolled iterations of the Sinkhorn algorithm. In our experiments, we tested 1, 3, and 5 iterations. For example:
```python
flot = FLOT(nb_iter=3)
```

The simpler model FLOT<sub>0</sub> is obtained by setting `nb_iter=0`. In this case, the OT module is equivalent to an attention layer.
```python
flot_0 = FLOT(nb_iter=0)
```

Input point clouds `pc1` and `pc2` can be passed to `flot` to estimate the flow from `pc1` to `pc2` as follows:
```python
scene_flow = flot([pc1, pc2])
```
The input point clouds `pc1` and `pc2` must be torch tensors of size `batch_size x nb_points x 3`.


### Making the current implementation faster

* Currently a nearest neighbour search, needed to perform convolutions on points, is done by an exhaustive search in the function `flot.models.FLOT.forward`. This search could be moved in the dataloader where fast algorithm can be used.

* The transport cost `C` in the function `flot.tools.ot.sinkhorn` is computed densely, whereas several values are masked right after by `support`. The computation of `support` could be done in the dataloader using fast nearest neighbours search. The computation of `C` could then be limited to the required entries.

* It is also possible to change the type convolution on point clouds used in `flot.models.FLOT.__init(self, nb_iter)__`. We provided an home-made, self-contained, implementation of PointNet++-like convolutions. One can use other convolutions on points for which faster implementation exists.
