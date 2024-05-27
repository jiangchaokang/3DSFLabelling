import os
import time
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from flot.datasets.generic import Batch
from flot.models.scene_flow import FLOT
from val_test import eval_model
import pdb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def compute_epe(est_flow, batch):
    """
    Compute EPE during training.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    epe : torch.Tensor
        Mean EPE for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    return epe


def compute_loss(est_flow, batch):
    """
    Compute training loss.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    loss = torch.mean(torch.abs(error))

    return loss


def train(scene_flow, trainloader, testloader, delta, optimizer, scheduler, path2log, nb_epochs):
    """
    Train scene flow model.

    Parameters
    ----------
    scene_flow : flot.models.FLOT
        FLOT model
    trainloader : flots.datasets.generic.SceneFlowDataset
        Dataset loader.
    delta : int
        Frequency of logs in number of iterations.
    optimizer : torch.optim.Optimizer
        Optimiser.
    scheduler :
        Scheduler.
    path2log : str
        Where to save logs / model.
    nb_epochs : int
        Number of epochs.

    """

    # Log directory
    if not os.path.exists(path2log):
        os.makedirs(path2log)
    writer = SummaryWriter(path2log)

    # Reload state
    total_it = 0
    epoch_start = 0

    # Train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # scene_flow = scene_flow.to(device, non_blocking=True)
    best_epe = 52.0
    for epoch in range(epoch_start, nb_epochs):

        # Init.
        running_epe = 0
        running_loss = 0

        # Train for 1 epoch
        start = time.time()
        scene_flow = scene_flow.train()
        for it, batch in enumerate(tqdm(trainloader)):

            # Send data to GPU
            batch = batch.to(device, non_blocking=True)

            # Gradient step
            optimizer.zero_grad()
            est_flow = scene_flow(batch["sequence"])
            loss = compute_loss(est_flow, batch)
            loss.backward()
            optimizer.step()

            # pdb.set_trace()
            # np.save("/parh/to/data_odometry_velodyne/visualization/nusc_pc1",batch["sequence"][0][0].cpu())
            # np.save("/parh/to/data_odometry_velodyne/visualization/nusc_pc2",batch["sequence"][1][0].cpu())
            # np.save("/parh/to/data_odometry_velodyne/visualization/nusc_pc3",batch["sequence"][0][0].cpu()+est_flow[0].detach().cpu()) 
            # Loss evolution
            running_loss += loss.item()
            train_epe = compute_epe(est_flow, batch).item()
            running_epe += train_epe
            print("loss.item():", loss.item())
            print("train_epe:", train_epe)
            # Logs
            if it % delta == delta - 1:
                # Print / save logs
                writer.add_scalar("Loss/epe", running_epe / delta, total_it)
                writer.add_scalar("Loss/loss", running_loss / delta, total_it)
                print(
                    "Epoch {0:d} - It. {1:d}: loss = {2:e}".format(
                        epoch, total_it, running_loss / delta
                    )
                )
                print(time.time() - start, "seconds")
                # Re-init.
                running_epe = 0
                running_loss = 0
                start = time.time()
        
        print("epoch:",epoch,"Current learning rate:", scheduler.get_last_lr())
        mean_epe, mean_outlier, mean_acc3d_relax, mean_acc3d_strict = eval_model(scene_flow,testloader)
        if mean_epe<best_epe:
            best_epe = mean_epe
            model_to_save = scene_flow.module
            state = {
                "nb_iter": model_to_save.nb_iter,
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            # savepath = os.path.join(path2log, str(epoch))
            # if not os.path.exists(savepath):
            #     os.makedirs(savepath)
            torch.save(state, os.path.join(path2log, "best_model_{}.tar".format(epoch)))
            print("The best model has been saved", "\n epe:",mean_epe)

            scene_flow = scene_flow.train()

                

            total_it += 1

        # Scheduler
        scheduler.step()

        # Save model after each epoch
        state = {
            "nb_iter": scene_flow.module.nb_iter,
            "model": scene_flow.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, os.path.join(path2log, "latest_model.tar"))

    #
    print("Finished Training")

    return None


def my_main(dataset_name, dataset_path, path2val, nb_iter, batch_size, max_points, nb_epochs, path2ckpt):
    """
    Entry point of the script.

    Parameters
    ----------
    dataset_name : str
        Version of FlyingThing3D used for training: 'HPLFlowNet' / 'flownet3d'.
    nb_iter : int
        Number of unrolled iteration of Sinkhorn algorithm in FLOT.
    batch_size : int
        Batch size.
    max_points : int
        Number of points in point clouds.
    nb_epochs : int
        Number of epochs.

    Raises
    ------
    ValueError
        If dataset_name is an unknow dataset.

    """

    # Path to current file
    pathroot = dataset_path  # os.path.dirname(__file__)

    # Path to dataset
    if dataset_name.lower() == "kitti_lidar".lower() or dataset_name.lower() == "kitti_lidar2".lower():
        from flot.datasets.kitti_lidar import lidarKITTI
        path2data = pathroot
        # lr_lambda = lambda epoch: 1.0 if epoch < 650 else 0.1
        lr_lambda = lambda epoch: (1 - epoch / nb_epochs) ** 0.9
        data_train = lidarKITTI(root_dir=path2data, nb_points=max_points, mode="train")
    elif dataset_name.lower() == "nuscenes_lidar".lower():
        from flot.datasets.nuscenes_lidar import lidarNuScenes
        path2data = pathroot
        lr_lambda = lambda epoch: (1 - epoch / nb_epochs) ** 0.9
        data_train = lidarNuScenes(root_dir=path2data, nb_points=max_points, mode="train")
    elif dataset_name.lower() == "argoverse_lidar".lower():
        from flot.datasets.argoverse_lidar import lidarArgoverse
        path2data = pathroot
        lr_lambda = lambda epoch: (1 - epoch / nb_epochs) ** 0.9
        data_train = lidarArgoverse(root_dir=path2data, nb_points=max_points, mode="train", dataset_name="argoverse")
    else:
        raise ValueError("Invalid dataset name: " + dataset_name)

    # Training dataset
    # ft3d_train = FT3D(root_dir=path2data, nb_points=max_points, mode="train")
    
    trainloader = DataLoader(
        data_train,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=16,
        collate_fn=Batch,
        drop_last=True,
    )

    if dataset_name.lower() == "kitti_lidar".lower():
            mode = "test"
            path2data = os.path.join(path2val, "KITTI_processed_occ_final")
            from flot.datasets.kitti_hplflownet import Kitti
            dataset_val = Kitti(root_dir=path2data, nb_points=max_points)
    elif dataset_name.lower() == "kitti_lidar2".lower():
            mode = "test"
            path2data = path2val
            from flot.datasets.dataset_eval import lidarKITTIEval
            dataset_val = lidarKITTIEval(root_dir=path2data, nb_points=max_points)
    elif dataset_name.lower() == "nuscenes_lidar".lower():
            mode = "test"
            path2data = "./sceneflow_eval_dataset/nuscenes/withNormal"
            from flot.datasets.dataset_eval import lidarEval
            dataset_val = lidarEval(root_dir=path2data, nb_points=max_points)
    elif dataset_name.lower() == "argoverse_lidar".lower():
            mode = "test"
            path2data = "./sceneflow_eval_dataset/argoverse/withNormal"
            from flot.datasets.dataset_eval import lidarEval
            dataset_val = lidarEval(root_dir=path2data, nb_points=max_points, dataset_name="argoverse")
            # mode = "test"
            # path2data = "./sceneflow_eval_dataset/waymo_flow_gt"
            # from flot.datasets.dataset_eval import lidarWaymoEval
            # dataset_val = lidarWaymoEval(root_dir=path2data, nb_points=max_points)
    else:
        raise ValueError("Unknown dataset " + dataset_name)
    print("\n\nDataset: " + path2data + " " + mode)
    
    testloader = DataLoader(
        dataset_val,
        batch_size=1,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        collate_fn=Batch,
        drop_last=False,
    )

    scene_flow = FLOT(nb_iter=nb_iter)

    file = torch.load(path2ckpt)
    scene_flow.nb_iter = file["nb_iter"]
    scene_flow.load_state_dict(file["model"])

    
    if torch.cuda.is_available():
        # 获取可用的CUDA设备数量
        num_devices = torch.cuda.device_count()
        # num_devices = 1
        if num_devices > 1:
            torch.backends.cudnn.benchmark = True
            scene_flow = torch.nn.DataParallel(scene_flow)
            scene_flow.cuda()
        else:
            # 将模型移动到默认的CUDA设备
            scene_flow.cuda()
    else:
        raise EnvironmentError("CUDA is not available.")

    
    # Optimizer
    optimizer = torch.optim.Adam(scene_flow.parameters(), lr=2e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Log directory
    now = datetime.now().strftime("%y_%m_%d-%H_%M_%S_%f")
    now += "__Iter_" + str(nb_iter)
    now += "__Pts_" + str(max_points)
    path2log = os.path.join("./SF_Model/FLOT/flot/log/Ablation_experiment",dataset_name, now)

    # Train
    print("Training started. Logs in " + path2log)
    train(scene_flow, trainloader, testloader, 500, optimizer, scheduler, path2log, nb_epochs)

    return None


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train FLOT.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HPLFlowNet",
        help="Training dataset. Either HPLFlowNet or " + "flownet3d.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/parh/to/data_odometry_velodyne/sf_kitti_lidar2",
        help="Training dataset Path.",
    )
    parser.add_argument(
        "--path2val",
        type=str,
        default="/parh/to/scene_flow",
        help="Training dataset Path.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--nb_epochs", type=int, default=40, help="Number of epochs.")
    parser.add_argument(
        "--nb_points",
        type=int,
        default=2048,
        help="Maximum number of points in point cloud.",
    )
    parser.add_argument(
        "--nb_iter",
        type=int,
        default=1,
        help="Number of unrolled iterations of the Sinkhorn " + "algorithm.",
    )
    parser.add_argument(
        "--path2ckpt",
        type=str,
        default="../pretrained_models/model_8192.tar",
        help="Path to saved checkpoint.",
    )
    args = parser.parse_args()

    # Launch training
    my_main(args.dataset, args.dataset_path, args.path2val, args.nb_iter, args.batch_size, args.nb_points, args.nb_epochs, args.path2ckpt)
