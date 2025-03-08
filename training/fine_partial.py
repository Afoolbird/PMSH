"""Module for training the fine matching module
"""
import sys
sys.path.append('/home/mll/2025/Text2Loc-main/')
from torch.utils.tensorboard import SummaryWriter
from loss.gaussian_loss import GaussianLoss
from loss.CommonOperations import *
import loss.Constants as constants

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections

import torch_geometric.transforms as T

import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import os
import os.path as osp
import tqdm

from models.cross_matcher_semi import CrossMatchSemi

from dataloading.kitti360pose.poses import Kitti360FineDataset, Kitti360FineDatasetMulti
from dataloading.kitti360pose.poses_semi import Kitti360FineSemiDataset, Kitti360FineSemiDatasetMulti


# from datapreparation.semantic3d.imports import COLORS as COLORS_S3D, COLOR_NAMES as COLOR_NAMES_S3D
from datapreparation.kitti360pose.utils import (
    COLORS as COLORS_K360,
    COLOR_NAMES as COLOR_NAMES_K360,
    SCENE_NAMES_TEST,
)
from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision, calc_pose_error2, calc_pose_error3, calc_pose_error4

"Training Process for fine localization"
def train_epoch_semi(model, dataloader, args, global_step):
    model.train()

    offset_lambda = args.offset_lambda
    
    stats = EasyDict(
        loss=[],
        loss_offsets=[],
        pose_offsets=[],

        loss_semi_offsets=[],
        pose_semi_offsets=[]
    )
        
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))
    for i_batch, batch in pbar:
        optimizer.zero_grad()
        texts = batch["texts"]

        output, sigmas = model(batch["objects"] + batch["semi_objects"], texts, batch["object_points"] + batch["semi_object_points"], sign_semi=True)

        new_output = torch.cat((output[:len(texts)], sigmas[:len(texts)]), dim=1)
        loss_1, loss_2, loss_offsets = criterion_offsets(
            new_output.unsqueeze(1), torch.tensor(np.array(batch["offsets"]), dtype=torch.float, device=device).unsqueeze(1)
        )
        
        new_semi_output = torch.cat((output[len(texts):], sigmas[len(texts):]), dim=1)
        loss_semi_1, loss_semi_2, loss_semi_offsets = criterion_offsets(
            new_semi_output.unsqueeze(1), torch.tensor(np.array(batch["semi_offsets"]), dtype=torch.float, device=device).unsqueeze(1)
        )

        loss =  offset_lambda * (loss_offsets + loss_semi_offsets)  
        # loss =  offset_lambda * (loss_offsets + loss_semi_offsets * 0.1)    
        # gpt:相当于是在调整学习率进行优化? 相当于0.0003*5
        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print()
            print(str(e))
            print()
            print(batch["all_matches"])

        stats.loss.append(loss.item())
        stats.loss_offsets.append(loss_offsets.item())
        stats.loss_semi_offsets.append(loss_semi_offsets.item())
        error = calc_pose_error2(           # 计算的是两个坐标的l2距离
                batch["objects"],
                batch["poses"],
                offsets=output[:len(texts)].detach().cpu().numpy(),
            )
        semi_error = calc_pose_error3(           # 计算的是两个坐标的l2距离
                np.array(batch["semi_offsets"]),
                offsets=output[len(texts):].detach().cpu().numpy(),
            )

        stats.pose_offsets.append(error)
        stats.pose_semi_offsets.append(semi_error)
        pbar.set_postfix(loss = loss_offsets.item() + loss_semi_offsets.item(), error = (error+semi_error)/2)

        writer.add_scalar('Train/iter_loss1', loss_1.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_loss2', loss_2.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_semi_loss1', loss_semi_1.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_semi_loss2', loss_semi_2.item(), global_step + i_batch)

        writer.add_scalar('Train/iter_loss', loss.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_loss_offsets', loss_offsets.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_loss_semi_offsets', loss_semi_offsets.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_pose_offsets', error, global_step + i_batch)
        writer.add_scalar('Train/iter_pose_semi_offsets', semi_error, global_step + i_batch)

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

def train_epoch(model, dataloader, args, global_step):
    model.train()

    offset_lambda = args.offset_lambda
    
    stats = EasyDict(
        loss=[],
        loss_offsets=[],
        pose_offsets=[],
    )
        
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))
    for i_batch, batch in pbar:
        optimizer.zero_grad()
        texts = batch["texts"]

        output, sigmas = model(batch["objects"], texts, batch["object_points"])

        new_output = torch.cat((output, sigmas), dim=1)
        loss_1, loss_2, loss_offsets = criterion_offsets(
            new_output.unsqueeze(1), torch.tensor(np.array(batch["offsets"]), dtype=torch.float, device=device).unsqueeze(1)
        )

        loss =  offset_lambda * loss_offsets
        # gpt:相当于是在调整学习率进行优化? 相当于0.0003*5
        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print()
            print(str(e))
            print()
            print(batch["all_matches"])

        stats.loss.append(loss.item())
        stats.loss_offsets.append(loss_offsets.item())
        error = calc_pose_error2(           # 计算的是两个坐标的l2距离
                batch["objects"],
                batch["poses"],
                offsets=output[:len(texts)].detach().cpu().numpy(),
            )

        stats.pose_offsets.append(error)
        pbar.set_postfix(loss = loss_offsets.item(), error = error)

        writer.add_scalar('Train/iter_loss1', loss_1.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_loss2', loss_2.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_loss', loss.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_loss_offsets', loss_offsets.item(), global_step + i_batch)
        writer.add_scalar('Train/iter_pose_offsets', error, global_step + i_batch)

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

@torch.no_grad()
def eval_epoch(model, dataloader, args,):
    model.eval() 
    
    stats = EasyDict(
        # recall=[],
        # precision=[],
        # pose_mid=[],
        # pose_mean=[],
        pose_offsets=[],
    )
    
    for i_batch, batch in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
        
        texts = batch["texts"]
        output, _ = model(batch["objects"], texts, batch["object_points"])     # bs,2
        stats.pose_offsets.append(
            calc_pose_error2(
                batch["objects"],
                # output.matches0.detach().cpu().numpy(),
                batch["poses"],
                offsets=output.detach().cpu().numpy(),
            )
        )

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

@torch.no_grad()
def train_save_uncertainty(model, dataloader, args,):
    assert args.continue_path is not None
    model.train()
    
    dir_path = os.path.dirname(args.continue_path)
    uncertainty = []
    semi_uncertainty = []
    semi_errors = []

    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))
    for i_batch, batch in pbar:
        texts = batch["texts"]
        output, sigmas = model(batch["objects"] + batch["semi_objects"], texts, batch["object_points"] + batch["semi_object_points"], sign_semi=True)
        uncertainty.append(compute_uncertainty(sigmas[:len(texts)].detach().cpu().unsqueeze(1)))
        semi_uncertainty.append(compute_uncertainty(sigmas[len(texts):].detach().cpu().unsqueeze(1)))
        
        semi_error = calc_pose_error4(           # 计算的是两个坐标的l2距离
                np.array(batch["semi_offsets"]),
                offsets=output[len(texts):].detach().cpu().numpy(),
            )
        semi_errors.append(semi_error)
    
    data_semi_error = np.concatenate(semi_errors, axis=0)
    save_semi_error_path = os.path.join(dir_path, 'semi_error.txt')
    np.savetxt(save_semi_error_path, data_semi_error, fmt='%.10f', delimiter=',')

    data_uncertainty = torch.cat(uncertainty, dim=0).numpy()
    data_semi_uncertainty = torch.cat(semi_uncertainty, dim=0).numpy()
    data_array = np.column_stack((data_uncertainty, data_semi_uncertainty))
    save_uncertainty_path = os.path.join(dir_path, 'uncertaintys.txt')
    np.savetxt(save_uncertainty_path, data_array, fmt='%.10f', delimiter=',')
    return

@torch.no_grad()
def compute_uncertainty(L_vect):
    batch_size            = L_vect.shape[0]
    L_mat = get_zero_variable((batch_size, 1, 2 ,2), L_vect)
    L_mat[:, :, 0, 0] = L_vect[:, :, 0]                                 
    L_mat[:, :, 1, 0] = L_vect[:, :, 1]                                 
    L_mat[:, :, 1, 1] = L_vect[:, :, 2]  
    Sigma = torch.matmul(L_mat, L_mat.transpose(2, 3))   
    det_Sigma = Sigma[:, :, 0, 0] * Sigma[:, :, 1, 1] - Sigma[:, :, 0, 1] * Sigma[:, :, 1, 0] + constants.EPSILON

    return det_Sigma.squeeze()

if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    folder_name = args.folder_name
    print("#####################")
    print("########   Folder Name: " + folder_name)
    print("#####################")
    dataset_name = "final"
    if not osp.isdir(f"./checkpoints/{dataset_name}/{folder_name}"):
        os.mkdir(f"./checkpoints/{dataset_name}/{folder_name}")

    # writer = SummaryWriter(log_dir=f"./checkpoints/{dataset_name}/{folder_name}")

    """
    Create data loaders
    """
    if args.dataset == "K360":
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360FineSemiDatasetMulti(
            args.base_path, SCENE_NAMES_TRAIN, train_transform, args, flip_pose=False
        ) 
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360FineSemiDataset.collate_fn,
            shuffle=args.shuffle,
        )

        dataset_val = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform, args,)
        dataloader_val = DataLoader(
            dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn
        )

        dataset_test = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform, args,)
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    best_val_offset = 1000  # Measured by mean of recall and precision
    last_model_save_path = None

    lr = args.learning_rate 

    train_stats_loss = {lr: []}
    train_stats_loss_offsets = {lr: []}
    train_stats_pose_offsets = {lr: []}
    val_stats_pose_offsets = {lr: []}
    test_stats_pose_offsets = {lr: []}

    model = CrossMatchSemi(
        dataset_train.get_known_classes(),
        COLOR_NAMES_K360,
        args,
    )
    if bool(args.continue_path):
        model_dic = torch.load(args.continue_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_dic, strict = False)
        print("load model success")

    model.to(device)

    criterion_offsets = GaussianLoss()

    # Warm-up
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

    num_epoch_warmup = 1
    global_step = 0
    num_epoch_wo_semi = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == num_epoch_warmup:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            if args.lr_scheduler == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
            elif args.lr_scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
                # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
            else:
                raise TypeError
    
        ########## 保存半正样本不确定性的 ##########
        train_save_uncertainty(model, dataloader_train, args)
        ########## 保存半正样本不确定性的 ##########

    #     if epoch <= num_epoch_wo_semi:
    #         train_out = train_epoch(model, dataloader_train, args, global_step)
    #     else:
    #         train_out = train_epoch_semi(model, dataloader_train, args, global_step)
    #         writer.add_scalar('Train/epoch_loss_semi_offsets', train_out.loss_semi_offsets, epoch)
    #         writer.add_scalar('Train/epoch_pose_semi_offsets', train_out.pose_semi_offsets, epoch)
    #     global_step += len(dataloader_train)
    #     writer.add_scalar('Train/epoch_loss', train_out.loss, epoch)
    #     writer.add_scalar('Train/epoch_loss_offsets', train_out.loss_offsets, epoch)
    #     writer.add_scalar('Train/epoch_pose_offsets', train_out.pose_offsets, epoch)
    #     writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

    #     train_stats_loss[lr].append(train_out.loss)
    #     train_stats_loss_offsets[lr].append(train_out.loss_offsets)
    #     train_stats_pose_offsets[lr].append(train_out.pose_offsets)

    #     val_out = eval_epoch(model, dataloader_val, args,)  # CARE: which loader for val!
    #     val_stats_pose_offsets[lr].append(val_out.pose_offsets)

    #     print()

    #     test_out = eval_epoch(model, dataloader_test, args,)  # CARE: which loader for test!
    #     test_stats_pose_offsets[lr].append(test_out.pose_offsets)

    #     print()

    #     writer.add_scalar('Eval/val_pose_offsets', val_out.pose_offsets, epoch)
    #     writer.add_scalar('Eval/test_pose_offsets', test_out.pose_offsets, epoch)

    #     if scheduler:
    #         scheduler.step()

    #     print(
    #         (
    #             f"\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} "
    #             f"t-offset {train_out.pose_offsets:0.3f} "
    #             f"v-offset {val_out.pose_offsets:0.3f} "
    #             f"e-offset {test_out.pose_offsets:0.3f} "
    #         ),
    #         flush=True,
    #     )

    #     offset = np.mean(val_out.pose_offsets)
    #     if offset < best_val_offset:
    #         model_path = f"./checkpoints/{dataset_name}/{folder_name}/fine_cont{cont}_epoch{epoch}_offset{offset:0.3f}_lr{args.learning_rate}_obj-{args.num_mentioned}-{args.pad_size}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_f-{feats}.pth"
    #         if not osp.isdir(osp.dirname(model_path)):
    #             os.mkdir(osp.dirname(model_path))

    #         print("Saving model to", model_path)
    #         try:
    #             model_dic = model.state_dict()
    #             out = collections.OrderedDict()
    #             for item in model_dic:
    #                 if "llm_model" not in item:
    #                     out[item] = model_dic[item]
    #             torch.save(out, model_path)
    #             if (
    #                 last_model_save_path is not None
    #                 and last_model_save_path != model_path
    #                 and osp.isfile(last_model_save_path)
    #             ):
    #                 print("Removing", last_model_save_path)
    #                 os.remove(last_model_save_path)
    #             last_model_save_path = model_path
    #         except Exception as e:
    #             print("Error saving model!", str(e))
    #         best_val_offset = offset
        
    # writer.close()
