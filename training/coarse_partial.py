"""Module for training the coarse cell-retrieval module
"""
import sys
sys.path.append('/home/mll/2025/Text2Loc-main/')
from torch.utils.tensorboard import SummaryWriter
from loss.iou_loss import IouLoss, DistanceLoss, IouUncertaintyLoss, DisUncertaintyLoss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
import collections

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os
import os.path as osp
import tqdm

from models.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells_semi import Kitti360CoarseSemiDatasetMulti, Kitti360CoarseSemiDataset
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss, ContrastiveLoss
from training.utils import plot_retrievals

"Training Process for global place recognition"
def train_epoch_iou(model, dataloader, args, global_step):
    model.train()
    epoch_losses = []
    epoch_losses_iou = []
    epoch_total_losses = []
    max_norm = 10

    batches = []
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))


    for i_batch, batch in pbar:

        optimizer.zero_grad()

        anchor = model.encode_text(batch["texts"])  # bs,256
        positive = model.encode_objects(batch["objects"]+batch["semi_objects"], batch["object_points"]+batch["semi_object_points"])   # bs,256

        loss = criterion(anchor, positive[:anchor.shape[0]])
        loss_iou = dis_uncertainty_criterion(anchor, positive[:anchor.shape[0]], positive[anchor.shape[0]:], 
                                    batch["poses"], batch["cells"], batch["semi_cells"], batch["semi_uncertainty"]) * 10
        # loss_iou = iou_uncertainty_criterion(anchor, positive[:anchor.shape[0]], positive[anchor.shape[0]:], 
                                    # batch["poses"], batch["cells"], batch["semi_cells"], batch["semi_uncertainty"]) * 10
        # loss_iou = dis_criterion(anchor, positive[:anchor.shape[0]], positive[anchor.shape[0]:], 
                                    # batch["poses"], batch["cells"], batch["semi_cells"])  * 10
        # loss_iou = iou_criterion(anchor, positive[:anchor.shape[0]], positive[anchor.shape[0]:], 
                                    # batch["poses"], batch["cells"], batch["semi_cells"])    * 10
        total_loss = loss + loss_iou

        total_loss.backward()

        ##### 记录梯度变化 #####
        total_gradient_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_gradient_norm += param.grad.norm(2).item() ** 2
        total_gradient_norm = total_gradient_norm ** 0.5
        writer.add_scalar('Gradients/Total_L2_norm', total_gradient_norm, global_step + i_batch)

        ##### 梯度裁剪 #####
        if total_gradient_norm > max_norm:
            scale = max_norm / total_gradient_norm
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(scale)

        total_gradient_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_gradient_norm += param.grad.norm(2).item() ** 2
        total_gradient_norm = total_gradient_norm ** 0.5
        # 将裁剪后的梯度L2范数记录到TensorBoard
        writer.add_scalar('Gradients/Total_L2_norm_after_clipping', total_gradient_norm, global_step + i_batch)

        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_losses_iou.append(loss_iou.item())
        epoch_total_losses.append(total_loss.item())

        torch.cuda.empty_cache()

        writer.add_scalar('Loss/loss_iter', loss.item(), global_step + i_batch)
        writer.add_scalar('Loss/loss_iou_iter', loss_iou.item(), global_step + i_batch)
        writer.add_scalar('Loss/total_loss_iter', total_loss.item(), global_step + i_batch)

    return np.mean(epoch_losses), np.mean(epoch_losses_iou), np.mean(epoch_total_losses), batches

def train_epoch(model, dataloader, args, global_step):
    model.train()
    epoch_losses = []

    batches = []
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))

    for i_batch, batch in pbar:
        optimizer.zero_grad()

        anchor = model.encode_text(batch["texts"])  # bs,256
        positive = model.encode_objects(batch["objects"], batch["object_points"])   # bs,256

        loss = criterion(anchor, positive)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        torch.cuda.empty_cache()

        writer.add_scalar('Loss/loss_iter', loss.item(), global_step + i_batch)

    return np.mean(epoch_losses), batches

@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False, return_distance=False):
    assert args.ranking_loss != "triplet"  # Else also update evaluation.pipeline

    model.eval()  
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(cells_dataset), model.embed_dim))        # 所有cell,256
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")                # 所有cell

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))   # 所有描述,256
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")        # 所有描述
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])   # num, 2(x,y)

    # Encode the query side
    t0 = time.time()
    index_offset = 0
    for batch in tqdm.tqdm(dataloader):

        text_enc = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    top_retrievals = {}  # {query_idx: top_cell_ids}
    if return_distance:
        dists_list = []
        scores_list = []
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]  # cell_encodings.shape[0]
            assert len(scores) == len(dataloader.dataset.all_cells) 
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0 : np.max(args.top_k)]

        # Best-cell hit accuracy
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        if return_distance:
            dists_list.append(dists[0:max(args.top_k)])
            scores_list.append(scores[sorted_indices])
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    if return_encodings:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings
    elif return_distance:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings, np.stack(dists_list), np.stack(scores_list)
    else:
        return accuracies, accuracies_close, top_retrievals

@torch.no_grad()
def calc_sim(model, dataloader, args, neighbour_range=128):
    assert args.ranking_loss != "triplet"  # Else also update evaluation.pipeline

    model.eval()

    unique_cells_dataset = dataloader.dataset.get_unique_cell_dataset()
    cells_dataloader = DataLoader(
        unique_cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in unique_cells_dataset.cells}
    cell_size = unique_cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(unique_cells_dataset), model.embed_dim))        # 所有cell,256
    db_cell_ids = np.zeros(len(unique_cells_dataset), dtype="<U32")                # 所有cell

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))   # 所有描述,256
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")        # 所有描述
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])   # num, 2(x,y)

    # Encode the query side
    t0 = time.time()
    index_offset = 0
    for batch in tqdm.tqdm(dataloader):

        text_enc = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")  

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    nearest_dict = dict()  # {query_idx: top_cell_ids}
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]  # [cell numbers,] = [cell numbers,256] @ [256,]
            assert len(scores) == len(cells_dataloader.dataset.cells) 
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0 : neighbour_range]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        mask = retrieved_cell_ids != target_cell_id
        nearest_dict[query_idx] = list(retrieved_cell_ids[mask])
    return nearest_dict

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

    writer = SummaryWriter(log_dir=f"./checkpoints/{dataset_name}/{folder_name}")
    custom_shuffle1 = True
    custom_shuffle2 = True
    eval_every_n_epoch = 4
    num_epoch_wo_iou = 0

    """
    Create data loaders
    """
    if args.dataset == "K360":
        # ['2013_05_28_drive_0003_sync', ]
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)    # 随机采样 256
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

        dataset_train = Kitti360CoarseSemiDatasetMulti(
            args.base_path,
            SCENE_NAMES_TRAIN,
            train_transform,
            shuffle_hints=True,
            flip_poses=True,
            
            exclude = False,     # False
            shuffle_batch_size = args.batch_size,
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseSemiDataset.collate_fn,
            shuffle=not custom_shuffle1,
            num_workers=args.cpus,
            # drop_last=True,
        )

        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform,)

        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

        dataset_test = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform,)

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data = dataset_train[0]
    assert len(data["debug_hint_descriptions"]) == args.num_mentioned
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    lr = args.learning_rate

    dict_loss = {lr: []}
    dict_acc = {k: [] for k in args.top_k}
    dict_acc_val = {k: [] for k in args.top_k}
    dict_acc_val_close = {k: [] for k in args.top_k}
    dict_acc_test = {k: [] for k in args.top_k}
    dict_acc_test_close = {k: [] for k in args.top_k}

    best_val_accuracy = -1
    last_model_save_path_val = None

    model = CellRetrievalNetwork(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args,
        )
    if args.continue_path:
        model_dic = torch.load(args.continue_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_dic, strict = False)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.ranking_loss == "pairwise":
        criterion = PairwiseRankingLoss(margin=args.margin)
    if args.ranking_loss == "hardest":
        criterion = HardestRankingLoss(margin=args.margin)
    if args.ranking_loss == "triplet":
        criterion = nn.TripletMarginLoss(margin=args.margin)
    if args.ranking_loss == "contrastive":
        criterion = ContrastiveLoss(temperature=args.temperature)
        iou_criterion = IouLoss()
        dis_criterion = DistanceLoss()
        iou_uncertainty_criterion = IouUncertaintyLoss()
        dis_uncertainty_criterion = DisUncertaintyLoss()

    if args.lr_scheduler == "exponential":  # 每个epoch后都会调整学习率
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    elif args.lr_scheduler == "step":   # 每过lr_step个epoch lr = lr * lr_gamma
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    else:
        raise TypeError

    global_step = 0
    if custom_shuffle1:
        dataloader_train.dataset.shuffle1()

    for epoch in range(1, args.epochs + 1):
        # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch
        if epoch <= num_epoch_wo_iou:
            loss, train_batches = train_epoch(model, dataloader_train, args, global_step)
        else:
            loss, loss_iou, total_loss, train_batches = train_epoch_iou(model, dataloader_train, args, global_step)
            writer.add_scalar('Loss/loss_iou_epoch', loss_iou, epoch)
            writer.add_scalar('Loss/total_loss_epoch', total_loss, epoch)
        global_step += len(dataloader_train)
        writer.add_scalar('Loss/loss_epoch', loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        dataloader_train.dataset.reset()        # 恢复原本顺序
        train_acc, train_acc_close, train_retrievals = eval_epoch(model, dataloader_train, args)  
        val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
        test_acc, test_acc_close, test_retrievals = eval_epoch(model, dataloader_test, args)

        key = lr
        dict_loss[key].append(loss)
        for k in args.top_k:
            dict_acc[k].append(train_acc[k])
            dict_acc_val[k].append(val_acc[k])
            dict_acc_val_close[k].append(val_acc_close[k])
            dict_acc_test[k].append(test_acc[k])
            dict_acc_test_close[k].append(test_acc_close[k])

            writer.add_scalar("Eval/train-"+str(k), train_acc[k], epoch)
            writer.add_scalar("Eval/val-"+str(k), val_acc[k], epoch)
            writer.add_scalar("Eval/test-"+str(k), test_acc[k], epoch)
    
        scheduler.step()
        print(f"\t lr {lr:0.4} loss {loss:0.3f} epoch {epoch} train-acc: ", end="")
        for k, v in train_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("val-acc: ", end="")
        for k, v in val_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("val-acc-close: ", end="")
        for k, v in val_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")

        print("test-acc: ", end="")
        for k, v in test_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("test-acc-close: ", end="")
        for k, v in test_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("\n", flush=True)

        if custom_shuffle1:
            if custom_shuffle2:
                if epoch % eval_every_n_epoch == 0:
                    sim_dict = calc_sim(model, dataloader_train, args, args.batch_size)
                if epoch >= eval_every_n_epoch:
                    dataloader_train.dataset.shuffle2(sim_dict, args.batch_size//2, args.batch_size)
                else:
                    dataloader_train.dataset.shuffle1()     # 应对最初的几回合的要shuffle
            else:
                dataloader_train.dataset.shuffle1()

        # Saving best model
        acc_val = val_acc[max(args.top_k)]
        if acc_val > best_val_accuracy:
            model_path = f"./checkpoints/{dataset_name}/{folder_name}/coarse_cont{cont}_epoch{epoch}_acc{acc_val:0.3f}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_loss-{args.ranking_loss}_f-{feats}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))

            print(f"Saving model at {acc_val:0.2f} to {model_path}")
            
            try:
                model_dic = model.state_dict()
                out = collections.OrderedDict()
                for item in model_dic:  # 排除模型中语言编码的固定参数的llm
                    if "llm_model" not in item:
                        out[item] = model_dic[item]
                torch.save(out, model_path)
                if (
                    last_model_save_path_val is not None
                    and last_model_save_path_val != model_path
                    and osp.isfile(last_model_save_path_val)
                ):  
                    print("Removing", last_model_save_path_val)
                    os.remove(last_model_save_path_val)
                
                last_model_save_path_val = model_path
                
            except Exception as e:
                print(f"Error saving model!", str(e))
            best_val_accuracy = acc_val
    
    writer.close()

