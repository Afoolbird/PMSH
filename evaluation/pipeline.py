import sys
sys.path.append('/home/mll/2025/Text2Loc-main/')
from loss.CommonOperations import *
import loss.Constants as constants

from datapreparation.kitti360pose.imports import Object3d
import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict
from copy import deepcopy
import pickle

import torch
from torch.utils.data import DataLoader
import time

from scipy.spatial.distance import cdist

from models.cell_retrieval import CellRetrievalNetwork
from models.cross_matcher_semi import CrossMatchSemi

from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360pose.eval import Kitti360TopKDataset

from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360

from training.coarse import eval_epoch as eval_epoch_retrieval
from training.utils import plot_retrievals

import torch_geometric.transforms as T
import tqdm

"""
TODO:
- Try to add num_matches*10 + sum(match_scores[correctly_matched])
"""


@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

        # Run retrieval model to obtain top-cells
        
    retrieval_accuracies, retrieval_accuracies_close, retrievals = eval_epoch_retrieval(
        model, dataloader, args
    )       # 目前看来这三个分别是检索到正确的cell_id 和 检索距离满足要求 和 根据文本检索到的子地图的 按匹配顺序的子地图id
    retrievals = [retrievals[idx] for idx in range(len(retrievals))]  # Dict -> list
    print("Retrieval Accs:")
    print(retrieval_accuracies)
    print("Retrieval Accs Close:")
    print(retrieval_accuracies_close)
    assert len(retrievals) == len(dataloader.dataset.all_poses)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))  # Predict cell-centers   将cell的中间位置作为预测位置进行计算准确率
        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies


@torch.no_grad()
def run_fine(model, retrievals, dataloader, args, transform_fine):
    # A batch in this dataset contains max(top_k) times the pose vs. each of the max(top_k) top-cells.

    # import pdb; pdb.set_trace()
    model.eval()
    dataset_topk = Kitti360TopKDataset(
        dataloader.dataset.all_poses, dataloader.dataset.all_cells, retrievals, transform_fine, args,)

    num_samples = max(args.top_k)

    t0 = time.time()
    matches = []
    offsets = []
    confidences = []
    cell_ids = []
    poses_w = []

    uncertaintys = []
    min_dists = []
    min_uncertains = []
    
    t0 = time.time()
    pbar = tqdm.tqdm(enumerate(dataset_topk), total = len(dataset_topk))
    for i_sample, sample in pbar:
        output, sigma = model(sample["objects"], sample["texts"], sample["object_points"])  # 10,3
        offsets.append(output.detach().cpu().numpy())
        uncertaintys.append(compute_uncertainty(sigma.detach().cpu().unsqueeze(1)))

        cell_ids.append([cell.id for cell in sample["cells"]])
        poses_w.append(sample["poses"][0].pose_w)
    print(f"Ran matching for {len(dataset_topk)} queries in {time.time() - t0:0.2f}.")

    assert len(offsets) == len(retrievals)
    cell_ids = np.array(cell_ids)

    t1 = time.time()
    print("ela:", t1 - t0)

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    # Gather the accuracies for each sample
    accuracies_offset = {k: {t: [] for t in args.threshs} for k in args.top_k}

    accuracies_offset_uncertainty = {1: {t: [] for t in args.threshs}}

    save_offsets = []
    for i_sample in tqdm.tqdm(range(len(retrievals))):
        pose = dataloader.dataset.all_poses[i_sample]   # gt x,y
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        sample_offsets = offsets[i_sample]
        uncertainty = uncertaintys[i_sample]

        if not np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample]):
            print()
            print([cell.id for cell in top_cells])
            print(cell_ids[i_sample])

        assert np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample])
        assert np.allclose(pose.pose_w, poses_w[i_sample])

        # Get objects, matches and offsets for each of the top-cells
        pos_in_cells_offsets = []
        for i_cell in range(len(top_cells)):
            # Copy the cell and pad it again, as the fine model might have matched a padding-object
            cell = deepcopy(top_cells[i_cell])
            while len(cell.objects) < args.pad_size:
                cell.objects.append(Object3d.create_padding())  # 没用

            cell_offsets = sample_offsets[i_cell]
            pos_in_cells_offsets.append(cell_offsets)
        pos_in_cells_offsets = np.array(pos_in_cells_offsets)

        accs_offsets, min_certainty, min_dist, min_dist_index = calc_sample_accuracies2(
            pose, top_cells, pos_in_cells_offsets, args.top_k, args.threshs, uncertainty
        )
        if min_dist != np.inf:
            uncertainty = uncertaintys[i_sample][min_dist_index]
            min_dists.append(min_dist)
            min_uncertains.append(uncertainty)

        for k in args.top_k:
            for t in args.threshs:
                accuracies_offset[k][t].append(accs_offsets[k][t])

        for t in args.threshs:
            accuracies_offset_uncertainty[1][t].append(min_certainty[1][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies_offset[k][t] = np.mean(accuracies_offset[k][t])
    
    for t in args.threshs:
        accuracies_offset_uncertainty[1][t] = np.mean(accuracies_offset_uncertainty[1][t])

    sorted_indices = sorted(range(len(min_uncertains)), key=lambda i: min_uncertains[i])
    # 根据排序索引对列表1和列表2进行排序
    sorted_list1 = [min_uncertains[i] for i in sorted_indices]
    sorted_list2 = [min_dists[i] for i in sorted_indices]

    data_array = np.column_stack((sorted_list1, sorted_list2))
    # np.savetxt('checkpoints/k360_30-10_scG_pd10_pc4_spY_all/record_semi/record_val.txt', data_array, fmt='%.10f', delimiter=',')
    # np.savetxt('checkpoints/k360_30-10_scG_pd10_pc4_spY_all/record_semi/record_test.txt', data_array, fmt='%.10f', delimiter=',')
    # np.savetxt('checkpoints/k360_30-10_scG_pd10_pc4_spY_all/record_semi/record_train.txt', data_array, fmt='%.10f', delimiter=',')

    return accuracies_offset, accuracies_offset_uncertainty

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

def calc_sample_accuracies2(pose, top_cells, pos_in_cells, top_k, threshs, uncertainty):
    pose_w = pose.pose_w
    assert len(top_cells) == max(top_k) == len(pos_in_cells)
    num_samples = len(top_cells)

    # Calc the pose-prediction in world coordinates for each cell
    pred_w = np.array(      # 每个单元格预测的真实坐标值
        [
            top_cells[i].bbox_w[0:2] + pos_in_cells[i, :] * top_cells[i].cell_size
            for i in range(num_samples)
        ]
    )

    # Calc the distances to the gt-pose
    dists = np.linalg.norm(pose_w[0:2] - pred_w, axis=1)    # 和真实的pose之间的距离
    assert len(dists) == max(top_k)

    # Discard close-by distances from different scenes      # 丢弃不同场景的近距离
    pose_scene_name = pose.cell_id.split("_")[0]
    cell_scene_names = np.array([cell.id.split("_")[0] for cell in top_cells])
    dists[pose_scene_name != cell_scene_names] = np.inf

    # Calculate the accuracy: is one of the top-k dists small enough?
    sorted_arr = np.sort(dists)
    sorted_indices1 = np.argsort(dists)

    sorted_uncertainty = np.sort(uncertainty)
    sorted_indices2 = np.argsort(uncertainty)

    return {k: {t: np.min(dists[0:k]) <= t for t in threshs} for k in top_k}, \
            {1:{t: dists[sorted_indices2[0]] <= t for t in threshs}}, \
                sorted_arr[0], sorted_indices1[0]

if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))

    # Load datasets
    if args.no_pc_augment:  # True
        transform = T.FixedPoints(args.pointnet_numpoints)  # 256
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()]) # 归一化到[-1,1]

    if args.no_pc_augment_fine: # True
        transform_fine = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform_fine = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])


    if args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False,
        )
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
        )
    
    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # Load models
    model_coarse_dic = torch.load(args.path_coarse, map_location=torch.device("cpu"))
    model_coarse = CellRetrievalNetwork(
                KNOWN_CLASS,
                COLOR_NAMES_K360,
                args,
            )
    model_coarse.load_state_dict(model_coarse_dic, strict = False)
    model_coarse.to(device)

    if args.path_fine:
        model_fine_dic = torch.load(args.path_fine, map_location=torch.device("cpu"))
        model_fine = CrossMatchSemi(
            KNOWN_CLASS,
            COLOR_NAMES_K360,
            args,
        )
        model_fine.load_state_dict(model_fine_dic, strict = False)
        model_fine.to(device)

    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_coarse, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

    # 从检索结果中可视化查询图像与相关检索图像之间的距离和相似性，并生成带有视觉标注（如距离、边框颜色等）的图像文件
    if args.plot_retrievals:
        plot_retrievals(retrievals, dataset_retrieval)

    # Run fine
    accuracies_offsets, accuracies_offsets_uncertainty  = run_fine(
        model_fine, retrievals, dataloader_retrieval, args, transform_fine
    )
    print_accuracies(accuracies_offsets, "Fine")
    print_accuracies(accuracies_offsets_uncertainty, "Fine Uncertainty")
