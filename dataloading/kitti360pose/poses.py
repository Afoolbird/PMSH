from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict
from numpy.lib.function_base import flip
import math
import json

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    CLASS_TO_INDEX,
    COLORS,
    COLOR_NAMES,
    SCENE_NAMES,
)
from datapreparation.kitti360pose.utils import sentence_style_t, sentence_style_n, sentence_style_s, sentence_style_w, sentence_style_e
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360pose.base import Kitti360BaseDataset
from dataloading.kitti360pose.utils import batch_object_points, flip_pose_in_cell
from dataloading.kitti360pose.utils import OBJECT_LIST


def load_pose_and_cell(pose: Pose, cell: Cell, hints, pad_size, transform, args, flip_pose=False,
                       horizontal_flip=False, vertical_flip = False):
    assert pose.cell_id == cell.id

    descriptions = pose.descriptions
    cell_objects_dict = {obj.id: obj for obj in cell.objects}

    matched_ids = [descr.object_id for descr in descriptions if descr.is_matched]
    matched_objects = [cell_objects_dict[matched_id] for matched_id in matched_ids]

    assert len(pose.descriptions) == args.num_mentioned     # 都是6个描述
    assert len(pose.descriptions) - pose.get_number_unmatched() == len(matched_objects)

    # Hints and descriptions have to be in same order
    for descr, hint in zip(descriptions, hints):
        assert descr.object_label in hint

    # Gather offsets
    # NOTE: Currently trains on best-offsets if available (matched)!
    if args.regressor_cell == "pose" and args.regressor_learn == "closest":
        offsets = np.array([descr.offset_closest for descr in descriptions])[:, 0:2]
    if args.regressor_cell == "pose" and args.regressor_learn == "center":
        offsets = np.array([descr.offset_center for descr in descriptions])[:, 0:2]
    if args.regressor_cell == "best" and args.regressor_learn == "closest":
        offsets = []
        for i_descr, descr in enumerate(descriptions):
            if descr.is_matched:
                offsets.append(descr.best_offset_closest[0:2])
            else:
                offsets.append(descr.offset_closest[0:2])
    if args.regressor_cell == "best" and args.regressor_learn == "center":
        offsets = []
        for i_descr, descr in enumerate(descriptions):
            if descr.is_matched:
                offsets.append(descr.best_offset_center[0:2])
            else:
                offsets.append(descr.offset_center[0:2])
    
    if args.regressor_cell == "all":    # True
        offsets = [(pose.pose_w[0] - cell.bbox_w[0]) / (cell.bbox_w[3] - cell.bbox_w[0]), (pose.pose_w[1] - cell.bbox_w[1]) / (cell.bbox_w[4]  - cell.bbox_w[1])]
    
    offsets = np.array(offsets)

    # Build best-center offsets for oracle
    offsets_best_center = []
    for i_descr, descr in enumerate(descriptions):
        if descr.is_matched:
            offsets_best_center.append(descr.best_offset_center[0:2])
        else:
            offsets_best_center.append(descr.offset_center[0:2])

    offsets_valid = np.sum(np.isnan(offsets)) == 0

    # Gather matched objects
    objects, matches = [], []  # Matches as [(obj_idx, hint_idx)]

    for i_descr, descr in enumerate(descriptions):
        if descr.is_matched:
            hint_obj = cell_objects_dict[descr.object_id]
            assert hint_obj.instance_id == descr.object_instance_id
            objects.append(hint_obj)

            obj_idx = len(objects) - 1
            hint_idx = i_descr
            matches.append((obj_idx, hint_idx))

    # Gather distractors, i.e. remaining objects
    for obj_index, obj in enumerate(cell.objects):
        if obj.id not in matched_ids:
            objects.append(obj)

    if len(objects) != len(cell.objects):
        print([obj.id for obj in objects])
        print([obj.id for obj in cell.objects])
        print(matched_ids)
    assert len(objects) == len(
        cell.objects
    ), f"Not all cell-objects have been gathered! {len(objects)}, {len(cell.objects)}, {cell.id}"

    # Pad or cut-off distractors (CARE: the latter would use ground-truth data!)
    if len(objects) > pad_size:
        objects = objects[0:pad_size]

    while len(objects) < pad_size:
        obj = Object3d.create_padding()
        objects.append(obj)
    
    # Build matches and all_matches
    # The matched objects are always put in first, however, our geometric models have no knowledge of these indices as they are permutation invariant.
    all_matches = matches.copy()

    # Add unmatched hints
    for i_descr, descr in enumerate(descriptions):
        if not descr.is_matched:
            obj_idx = len(objects)  # Match to objects-side bin
            hint_idx = i_descr
            all_matches.append((obj_idx, hint_idx))

    # Add unmatched objects
    for obj_idx, obj in enumerate(objects):
        if obj.id not in matched_ids:
            hint_idx = len(descriptions)  # Match to hints-side bin
            all_matches.append((obj_idx, hint_idx))

    matches, all_matches = np.array(matches), np.array(all_matches)
    assert len(matches) == len(matched_ids)
    assert len(all_matches) == len(objects) + len(descriptions) - len(matches)
    assert np.sum(all_matches[:, 1] == len(descriptions)) == len(objects) - len(
        matched_ids
    )  # Binned objects
    assert np.sum(all_matches[:, 0] == len(objects)) == len(descriptions) - len(
        matched_ids
    )  # Binned hints


    text = " ".join(hints)

    object_points = batch_object_points(objects, transform)

    object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in objects]
    object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in objects]

    return {
        "poses": pose,
        "cells": cell,
        "objects": objects,
        "object_points": object_points,
        "hint_descriptions": hints,
        "texts": text,
        "matches": matches,
        "all_matches": all_matches,
        "offsets": np.array(offsets),
        "offsets_best_center": np.array(offsets_best_center),
        # 'offsets_valid': offsets_valid,
        "object_class_indices": object_class_indices,
        "object_color_indices": object_color_indices,
    }


def load_pose_and_cell_aug2(pose: Pose, cell: Cell, hints, pad_size, transform, args, flip_pose=False,
                       horizontal_flip=False, vertical_flip = False, new_matching = None):
    assert new_matching is not None

    descriptions = pose.descriptions
    cell_objects_dict = {obj.id: obj for obj in cell.objects}

    matched_ids = [idx for idx in new_matching if idx is not None]
    matched_objects = [cell_objects_dict[matched_id] for matched_id in matched_ids]

    assert len(pose.descriptions) == args.num_mentioned
    # assert len(pose.descriptions) - pose.get_number_unmatched() == len(matched_objects)

    # Hints and descriptions have to be in same order
    for descr, hint in zip(descriptions, hints):
        assert descr.object_label in hint

    # Gather offsets
    # NOTE: Currently trains on best-offsets if available (matched)!
    if args.regressor_cell == "pose" and args.regressor_learn == "closest":
        offsets = np.array([descr.offset_closest for descr in descriptions])[:, 0:2]
    if args.regressor_cell == "pose" and args.regressor_learn == "center":
        offsets = np.array([descr.offset_center for descr in descriptions])[:, 0:2]

    
    if args.regressor_cell == "all":    # True
        offsets = [(pose.pose_w[0] - cell.bbox_w[0]) / (cell.bbox_w[3] - cell.bbox_w[0]), (pose.pose_w[1] - cell.bbox_w[1]) / (cell.bbox_w[4]  - cell.bbox_w[1])]
    
    offsets = np.array(offsets)

    # Build best-center offsets for oracle
    # TODO: This part has error
    offsets_best_center = []
    for i_descr, descr in enumerate(descriptions):
        if descr.is_matched:
            offsets_best_center.append(descr.best_offset_center[0:2])
        else:
            offsets_best_center.append(descr.offset_center[0:2])

    offsets_valid = np.sum(np.isnan(offsets)) == 0

    # Gather matched objects
    objects, matches = [], []  # Matches as [(obj_idx, hint_idx)]
    for i_descr, descr in enumerate(descriptions):
        if new_matching[i_descr] is not None:
            hint_obj = cell_objects_dict[new_matching[i_descr]]
            # assert hint_obj.instance_id == descr.object_instance_id
            objects.append(hint_obj)

            obj_idx = len(objects) - 1
            hint_idx = i_descr
            matches.append((obj_idx, hint_idx))

    # Gather distractors, i.e. remaining objects
    for obj_index, obj in enumerate(cell.objects):
        if obj.id not in matched_ids:
            objects.append(obj)


    if len(objects) != len(cell.objects):
        print([obj.id for obj in objects])
        print([obj.id for obj in cell.objects])
        print(matched_ids)
    assert len(objects) == len(
        cell.objects
    ), f"Not all cell-objects have been gathered! {len(objects)}, {len(cell.objects)}, {cell.id}"

    # Pad or cut-off distractors (CARE: the latter would use ground-truth data!)
    if len(objects) > pad_size:
        objects = objects[0:pad_size]

    while len(objects) < pad_size:
        obj = Object3d.create_padding()
        objects.append(obj)
    
    # Build matches and all_matches
    # The matched objects are always put in first, however, our geometric models have no knowledge of these indices as they are permutation invariant.
    all_matches = matches.copy()    # [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    # Add unmatched hints
    for i_descr, descr in enumerate(descriptions):
        if new_matching[i_descr] is None:
            obj_idx = len(objects)  # Match to objects-side bin
            hint_idx = i_descr
            all_matches.append((obj_idx, hint_idx))

    # Add unmatched objects
    for obj_idx, obj in enumerate(objects):
        if obj.id not in matched_ids:
            hint_idx = len(descriptions)  # Match to hints-side bin
            all_matches.append((obj_idx, hint_idx))

    matches, all_matches = np.array(matches), np.array(all_matches)
    assert len(matches) == len(matched_ids)
    assert len(all_matches) == len(objects) + len(descriptions) - len(matches)
    assert np.sum(all_matches[:, 1] == len(descriptions)) == len(objects) - len(
        matched_ids
    )  # Binned objects
    assert np.sum(all_matches[:, 0] == len(objects)) == len(descriptions) - len(
        matched_ids
    )  # Binned hints

    

    text = " ".join(hints)

    object_points = batch_object_points(objects, transform)

    object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in objects]
    object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in objects]

    return {
        "poses": pose,      # gt-pose
        "cells": cell,
        "objects": objects,     # 排序之后的objects
        "object_points": object_points, # 排序之后的object_points
        "hint_descriptions": hints,
        "texts": text,          # 描述的文本
        "matches": matches,
        "all_matches": all_matches,
        "offsets": np.array(offsets),   # 偏差
        "offsets_best_center": np.array(offsets_best_center),
        # 'offsets_valid': offsets_valid,
        "object_class_indices": object_class_indices,
        "object_color_indices": object_color_indices,
    }

def load_pose_and_cell_aug(pose: Pose, cell: Cell, hints, pad_size, transform, args, flip_pose=False,
                       horizontal_flip=False, vertical_flip = False):
    # assert pose.cell_id == cell.id

    descriptions = pose.descriptions

    # Gather offsets
    # NOTE: Currently trains on best-offsets if available (matched)!
    if args.regressor_cell == "pose" and args.regressor_learn == "closest":
        offsets = np.array([descr.offset_closest for descr in descriptions])[:, 0:2]
    if args.regressor_cell == "pose" and args.regressor_learn == "center":
        offsets = np.array([descr.offset_center for descr in descriptions])[:, 0:2]
    if args.regressor_cell == "best" and args.regressor_learn == "closest":
        offsets = []
        for i_descr, descr in enumerate(descriptions):
            if descr.is_matched:
                offsets.append(descr.best_offset_closest[0:2])
            else:
                offsets.append(descr.offset_closest[0:2])
    if args.regressor_cell == "best" and args.regressor_learn == "center":
        offsets = []
        for i_descr, descr in enumerate(descriptions):
            if descr.is_matched:
                offsets.append(descr.best_offset_center[0:2])
            else:
                offsets.append(descr.offset_center[0:2])
    
    if args.regressor_cell == "all":
        offsets = [(pose.pose_w[0] - cell.bbox_w[0]) / (cell.bbox_w[3] - cell.bbox_w[0]) , (pose.pose_w[1] - cell.bbox_w[1]) / (cell.bbox_w[4]  - cell.bbox_w[1])]

    
    objects = []
    for obj_index, obj in enumerate(cell.objects):
        objects.append(obj)

    if len(objects) > pad_size:
        objects = objects[0:pad_size]

    while len(objects) < pad_size:
        obj = Object3d.create_padding()
        objects.append(obj)


    text = " ".join(hints)

    object_points = batch_object_points(objects, transform)

    object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in objects]
    object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in objects]

    return {
        "poses": pose,
        "cells": cell,
        "objects": objects,
        "object_points": object_points,
        "hint_descriptions": hints,
        "texts": text,
        # "matches": matches,
        # "all_matches": all_matches,
        "offsets": np.array(offsets),
        # "offsets_best_center": np.array(offsets_best_center),
        # 'offsets_valid': offsets_valid,
        "object_class_indices": object_class_indices,
        "object_color_indices": object_color_indices,
    }

class Kitti360FineDataset(Kitti360BaseDataset):     # 只在训练时用过
    def __init__(self, base_path, scene_name, transform, args, flip_pose=False,
                 pmc_prob = 0.0,
                 pmc_threshold = 0.4,
                 count_threshold = 1,
                 ):
        """Dataset to train the fine module.

        Args:
            base_path: Data root path
            scene_name: Scene name
            transform: PyG transform on object points
            args: Global training arguments
            flip_pose (bool, optional): Flip poses to opposite site of the cell (including the hint direction). Defaults to False.
            pmc_prob (float, optional): Probability of prototype-based map cloning. Defaults to 0.0 (no prototype-based map cloning).
            pmc_threshold (float, optional): Distance limitation between the ground turth target and submap center. Defaults to 0.4 (distance limitation = 12 m).
            count_threshold (integer, optional): The permissible number of mismatched instance. Defaults to 1 (only 1 instance missing).
        """
        

        super().__init__(base_path, scene_name)

        self.pad_size = args.pad_size
        self.transform = transform
        self.flip_pose = flip_pose
        self.pmc_prob = pmc_prob
        self.pmc_threshold = pmc_threshold
        self.count_threshold = count_threshold
        self.rematch = pmc_prob > 0.0
        if pmc_prob > 0.0:
            with open(osp.join(base_path, "direction", f"{scene_name}.json")) as json_file:
                self.direction_map = json.load(json_file)   # 和每个cell有关的 周围cell的记录

        self.args = args

    def __getitem__(self, idx):
        pose = self.poses[idx]
        if self.pmc_prob > 0.0:
            """
            Prototype-based map cloning (PMC)
            """
            mapping = self.direction_map[pose.cell_id]  # 存放的都是center距离best_cell 15距离以内的 对应文中的alpha=15
            if np.random.choice((True, False), p=[ self.pmc_prob, 1 - self.pmc_prob]):  # 给定概率pmc_prob 从布尔值True和False选择一个 p表示概率分布
                new_id_list = [value for value in mapping.values() if value is not None]
                new_valid_id_list = []
                new_valid_length_list = []
                for new_poss_id in new_id_list:     # 在周围的cell中找到符合pose描述条件的cell 不匹配的obj最多一个
                    cell = self.cells_dict[new_poss_id]
                    length = np.max(np.abs(pose.pose_w[:2] - cell.get_center()[:2]) / (cell.bbox_w[3] - cell.bbox_w[0]))
                    if length < self.pmc_threshold: # cell中心距离pose小于12    对应文中的beta=12
                        count = 0
                        new_pose = (pose.pose_w - cell.bbox_w[:3]) / (cell.bbox_w[3] - cell.bbox_w[0])  # pose在新的cell(附近的)的相对位置
                        obj_list = []
                        for descr in pose.descriptions:     # 针对每个描述的object
                            object_label = descr.object_label
                            offset_closest = descr.offset_closest
                            
                            for obj_index, obj in enumerate(cell.objects):  # 循环过程中找到一个符合条件的对象就会执行break，else语句就不会执行。如果for循环没有遇到break语句而是正常结束，则会执行else块。
                                obj_label = obj.label
                                obj_offset = (new_pose - obj.get_closest_point(new_pose))[:2]   # 新的pose距离obj instance的最近点的距离
                                # obj_color = obj.get_color_text()
                                if object_label == obj_label and np.linalg.norm(offset_closest - obj_offset) < 1e-7 and obj_index not in obj_list:
                                    # new_matching.append(obj_index)
                                    obj_list.append(obj_index)
                                    break
                            else:
                                count += 1
                        if count <= self.count_threshold:
                            new_valid_id_list.append(new_poss_id)
                            new_valid_length_list.append(np.linalg.norm((pose.pose_w[:2] - cell.get_center()[:2]) / (cell.bbox_w[3] - cell.bbox_w[0])))

                if new_valid_id_list == []:
                    cell = self.cells_dict[pose.cell_id]
                    cell_id = pose.cell_id
                else:
                    new_valid_length = 1 / np.array(new_valid_length_list) ** 2 # ** (5/4)
                    new_valid_length /= np.sum(new_valid_length)    # 把符合条件的cell中心距离pose的距离分配到0-1概率中
                    new_id = np.random.choice(new_valid_id_list, p = new_valid_length)  # 选择新的id
                    cell = self.cells_dict[new_id]
                    cell_id = new_id

            else:   # 正常数据集中的cell和cell_id
                cell = self.cells_dict[pose.cell_id]
                cell_id = pose.cell_id
        else:   # 正常数据集中的cell和cell_id
            cell = self.cells_dict[pose.cell_id]
            cell_id = pose.cell_id
        
        if self.rematch:
            if cell_id == pose.cell_id:
                new_matching = None
            else:
                new_matching = []
                new_pose = (pose.pose_w - cell.bbox_w[:3]) / (cell.bbox_w[3] - cell.bbox_w[0]) # pose在新的cell(附近的)的相对位置
                for descr in pose.descriptions:
                    object_label = descr.object_label
                    offset_closest = descr.offset_closest
                    # object_color = descr.object_color_text

                    for obj_index, obj in enumerate(cell.objects):
                        obj_label = obj.label
                        obj_offset = (new_pose - obj.get_closest_point(new_pose))[:2]
                        # obj_color = obj.get_color_text()
                        if object_label == obj_label and np.linalg.norm(offset_closest - obj_offset) < 1e-7 and \
                           obj_index not in new_matching:   
                           
                            new_matching.append(obj_index)
                            break
                    else:
                        new_matching.append(None)

        hints = self.hint_descriptions[idx]

        horizontal_flip = False
        vertical_flip = False

        if self.flip_pose:
            if np.random.choice((True, False)): # Horizontal
                horizontal_flip = True
            if np.random.choice((True, False)): # Vertical
                vertical_flip = True
     
        if self.pmc_prob > 0.0:
            if not self.rematch:
                out = load_pose_and_cell_aug(
                    pose, cell, hints, self.pad_size, self.transform, self.args, flip_pose=self.flip_pose,
                    horizontal_flip=horizontal_flip, vertical_flip = vertical_flip,  
                )
            elif new_matching is None:
                out = load_pose_and_cell(   # 原先的加载pose和cell的方式
                    pose, cell, hints, self.pad_size, self.transform, self.args, flip_pose=self.flip_pose,
                    horizontal_flip=horizontal_flip, vertical_flip = vertical_flip,
                )

            else:
                out = load_pose_and_cell_aug2(
                    pose, cell, hints, self.pad_size, self.transform, self.args, flip_pose=self.flip_pose,
                    horizontal_flip=horizontal_flip, vertical_flip = vertical_flip, new_matching = new_matching
                )

        else:
            out = load_pose_and_cell(   # 原先的加载pose和cell的方式
                pose, cell, hints, self.pad_size, self.transform, self.args, flip_pose=self.flip_pose,
                horizontal_flip=horizontal_flip, vertical_flip = vertical_flip,
            )

        return out 

    def __len__(self):
        return len(self.poses)

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


class Kitti360FineDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, args, flip_pose=False,
                 pmc_prob = 0.0,
                 pmc_threshold = 0.4,
                 count_threshold = 1,
                 ):
        """Multi-scene version of Kitti360FineDataset.

        Args:
            base_path: Data root path
            scene_name: Scene name
            transform: PyG transform on object points
            args: Global training arguments
            flip_pose (bool, optional): Flip poses to opposite site of the cell (including the hint direction). Defaults to False.
        """
        self.scene_names = scene_names
        self.flip_pose = flip_pose
        self.datasets = [
            Kitti360FineDataset(base_path, scene_name, transform, args, flip_pose, pmc_prob, pmc_threshold, count_threshold)
            for scene_name in scene_names
        ]

        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses]  # For stats
        self.all_cells = [
            cell for dataset in self.datasets for cell in dataset.cells
        ]  # For eval stats

        print(str(self))

    def __getitem__(self, idx):
        count = 0
        for dataset in self.datasets:
            idx_in_dataset = idx - count
            if idx_in_dataset < len(dataset):
                return dataset[idx_in_dataset]
            else:
                count += len(dataset)
        assert False

    def __repr__(self):
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(
            np.unique(poses, axis=0)
        )  # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps
        return f"Kitti360FineDatasetMulti: {len(self)} descriptions for {num_poses} unique poses from {len(self.datasets)} scenes, {len(self.all_cells)} cells, flip: {self.flip_pose}."

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    # def get_known_words(self):
    #     known_words = []
    #     for ds in self.datasets:
    #         known_words.extend(ds.get_known_words())
    #     return list(np.unique(known_words))

    def get_known_classes(self):
        known_classes = []
        for ds in self.datasets:
            known_classes.extend(ds.get_known_classes())
        return list(np.unique(known_classes))


if __name__ == "__main__":
    base_path = "./data/k360_30-10_scG_pd10_pc4_spY_all"
    folder_name = "2013_05_28_drive_0003_sync"

    args = EasyDict(pad_size=8, num_mentioned=6, ranking_loss="pairwise", regressor_cell="pose", regressor_learn="center", regressor_eval="center")
    transform = T.Compose([T.FixedPoints(1024), T.NormalizeScale()])

    dataset = Kitti360FineDatasetMulti(
        base_path,
        [
            folder_name,
        ],
        transform,
        args,
    )
    data = dataset[0]
