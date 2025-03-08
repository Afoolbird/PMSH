from typing import List

import os, json
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
)
from datapreparation.kitti360pose.utils import CLASS_TO_INDEX, COLOR_NAMES
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
from dataloading.kitti360pose.base import Kitti360BaseDataset
from dataloading.kitti360pose.utils import batch_object_points

import time, copy, random
from tqdm import tqdm
from collections import defaultdict

class Kitti360CoarseSemiDataset(Kitti360BaseDataset):
    def __init__(
        self,
        base_path,
        scene_name,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
        exclude=False,
    ):
        """Dataset variant for coarse module training.
        Returns one item per pose.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_name: Scene name
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
            exclude (bool, optional): 代表是否排除为半正样本为None的数据.
        """
        super().__init__(base_path, scene_name)
        self.shuffle_hints = shuffle_hints
        self.transform = transform
        self.flip_poses = flip_poses

        self.sample_close_cell = sample_close_cell
        self.cell_centers = np.array([cell.get_center()[0:2] for cell in self.cells])

        with open(osp.join(base_path, "semi_cell", f"{scene_name}.pkl"), 'rb') as f:
            self.semi_cell = list(pickle.load(f).items())
        self.exclude = exclude
        if self.exclude:
            none_indexes = [index for index, (key, value) in enumerate(self.semi_cell) if value is None]
            for index in sorted(none_indexes, reverse=True):    # 需要从后往前删除以保持index的正确性
                self.poses.pop(index)
                self.hint_descriptions.pop(index)
                self.semi_cell.pop(index)

    def __getitem__(self,):
        raise Exception("Not implemented: abstract class.")

    def __len__(self):
        return len(self.poses)

class Kitti360CoarseSemiDatasetMulti(Dataset):
    def __init__(
        self,
        base_path,
        scene_names,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,

        exclude = False,
        shuffle_batch_size = 32,
    ):
        """Multi-scene variant of Kitti360CoarseDataset.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_names: List of scene names
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        self.exclude = exclude                # 这个参数代表是否排除为半正样本为None的数据

        self.scene_names = scene_names
        self.transform = transform
        self.flip_poses = flip_poses
        self.sample_close_cell = sample_close_cell
        self.shuffle_hints = shuffle_hints
        self.datasets = [
            Kitti360CoarseSemiDataset(      # 每个场景中的数据
                base_path, scene_name, transform, shuffle_hints, flip_poses, sample_close_cell, self.exclude
            )
            for scene_name in scene_names
        ]

        self.all_cells = [
            cell for dataset in self.datasets for cell in dataset.cells
        ]  # For cell-only dataset
        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses]  # For eval
        self.all_semi_cells_id = [semi_cell[1] for dataset in self.datasets for semi_cell in dataset.semi_cell]

        cell_ids = [cell.id for cell in self.all_cells]
        assert len(np.unique(cell_ids)) == len(self.all_cells)  # IDs should not repeat
        assert len(self.all_semi_cells_id) == len(self.all_poses)

        ##### load uncertaintys #####
        # record_path = 'checkpoints/k360_30-10_scG_pd10_pc4_spY_all/fine_semi/1/uncertaintys.txt'
        record_path = 'checkpoints/k360_30-10_scG_pd10_pc4_spY_all/fine_semi/30/uncertaintys.txt'
        self.semi_uncertaintys = np.loadtxt(record_path, delimiter=',', dtype=float)[:, 1]
        if self.exclude:
            ############# 这个排除掉了为None的半正样本的不确定性 #############
            with open('data/k360_30-10_scG_pd10_pc4_spY_all/semi_record.txt', 'r') as file:
                loaded_dict = json.load(file)
            none_indexes = [index for index, (key, value) in enumerate(loaded_dict.items()) if value is None]
            self.semi_uncertaintys = np.delete(self.semi_uncertaintys, none_indexes)
        assert len(self.all_poses) == len(self.semi_uncertaintys)
        ##### load uncertaintys #####

        ############# select hard sample #############
        self.shuffle_batch_size = shuffle_batch_size

        self.sample_cells_dict = {}
        self.hint_descriptions = [hint_description for dataset in self.datasets for hint_description in dataset.hint_descriptions]
        for dataset in self.datasets:
            self.sample_cells_dict.update(dataset.cells_dict)
        assert len(np.unique(cell_ids)) == len(self.sample_cells_dict) and len(self.all_poses) == len(self.hint_descriptions)
        
        self.samples = [(index, self.all_poses[index].cell_id) for index in range(len(self.all_poses))]
        self.cell_id2pairs = defaultdict(list)
        for pair in self.samples:      
            self.cell_id2pairs[pair[1]].append(pair)
        ############# select hard sample #############

        print(str(self))

    def __getitem__(self, idx):     # 相当于分别从相应的5个区域中获取到相应的位置信息

        pose_index, cell_id = self.samples[idx]
        pose = self.all_poses[pose_index]
        assert cell_id == pose.cell_id

        cell = self.sample_cells_dict[pose.cell_id]
        hints = self.hint_descriptions[pose_index]

        if self.shuffle_hints:
            hints = np.random.choice(hints, size=len(hints), replace=False)

        text = " ".join(hints)

        semi_cell_id = self.all_semi_cells_id[pose_index]
        if semi_cell_id != None:
            semi_cell = self.sample_cells_dict[semi_cell_id]
        else:
            semi_cell = cell

        # NOTE: hints are currently not flipped! (Only the text.)
        if self.flip_poses:
            if np.random.choice((True, False)):  # Horizontal
                pose, cell, semi_cell, text = flip_pose_in_cell(pose, cell, semi_cell, text, 1)
            if np.random.choice((True, False)):  # Vertical
                pose, cell, semi_cell, text = flip_pose_in_cell(pose, cell, semi_cell, text, -1)

        object_points = batch_object_points(cell.objects[:28], self.transform)
        semi_object_points = batch_object_points(semi_cell.objects[:28], self.transform)

        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell.objects]
        object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in cell.objects]
        
        assert pose.pose_w[0] > semi_cell.bbox_w[0] and pose.pose_w[0] < semi_cell.bbox_w[3]
        assert pose.pose_w[1] > semi_cell.bbox_w[1] and pose.pose_w[1] < semi_cell.bbox_w[4]

        return {
            "poses": pose,
            "cells": cell,
            "objects": cell.objects[:28],
            "object_points": object_points, # Batch(batch=[7680], pos=[7680, 3], ptr=[31], x=[7680, 3])
            "texts": text,
            "cell_ids": pose.cell_id,
            "object_class_indices": object_class_indices,
            "object_color_indices": object_color_indices,
            "debug_hint_descriptions": hints,  # Care: Not shuffled etc!

            "semi_cells": semi_cell,
            "semi_objects": semi_cell.objects[:28],
            "semi_object_points": semi_object_points,
            "semi_uncertainty": self.semi_uncertaintys[pose_index]
        }

    def __len__(self):
        # return np.sum([len(ds) for ds in self.datasets])
        return len(self.samples)

    def __repr__(self): # 自我描述的方法
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(
            np.unique(poses, axis=0)
        )  # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps
        return f"Kitti360CellSemiDatasetMulti: {len(self.scene_names)} scenes, {len(self)} descriptions for {num_poses} unique poses, {len(self.all_cells)} cells, flip {self.flip_poses}, close-cell {self.sample_close_cell}"

    def get_known_classes(self):
        known_classes = []
        for ds in self.datasets:
            known_classes.extend(ds.get_known_classes())
        return list(np.unique(known_classes))

    def get_cell_dataset(self):
        return Kitti360CoarseCellOnlyDataset(self.all_cells, self.transform)

    def get_unique_cell_dataset(self):
        cell_ids = [pose.cell_id for pose in self.all_poses]
        unique_cell_ids = list(set(cell_ids))
        unique_cells = [self.sample_cells_dict[cell_id] for cell_id in unique_cell_ids]
        return Kitti360CoarseCellOnlyDataset(unique_cells, self.transform)

    def reset(self,):
        self.samples = [(index, self.all_poses[index].cell_id) for index in range(len(self.all_poses))]

    # 去除相同cell_id样本的shuffle
    def shuffle1(self,):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''
        
        print("\nShuffle Dataset:")
        
        pair_pool = [(index, self.all_poses[index].cell_id) for index in range(len(self.all_poses))]
            
        # Shuffle pairs order
        random.shuffle(pair_pool)
        
        # Lookup if already used in epoch
        pairs_epoch = set()   
        idx_batch = set()
    
        # buckets
        batches = []
        current_batch = []
            
        # counter
        break_counter = 0
        
        # progressbar
        pbar = tqdm()

        while True:
            
            pbar.update()
            
            if len(pair_pool) > 0:              # 只要pool中有数据就代表还没有完全分配完
                pair = pair_pool.pop(0)
                pose_index, cell_id = pair

                if cell_id not in idx_batch and pair not in pairs_epoch:
                    
                    idx_batch.add(cell_id)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
        
                    break_counter = 0
                    
                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                        
                    break_counter += 1
                    
                if break_counter >= 256:
                    break                   # 连续有256次不合适     剩下的数据不足以在组成一个batch的数据了所以舍弃？
                
            else:           
                break       # pool中没有数据 已分配完毕

            if len(current_batch) >= self.shuffle_batch_size:
            
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []
    
        pbar.close()
        
        # wait before closing progress bar
        time.sleep(0.3)
        
        self.samples = batches
        print("SHUFFLE--1")
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.all_poses), len(self.samples))) 
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.all_poses) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))

    # 寻找难样本的shuffle
    def shuffle2(self, sim_dict=None, neighbour_select=32, neighbour_range=64):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''
        
        print("\nShuffle Dataset:")
        pair_pool = [(index, self.all_poses[index].cell_id) for index in range(len(self.all_poses))]
        cell_id2pairs_pool = copy.deepcopy(self.cell_id2pairs)

        neighbour_split = neighbour_select // 2
                    
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)   # len(self.all_poses)
        
        # Shuffle pairs order
        random.shuffle(pair_pool)
        # Lookup if already used in epoch
        pairs_epoch = set()   
        idx_batch = set()
        # buckets
        batches = []
        current_batch = []
        # counter
        break_counter = 0
        # progressbar
        pbar = tqdm()

        while True:
            pbar.update()
            
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                pose_index, cell_id = pair

                if cell_id not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    idx_batch.add(cell_id)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    # remove from pool used for sim-sampling
                    cell_id2pairs_pool[cell_id].remove(pair)

                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        near_similarity = copy.deepcopy(similarity_pool[pose_index][:neighbour_range])
                        near_always = copy.deepcopy(near_similarity[:neighbour_split])
                        near_random = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(near_random)
                        near_random = near_random[:neighbour_split]
                        near_similarity_select = near_always + near_random

                        for cell_id in near_similarity_select:
                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            if cell_id not in idx_batch:
                                near_pairs = copy.deepcopy(cell_id2pairs_pool[cell_id])

                                 # up to 2 for one sat view 
                                random.shuffle(near_pairs)
                                for near_pair in near_pairs:
                                                                                
                                    idx_batch.add(cell_id)
                                    current_batch.append(near_pair)
                                    pairs_epoch.add(near_pair)
                                    
                                    cell_id2pairs_pool[cell_id].remove(near_pair)
                                    similarity_pool[pose_index].remove(cell_id)
                                    
                                    # only select one view
                                    break
                            
                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                        
                    break_counter += 1
                    
                if break_counter >= 256:
                    break
                
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
            
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("SHUFFLE--2")
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.all_poses), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.all_poses) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))


class Kitti360CoarseCellOnlyDataset(Dataset):
    def __init__(self, cells: List[Cell], transform):
        """Dataset to return only the cells for encoding during evaluation
        NOTE: The way the cells are read from the Cells-Only-Dataset, they may have been augmented differently during the actual training. Cells-Only does not flip and shuffle!
        """
        super().__init__()

        self.cells = cells
        self.transform = transform

    def __getitem__(self, idx):
        cell = self.cells[idx]
        assert len(cell.objects) >= 1
        object_points = batch_object_points(cell.objects, self.transform)

        return {
            "cells": cell,
            "cell_ids": cell.id,
            "objects": cell.objects,
            "object_points": object_points,
        }

    def __len__(self):
        return len(self.cells)


def flip_pose_in_cell(pose: Pose, cell: Cell, semi_cell: Cell, text, direction, hints=None, offsets=None):
    """Flips the cell horizontally or vertically
    CARE: Needs adjustment for non-compass directions
    CARE: Description.object_closest_point is flipped but direction in description is not flipped.

    Args:
        pose (Pose): The pose to flip, is copied before modification
        cell (Cell): The cell to flip, is copied before modification
        text (str): The text description to flip
        direction (int): Horizontally (+1) or vertically (-1)

    Returns:
        Pose: flipped pose
        Cell: flipped cell
        str: flipped text
    """
    assert direction in (-1, 1)
    assert sum([hints is None, offsets is None]) != 1  # Either both or none

    pose = deepcopy(pose)
    cell = deepcopy(cell)
    semi_cell = deepcopy(semi_cell)
    if offsets is not None:
        offsets = offsets.copy()

    if direction == 1:  # Horizontally
        pose.pose[0] = 1.0 - pose.pose[0]
        for obj in cell.objects:
            obj.xyz[:, 0] = 1 - obj.xyz[:, 0]
        for obj in semi_cell.objects:
            obj.xyz[:, 0] = 1 - obj.xyz[:, 0]
        for descr in pose.descriptions:
            descr.closest_point[0] = 1.0 - descr.closest_point[0]

        text = (
            text.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )                   

        if hints is not None:
            hints = [
                hint.replace("east", "east-flipped")
                .replace("west", "east")
                .replace("east-flipped", "west")
                for hint in hints
            ]
            offsets[:, 0] *= -1

    elif direction == -1:  # Vertically
        pose.pose[1] = 1.0 - pose.pose[1]
        for obj in cell.objects:
            obj.xyz[:, 1] = 1 - obj.xyz[:, 1]
        for obj in semi_cell.objects:
            obj.xyz[:, 0] = 1 - obj.xyz[:, 0]
        for descr in pose.descriptions:
            descr.closest_point[1] = 1.0 - descr.closest_point[1]

        text = (
            text.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )
                    
        if hints is not None:
            hints = [
                hint.replace("north", "north-flipped")
                .replace("south", "north")
                .replace("north-flipped", "south")
                for hint in hints
            ]
            offsets[:, 1] *= -1

    assert "flipped" not in text

    if hints is not None:
        return pose, cell, semi_cell, text, hints, offsets
    else:
        return pose, cell, semi_cell, text


if __name__ == "__main__":
    base_path = "./data/k360_30-10_scG_pd10_pc8_spY_all_nm6/"

    transform = T.FixedPoints(256)

    for scene_names in (SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST):
        dataset = Kitti360CoarseSemiDatasetMulti(
            base_path, scene_names, transform, shuffle_hints=False, flip_poses=False
        )
        # data = dataset[0]
        # pose, cell, text = data['poses'], data['cells'], data['texts']
        # offsets = np.array([descr.offset_closest for descr in pose.descriptions])
        # hints = text.split('.')
        # pose_f, cell_f, text_f, hints_f, offsets_f = flip_pose_in_cell(pose, cell, text, 1, hints=hints, offsets=offsets)

        # Gather information about duplicate descriptions
        descriptors = []
        for pose in dataset.all_poses:
            mentioned = sorted(
                [f"{d.object_label}_{d.object_color_text}_{d.direction}" for d in pose.descriptions]
            )
            descriptors.append(mentioned)

        unique, counts = np.unique(descriptors, return_counts=True)
        # for d in descriptors[0:10]:
        #     print('\t',d)
        print(
            f"{len(descriptors)} poses, {len(unique)} uniques, {np.max(counts)} max duplicates, {np.mean(counts):0.2f} mean duplicates"
        )
        print("---- \n\n")