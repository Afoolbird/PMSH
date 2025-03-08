import torch
import math
import numpy as np

# 有问题的iou计算方法
def circle_iou_fixed_radius(batch1, batch2, radius=15):
    """
    计算两个batch的圆的IoU (向量化，固定半径)
    :param batch1: 形状为 (bs, 2) 的数组，每行表示圆心坐标 (x1, y1)
    :param batch2: 形状为 (bs, 2) 的数组，每行表示圆心坐标 (x2, y2)
    :param radius: 圆的固定半径，默认为 15
    :return: 形状为 (bs,) 的数组，每个值是对应圆的IoU
    """
    x1, y1 = batch1[:, 0], batch1[:, 1]
    x2, y2 = batch2[:, 0], batch2[:, 1]

    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    area = np.pi * radius ** 2
    intersection_area = np.zeros_like(d)
    contains_mask = d <= 0
    intersection_area[contains_mask] = area
    overlap_mask = (d < 2 * radius) & (d > 0)
    d_overlap = d[overlap_mask]
    part1 = radius ** 2 * np.arccos((d_overlap ** 2) / (2 * d_overlap * radius))
    part2 = radius ** 2 * np.arccos((d_overlap ** 2) / (2 * d_overlap * radius))
    part3 = 0.5 * np.sqrt((2 * radius - d_overlap) * (2 * radius + d_overlap) * d_overlap)
    intersection_area[overlap_mask] = part1 + part2 - part3
    union_area = 2 * area - intersection_area
    iou = intersection_area / union_area
    return iou

def circle_iou(circle1, circle2, radius):
    """
    计算两个圆的IoU
    :param circle1: (x1, y1, r1) 第一个圆的圆心坐标(x1, y1)和半径r1
    :param circle2: (x2, y2, r2) 第二个圆的圆心坐标(x2, y2)和半径r2
    :return: IoU值
    """
    r1 = r2 = radius
    x1, y1 = circle1
    x2, y2 = circle2
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        intersection_area = math.pi * min(r1, r2) ** 2
    else:
        part1 = r1 ** 2 * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
        part2 = r2 ** 2 * math.acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
        part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        intersection_area = part1 + part2 - part3
    circle1_area = math.pi * r1 ** 2
    circle2_area = math.pi * r2 ** 2
    union_area = circle1_area + circle2_area - intersection_area
    return intersection_area / union_area

def compute_iou( pose_center, pos_center, semi_center):
    batch_size = pose_center.shape[0]
    IOU_pos = []
    IOU_semi = []
    for index in range(batch_size):
        IOU_pos.append(circle_iou(pose_center[index], pos_center[index], 15))
        IOU_semi.append(circle_iou(pose_center[index], semi_center[index], 15))
    IOU_pos = np.array(IOU_pos)
    IOU_semi = np.array(IOU_semi)

    return IOU_semi / IOU_pos
# 我找的半正的数据中只有一个Iou之比的值是大于1的

def compute_distance(pose_center, pos_center, semi_center):
    dis_pos = np.linalg.norm(pose_center - pos_center, axis=1)
    dis_semi = np.linalg.norm(pose_center - semi_center, axis=1)
    return dis_pos / dis_semi
# 我找的半正的数据中只有一个Distance之比的值是大于1的 和上面是对应的

class IouLoss(torch.nn.Module):
    def __init__(self,):
        """
        IouLoss for retrieval training.
        Because the intersection and union ratio of a square and a circle is not easy to calculate, 
        the intersection and union ratio of two circles is chosen as the next best choice.
        Args:

        强制让相似度之比 靠近IOU之比
        """
        super(IouLoss, self).__init__()
        self.radius = 15

    def forward(self, anchor, positive, semi_positive, poses, pos_cells, semi_cells):  # Norming the input (as in paper) is actually not helpful
        sim_pos = torch.sum(positive * anchor, dim=1)
        sim_semi = torch.sum(semi_positive * anchor, dim=1)

        poses_center = np.array([pose.pose_w for pose in poses])    # bs, 3
        pos_cells_center = np.array([cell.get_center() for cell in pos_cells]) 
        semi_cells_center = np.array([cell.get_center() for cell in semi_cells]) 
        ratio = self.compute_iou(poses_center[:,:2], pos_cells_center[:,:2], semi_cells_center[:,:2])
        
        error = ( sim_semi / sim_pos ) - torch.tensor(ratio, dtype=torch.float, device=anchor.device)
        loss = torch.mean(error * error)
        return loss
    
    def compute_iou(self, pose_center, pos_center, semi_center):
        batch_size = pose_center.shape[0]
        IOU_pos = []
        IOU_semi = []
        for index in range(batch_size):
            IOU_pos.append(circle_iou(pose_center[index], pos_center[index], self.radius))
            IOU_semi.append(circle_iou(pose_center[index], semi_center[index], self.radius))
        IOU_pos = np.array(IOU_pos)
        IOU_semi = np.array(IOU_semi)

        return IOU_semi / IOU_pos

class DistanceLoss(torch.nn.Module):
    def __init__(self,):
        """
         DistanceLoss for retrieval training.   
        Args:
        """
        super(DistanceLoss, self).__init__()

    def forward(self, anchor, positive, semi_positive, poses, pos_cells, semi_cells):  # Norming the input (as in paper) is actually not helpful
        sim_pos = torch.sum(positive * anchor, dim=1)
        sim_semi = torch.sum(semi_positive * anchor, dim=1)

        poses_center = np.array([pose.pose_w for pose in poses])    # bs, 3
        pos_cells_center = np.array([cell.get_center() for cell in pos_cells]) 
        semi_cells_center = np.array([cell.get_center() for cell in semi_cells]) 
        ratio = self.compute_distance(poses_center[:,:2], pos_cells_center[:,:2], semi_cells_center[:,:2])
        
        error = ( sim_semi / sim_pos ) - torch.tensor(ratio, dtype=torch.float, device=anchor.device)
        loss = torch.mean(error * error)
        return loss
    
    def compute_distance(self, pose_center, pos_center, semi_center):
        dis_pos = np.linalg.norm(pose_center - pos_center, axis=1)
        dis_semi = np.linalg.norm(pose_center - semi_center, axis=1)
        return dis_pos / dis_semi

class IouUncertaintyLoss(torch.nn.Module):
    def __init__(self,):
        """
        IouLoss for retrieval training.
        Because the intersection and union ratio of a square and a circle is not easy to calculate, 
        the intersection and union ratio of two circles is chosen as the next best choice.
        Args:
        """
        super(IouUncertaintyLoss, self).__init__()
        self.radius = 15

        # fine_semi/1的
        # self.alpha = 100
        # self.beta = 60
        # self.gamma = 0.04

        # fine_semi/30的
        # self.alpha = 0.5
        # self.beta = 300
        # self.gamma = 0.03

        # fine_semi/30的 final
        # self.alpha = 0.25
        # self.beta = 250
        # self.gamma = 0.028        # 50%

        self.alpha = 0.25
        self.beta = 250
        # self.gamma = 0.022      # 20%
        self.gamma = 0.037      # 80%


    def forward(self, anchor, positive, semi_positive, poses, pos_cells, semi_cells, semi_uncertainty):  # Norming the input (as in paper) is actually not helpful
        sim_pos = torch.sum(positive * anchor, dim=1)
        sim_semi = torch.sum(semi_positive * anchor, dim=1)

        poses_center = np.array([pose.pose_w for pose in poses])    # bs, 3
        pos_cells_center = np.array([cell.get_center() for cell in pos_cells]) 
        semi_cells_center = np.array([cell.get_center() for cell in semi_cells]) 
        ratio = torch.tensor(self.compute_iou(poses_center[:,:2], pos_cells_center[:,:2], semi_cells_center[:,:2]), dtype=torch.float, device=anchor.device) 
        
        weight = self.get_weight(sim_pos, sim_semi, ratio, semi_uncertainty)
        error = weight * ( sim_semi / sim_pos ) - ratio

        loss = torch.mean(error * error)
        return loss
    
    def compute_iou(self, pose_center, pos_center, semi_center):
        batch_size = pose_center.shape[0]
        IOU_pos = []
        IOU_semi = []
        for index in range(batch_size):
            IOU_pos.append(circle_iou(pose_center[index], pos_center[index], self.radius))
            IOU_semi.append(circle_iou(pose_center[index], semi_center[index], self.radius))
        IOU_pos = np.array(IOU_pos)
        IOU_semi = np.array(IOU_semi)

        return IOU_semi / IOU_pos

    def delta(self, epsilon):
        # 原本是np.abs(epsilon) 但epsilon是大于0的数据就删除了
        return 1 / (1 + self.alpha * np.exp(self.beta * (self.gamma - epsilon)))
    
    def get_weight(self, sim_pos, sim_semi, ratio, semi_uncertainty):
        F = torch.tensor(self.delta(np.array(semi_uncertainty)), dtype=torch.float, device=ratio.device)
        W = ratio*sim_pos/sim_semi - 1
        weight = 1 + F * W
        return weight

class DisUncertaintyLoss(torch.nn.Module):
    def __init__(self,):
        """
        IouLoss for retrieval training.
        Because the intersection and union ratio of a square and a circle is not easy to calculate, 
        the intersection and union ratio of two circles is chosen as the next best choice.
        Args:
        """
        super(DisUncertaintyLoss, self).__init__()

        # fine_semi/30的
        # self.alpha = 0.25
        # self.beta = 250
        # self.gamma = 0.028      # 50%

        self.alpha = 0.25
        self.beta = 250
        # self.gamma = 0.022      # 20%
        self.gamma = 0.037      # 80%

    def forward(self, anchor, positive, semi_positive, poses, pos_cells, semi_cells, semi_uncertainty):  # Norming the input (as in paper) is actually not helpful
        sim_pos = torch.sum(positive * anchor, dim=1)
        sim_semi = torch.sum(semi_positive * anchor, dim=1)

        poses_center = np.array([pose.pose_w for pose in poses])    # bs, 3
        pos_cells_center = np.array([cell.get_center() for cell in pos_cells]) 
        semi_cells_center = np.array([cell.get_center() for cell in semi_cells]) 
        ratio = torch.tensor(self.compute_distance(poses_center[:,:2], pos_cells_center[:,:2], semi_cells_center[:,:2]), dtype=torch.float, device=anchor.device) 
        
        weight = self.get_weight(sim_pos, sim_semi, ratio, semi_uncertainty)
        error = weight * ( sim_semi / sim_pos ) - ratio

        loss = torch.mean(error * error)
        return loss
    
    def compute_distance(self, pose_center, pos_center, semi_center):
        dis_pos = np.linalg.norm(pose_center - pos_center, axis=1)
        dis_semi = np.linalg.norm(pose_center - semi_center, axis=1)
        return dis_pos / dis_semi

    def delta(self, epsilon):
        # 原本是np.abs(epsilon) 但epsilon是大于0的数据 所以就删除了
        return 1 / (1 + self.alpha * np.exp(self.beta * (self.gamma - epsilon)))
    
    def get_weight(self, sim_pos, sim_semi, ratio, semi_uncertainty):
        F = torch.tensor(self.delta(np.array(semi_uncertainty)), dtype=torch.float, device=ratio.device)
        W = ratio*sim_pos/sim_semi - 1
        weight = 1 + F * W
        return weight

if __name__ == "__main__":
    # 示例
    circle1 = (0, 0)  # 圆心(0, 0)，半径15
    circle2 = (5, 0)  # 圆心(5, 0)，半径15

    iou = circle_iou(circle1, circle2, 15)
    print(f"IoU: {iou}")

    circle1 = (30, 30)  # 圆心(0, 0)，半径15
    circle2 = (50, 50)  # 圆心(5, 0)，半径15

    iou = circle_iou(circle1, circle2, 15)
    print(f"IoU: {iou}")

    batch1 = np.array([[0, 0], [10, 10], [30, 30]])
    batch2 = np.array([[5, 0], [10, 10], [50, 50]])

    iou_results = circle_iou_fixed_radius(batch1, batch2)
    print("IoU results:", iou_results)
