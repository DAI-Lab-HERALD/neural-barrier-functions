import json

import torch


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def overlap_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    # Separating axis theorem
    return (partition_upper[..., 0] >= rect_lower[0]) & (partition_lower[..., 0] <= rect_upper[0]) & \
           (partition_upper[..., 1] >= rect_lower[1]) & (partition_lower[..., 1] <= rect_upper[1])


def overlap_circle(partition_lower, partition_upper, center, radius):
    closest_point = torch.max(partition_lower, torch.min(partition_upper, center))
    distance = (closest_point - center).norm(dim=-1)
    return distance <= radius


def overlap_outside_rectangle(partition_lower, partition_upper, rect_lower, rect_upper):
    return (partition_upper[..., 0] >= rect_upper[0]) | (partition_lower[..., 0] <= rect_lower[0]) | \
           (partition_upper[..., 1] >= rect_upper[1]) | (partition_lower[..., 1] <= rect_lower[1])


def overlap_outside_circle(partition_lower, partition_upper, center, radius):
    farthest_point = torch.where((partition_lower - center).abs() > (partition_upper - center).abs(), partition_lower, partition_upper)
    distance = (farthest_point - center).norm(dim=-1)
    return distance >= radius
