import os

import pickle
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_region_indices():
    """
    解析并提取嘴部、眼部和其他区域的索引
    """
    pkl_path = "vocaset/regions/FLAME_masks.pkl"
    pkl_path = os.path.join(project_root, pkl_path)

    num_vertices=5023
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到区域划分文件: {pkl_path.resolve()}")
        
    with open(pkl_path, 'rb') as f:
        region_dict = pickle.load(f, encoding='latin1')

    # 嘴部索引
    lip_region = torch.from_numpy(region_dict['lips']).long()
    lip_region = torch.unique(lip_region)

    # 眼部索引
    eye_region = torch.from_numpy(region_dict['eye_region']).long()
    eye_region = torch.unique(eye_region)

    # 其他区域索引 (利用布尔掩码反算)
    all_indices = torch.arange(num_vertices)
    used_id = torch.cat([lip_region, eye_region])
    full_mask = torch.ones(num_vertices, dtype=torch.bool)
    full_mask[used_id] = False 
    other_region = all_indices[full_mask].long()

    return (lip_region, eye_region, other_region)
