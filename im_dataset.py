import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import scipy.io
from collections import Counter  

def get_imbalance_dataset(dataset,num_classes=10, imbalance_ratio=0.5, transform=None, split="train", imb_type='exp'):
    train = split == "train"
    if train:
        img_num_list = get_img_num_per_cls(dataset.targets, num_classes, imb_type, imbalance_ratio)
        data_list = dataset.data 
        new_data_list, new_targets = gen_imbalanced_data(data_list, dataset.targets, img_num_list)
        dataset.data = [new_data.astype(dtype) for new_data, dtype in zip(new_data_list, [d.dtype for d in dataset.data])]
        dataset.targets = np.array(new_targets, dtype=dataset.targets.dtype)
        data_length = len(dataset.data)
        if data_length == 2:
            dataset.V1, dataset.V2 = dataset.data
        elif data_length == 3:
            dataset.V1, dataset.V2, dataset.V3 = dataset.data
        elif data_length == 4:
            dataset.V1, dataset.V2, dataset.V3, dataset.V4 = dataset.data
        elif data_length == 5:
            dataset.V1, dataset.V2, dataset.V3, dataset.V4, dataset.V5 = dataset.data
        else:
            raise ValueError("Unsupported number of views in dataset.data")
        dataset.Y = dataset.targets

    from collections import Counter
    print(f"Imbalance ratio: {imbalance_ratio}, Class distribution: {Counter(dataset.targets)}")
    total_samples = len(dataset.targets)
    
    return dataset, total_samples


def _get_class_dict():
    class_dict = dict()
    for i, anno in enumerate(get_annotations()):
        cat_id = anno["category_id"]
        if not cat_id in class_dict:
            class_dict[cat_id] = []
        class_dict[cat_id].append(i)
    return class_dict


def get_img_num_per_cls(targets, cls_num, imb_type, imb_factor):
    img_max = len(targets) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(round(num)))  
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def gen_imbalanced_data(data_list, targets, img_num_per_cls):
    np.random.seed(0)
    num_views = len(data_list)
    new_data_list = [[] for _ in range(num_views)]
    new_targets = []
    targets_np = np.array(targets)  
    classes = np.unique(targets_np)

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]

        for view_idx in range(num_views):
            data = data_list[view_idx]
            selected_data = data[selec_idx]
            new_data_list[view_idx].append(selected_data)
        new_targets.extend(targets_np[selec_idx])

    for view_idx in range(num_views):
        new_data_list[view_idx] = np.concatenate(new_data_list[view_idx], axis=0).astype(data_list[view_idx].dtype)

    new_targets = np.array(new_targets, dtype=targets_np.dtype)

    return new_data_list, new_targets

def get_annotations(labels):
    annos = []
    for label in labels:
        annos.append({'category_id': int(label)})
    return annos

def get_cls_num_list(cls_num,num_per_cls_dict):
    cls_num_list = []
    for i in range(cls_num):
        cls_num_list.append(num_per_cls_dict[i])
    return cls_num_list

