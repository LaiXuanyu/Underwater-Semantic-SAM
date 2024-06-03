import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2



def get_bounding_boxes(ground_truth_maps: np.array) -> list:
    """
    Get the bounding boxes for multiple ground truth masks

    Arguments:
        ground_truth_maps: Take ground truth masks in array format [num_classes, H, W]

    Return:
        bboxes: List of bounding boxes for each mask [[x_min, y_min, x_max, y_max], ...]
    """
    bboxes = []
    num_classes, H, W = ground_truth_maps.shape

    for class_idx in range(num_classes):
        ground_truth_map = ground_truth_maps[class_idx]
        idx = np.where(ground_truth_map > 0)
        
        if len(idx[0]) == 0 or len(idx[1]) == 0:
            # If there are no positive pixels for this class, skip it
            bboxes.append([0, 0, 0, 0])
            continue

        x_indices = idx[1]
        y_indices = idx[0]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        
        bbox = [x_min, y_min, x_max, y_max]
        bboxes.append(bbox)

    return bboxes

def stacking_batch(batch, outputs):
    """
    Given the batch and outputs of SAM, stacks the tensors to compute the loss. We stack by adding another dimension.

    Arguments:
        batch(list(dict)): List of dict with the keys given in the dataset file
        outputs: list(dict): List of dict that are the outputs of SAM
    
    Return: 
        stk_gt: Stacked tensor of the ground truth masks in the batch. Shape: [batch_size, H, W] -> We will need to add the channels dimension (dim=1)
        stk_out: Stacked tensor of logits mask outputed by SAM. Shape: [batch_size, 1, 1, H, W] -> We will need to remove the extra dimension (dim=1) needed by SAM 
    """
    stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
    stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
        
    return stk_gt, stk_out


def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)


def init_point_sampling(mask, get_point=1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]
    fg_size = len(fg_coords)
    bg_size = len(bg_coords)
    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices],
                                                                                              dtype=torch.int)
        return coords, labels
    
def move_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch