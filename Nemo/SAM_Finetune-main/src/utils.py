import numpy as np
import torch



def get_bounding_box(ground_truth_map: np.array) -> list:
  """
  Get the bounding box of the image with the ground truth mask
  
    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

  """
  # get bounding box from mask
  idx = np.where(ground_truth_map > 0)
  x_indices = idx[1]
  y_indices = idx[0]
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

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