import torch
import numpy as np
from torch.utils.data import Dataset
import src.utils as utils


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        mask = item["target_mask"]
        mask = mask.convert('1')     
        ground_truth_mask =  np.array(mask)
        np.unique(ground_truth_mask)

        image = item["input_image"]
        original_size = tuple(image.size)[::-1]
        box = utils.get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, original_size, box)

        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)
        return inputs


    
def collate_fn(batch: torch.utils.data) -> list:
    return list(batch)