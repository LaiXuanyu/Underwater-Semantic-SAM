import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from .base_dataset import BaseDataset
import torchvision.transforms.functional as F

"""
RGB color code and object categories:
------------------------------------
000 BW: Background waterbody
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)
"""

class SuimDataset(Dataset):

    def __init__(self,
                 image_dir,
                 mask_dir,
                 img_suffix='.jpg',
                 mask_suffix='.bmp',
                 ignore_label=-1,
                 transform = None,
                 target_size=(480, 640),
                 num_classes=8):
        
        """
        Args:
            image_dir (string): 图像文件的路径。
            mask_dir (string): 掩码文件的路径。
            img_suffix (string): 图像文件的后缀名。
            mask_suffix (string): 掩码文件的后缀名。
            ignore_label (int): 忽略标签的值。
            num_classes (int): 类别数量。

        """
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.num_classes = num_classes
        self.transform = transform  
        self.target_size = target_size     
        self.ignore_label = ignore_label
        self.img_names = [f.split(self.img_suffix)[0] for f in os.listdir(image_dir) if f.endswith(self.img_suffix)]

        self.files = []

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_dir, img_name + self.img_suffix)
        mask_path = os.path.join(self.mask_dir, img_name + self.mask_suffix)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        # Resize image and mask to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        image = F.to_tensor(image)
        mask = np.array(mask)  # 转换为 numpy 数组以便后续处理
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        mask = torch.tensor(mask, dtype=torch.float32)  # 确保 mask 是一个浮点型张量
        categories = self.get_robot_fish_human_reef_wrecks_pytorch(mask)


        return image, mask.long(), categories

    def get_robot_fish_human_reef_wrecks_pytorch(self, mask):
        # 确保 mask 的形状是 [H, W, 3]
        if mask.ndim == 2:
            mask = mask[:, :, None]
        elif mask.shape[2] != 3:
            raise ValueError(f"Expected mask shape [H, W, 3], got {mask.shape}")

        categories = torch.zeros(mask.shape[0], mask.shape[1], 5, dtype=torch.float32, device=mask.device)

        human_color = torch.tensor([0, 0, 1], device=mask.device)
        robot_color = torch.tensor([1, 0, 0], device=mask.device)
        fish_color = torch.tensor([1, 1, 0], device=mask.device)
        reef_color = torch.tensor([1, 0, 1], device=mask.device)
        wreck_color = torch.tensor([0, 1, 1], device=mask.device)

        categories[:, :, 0] = torch.all(mask == robot_color, dim=-1).float()  # Robot
        categories[:, :, 1] = torch.all(mask == fish_color, dim=-1).float()   # Fish
        categories[:, :, 2] = torch.all(mask == human_color, dim=-1).float()  # Human
        categories[:, :, 3] = torch.all(mask == reef_color, dim=-1).float()   # Reef
        categories[:, :, 4] = torch.all(mask == wreck_color, dim=-1).float()  # Wreck

        return categories.permute(2, 0, 1)  # 调整通道顺序为 [类别, H, W]