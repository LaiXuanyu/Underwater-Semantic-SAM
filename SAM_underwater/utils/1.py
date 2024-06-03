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

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        #对读取的图像进行处理
        # 转换PIL图像为PyTorch张量
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # 使用PyTorch实现的分类函数处理掩码
        categories = self.get_robot_fish_human_reef_wrecks_pytorch(mask)

        #数据增强
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask , categories
    

    def get_robot_fish_human_reef_wrecks_pytorch(self,mask):
        # 通过To tensor函数 mask 是形状为 [3, H, W] 的 PyTorch 张量，并且数值是0或1
        categories = torch.zeros(mask.shape[1], mask.shape[2], 5, dtype=mask.dtype, device=mask.device)

        # 定义颜色映射
        human_color = torch.tensor([0, 0, 1], device=mask.device)
        robot_color = torch.tensor([1, 0, 0], device=mask.device)
        fish_color = torch.tensor([1, 1, 0], device=mask.device)
        reef_color = torch.tensor([1, 0, 1], device=mask.device)
        wreck_color = torch.tensor([0, 1, 1], device=mask.device)

        # 应用逻辑索引来填充各类别
        categories[:, :, 0] = torch.all(mask == robot_color, dim=-1).float()  # Robot
        categories[:, :, 1] = torch.all(mask == fish_color, dim=-1).float()   # Fish
        categories[:, :, 2] = torch.all(mask == human_color, dim=-1).float()  # Human
        categories[:, :, 3] = torch.all(mask == reef_color, dim=-1).float()   # Reef
        categories[:, :, 4] = torch.all(mask == wreck_color, dim=-1).float()  # Wreck

        return categories.permute(2, 0, 1)  # 调整通道顺序为 [类别, H, W]