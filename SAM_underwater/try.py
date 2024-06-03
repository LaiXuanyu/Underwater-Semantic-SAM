import os   
from os.path import join, exists
import numpy as np
import torch
import monai
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.nn import functional as F
from torch.optim import Adam
from utils.suim import SuimDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from segment_anything import build_sam_vit_b
from utils.processor import Samprocessor


COLOR_MAP = {
    0: [0, 0, 0],        # 背景 (黑色)
    1: [255, 0, 0],      # 类别1 (红色)
    2: [0, 255, 0],      # 类别2 (绿色)
    3: [0, 0, 255],      # 类别3 (蓝色)
    4: [255, 255, 0],    # 类别4 (黄色)
    5: [255, 0, 255]     # 类别5 (品红)
}

def apply_color_map(mask, color_map):
    """
    根据类别掩码和颜色映射应用颜色

    Args:
        mask (torch.Tensor): 掩码张量，形状为 [C, H, W]
        color_map (dict): 颜色映射字典

    Returns:
        np.array: 彩色掩码图像
    """
    H, W = mask.shape[1:]
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_index in range(mask.shape[0]):
        class_mask = mask[class_index].numpy()
        color = color_map[class_index + 1]  # 类别从1开始
        color_mask[class_mask > 0] = color
    
    return color_mask

def visualize_image_mask_and_bbox(image, mask, bboxes):
    """
    显示原始图像、彩色掩码以及边界框的叠加效果

    Args:
        image (torch.Tensor): 图像张量，形状为 [C, H, W]
        mask (torch.Tensor): 掩码张量，形状为 [C, H, W]
        bboxes (list): 边界框列表，每个元素是一个 [tensor([x_min, y_min]), tensor([x_max, y_max])]
    """
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)  # 将图像从 [C, H, W] 转换为 [H, W, C]
    color_mask = apply_color_map(mask, COLOR_MAP)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原始图像
    ax[0].imshow(image)
    ax[0].set_title('Original Image')

    # 显示叠加彩色掩码和边界框的图像
    ax[1].imshow(image)
    ax[1].imshow(color_mask, alpha=0.5)  # 叠加彩色掩码，使用 alpha 透明度进行叠加

    # 添加边界框
    for bbox in bboxes:
        if all(coord.numel() == 2 for coord in bbox):  # 确保每个边界框有四个值
            x_min, y_min = bbox[0]
            x_max, y_max = bbox[2]
            rect = patches.Rectangle((x_min.item(), y_min.item()), x_max.item() - x_min.item(), y_max.item() - y_min.item(), linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
    
    ax[1].set_title('Image with Colored Mask and BBox')

    plt.show()

def main():


    dataset_name = "suim"
    image_dir = "/FYP/SUIM/train_val/train_val/images"
    mask_dir = "/FYP/SUIM/train_val/train_val/masks"
    model_path = "./checkpoint/sam_vit_b_01ec64.pth"
    multimask_outputs = True

    ckpt_dir = "ckpt/"
    im_res_ = (320, 240, 3)
    ckpt_name = "SAM_suim.basic"
    model_ckpt_name = join(ckpt_dir, ckpt_name)

    # 创建检查点目录（如果不存在）
    if not exists(ckpt_dir): 
        os.makedirs(ckpt_dir)

    print("检查点文件路径:", model_ckpt_name)


    '''
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop((240, 320)),
        transforms.ToTensor()
    ])
    '''


    sam = build_sam_vit_b(checkpoint=model_path)
    processor = Samprocessor(sam)
    model = sam.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集实例
    dataset = SuimDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_suffix='.jpg',
        mask_suffix='.bmp',
        processor= None ,
        transform= None
    )

    #创建dataloader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # 损失函数
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # 优化器
    optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)

    for batch in train_loader:
        image = batch['image'][0]  # 获取批次中的第一个图像
        mask = batch['label'][0]   # 获取批次中的第一个掩码
        print(mask.shape)
        bboxes = batch['bbox'][0]  # 获取批次中的第一个边界框

        visualize_image_mask_and_bbox(image, mask, bboxes)
        break  # 只显示一个批次

if __name__ == "__main__":
    main()