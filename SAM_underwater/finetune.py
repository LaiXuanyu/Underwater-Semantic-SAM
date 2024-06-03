import torch.nn as nn
import torchvision.models as models
import os
import argparse     
from os.path import join, exists
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import monai
from torch.nn import functional as F
from torch.optim import Adam
from utils.suim import SuimDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from segment_anything import build_sam_vit_b
from utils.processor import Samprocessor
from utils.tools import move_to_device

def train_one_epoch(model, dataloader, criterion, optimizer, device, multimask):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        batched_input = move_to_device(batch, device)
        # print("batched_input after move to device:", batched_input)
        images = batch['image']  # 将图像移至适当设备
        # masks = batch['label'] # 将掩码移至适当设备
        # bboxes = batch['bbox'] # 将边界框移至适当设备

        optimizer.zero_grad()
        # batched_input = [{'image': img, 'masks': msk, 'bboxes': bbox} for img, msk, bbox in zip(images, masks, bboxes)]
        outputs = model(batched_input=batched_input, multimask_output=False)
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Mean loss training: {epoch_loss}')    
    return epoch_loss

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks, categories in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a dataset for Hugging Face.")
    parser.add_argument("--push_to_hub", action="store_true", help="Flag to push the dataset to Hugging Face Hub.")
    parser.add_argument("--model_path", type=str, default = "./checkpoint/sam_vit_b_01ec64.pth")
    parser.add_argument("--rank", type=int, default = 512)
    parser.add_argument("--dataset_name", type=str, help= "huggingface dataset name", default = "BoooomNing/SAM_fashion")
    parser.add_argument("--batch_size", type=int, default = 1)
    parser.add_argument("--lr", type = float, default = 0.0001)   
    parser.add_argument("--num_epochs", type=int, default = 1)
    return parser.parse_args()



def main():

    dataset_name = "suim"
    image_dir = "/FYP/SUIM/train_val/train_val/images"
    mask_dir = "/FYP/SUIM/train_val/train_val/masks"
    model_path = "./checkpoint/sam_vit_b_01ec64.pth"
    multimask_outputs = True

    ckpt_dir = "ckpt/"
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
    #processor = Samprocessor(sam)
    model = sam.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建数据集实例
    dataset = SuimDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_suffix='.jpg',
        mask_suffix='.bmp',
        processor= None,
        device= device,
        transform= None
    )

    #创建dataloader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # 损失函数
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # 优化器
    optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, seg_loss, optimizer, device,multimask_outputs)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        
        # 保存检查点
        ckpt_dir = "ckpt/"
        ckpt_name = f"fcn8_vgg_epoch_{epoch+1}.pth"
        model_ckpt_name = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save(model.state_dict(), model_ckpt_name)


if __name__ == "__main__":
    main()