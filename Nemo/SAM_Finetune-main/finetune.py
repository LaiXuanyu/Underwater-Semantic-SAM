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
from suim import SuimDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from segment_anything import build_sam_vit_b
from src.lora import LoRA_sam
from src.processor import Samprocessor


def train_one_epoch(model, dataloader, criterion, optimizer, device, multimask):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images, masks, categories = batch
        print(f"images: {images}")
        print(f"images shape: {images.shape}")
        print(batch)
        images = images.to(device)
        masks = categories.to(device)

        optimizer.zero_grad()
        outputs = model(images,multimask)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
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
    model_path = "/FYP/checkpoint/sam_vit_b_01ec64.pth"
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


    sam = build_sam_vit_b(checkpoint = model_path)
    sam_lora = LoRA_sam(sam, args.rank)  
    model = sam_lora.sam
    processor = Samprocessor(model)
    
    # Process the dataset
    dataset = load_dataset(args.dataset_name , split="train")
    train_ds = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size , shuffle=True, collate_fn=collate_fn)


    # Initialize optimize and Loss
    optimizer = Adam(model.image_encoder.parameters(), lr=args.lr, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


    # Set model to train and into the device
    model.train()
    model.to(device)

    # 创建数据集实例
    dataset = SuimDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_suffix='.jpg',
        mask_suffix='.bmp',
        transform= None
    )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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