import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import Adam
import src.utils as utils
from src.dataloader import collate_fn, SAMDataset
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam
from datasets import load_dataset
import os
import argparse
from huggingface_hub import Repository, HfApi
import urllib.request




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


def ensure_model_file(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading model to {path}...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, path)
        print("Download complete.")

def main():
    args = parse_args()
    
    ensure_model_file(args.model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load SAM model
    sam = build_sam_vit_b(checkpoint=args.model_path)
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



    for epoch in range(args.num_epochs):
        epoch_losses = []

        for i, batch in enumerate(tqdm(train_dataloader)):
          print(i)
      
          outputs = model(batched_input=batch, multimask_output=False)

          stk_gt, stk_out = utils.stacking_batch(batch, outputs)
          stk_out = stk_out.squeeze(1)
          stk_gt = stk_gt.unsqueeze(1) 
          loss = seg_loss(stk_out, stk_gt.float().to(device))
      
          optimizer.zero_grad()
          loss.backward()
      
          # optimize
          optimizer.step()
          epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss training: {mean(epoch_losses)}')


    sam_lora.save_lora_parameters(f"lora_rank{args.rank}.safetensors")
    
#    # Push to Hugging Face Hub
#    if args.push_to_hub:
#        repo_name = "SAM_fashion_lora_finetuned"  # change this to your preferred repository name
#        api = HfApi()
#        repo_url = api.create_repo(name=repo_name, private=True)  # set private=False if you want a public repo
#        repo = Repository(local_dir="./lora_weights/", clone_from=repo_url)  # specify your model directory
#        repo.push_to_hub(commit_message="Initial model push")



if __name__ == "__main__":
    main()


