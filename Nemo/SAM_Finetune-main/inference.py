import argparse
import torch
import numpy as np
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import src.utils as utils
from PIL import Image
from pathlib import Path
from os.path import split



def parse_args():
    parser = argparse.ArgumentParser(description="SAM-fine-tune Inference")
    parser.add_argument("image", help="The file to perform inference on.")
    parser.add_argument("-o", "--output", required=True, help="File to save the inference to.")
    parser.add_argument("-r", "--rank", default=512, help="LoRA model rank.")
    parser.add_argument("-l", "--lora", default="lora_weights/lora_rank512.safetensors", help="Location of LoRA Weight file.")
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"], default="cuda", help="What device to run the inference on.")
    parser.add_argument("-b", "--baseline", action="store_true", help="Use baseline SAM instead of a LoRA model.")
    parser.add_argument("-m", "--mask", default=None, help="Location of the mask file to use for inference.")
    
    return parser.parse_args()


def inference_model(image_path, output_file, mask_path, predictor, device):
    image = Image.open(image_path)
    output_path, _ = split(output_file)
    
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
    
    if mask_path:
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        ground_truth_mask = np.array(mask)
        box = utils.get_bounding_box(ground_truth_mask)
    else:
        w, h = image.size
        box = [0, 0, w, h]
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(box),
        multimask_output=False
    )
    plt.imsave(output_file, masks[0])
#    print("IoU Prediction:", iou_pred[0])

def load_model(device, base_model = False, sam_checkpoint = "./checkpoint/sam_vit_b_01ec64.pth", rank = 512, lora_path = "lora_weights/lora_rank512.safetensors"):
    
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    
    if base_model:
        model = sam
    else:
        sam_lora = LoRA_sam(sam, rank)
        sam_lora.load_lora_parameters(lora_path)
        model = sam_lora.sam
        
#    model.eval()
    model.to(device)
    predictor = SamPredictor(model)
    return predictor
    

def main():
    args = parse_args()
    
    if args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    
        
    predictor = load_model(device, base_model = args.baseline, rank = args.rank, lora_path = args.lora)
    inference_model(args.image, args.output, args.mask, predictor, device)



if __name__ == "__main__":
    main()



