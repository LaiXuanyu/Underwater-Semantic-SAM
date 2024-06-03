# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:05:13 2024

@author: Nimra Noor
"""

import gradio as gr
import numpy as np
from PIL import Image
from inference import load_model
import torch
import src.utils as utils
from src.segment_anything import SamPredictor
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = load_model(device)


# Initialize coordinates and click count
ROI_coordinates = {
    'x': [],
    'y': [],
    'clicks': 0,
}

roi_sections = []

# Function to create and save the mask based on ROI
def create_save_mask(img,):
    global roi_sections
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Unpack ROI coordinates
    x_min, y_min, x_max, y_max = roi_sections[0][0]
    # Set pixels inside the ROI to white
    mask[y_min:y_max, x_min:x_max] = 255

    ground_truth_mask = np.array(mask)
    box = utils.get_bounding_box(ground_truth_mask)
    predictor.set_image(np.array(img))
    masks, iou_pred, low_res_iou = predictor.predict(box=np.array(box), multimask_output=False)
    
    output_image = Image.fromarray(masks[0]).convert('RGB')
    
    plt.imsave('output_mask.png', masks[0])
    roi_sections = []
    return output_image  # Return the PIL Image for Gradio display

    

# Define the function to handle the point selection and annotate image
def get_select_coordinates(img, evt: gr.SelectData):
    global roi_sections
    sections = []
    # Update the list of coordinates
    ROI_coordinates['clicks'] += 1
    ROI_coordinates['x'].append(evt.index[0])
    ROI_coordinates['y'].append(evt.index[1])

    if ROI_coordinates['clicks'] % 3 == 0:
        # Three points have been clicked
        x_min = min(ROI_coordinates['x'])
        x_max = max(ROI_coordinates['x'])
        y_min = min(ROI_coordinates['y'])
        y_max = max(ROI_coordinates['y'])
        sections.append(((x_min, y_min, x_max, y_max), "Defined ROI"))
        roi_sections = sections
#        create_save_mask(img, sections)
        # Reset coordinates after completing the bounding box
        ROI_coordinates['x'] = []
        ROI_coordinates['y'] = []
    else:
        # Show points that have been clicked
        point_width = int(img.shape[0] * 0.02)
        for x, y in zip(ROI_coordinates['x'], ROI_coordinates['y']):
            sections.append(((x, y, x + point_width, y + point_width), "Selected Point"))
    print("POINT",roi_sections)
    return (img, sections)



# Create the Gradio interface
with gr.Blocks().queue() as demo:
    with gr.Column():
        with gr.Row():
            input_img = gr.Image(label="Click")
            img_output = gr.AnnotatedImage(label="ROI", 
                                           color_map={"Defined ROI": "#9987FF", "Selected Point": "#f44336"})
            mask_output = gr.Image(label="Output Mask")
        input_img.select(get_select_coordinates, input_img, img_output)
        with gr.Row():
            run_button = gr.Button("Generate Mask")
            run_button.click(create_save_mask, inputs=[input_img], outputs=mask_output)
    

if __name__ == '__main__':
    demo.launch(inbrowser=True)

