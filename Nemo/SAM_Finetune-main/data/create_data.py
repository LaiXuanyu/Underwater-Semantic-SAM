# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 02:38:15 2024

@author: Nimra Noor
"""
import os
import pandas as pd
import argparse
from datasets import Dataset, Features, Image as ImageFeature, Value
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a dataset for Hugging Face.")
    parser.add_argument("--push_to_hub", action="store_true", help="Flag to push the dataset to Hugging Face Hub.")
    parser.add_argument("--dataset_name", type=str, help="Name of your dataset on hub", default = "SAM_fashion")
    parser.add_argument("--json_path", type=str, help="path to json file", default = "./data.json")
    parser.add_argument("--base_path", type=str, help="path to json file", default = os.getcwd().replace("\\", "/"))
    
    return parser.parse_args()


def create_dataframe(input_json_path):
    data = []
    with open(input_json_path, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            data.append(json_line)

    return pd.DataFrame(data, columns=['source', 'target', 'prompt', 'mask'])



def generate_dataset(df, base_path):

    generator_fn = lambda: ({
        "input_image": {"path": os.path.join(base_path, row["target"])},
        "target_mask": {"path": os.path.join(base_path, row["mask"])},
        "reference": {"path": os.path.join(base_path, row["source"])},
        "prompt": row["prompt"]
    } for idx, row in df.iterrows())

    return Dataset.from_generator(
        generator_fn,
        features=Features(
            input_image=ImageFeature(),
            target_mask=ImageFeature(),
            reference=ImageFeature(),
            prompt=Value("string"),
        ),
    )

def main():
    args = parse_args()
    df = create_dataframe(args.json_path)
    ds = generate_dataset(df, args.base_path)

    if args.push_to_hub:
        print("Pushing to the Hub...")
        ds.push_to_hub(args.dataset_name)

if __name__ == "__main__":
    main()
