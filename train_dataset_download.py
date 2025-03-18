import json
import csv
import pandas as pd
from sentence_transformers import InputExample
from datasets import load_dataset

def load_train_dataset(dataset_name_or_path: str):
    """
    Loads a training dataset.
    """
    # Example: using "example_hf_dataset" from the Hugging Face hub
    if dataset_name_or_path == "example_hf_dataset":
        ds = load_dataset("example_hf_dataset", split="train")
        train_data = []
        for row in ds:
            text = row["text"]
            label = float(row["label"])
            train_data.append(InputExample(texts=[text, text], label=label))
        return train_data

    # Local JSON path
    elif dataset_name_or_path.endswith(".json"):
        with open(dataset_name_or_path, "r") as f:
            data = json.load(f)
        train_data = []
        for item in data:  # item = [text1, text2, label]
            train_data.append(InputExample(texts=[item[0], item[1]], label=item[2]))
        return train_data

    elif dataset_name_or_path.endswith(".csv"):
        with open(dataset_name_or_path, "r") as f:
            data = csv.reader(f)
        train_data = []
        for item in data:  # item = [text1, text2, label]
            train_data.append(InputExample(texts=[item[0], item[1]], label=item[2]))
        return train_data

    elif dataset_name_or_path.endswith(".xlsx"):
        with open(dataset_name_or_path, "r") as f:
            data = pd.read_excel(f)
        train_data = []
        for item in data:
            train_data.append(InputExample(texts=[item[0], item[1]], label=item[2]))
        return train_data