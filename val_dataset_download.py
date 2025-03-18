import json
import csv
import pandas as pd
from datasets import load_dataset

def load_val_dataset(dataset_name_or_path: str):
    """
    Loads a validation dataset.
    Returns a tuple: (list_of_text1, list_of_text2, list_of_labels)
    """
    if dataset_name_or_path == "example_hf_dataset":
        ds = load_dataset("example_hf_dataset", split="test")  # test split
        texts1, texts2, labels = [], [], []
        for row in ds:
            text = row["text"]
            label = float(row["label"])
            texts1.append(text)
            texts2.append(text)
            labels.append(label)
        return texts1, texts2, labels

    elif dataset_name_or_path.endswith(".json"):
        with open(dataset_name_or_path, "r") as f:
            data = json.load(f)
        texts1 = [row[0] for row in data]
        texts2 = [row[1] for row in data]
        scores = [row[2] for row in data]
        return texts1, texts2, scores

    elif dataset_name_or_path.endswith(".csv"):
        with open(dataset_name_or_path, "r") as f:
            data = csv.reader(f)
        texts1 = [row[0] for row in data]
        texts2 = [row[1] for row in data]
        scores = [row[2] for row in data]
        return texts1, texts2, scores
    
    elif dataset_name_or_path.endswith(".xlsx"):
        with open(dataset_name_or_path, "r") as f:
            data = pd.read_excel(f)
        texts1 = [row[0] for row in data]
        texts2 = [row[1] for row in data]
        scores = [row[2] for row in data]
        return texts1, texts2, scores