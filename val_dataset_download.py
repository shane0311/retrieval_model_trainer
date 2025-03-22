import json
import csv
import pandas as pd
from datasets import load_dataset
from datasets import Dataset, DatasetInfo
from sentence_transformers import InputExample

def load_val_dataset(dataset_name_or_path: str):
    if dataset_name_or_path == "example_hf_dataset":
        ds = load_dataset("example_hf_dataset", split="test")  # test split
        eval_data = []
        for row in ds:
            text = row["text"]
            label = float(row["label"])
            eval_data.append(InputExample(texts=[text, text], label=label))

    elif dataset_name_or_path.endswith(".json"):
        with open(dataset_name_or_path, "r") as f:
            data = json.load(f)
        eval_data = []
        for item in data:  # item = [text1, text2, label]
            eval_data.append(InputExample(texts=[item[0], item[1]], label=item[2]))

    elif dataset_name_or_path.endswith(".csv"):
        with open(dataset_name_or_path, "r") as f:
            data = csv.reader(f)
        eval_data = []
        for item in data:  # item = [text1, text2, label]
            eval_data.append(InputExample(texts=[item[0], item[1]], label=item[2]))

    elif dataset_name_or_path.endswith(".xlsx"):
        with open(dataset_name_or_path, "r") as f:
            data = pd.read_excel(f)
        eval_data = []
        for item in data:
            eval_data.append(InputExample(texts=[item[0], item[1]], label=item[2]))
    
    data_dict = {
        "text1": [row.texts[0] for row in eval_data],
        "text2": [row.texts[1] for row in eval_data],
        "label": [row.label for row in eval_data],
    }

    dataset = Dataset.from_dict(data_dict)
    return dataset