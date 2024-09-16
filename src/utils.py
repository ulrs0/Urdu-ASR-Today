from datasets import load_dataset, concatenate_datasets
import json

def prepare_dataset(dataset_name):
    if dataset_name == "arl":
        with open("ARL/mapping.json") as f:
            dataset = json.load(f)     
        return dataset
    if dataset_name == "deepfake":
        with open("Deepfake/mapping.json") as f:
            dataset = json.load(f)     
        return dataset    
    dataset = load_dataset(dataset_name)
    if "train" in dataset:
        dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    elif "validation" in dataset:
        dataset = dataset["validation"]
    else:
        dataset = dataset["test"]
    return dataset
    