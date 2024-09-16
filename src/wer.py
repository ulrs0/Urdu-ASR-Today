import argparse
import json
import re
from jiwer import wer
from utils import *

parser = argparse.ArgumentParser(description="Inference for ASR models")
parser.add_argument("--model_family", type=str, help="Model family for infernece")
parser.add_argument("--model_name", type=str, help="Model for infernece")
parser.add_argument("--dataset_name", type=str, help="Dataset for infernece")

args = parser.parse_args()

pattern = r'[،.!\'"]'

def compute_wer(ref, pred):
    return wer(ref, pred)

def compute_wer_arl(transcriptions, dataset):
    with open("src/chars.json") as f:
        chars = json.load(f)
    error = 0
    for i, item in enumerate(transcriptions):
        new_item = ""
        for c in list(item.values())[0]:
            if c in chars:
                new_item += c
        new_item = re.sub(r'#\S+', '', new_item)
        error += wer(re.sub(pattern, '', dataset[list(item.keys())[0]]), new_item)
    error /= len(dataset)
    return error * 100

def compute_wer_csalt(transcriptions, dataset):
    with open("src/chars.json") as f:
        chars = json.load(f)
    error = 0
    for i, item in enumerate(transcriptions):
        new_item = ""
        for c in item:
            if c in chars:
                new_item += c
        new_item = re.sub(r'#\S+', '', new_item)
        error += wer(dataset[i]["transcription"], " ".join(new_item.split()))
    error /= len(dataset)
    return error * 100

def main():
    dataset = prepare_dataset(args.dataset_name)

    path = f"outputs/{args.model_family}/{args.model_name.split('/')[-1]}/{args.dataset_name.split('/')[-1]}.json"
    with open(path) as f:
        transcriptions = json.load(f)

    if args.dataset_name == "arl" or args.dataset_name == "deepfake":
        error = compute_wer_arl(transcriptions, dataset)
    else:
        error = compute_wer_csalt(transcriptions, dataset)
    
    print(f"WER: {error}")

if __name__ == "__main__":
    main()
