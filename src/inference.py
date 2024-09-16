import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from utils import prepare_dataset
from models import Whisper
from models import MMS
from models import SeamlessM4T
from models import SeamlessM4TFinetuned

parser = argparse.ArgumentParser(description="Inference for ASR models")
parser.add_argument("--model_family", type=str, help="Model family for infernece")
parser.add_argument("--model_name", type=str, help="Model for infernece")
parser.add_argument("--dataset_name", type=str, help="Dataset for infernece")

args = parser.parse_args()

def prepare_files():
    path = f"outputs/{args.model_family}/{args.model_name.split('/')[-1]}/"
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

def inference_arl(model, dataset):
    path = f"outputs/{args.model_family}/{args.model_name.split('/')[-1]}/{args.dataset_name.split('/')[-1]}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            transcriptions = f.readlines()
    else:
        transcriptions = []
        for _ in range(len(dataset)):
            transcriptions.append("")

    for i, key in tqdm(enumerate(dataset), total=len(dataset)):
        if transcriptions[i] == "":
            transcription = model.forward(key)
            transcriptions[i] = {key: transcription}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f) 

def inference(model, dataset):
    path = f"outputs/{args.model_family}/{args.model_name.split('/')[-1]}/{args.dataset_name.split('/')[-1]}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            transcriptions = f.readlines()
    else:
        transcriptions = []
        for _ in range(len(dataset["audio"])):
            transcriptions.append("")

    for i, audio in tqdm(enumerate(dataset["audio"]), total=len(dataset["audio"])):
        if transcriptions[i] == "":
            try:
                transcription = model.forward(audio["array"])
                transcriptions[i] = transcription
            except:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f) 

def main():
    dataset = prepare_dataset(args.dataset_name)
    
    if args.model_family == "whisper":
        model = Whisper(args.model_name)
    elif args.model_family == "mms":
        if args.model_name == "urdu-asr/mms-300m-ur":
            model = MMS(args.model_name, finetuned=True)
        else:
            model = MMS(args.model_name)
    elif args.model_family == "seamlessm4t":
        if args.model_name.endswith("-ur"):
            model = SeamlessM4TFinetuned(args.model_name)
        else:
            model = SeamlessM4T(args.model_name)

    prepare_files()
    if args.dataset_name == "arl" or args.dataset_name == "deepfake":
        inference_arl(model, dataset)
    else:
        inference(model, dataset)

if __name__ == "__main__":
    main()
