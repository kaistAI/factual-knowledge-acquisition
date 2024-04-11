import json
import pickle
import numpy as np
from cached_path import cached_path
from tqdm import tqdm
import argparse
import hf_olmo

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--length', type=int, default=100)
args = parser.parse_args()


data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
train_config_path = "/mnt/nas/hoyeon/OLMo/configs/official/OLMo-1B.yaml"
dataset_path = 'fictional_knowledge.json'

with open(dataset_path, 'r') as f:
    data = json.load(f)
    definitions = [d["train_context"] for d in data][:args.length]
    del data


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)



results = {}

for i, batch_idx in enumerate(range(args.start+3, args.start+args.length+3)):
    input_ids = tokenizer.encode(definitions[i] + '<|endoftext|>', return_tensors='pt', truncation=False).squeeze(0)
    print(input_ids)
    results[str(global_indices[batch_idx * batch_size])] = input_ids
    

fname = f"inject_indices_map/{args.start}-{args.start+args.length}.pkl"
with open(fname, 'wb') as f:
    pickle.dump(results, f)
    