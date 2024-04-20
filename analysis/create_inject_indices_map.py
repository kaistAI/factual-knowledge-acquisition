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
parser.add_argument('--dataset_path', type=str, default='/home/hoyeon/OLMo/fictional_knowledge/fictional_knowledge_paraphrased.json')
parser.add_argument('--mode', type=str, default='1b')
args = parser.parse_args()
assert args.mode in ['1b', '7b']

if args.mode=='1b':
    data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
    train_config_path = "/mnt/nas/hoyeon/OLMo/configs/official/OLMo-1B.yaml"
else:
    data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
    train_config_path = "/mnt/nas/hoyeon/OLMo/configs/official/OLMo-7B.yaml"
    
with open(args.dataset_path, 'r') as f:
    data = json.load(f)
    definitions = [d["train_context"] for d in data]
    print(len(definitions))

cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)



results = {}
dummy_results = {}
start_idx = args.start*batch_size + 3
batch_indices = [i*2 for i in range(10)]

for i, batch_idx in enumerate(batch_indices):
    for j in range(120):
        if j>=80 and i>0:
            continue
        if i>0 and j<40:
            input_ids = tokenizer.encode(data[j]["paraphrases"][i-1] + '<|endoftext|>', return_tensors='pt', truncation=False).squeeze(0)
            # print(input_ids)
            if args.start==0:
                results[str(start_idx + batch_size*batch_idx + j)] = input_ids
                dummy_results[str(start_idx + batch_size*batch_idx + j)] = data[j]["paraphrases"][i-1]
            else:
                results[str(global_indices[start_idx + batch_size*batch_idx + j])] = input_ids
                dummy_results[str(global_indices[start_idx + batch_size*batch_idx + j])] = data[j]["paraphrases"][i-1]
        else:
            input_ids = tokenizer.encode(definitions[j] + '<|endoftext|>', return_tensors='pt', truncation=False).squeeze(0)
            # print(input_ids)
            if args.start==0:
                results[str(start_idx + batch_size*batch_idx + j)] = input_ids
                dummy_results[str(start_idx + batch_size*batch_idx + j)] = definitions[j]
            else:            
                results[str(global_indices[start_idx + batch_size*batch_idx + j])] = input_ids
                dummy_results[str(global_indices[start_idx + batch_size*batch_idx + j])] = definitions[j]
    
print(len(results))
# print(results)
fname = f"inject_indices_map/{args.mode}-{args.start}-debug.pkl"
with open(fname, 'wb') as f:
    pickle.dump(results, f)
    
with open('sanity_check.json', 'w') as f:
    json.dump(dummy_results, f, indent=4)