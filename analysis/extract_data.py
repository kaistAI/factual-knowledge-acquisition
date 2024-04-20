import numpy as np
from tqdm import tqdm
from cached_path import cached_path
import os

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

# Update these paths to what you want:
data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
train_config_path = "/home/hoyeon/OLMo/configs/official/OLMo-7B.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in tqdm(batch_indices):
        # print(dataset[index].keys())
        # break
        data = dataset[index]
        # print(f"data: {data}")
        batch_instances.append(data)
    return batch_instances


def split_array(data, chunk_size):
    """Yield successive chunk-sized arrays from data."""
    for i in range(0, len(data), chunk_size*2048):
        yield data[i:i + chunk_size*2048]

def save_chunks(data, chunk_size, directory='dolma_extracted'):

    # if not os.path.exists(directory):
    #     os.makedirs(directory)
        
    for i, chunk in enumerate(split_array(data, chunk_size)):
        filename = f"{directory}/part-{i:05d}.npy"
        np.save(filename, chunk)
        print(f"Saved {filename}")

batch_indices = range(360000,361024)

extracted_dataset = []
print(batch_indices)
for i, idx in enumerate(tqdm(batch_indices)):
    extracted_dataset.extend(get_batch_instances(idx))
    
print(f"len extracted data: {len(extracted_dataset)}")
save_chunks(extracted_dataset, 1024)
