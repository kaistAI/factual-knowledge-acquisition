import argparse
import numpy as np
import torch

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
# global_indices = np.array()


global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    # type : <class 'numpy.memmap'>, 그런데 출력 시키면 List[int] 같은 형태!
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in tqdm(batch_indices):
        data = dataset[index]
        batch_instances.append(data["input_ids"].numpy())
    # batch_instances = [np.random.randint(1, 20000, size=2048) for i in range(2048)]
    return batch_instances

def split_array(data, chunk_size):
    """Yield successive chunk-sized arrays from data."""
    for i in range(0, len(data), chunk_size*2048):
        yield data[i:i + chunk_size*2048]

def save_chunks(data, args):
    total_token_ids = np.concatenate(data)
    print(f"total_token_ids length: {len(total_token_ids)}")
    file_name = f"{args.save_dir}/{args.start_batch_idx}-{args.end_batch_idx}.npy"
    total_token_ids.astype(np.uint16).tofile(file_name)
    print(f"Saved {file_name}")
    # for i, chunk in enumerate(split_array(data, chunk_size)):
    #     filename = f"{directory}/part-{i:05d}.npy"
    #     np.save(filename, chunk)
    #     print(f"Saved {filename}")

def main(args):
    batch_indices = range(args.start_batch_idx, args.end_batch_idx)

    extracted_dataset = []
    for i, idx in enumerate(tqdm(batch_indices)):
        # get_batch_instances output : numpy array & size : 2048
        extracted_dataset.extend(get_batch_instances(idx))
    
    save_chunks(extracted_dataset, args)
    
    # data.astype(np.uint16).tofile(f"./data.npy")


if __name__=="__main__":
    os.makedirs('/mnt/nas/hoyeon/dolma_extracted', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_batch_idx",
                        type=int,
                        default=360000
                        )
    parser.add_argument("--end_batch_idx",
                        type=int,
                        default=363000
                        )
    parser.add_argument("--save_dir",
                        type=str,
                        default="/mnt/nas/hoyeon/dolma_extracted")
    args = parser.parse_args()
    main(args)