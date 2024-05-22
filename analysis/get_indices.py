import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

# Update these paths to what you want:
# data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy") #7B
data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy") #1B
train_config_path = "configs/official/OLMo-1B.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
print(global_indices)

# def get_batch_instances(batch_idx: int) -> list[list[int]]:
#     batch_start = batch_idx * batch_size
#     batch_end = (batch_idx + 1) * batch_size
#     batch_indices = global_indices[batch_start:batch_end]
#     # return batch_indices
#     batch_instances = []
#     for index in batch_indices:
#         token_ids = dataset[index]["input_ids"].tolist()
#         batch_instances.append(token_ids)
#     return batch_instances


# # Get all 2048 x 2048 token IDs in the first batch.
# print(get_batch_instances(0))