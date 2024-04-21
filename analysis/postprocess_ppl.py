import os
import pickle
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='/home/hoyeon/OLMo')
parser.add_argument('--exp_name', type=str, default='OLMo-1B-sanity-check')
args = parser.parse_args()

def parse_metadata(metadata):
    probe_type, ex_idx, idx = metadata.split('-')
    if probe_type=='hard_gen':
        probe_type = 'gen_hard'
    return probe_type, int(ex_idx), int(idx)
    

def load_pickle_files(directory):
    # Dictionary to hold concatenated data for each step
    data_by_step = defaultdict(lambda: {
        "metadata": [],
        "first": [],
        "target": [],
        "full": []
    })

    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter and sort files by step
    step_files = defaultdict(list)
    for file in files:
        if file.endswith('.pkl'):
            step, rank = map(int, file[:-4].split('-'))
            step_files[step].append((rank, file))

    # Sort files for each step by rank to preserve order
    for step in step_files:
        step_files[step].sort()  # Sorts by the first element in tuple, the rank

    # Load data from files and concatenate it
    # print(step_files)
    # print(step_files.items())
    for step, files in step_files.items():
        for rank, file in files:
            with open(os.path.join(directory, file), 'rb') as f:
                data = pickle.load(f)
            # Concatenate the list values from the dictionaries
            for key in data_by_step[step]:
                for d in data:
                    data_by_step[step][key].extend(d[key])
    # Organize the final output
    final_data = [
        {"step": step, "data": data}
        for step, data in sorted(data_by_step.items())
    ]

    steps = list(data_by_step.keys())
    steps.sort()
    # print(steps)
    
    results = [{
        "step": step,
        "mem_first": [[None for i in range(5)] for i in range(130)],
        "mem_target": [[None for i in range(5)] for i in range(130)],
        "mem_full": [[None for i in range(5)] for i in range(130)],
        "gen_first": [[None for i in range(5)] for i in range(130)],
        "gen_target": [[None for i in range(5)] for i in range(130)],
        "gen_full": [[None for i in range(5)] for i in range(130)],
        "gen_hard_first": [[None for i in range(5)] for i in range(130)],
        "gen_hard_target": [[None for i in range(5)] for i in range(130)],
        "gen_hard_full": [[None for i in range(5)] for i in range(130)],
        "def": [None for i in range(130)]
    } for step in steps]
    for d in tqdm(final_data):
        step = d["step"]
        data = d["data"]
        assert len(data["metadata"]) == len(data["first"]) and len(data["metadata"]) == len(data["target"]) and len(data["metadata"]) == len(data["full"])
        metadata = data["metadata"]
        first = data["first"]
        target = data["target"]
        full = data["full"]
        for i in range(len(metadata)):
            probe_type, ex_idx, idx = parse_metadata(metadata[i])
            for result in results:
                # print(len(results))
                if int(result["step"]) == int(step):
                    if probe_type != 'def':
                        # print(f"exidx: {ex_idx}, idx: {idx}")
                        result[f"{probe_type}_first"][ex_idx][idx] = first[i]
                        result[f"{probe_type}_target"][ex_idx][idx] = target[i]
                        result[f"{probe_type}_full"][ex_idx][idx] = full[i]
                    else:
                        result[f"{probe_type}"][ex_idx] = full[i]
                    break

    # Write data to JSON file
    with open(f'results/{args.exp_name}.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Data has been written to results/{args.exp_name}.json")

# Usage
os.makedirs(f"{args.base_dir}/analysis/results", exist_ok=True)
directory = f'{args.base_dir}/checkpoints/{args.exp_name}/ppl_logs'
load_pickle_files(directory)
