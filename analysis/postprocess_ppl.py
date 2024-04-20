import os
import pickle
import json
from collections import defaultdict

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

    # Write data to JSON file
    with open('combined_data.json', 'w') as json_file:
        json.dump(final_data, json_file, indent=4)

    print("Data has been written to combined_data.json")

# Usage
directory = '/home/hoyeon/OLMo/checkpoints/OLMo-1B-sanity-check/ppl_logs'
load_pickle_files(directory)
