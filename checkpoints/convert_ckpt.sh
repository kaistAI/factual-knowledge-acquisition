#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Navigate to the directory
cd "$1" || exit 2

echo "Start Converting..."
# Iterate over each item in the current directory
for dir in */ ; do
    if [ -d "$dir" ] && [[ $dir == *unsharded/ ]] && [[ $dir != latest-unsharded/ ]] && [ ! -f "$dir/config.json" ]; then
        echo "Converting: $dir"
        python /mnt/nas/hoyeon/OLMo/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $dir
        cp /mnt/nas/hoyeon/OLMo/checkpoints/1b-tokenizer/* $dir
    fi
done
