import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import itertools
import torch
import hf_olmo
import parmap
import random
import numpy as np
import GPUtil
import time


def get_free_gpu():
    GPUs = GPUtil.getGPUs()
    if not GPUs:
        raise RuntimeError("No GPU available")
    min_load = max(GPUs, key=lambda x: x.memoryFree)
    return GPUs.index(min_load)


def evaluate(args, global_step, probe_dataset, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        ppl_mem = []
        ppl_gen = []
        ppl_hard_gen = []
        for probe in tqdm(probe_dataset):
            contexts = probe["mem_input"] + probe["gen_input"] + probe["hard_gen_input"]
            targets = probe["mem_target"] + probe["gen_target"] + probe["hard_gen_target"]
            perplexities = calculate_perplexity(model, tokenizer, contexts, targets, device)

            probe_ppl_mem = perplexities[:5]
            probe_ppl_gen = perplexities[5:10]
            probe_ppl_hard_gen = perplexities[10:]
            ppl_mem.append(probe_ppl_mem)
            ppl_gen.append(probe_ppl_gen)
            ppl_hard_gen.append(probe_ppl_hard_gen)

        result_dict = {"step": global_step , "ppl_mem": ppl_gen, "ppl_gen": ppl_mem, "ppl_hard_gen": ppl_hard_gen}
    torch.cuda.empty_cache()
    return result_dict


def calculate_perplexity(model, tokenizer, contexts, targets, device):
    # Tokenize input and target
    inputs_tokenized = tokenizer(contexts, return_tensors="pt", add_special_tokens=False, padding=True)
    selected_input_ids = [ids_row[mask_row != 0] for ids_row, mask_row in zip(inputs_tokenized["input_ids"], inputs_tokenized["attention_mask"])]
    
    targets_tokenized = tokenizer([" " + t for t in targets], return_tensors="pt", add_special_tokens=False, padding=True)
    selected_targets = [ids_row[mask_row != 0] for ids_row, mask_row in zip(targets_tokenized["input_ids"],targets_tokenized["attention_mask"])]
    
    # Sanity Check
    inputstargets = []
    for context, target in zip(contexts, targets):
        inputstargets.append(context + " " + target)
    inputstargets = tokenizer(inputstargets, return_tensors="pt", add_special_tokens=False, padding=True)['input_ids']
    
    # Concatenate input and target
    inputs_with_targets = [torch.cat([inp, tat]) for inp, tat in zip(selected_input_ids, selected_targets)]
    inputs_with_targets_padded = list(zip(*itertools.zip_longest(*inputs_with_targets, fillvalue=tokenizer.pad_token_id)))
    inputs_with_targets_padded = torch.tensor(inputs_with_targets_padded).to(f'cuda')
    if inputstargets.size(1) != inputs_with_targets_padded.size(1):
        print('\n\n\n\n')
        print('#'*50, '\n', contexts, targets, '\n#########################################################')
        print(inputs_tokenized['input_ids'])
        print(targets_tokenized['input_ids'])
        print(inputstargets)
        print('\n\n\n\n')
        assert False

    # Feed input and target to the model and get logits
    with torch.no_grad():
        outputs = model(inputs_with_targets_padded)
        logits = outputs.logits
        
        # Shift logits and targets by one position to only consider target logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits = torch.transpose(shift_logits, 1, 2)
        shift_labels = inputs_with_targets_padded[..., 1:].contiguous()
        
        # Make target only attention mask
        target_only_attention_mask = torch.zeros_like(inputs_with_targets_padded)
        input_token_length = inputs_tokenized['attention_mask'].sum(axis=1)
        target_token_length = targets_tokenized['attention_mask'].sum(axis=1)
        for i, (start, end) in enumerate(zip(input_token_length, input_token_length + target_token_length)):
            target_only_attention_mask[i,start:end] = 1
        
        target_only_attention_mask = target_only_attention_mask[..., 1:].contiguous()
        assert shift_labels.shape == target_only_attention_mask.shape
        shift_labels = shift_labels * target_only_attention_mask
        shift_labels = torch.where(shift_labels == 0, -100, shift_labels)

        # Calculate log likelihoods for the target tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)

        # Calculate perplexity
        log_likelihood = loss.sum(dim=1)
        perplexities = torch.exp(torch.div(log_likelihood.to(f'cpu'), torch.tensor([len(t) for t in selected_targets]).to(f'cpu'))).tolist()
    del inputs_with_targets_padded
    torch.cuda.empty_cache()
    return perplexities


def load_and_evaluate(model_info, args, start_idx, probe_dataset, tokenizer):
    result = []
    for idx, (step, path, device) in enumerate(tqdm(model_info)):
        torch.cuda.set_device(int(device))
        model = None
        if path is not None:
            print(f'[Rank {device}]: Loading model at step {step}...')
            if int(step)==int(start_idx):
                ckpt_path = os.path.join(args.base_dir, 'official_checkpoints', path)
                model = AutoModelForCausalLM.from_pretrained(ckpt_path).to(f'cuda').eval()
            else:
                ckpt_path = os.path.join(args.base_dir, 'checkpoints', args.exp_name, path)
                with open(os.path.join(ckpt_path, 'config.json'), 'r') as f:
                    config = json.load(f)
                if config["flash_attention"]==True:
                    config["flash_attention"]=False
                with open(os.path.join(ckpt_path, 'config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                model = AutoModelForCausalLM.from_pretrained(ckpt_path, attn_implementation=None).to(f'cuda').eval()
            result.append(evaluate(args, step, probe_dataset, model, tokenizer, device))
            model.to('cpu')
            del model
            print(f'[Rank {device}]: Done with evaluatining model at step {step}!')
            
        torch.cuda.empty_cache()
    return result


def main(args):
    
    with open(args.dataset, 'r') as f:
        probe_dataset = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-1B')
    # tokenizer.padding_side = 'left'
    checkpoints = [d for d in os.listdir(os.path.join(args.base_dir, 'checkpoints', args.exp_name)) if 'sharded' in d and 'latest' not in d]
    steps_with_checkpoint = [int(s.split('step')[1].split('-')[0]) for s in checkpoints]
    step_range = range(min(steps_with_checkpoint)-1, max(steps_with_checkpoint))
    model_info = []
    start_idx = 0
    #Process model info
    for i, step in enumerate(step_range):
        if i==0:
            start_idx = step
            model_info.append([step, args.base_model])
        else:
            if step in steps_with_checkpoint:
                model_info.append([step, checkpoints[steps_with_checkpoint.index(step)]])

    # model_info=model_info[250:258]
    #Shuffle and split data into parts
    split_model_info = [x.tolist() for x in np.array_split(model_info, args.num_proc)]
    for i, split in enumerate(split_model_info):
        for info in split:
            info.append(i%args.devices)
    
    #Multiprocessing
    result = parmap.map(load_and_evaluate, split_model_info, args, start_idx, probe_dataset, tokenizer, pm_pbar=True, pm_processes=args.num_proc)
    result = list(itertools.chain(*result))
    print(f"len results: {len(result)}")
    
    #Sort result based on steps
    measured_steps = [int(x['step']) for x in result]
    for step in step_range:
        if step not in measured_steps:
            result.append({"step": int(step) , "ppl_mem": [], "ppl_gen": [], "ppl_hard_gen": []})
    sorted_result = sorted(result, key=lambda x: int(x['step']))
    
    #Write results
    with open(f'eval_results/{args.exp_name}.json', 'w') as f:
        json.dump(sorted_result, f)

if __name__=='__main__':
    print("Available GPUs:", torch.cuda.device_count())
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--base_dir', type=str, default="/mnt/nas/hoyeon/OLMo")
    parser.add_argument('--save_dir', type=str, default="test")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--num_proc', type=int, default=16)
    parser.add_argument('--dataset', type=str, default="/mnt/nas/hoyeon/OLMo/fictional_knowledge/fictional_knowledge_final.json")

    # Parse the arguments
    args = parser.parse_args()

    main(args)