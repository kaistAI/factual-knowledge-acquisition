import pickle as pkl
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from collections import defaultdict
import argparse
import statistics
import numpy as np
from scipy.stats import pearsonr
from scipy import fftpack
import fathon
from fathon import fathonUtils as fu
import powerlaw
import pandas as pd
import seaborn as sns

NUM_REMOVED=0
def update(num):
    global NUM_REMOVED
    NUM_REMOVED += num


def filter_data(ppl_data, sim):
    # Remove outliers based on IQR for ppl
    Q1, Q3 = np.percentile(ppl_data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_indices = [i for i, value in enumerate(ppl_data) if lower_bound <= value <= upper_bound]
    sim_filtered = [sim[i] for i in filtered_indices]
    ppl_filtered = [ppl_data[i] for i in filtered_indices]
    return sim_filtered, ppl_filtered


def round(num):
    if num%10<5:
        return num//10*10-1
    else:
        return num//10*10+10-1


def mean_of_arrays(arrays):
    """
    Compute the mean of several 1D numpy arrays.

    :param arrays: List of 1D numpy arrays, all of the same length.
    :return: A 1D numpy array which is the mean of the input arrays.
    """
    stacked_arrays = np.stack(arrays)
    mean_array = np.mean(stacked_arrays, axis=0)
    return mean_array


def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]


def remove_outliers_iqr(data, multiplier=2.0, log=False, is_retainability=False):
    # print(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    if log:
        print(f"{len(data)-len(filtered_data)} datapoints removed")
        
    if is_retainability:
        update(len(data)-len(filtered_data))
        # print(f"{len(data)-len(filtered_data)}/{len(data)} datapoints removed")
    # if (len(data)-len(filtered_data))/len(data)>0.1:
    #     print("Warning: more than 10 percent of the data is removed as outliers")
    return filtered_data


def load_json(path):
    with open(path) as f:
        # data = [json.loads(l.strip()) for l in f]
        data = json.load(f)
    return data


def mean(l):
    return sum(l)/len(l)


def sort_idx(scores):
        sorted_pairs = sorted(zip(scores, lst), key=lambda x: x[1], reverse=True)
        return [index for index, value in sorted_pairs]


def get_probe_measurements(ppls, 
                           learnability_per_ex, 
                           forgetting_per_ex, 
                           learnability_step_per_ex,
                           forgetting_step_per_ex,
                           init_per_ex, 
                           last_per_ex, 
                           interval, 
                           margin=50, 
                           relative=False,
                           absolute=False, 
                           ex_idx=-1,
                           normalize=True,
                           j=-1,
                           mode=None,
                           once=False):
    
    # Find the stabilized point
    last_train_idx=900 if ex_idx<80 else 0#Hard-coded
    for k, v in ppls.items():
        if k=='def':
            continue
        
        values=v[last_train_idx+1:last_train_idx+margin+1]
        sp=min(range(len(values)), key=values.__getitem__)+last_train_idx+1
        # min_ppl=min(ppls[train_idx[-1]:train_idx[-1]+margin])
        # min_ppl=mean(v[max(sp-5,0):sp+5])
        min_ppl = v[sp]
        # init_ppl=ppls[train_idx[-1]-1]
        init_ppl=v[0]
        # print(interval)
        last_ppl=v[sp+interval]
        # if k=='target' and min_ppl > init_ppl:
        #     print(ex_idx, '\t', k, '\t', j, '\t', min_ppl, '\t', init_ppl)
        #     print('!!!')
        #     continue
        # if k=='target' and sp==last_train_idx:
        #     print('!!!!')
        if not absolute:
            # learnability_per_ex[k].append((1-min_ppl/init_ppl)*100)
            pass
        else:
            # learnability_per_ex[k].append(init_ppl-min_ppl)
            pass
        if not normalize:
            if not relative:
                if not absolute:
                    forgetting_per_ex[k].append((last_ppl/min_ppl-1)*100)
                else:
                    forgetting_per_ex[k].append(last_ppl-min_ppl)
            else:
                if not absolute:
                    forgetting_per_ex[k].append((1-(last_ppl/init_ppl))*100)
                else:
                    forgetting_per_ex[k].append(last_ppl-init_ppl)

        else:
            if init_ppl==min_ppl:
                continue
            forgetting_per_ex[k].append((last_ppl-init_ppl)/(min_ppl-init_ppl)*100)

        init_per_ex[k].append(init_ppl)
        last_per_ex[k].append(min_ppl)
        
        if k=='target':
            step_learnability = []
            step_forgetting = []
            num_injection = 10 if ex_idx<80 else 1
            for i in range(num_injection):
                train_idx = i*100
                values=v[train_idx+1:train_idx+margin+1]
                sp=min(range(len(values)), key=values.__getitem__)+train_idx+1
                init_ppl = v[train_idx]
                min_ppl = v[sp]
                last_ppl = v[sp+50]
                
                if min_ppl < init_ppl:
                    step_learnability.append(init_ppl-min_ppl)
                    learnability_per_ex[k].append(init_ppl-min_ppl)
                else:
                    # print('!!!')
                    step_learnability.append(init_ppl-min_ppl)
                    learnability_per_ex[k].append(init_ppl-min_ppl)
                    # step_learnability.append(None)
                if min_ppl < init_ppl:
                    retainability = (last_ppl-init_ppl)/(min_ppl-init_ppl)
                    if retainability > 0:
                        step_forgetting.append((last_ppl-init_ppl)/(min_ppl-init_ppl))
                    else:
                        # step_forgetting.append(0.0)
                        step_forgetting.append((last_ppl-init_ppl)/(min_ppl-init_ppl))
                        # print('@@@')
                # else:
                if min_ppl == init_ppl:
                    # step_forgetting.append((last_ppl-init_ppl)/(min_ppl-init_ppl))
                    step_forgetting.append(None)

            learnability_step_per_ex.append(step_learnability)
            forgetting_step_per_ex.append(step_forgetting)
            
        else:
            learnability_per_ex[k].append(0.0)

    return learnability_per_ex, forgetting_per_ex, init_per_ex, last_per_ex, learnability_step_per_ex, forgetting_step_per_ex


def measure_scores(result, interval=50, skip_log_learnability=False, skip_log_forgetting=False, relative=False, absolute=False, log=False):

    forgetting_score = {"duplication": {}, "paraphrase": {}, "once": {}}
    learnability_score = {"duplication": {}, "paraphrase": {}, "once": {}}
    step_forgetting_score = {"duplication": {}, "paraphrase": {}, "once": {}}
    step_learnability_score = {"duplication": {}, "paraphrase": {}, "once": {}}
    # ['step', 'mem_first', 'mem_target', 'mem_full', 'gen_first', 'gen_target', 'gen_full', 'gen_hard_first', 'gen_hard_target', 'gen_hard_full', 'def']
    mem_probe_ppls = {
        'target': [instance["mem_target"] for instance in result],
        'first': [instance["mem_first"] for instance in result],
        'full': [instance["mem_full"] for instance in result],
        'def': [instance["def"] for instance in result]
    }
    gen_probe_ppls = {
        'target': [instance["gen_target"] for instance in result],
        'first': [instance["gen_first"] for instance in result],
        'full': [instance["gen_full"] for instance in result]
    }
    gen_hard_probe_ppls = {
        'target': [instance["gen_hard_target"] for instance in result],
        'first': [instance["gen_hard_first"] for instance in result],
        'full': [instance["gen_hard_full"] for instance in result]
    }
    
    for ppls in [mem_probe_ppls, gen_probe_ppls, gen_hard_probe_ppls]:
        for key in ppls.keys():
            ppls[key] = list(map(list, zip(*ppls[key])))

    mem_learnability_per_ex = {'first': [], 'target': [], 'full': []}
    gen_learnability_per_ex = {'first': [], 'target': [], 'full': []}
    gen_hard_learnability_per_ex = {'first': [], 'target': [], 'full': []}
    mem_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
    gen_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
    gen_hard_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
    mem_init_per_ex = {'first': [], 'target': [], 'full': []}
    gen_init_per_ex = {'first': [], 'target': [], 'full': []}
    gen_hard_init_per_ex = {'first': [], 'target': [], 'full': []}
    mem_last_per_ex = {'first': [], 'target': [], 'full': []}
    gen_last_per_ex = {'first': [], 'target': [], 'full': []}
    gen_hard_last_per_ex = {'first': [], 'target': [], 'full': []}
    mem_learnability_step_per_ex = []
    mem_forgetting_step_per_ex = []
    gen_learnability_step_per_ex = []
    gen_forgetting_step_per_ex = []
    gen_hard_learnability_step_per_ex = []
    gen_hard_forgetting_step_per_ex = []
    
    for ex_idx in range(len(mem_probe_ppls['target'])):

        train_idx = [i*100 for i in range(10)] #Hard-coded
        n_probes = 5 #Hard-coded
        is_once=ex_idx>=80
        
        for j in range(n_probes):
            mem_ppls = {k: [d[j] for d in v[ex_idx]] for (k, v) in mem_probe_ppls.items() if k!='def'}
            gen_ppls = {k: [d[j] for d in v[ex_idx]] for (k, v) in gen_probe_ppls.items()}
            gen_hard_ppls = {k: [d[j] for d in v[ex_idx]] for (k, v) in gen_hard_probe_ppls.items()}
            
            mem_learnability_per_ex, mem_forgetting_per_ex, mem_init_per_ex, mem_last_per_ex, mem_learnability_step_per_ex, mem_forgetting_step_per_ex = get_probe_measurements(mem_ppls, mem_learnability_per_ex, mem_forgetting_per_ex, mem_learnability_step_per_ex, mem_forgetting_step_per_ex, mem_init_per_ex, mem_last_per_ex, interval, relative=relative, absolute=absolute, ex_idx=ex_idx, j=j, mode='mem', once=is_once)
            gen_learnability_per_ex, gen_forgetting_per_ex, gen_init_per_ex, gen_last_per_ex, gen_learnability_step_per_ex, gen_forgetting_step_per_ex = get_probe_measurements(gen_ppls, gen_learnability_per_ex, gen_forgetting_per_ex, gen_learnability_step_per_ex, gen_forgetting_step_per_ex, gen_init_per_ex, gen_last_per_ex, interval, relative=relative, absolute=absolute, ex_idx=ex_idx, j=j, mode='gen', once=is_once)
            gen_hard_learnability_per_ex, gen_hard_forgetting_per_ex, gen_hard_last_per_ex, gen_hard_last_per_ex, gen_hard_learnability_step_per_ex, gen_hard_forgetting_step_per_ex = get_probe_measurements(gen_hard_ppls, gen_hard_learnability_per_ex, gen_hard_forgetting_per_ex, gen_hard_learnability_step_per_ex, gen_hard_forgetting_step_per_ex, gen_hard_init_per_ex, gen_hard_last_per_ex, interval, absolute=absolute, relative=relative, ex_idx=ex_idx, j=j, mode='hard-gen', once=is_once)

        if ex_idx+1 in [40, 80, 120]:
            # remove outliers
            for k in mem_learnability_per_ex.keys():
                mem_learnability_per_ex[k] = remove_outliers_iqr(mem_learnability_per_ex[k], log=log)
                gen_learnability_per_ex[k] = remove_outliers_iqr(gen_learnability_per_ex[k], log=log)
                gen_hard_learnability_per_ex[k] = remove_outliers_iqr(gen_hard_learnability_per_ex[k], log=log)
                
                gen_forgetting_per_ex[k] = remove_outliers_iqr(gen_forgetting_per_ex[k], log=log, is_retainability=k=='target')
                mem_forgetting_per_ex[k] = remove_outliers_iqr(mem_forgetting_per_ex[k], log=log, is_retainability=k=='target')
                gen_hard_forgetting_per_ex[k] = remove_outliers_iqr(gen_hard_forgetting_per_ex[k], log=log, is_retainability=k=='target')

            mem_learnability_step_per_ex = [remove_outliers_iqr([ai for ai in a if ai is not None]) for a in zip(*mem_learnability_step_per_ex)]
            gen_learnability_step_per_ex = [remove_outliers_iqr([ai for ai in a if ai is not None]) for a in zip(*gen_learnability_step_per_ex)]
            gen_hard_learnability_step_per_ex = [remove_outliers_iqr([ai for ai in a if ai is not None]) for a in zip(*gen_hard_learnability_step_per_ex)]
            
            gen_forgetting_step_per_ex = [remove_outliers_iqr([ai for ai in a if ai is not None]) for a in zip(*gen_forgetting_step_per_ex)]
            mem_forgetting_step_per_ex = [remove_outliers_iqr([ai for ai in a if ai is not None]) for a in zip(*mem_forgetting_step_per_ex)]
            gen_hard_forgetting_step_per_ex = [remove_outliers_iqr([ai for ai in a if ai is not None]) for a in zip(*gen_hard_forgetting_step_per_ex)]
                
                
            if ex_idx+1==40:
                forgetting_score["paraphrase"]["mem"] = mean(mem_forgetting_per_ex['target'])
                forgetting_score["paraphrase"]["gen"] = mean(gen_forgetting_per_ex['target'])
                forgetting_score["paraphrase"]["gen_hard"] = mean(gen_hard_forgetting_per_ex['target'])
                learnability_score["paraphrase"]["mem"] = mean(mem_learnability_per_ex['target'])
                learnability_score["paraphrase"]["gen"] = mean(gen_learnability_per_ex['target'])
                learnability_score["paraphrase"]["gen_hard"] = mean(gen_hard_learnability_per_ex['target'])
                step_forgetting_score["paraphrase"]["mem"] = [mean(a) for a in mem_forgetting_step_per_ex]
                step_forgetting_score["paraphrase"]["gen"] = [mean(a) for a in gen_forgetting_step_per_ex]
                step_forgetting_score["paraphrase"]["gen_hard"] = [mean(a) for a in gen_hard_forgetting_step_per_ex]
                step_learnability_score["paraphrase"]["mem"] = [mean(a) for a in mem_learnability_step_per_ex]
                step_learnability_score["paraphrase"]["gen"] = [mean(a) for a in gen_learnability_step_per_ex]
                step_learnability_score["paraphrase"]["gen_hard"] = [mean(a) for a in gen_hard_learnability_step_per_ex]
            elif ex_idx+1==80:
                forgetting_score["duplication"]["mem"] = mean(mem_forgetting_per_ex['target'])
                forgetting_score["duplication"]["gen"] = mean(gen_forgetting_per_ex['target'])
                forgetting_score["duplication"]["gen_hard"] = mean(gen_hard_forgetting_per_ex['target'])
                learnability_score["duplication"]["mem"] = mean(mem_learnability_per_ex['target'])
                learnability_score["duplication"]["gen"] = mean(gen_learnability_per_ex['target'])
                learnability_score["duplication"]["gen_hard"] = mean(gen_hard_learnability_per_ex['target'])
                step_forgetting_score["duplication"]["mem"] = [mean(a) for a in mem_forgetting_step_per_ex]
                step_forgetting_score["duplication"]["gen"] = [mean(a) for a in gen_forgetting_step_per_ex]
                step_forgetting_score["duplication"]["gen_hard"] = [mean(a) for a in gen_hard_forgetting_step_per_ex]
                step_learnability_score["duplication"]["mem"] = [mean(a) for a in mem_learnability_step_per_ex]
                step_learnability_score["duplication"]["gen"] = [mean(a) for a in gen_learnability_step_per_ex]
                step_learnability_score["duplication"]["gen_hard"] = [mean(a) for a in gen_hard_learnability_step_per_ex]
            elif ex_idx+1==120:
                forgetting_score["once"]["mem"] = mean(mem_forgetting_per_ex['target'])
                forgetting_score["once"]["gen"] = mean(gen_forgetting_per_ex['target'])
                forgetting_score["once"]["gen_hard"] = mean(gen_hard_forgetting_per_ex['target'])
                learnability_score["once"]["mem"] = mean(mem_learnability_per_ex['target'])
                learnability_score["once"]["gen"] = mean(gen_learnability_per_ex['target'])
                learnability_score["once"]["gen_hard"] = mean(gen_hard_learnability_per_ex['target'])
                step_forgetting_score["once"]["mem"] = [a for a in mem_forgetting_step_per_ex]
                step_forgetting_score["once"]["gen"] = [a for a in gen_forgetting_step_per_ex]
                step_forgetting_score["once"]["gen_hard"] = [a for a in gen_hard_forgetting_step_per_ex]
                step_learnability_score["once"]["mem"] = [a for a in mem_learnability_step_per_ex]
                step_learnability_score["once"]["gen"] = [a for a in gen_learnability_step_per_ex]
                step_learnability_score["once"]["gen_hard"] = [a for a in gen_hard_learnability_step_per_ex]
                
            # store mean values
            # mem_learnability.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_learnability.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_hard_learnability.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # mem_forgetting.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_forgetting.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_hard_forgetting.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})

            # print(f"memorizability: mean {mean(memorizability)} / {statistics.pstdev(memorizability)}")
            if not skip_log_learnability:
                if ex_idx+1==40:
                    print('==========\nParaphrased\n==========')
                elif ex_idx+1==80:
                    print('==========\nDuplicated\n==========')
                else:
                    print('==========\nOnce\n==========')
                    
                print(f"mem_learnability: {mean(mem_learnability_per_ex['target']):.2f}")
                # print(f"mem_init: {mean(mem_init_per_ex['target']):.2f}\nmem_last: {mean(mem_last_per_ex['target']):.2f}")
                # print(f"{statistics.pstdev(mem_learnability_per_ex['target']):.2f}")
                print('-'*50)
                print(f"gen_learnability: {mean(gen_learnability_per_ex['target']):.2f}")
                # print(f"gen_init: {mean(gen_init_per_ex['target']):.2f}\ngen_last: {mean(gen_last_per_ex['target']):.2f}")
                print('-'*50)
                print(f"gen_hard_learnability: {mean(gen_hard_learnability_per_ex['target']):.2f}")
                # print(f"gen_hard_init: {mean(gen_hard_init_per_ex['target']):.2f}\ngen_hard_last: {mean(gen_hard_last_per_ex['target']):.2f}")
                print()
                print('='*50)
                print()
            
            if ex_idx+1==120:
                break
            
            # reset values
            mem_learnability_per_ex = {'first': [], 'target': [], 'full': []}
            gen_learnability_per_ex = {'first': [], 'target': [], 'full': []}
            gen_hard_learnability_per_ex = {'first': [], 'target': [], 'full': []}
            mem_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
            gen_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
            gen_hard_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
            
            mem_learnability_step_per_ex  = []
            mem_forgetting_step_per_ex = []
            gen_learnability_step_per_ex = []
            gen_forgetting_step_per_ex = []
            gen_hard_learnability_step_per_ex = []
            gen_hard_forgetting_step_per_ex = []
    
    if skip_log_forgetting:
        with open(f"step_eval/{args.exp_name}", 'w') as f:
            json.dump({'effectivity': step_learnability_score, 'retainability': step_forgetting_score}, f, indent=4)
        with open(f"learnability_eval/{args.exp_name}", 'w') as f:
            json.dump(learnability_score, f, indent=4)
    return forgetting_score


    # print(len(gen_learnability_all_per_ex))
    # print(len(gen_learnability_easy_per_ex)+len(gen_learnability_hard_per_ex))

    # # Filter out -1 and get the indices
    # filtered_data_with_indices = [(value, index) for index, value in enumerate(gen_learnability_all_per_ex) if value != -1]

    # # Sort the list by value
    # sorted_data = sorted(filtered_data_with_indices)

    # # Get the indices of the lowest and highest 10 values
    # lowest_10_indices = [(index, value) for value, index in sorted_data[:10]]
    # top_10_indices = [(index, value) for value, index in sorted_data[-10:]]

    # # Store the information in a dictionary
    # result = {
    #     'top_10_indices': top_10_indices,
    #     'lowest_10_indices': lowest_10_indices
    # }

    # with open('indices_info.json', 'w') as f:
    #     json.dump(result, f, indent=4)

    # Fit the data to a power-law distribution

    # fit_powerlaw(pre_gen_fluc_per_ex, mode='pre_gen')
    # fit_powerlaw(gen_fluc_per_ex, mode='gen')
    # fit_powerlaw(pre_mem_fluc_per_ex, mode='pre-mem')
    # fit_powerlaw(mem_fluc_per_ex, mode='mem')


def plot_perplexity(rows, cols, plot_number, steps, x_mem, x_gen, xlabel, ylabel, scatter_data=None, local=False, x_hard_gen=None, avg=False, once=False, mode=None):
    # steps = steps[:2000]
    steps = range(0,2000)
    x_mem = x_mem[:2000]
    x_gen = x_gen[:2000]
    if x_hard_gen is not None:
        x_hard_gen = x_hard_gen[:2000]
    ax = plt.subplot(rows, cols, plot_number)
    if local:
        steps = range(-30,90)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        xlabel = 'Training Steps (t)'
        ax.set_xlabel(xlabel, fontsize=20)
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
        xlabel = xlabel.split('\n')
        xlabel = "\n".join([""+x for x in xlabel])
    if local:
        # ymin, ymax = ax.get_ylim()
        ymin, ymax = -0.2, 0.8
        ax.set_ylim(ymin, ymax)
        ax.plot(steps, [-x+x_mem[30] for x in x_mem], color='blue', label=r'$\ell(q)$')
        ax.fill_between(steps, ymin, ymax, where=(np.array(steps) >= 0) & (np.array(steps) <= 50), color='lightgreen', alpha=0.5, label='Window')
        ax.axvline(x=29, color='red', linestyle='-', linewidth=3)
        ax.annotate(r'$t_{LAM}(q,i)$', xy=(30, ymin+0.02), xytext=(30, ymin + 0.12),
             color='red', arrowprops=dict(color='red', arrowstyle='simple'), fontsize=18)
        
        # Add vertical line at x=-1
        y_value_at_t29 = -x_mem[59] + x_mem[30] + 0.2
        y_value_at_t60 = -x_mem[89] + x_mem[30] + 0.2
        #59.5
        offset = 0.012
        ax.axvline(x=-2, ymin=0.2, ymax=y_value_at_t29, color='dimgray', linestyle='-', linewidth=3)
        ax.axhline(y=y_value_at_t29-0.2, xmin=32/120-offset, xmax=33/120-offset, color='dimgray', linestyle='-', linewidth=3)
        ax.axhline(y=0, xmin=32/120-offset, xmax=33/120-offset, color='dimgray', linestyle='-', linewidth=3)
        ax.text(-3, y_value_at_t29 / 2 - 0.1, r'$\mathcal{E}(q,i)$', color='dimgray', ha='right', va='center', fontsize=18)
        
        offset = -0.455
        ax.axvline(x=59, ymin=0.2, ymax=y_value_at_t60, color='navy', linestyle='-', linewidth=3)
        ax.axhline(y=y_value_at_t60-0.2, xmin=32/120-offset, xmax=33/120-offset, color='navy', linestyle='-', linewidth=3)
        ax.axhline(y=0, xmin=32/120-offset, xmax=33/120-offset, color='navy', linestyle='-', linewidth=3)
        ax.text(71, y_value_at_t60 / 2 - 0.1, r'$\mathcal{R}(q,30)$', color='navy', ha='right', va='center', fontsize=18)
        
        ax.axhline(y=y_value_at_t29-0.2, color='black', linestyle='-.', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-.', linewidth=1)

        # Annotate the height of the line
        

    else:
        ax.plot(steps, [-x+x_mem[0] for x in x_mem], color='blue', label='Memorization')
        ax.plot(steps, [-x+x_gen[0] for x in x_gen], color='orange', label='Semantic')
    if avg:
        ax.plot(steps, [-x+x_hard_gen[0] for x in x_hard_gen], color='red', label='Composition')
    if scatter_data:
        x_vals, y_vals, colors, sizes = scatter_data
        ax.scatter(x_vals, y_vals, color=colors, s=sizes)
    
    # Set major ticks formatter and locator
    
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))  # Format tick labels as 'k' units

    # ymin, ymax = 0.0, 2.5
    ymin, ymax = -0.1, 1.8
    ax.set_ylim(ymin, ymax)
    # ymin, ymax = ax.get_ylim()
    # ymax=500
    x_positions = [0] if local or once else [steps[i] for i in [i*100 for i in range(10)]]
    plt.vlines(x=x_positions, ymin=ymin, ymax=ymax, colors='black', linestyles='dotted', label='Injection', linewidth=3)
    
    # ax.set_ylim(0, 500)
    if avg and mode=='once':
        ax.set_xlabel('Training Steps', fontsize=24)
        # pass
    if local:
        ax.set_ylabel(r'$\Delta\ell(q)$', fontsize=24)
    else:
        ax.set_ylabel(r'Avg. $\Delta\ell(q)$', fontsize=24)
    ax.grid(True)
    if not avg or mode=='dup':
        ax.legend(loc='upper right', prop={'size': 18})
    ax.tick_params(axis='both', labelsize=20)

def plot_difference(rows, cols, plot_number, steps, x_mem, x_gen, xlabel, ylabel):
    steps = steps[:2000]
    x_mem = x_mem[:2000]
    x_gen = x_gen[:2000]
    ax = plt.subplot(rows, cols, plot_number)
    if x_gen is not None:
        differences = [g - m for m, g in zip(x_mem, x_gen)]
        color='green'
    else:
        differences = x_mem
        color='red'
    ax.plot(steps, differences, color=color)
    
    # Set major ticks formatter and locator
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))  # Format tick labels as 'k' units

    ymin, ymax = ax.get_ylim()
    # ymax=500
    x_positions = [steps[i] for i in [i*100 for i in range(10)]]
    plt.vlines(x=x_positions, ymin=ymin, ymax=ymax, colors='black', linestyles='dotted', label='Injection')

    # ax.set_ylim(0, 500)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_ppl_with_trained_at(result, save_dir, min_step, draw_single, draw_avg):
    
    steps = [data["step"] for data in result]
    all_mem_ppls = []
    all_gen_ppls = []
    all_hard_gen_ppls = []
    
    # gen_ppls = [instance["ppl_probe"] for instance in result]
    keys = ['mem_first', 'mem_target', 'mem_full', 'gen_first', 'gen_target', 'gen_full', 'gen_hard_first', 'gen_hard_target', 'gen_hard_full', 'def']
    ppl_data = {key: [instance[key] for instance in result] for key in keys}
    # print(len(ppl_data['gen_hard_full']),len(ppl_data['gen_hard_full'][0]),len(ppl_data['gen_hard_full'][0][0]))
    for key in ppl_data:
        # if key != 'def':
        ppl_data[key] = list(map(list, zip(*ppl_data[key])))
    # print(len(ppl_data['gen_hard_full'][0][0]))

    # plt.figure(figsize=(16, 20))
    with open(os.path.join(args.base_dir, 'fictional_knowledge/fictional_knowledge_paraphrased.json'), 'r') as f:
        dataset = json.load(f)
    
    # assert len(dataset)==len(all_gen_ppls[0])
    if draw_avg:
        par_ppl_mem_avg = np.mean(np.array([[mean(d) for d in ppl_data["mem_target"][ex_idx]] for ex_idx in range(0,40)]), axis=0)
        par_ppl_gen_avg = np.mean(np.array([[mean(d) for d in ppl_data["gen_target"][ex_idx]] for ex_idx in range(0,40)]), axis=0)
        par_ppl_hard_gen_avg = np.mean(np.array([[mean(d) for d in ppl_data["gen_hard_target"][ex_idx]] for ex_idx in range(0,40)]), axis=0)
        
        dup_ppl_mem_avg = np.mean(np.array([[mean(d) for d in ppl_data["mem_target"][ex_idx]] for ex_idx in range(40,80)]), axis=0)
        dup_ppl_gen_avg = np.mean(np.array([[mean(d) for d in ppl_data["gen_target"][ex_idx]] for ex_idx in range(40,80)]), axis=0)
        dup_ppl_hard_gen_avg = np.mean(np.array([[mean(d) for d in ppl_data["gen_hard_target"][ex_idx]] for ex_idx in range(40,80)]), axis=0)
        
        once_ppl_mem_avg = np.mean(np.array([[mean(d) for d in ppl_data["mem_target"][ex_idx]] for ex_idx in range(80,120)]), axis=0)
        once_ppl_gen_avg = np.mean(np.array([[mean(d) for d in ppl_data["gen_target"][ex_idx]] for ex_idx in range(80,120)]), axis=0)
        once_ppl_hard_gen_avg = np.mean(np.array([[mean(d) for d in ppl_data["gen_hard_target"][ex_idx]] for ex_idx in range(80,120)]), axis=0)
        
        plt.figure(figsize=(30, 4))
        plot_perplexity(1, 1, 1, steps, par_ppl_mem_avg, par_ppl_gen_avg, '', 'Log Probability', x_hard_gen=par_ppl_hard_gen_avg, avg=True, mode='par')
        plt.savefig(os.path.join(save_dir, args.exp_name[:-5]+'_par.pdf'), bbox_inches='tight')
        plt.close()
    
        plt.figure(figsize=(30, 4))
        plot_perplexity(1, 1, 1, steps, dup_ppl_mem_avg, dup_ppl_gen_avg, '', 'Log Probability', x_hard_gen=dup_ppl_hard_gen_avg, avg=True, mode='dup')
        plt.savefig(os.path.join(save_dir, args.exp_name[:-5]+'_dup.pdf'), bbox_inches='tight')
        plt.close()
    
        plt.figure(figsize=(30, 4))
        plot_perplexity(1, 1, 1, steps, once_ppl_mem_avg, once_ppl_gen_avg, '', 'Log Probability', x_hard_gen=once_ppl_hard_gen_avg, avg=True, once=True, mode='once')
        plt.savefig(os.path.join(save_dir, args.exp_name[:-5]+'_once.pdf'), bbox_inches='tight')
        plt.close()
    
        plt.figure(figsize=(30, 4))
        plot_difference(1, 1, 1, steps, once_ppl_mem_avg, once_ppl_gen_avg, 'Step', 'Log Probability')
        plt.savefig(os.path.join(save_dir, args.exp_name[:-5]+'_diff.pdf'), bbox_inches='tight')
        plt.close()
    
    
    else:
        for ex_idx in tqdm(range(len(ppl_data['def']))):
            num_probes = 5  # Assuming all results have the same structure

            probes = list(zip([dataset[ex_idx]["mem_input"][i] + " " + "\""+dataset[ex_idx]["mem_target"][i]+"\"" for i in range(5)], [dataset[ex_idx]["gen_input"][i] + " " + "\""+dataset[ex_idx]["gen_target"][i]+"\"" for i in range(5)], [dataset[ex_idx]["hard_gen_input"][i] + " " + "\""+dataset[ex_idx]["hard_gen_target"][i]+"\"" for i in range(5)]))
            texts = [f"Mem probe: {m}\nGen_probe: {g}" for (m, g, h) in probes]
            hard_texts = [f"Hard_Gen_probe: {h}" for (m, g, h) in probes]
            if draw_single:
                if ex_idx!=6:
                    continue
                plt.figure(figsize=(15, 4))
                ppl_mem_target = [d[0] for d in ppl_data["mem_target"][ex_idx]]
                ppl_gen_target = [d[0] for d in ppl_data["gen_target"][ex_idx]]
                plot_perplexity(1, 1, 1, steps[170:290], ppl_mem_target[170:290], ppl_gen_target[170:290], '', 'Log Probability', local=True)
            else:
                plt.figure(figsize=(60, 30))
            # plt.figure(figsize=(16, 30))
                    
                for j in range(num_probes):
                    # Generate subplot indices for the current subplot row
                    first_subplot_idx = 2 + j * 6
                    target_subplot_idx = 1 + j * 6
                    full_subplot_idx = 3 + j * 6
                    hard_gen_subplot_idx = 4 + j * 6

                    # Get data for current probe across all examples
                    ppl_mem_first = [d[j] for d in ppl_data["mem_first"][ex_idx]]
                    ppl_mem_target = [d[j] for d in ppl_data["mem_target"][ex_idx]]
                    ppl_mem_full = [d[j] for d in ppl_data["mem_full"][ex_idx]]
                    
                    ppl_gen_first = [d[j] for d in ppl_data["gen_first"][ex_idx]]
                    ppl_gen_target = [d[j] for d in ppl_data["gen_target"][ex_idx]]
                    ppl_gen_full = [d[j] for d in ppl_data["gen_full"][ex_idx]]
                    
                    ppl_hard_gen_first = [d[j] for d in ppl_data["gen_hard_first"][ex_idx]]
                    ppl_hard_gen_target = [d[j] for d in ppl_data["gen_hard_target"][ex_idx]]
                    ppl_hard_gen_full = [d[j] for d in ppl_data["gen_hard_full"][ex_idx]]
                    
                    ppl_def = [ppl_data["def"][ex_idx]]

                    # Plot perplexity and differences
                    plot_perplexity(num_probes+1, 6, target_subplot_idx, steps, ppl_mem_target, ppl_gen_target, texts[j], 'Perplexity (Target)')
                    plot_perplexity(num_probes+1, 6, first_subplot_idx, steps, ppl_mem_first, ppl_gen_first, '', 'Perplexity (First)')
                    plot_perplexity(num_probes+1, 6, full_subplot_idx, steps, ppl_mem_full, ppl_gen_full, '', 'Perplexity (Full)')
                    # plot_difference(num_probes+1, 4, diff_subplot_idx, steps, x_mem, x_gen, '', 'Difference Perplexity')
                    plot_difference(num_probes+1, 6, hard_gen_subplot_idx, steps, ppl_hard_gen_target, None, hard_texts[j], 'Hard-Gen Perplexity (Target)')
                    plot_difference(num_probes+1, 6, hard_gen_subplot_idx+1, steps, ppl_hard_gen_first, None, '', 'Hard-Gen Perplexity (First)')
                    plot_difference(num_probes+1, 6, hard_gen_subplot_idx+2, steps, ppl_hard_gen_full, None, '', 'Hard-Gen Perplexity (Full)')
                
                def_idx = 1+5*6
                plot_difference(num_probes+1, 6, def_idx, steps, ppl_def[0], None, '', 'Def Perplexity')
                
                plt.tight_layout()  # Adjust layout to make room for all plots
            
            # Annotate each row with descriptive text
            # for i, text in enumerate(texts):
            #     plt.figtext(0.9, 0.15 * (len(texts) - i), text, fontsize=12)
            

            # Save the figure to a file
            plt.savefig(os.path.join(save_dir, args.exp_name[:-5], str(ex_idx)+'.pdf'), bbox_inches='tight')
            plt.close()

def preprocess_result(result):
    new_result = []
    for res in result:
        instance = {k: v for (k,v) in res.items()}
        # ['step', 'mem_first', 'mem_target', 'mem_full', 'gen_first', 'gen_target', 'gen_full', 'gen_hard_first', 'gen_hard_target', 'gen_hard_full', 'def']
        for k in instance.keys():
            if k=='step':
                continue
            if k=='def':
                instance[k] = [np.exp(i) for i in instance[k]]
            else:
                instance[k] = [[np.exp(i) for i in instance[k][j]] for j in range(len(instance[k]))]
        new_result.append(instance)
    return new_result

def main(args):

    measure_indices = range(156)

    result=load_json(os.path.join(args.base_dir, 'analysis/results', args.exp_name))
    if not args.no_take_exp:
        print('Take exp to get ppl values')
        result = preprocess_result(result=result)
    else:
        print('Analysis performed based on CE loss')
    min_step = min([int(d["step"]) for d in result])
    print(min_step)

    if args.mode=='draw_figures':
        os.makedirs(args.save_dir, exist_ok=-True)
        plot_indices = range(156,196)
        plot_ppl_with_trained_at(result, save_dir=args.save_dir, min_step=min_step, draw_single=args.draw_single, draw_avg=args.draw_avg)
    
    elif args.mode=='measure_scores':

        if args.skip_log_forgetting:
            measure_scores(result, 
                        interval=50, 
                        skip_log_learnability=args.skip_log_learnability, 
                        skip_log_forgetting=args.skip_log_forgetting,
                        relative=args.relative, 
                        absolute=args.absolute)
        else:
            print(len(result))
            interval = len(result)-950
            print(f"Max interval: {interval}")
            
            total_result = {"duplication": {'mem': [], 'gen': [], 'gen_hard': []}, 
                            "paraphrase": {'mem': [], 'gen': [], 'gen_hard': []}, 
                            "once": {'mem': [], 'gen': [], 'gen_hard': []}}
            for i in tqdm(range(interval)):
                single_result = measure_scores(result, 
                        interval=i, 
                        skip_log_learnability=args.skip_log_learnability, 
                        skip_log_forgetting=args.skip_log_forgetting,
                        relative=args.relative, 
                        absolute=args.absolute)
                
                for k, v in single_result.items():
                    total_result[k]["mem"].append(v["mem"])
                    total_result[k]["gen"].append(v["gen"])
                    total_result[k]["gen_hard"].append(v["gen_hard"])
                    
                # global NUM_REMOVED
                # print(NUM_REMOVED)
                    
            os.makedirs('forgetting_measurements', exist_ok=True)
            ppl_or_loss = 'loss' if args.no_take_exp else 'ppl'
            absolute_or_percentage = 'absolute' if args.absolute else 'percentage'
            learnability_or_forgetting = 'learnability' if args.relative else 'forgetting'
            # fname = f'forgetting_measurements/{args.exp_name[:-5]}_{ppl_or_loss}_{absolute_or_percentage}_{learnability_or_forgetting}.json'
            fname = f'forgetting_measurements/{args.exp_name[:-5]}_{ppl_or_loss}_regularized.json'
            with open(fname, 'w') as f:
                json.dump(total_result, f, indent=4)
                
            print(f"num_removed: {NUM_REMOVED}")
            print(f"percentage: {(NUM_REMOVED/(interval*15*120))*100}")

    else:
        raise NotImplementedError
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # exp_name = "ft_medium_8e-6"
    # data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"


    # Add arguments
    parser.add_argument('--base_dir', type=str, default='/home/hoyeon/OLMo')
    parser.add_argument('--save_dir', type=str, default="figs")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--mode', type=str, default="draw_figures")
    parser.add_argument('--no_take_exp', action='store_true')
    parser.add_argument('--skip_log_learnability', action='store_true')
    parser.add_argument('--skip_log_forgetting', action='store_true')
    parser.add_argument('--draw_single', action='store_true')
    parser.add_argument('--draw_avg', action='store_true')
    parser.add_argument('--relative', action='store_true')
    parser.add_argument('--absolute', action='store_true')

    # Parse the arguments
    args = parser.parse_args()
    
    main(args)
    
