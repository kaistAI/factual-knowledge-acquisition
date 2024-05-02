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


def fraction_larger_than_first(lst):
    if len(lst) <= 1:
        return 0  # Avoid division by zero if the list has one or no elements
    
    first_value = lst[0]
    count_larger = sum(1 for x in lst[1:] if x > first_value)
    
    return count_larger / (len(lst) - 1)


def check_success(ppls, threshold=0.2, debug=False):
    # print(ppls)
    # print((ppls[0]-min(ppls[:50])), '\t', abs(max(ppls[-50:])-min(ppls[-50:])))
    # assert False
    if (ppls[0]-min(ppls[:25])) > 1.5*(abs(max(ppls[-25:])-min(ppls[-25:]))):
        # if fraction_larger_than_first(ppls)<threshold:
        return True
    else:
        if debug:
            if ppls[0]>ppls[1]:
                print("Fail due to the initial ppl increase")
            if fraction_larger_than_first(ppls)<threshold:
                print("Fail due to the threshold condition")
        return False


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


def draw_violin_contain(contain, hard_contain, ppl, ppl_hard, interval, exp_name):

    assert len(contain)==len(ppl) and len(hard_contain)==len(ppl_hard)

    contain_combined = contain + hard_contain
    contain_group = []
    for c in contain_combined:
        if c:
            contain_group.append('Contianed')
        else:
            contain_group.append('Not contained')
    ppl_combined = ppl + ppl_hard

    df = pd.DataFrame({'category': contain_group, 'value': ppl_combined})

    # Calculate the Q1, Q3, and IQR for each category
    Q1 = df.groupby('category')['value'].quantile(0.25)
    Q3 = df.groupby('category')['value'].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data based on the IQR criterion for each category
    filtered_df = df.groupby('category').apply(
        lambda group: group[(group['value'] >= lower_bound[group.name]) & (group['value'] <= upper_bound[group.name])]
    ).reset_index(drop=True)

# 
    # fig, axes = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1]})
# 
    # Violin plot
    sns.violinplot(x='category', y='value', data=filtered_df)
    # axes[0, 0].set_xlabel('Bins of sim (intervals)')
    # axes[0, 0].set_ylabel('ppl values')
    # axes[0, 0].tick_params(axis='x', rotation=45)

    # # Histogram
    # sns.histplot(ax=axes[1, i], x=sim_filtered, bins=bins, kde=False)
    # axes[1, 0].set_ylabel('Count')
    # axes[1, 0].set_xticks(bins)
    # axes[1, 0].tick_params(axis='x', rotation=45)

    # plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(f"violin/{exp_name}", exist_ok=True)
    filename = f"violin/{exp_name}/{interval}_contain.png"
    plt.savefig(filename)


# def draw_violin_len(ppl, mode, interval, exp_name, train_indices):
#     with open('/mnt/nas/hoyeon/trl-pretrain/scratch/length_data.json', 'r') as f:
#         len_data = json.load(f)
#     # Setup the figure with subplots
    
#     fig, axes = plt.subplots(2, 1, figsize=(24, 16), gridspec_kw={'height_ratios': [3, 1]})

#     len_filtered = []
#     for i, d in enumerate(len_data):
#         if len(train_indices[i])>0:
#             if mode == 'all':
#                 len_filtered.extend(d)
#             elif mode == 'hard':
#                 len_filtered.extend(d[5:])
#             else:
#                 len_filtered.extend(d[:5])


    # print(len(ppl))
    # print(len(len_filtered))

    for i, (len_filtered, ppl_filtered) in enumerate([(len_filtered, ppl)]):
        # Binning the data
        bins = np.linspace(min(len_filtered), max(len_filtered), num=9)
        bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
        bin_map = {label: bins[i] for i, label in enumerate(bin_labels)}
        sim_binned = np.digitize(len_filtered, bins, right=True)
        sim_binned = [min(i, len(bin_labels)) for i in sim_binned]

        # Creating a DataFrame for plotting
        data = pd.DataFrame({
            'Group': [bin_labels[i-1] for i in sim_binned],
            'Value': ppl_filtered
        })

        # Add a numerical column for sorting
        data['SortKey'] = data['Group'].map(bin_map)
        data.sort_values('SortKey', inplace=True)

        # Now use 'Group' for plotting labels but sort by 'SortKey'
        sns.violinplot(ax=axes[0], x='Group', y='Value', data=data, order=sorted(data['Group'].unique(), key=lambda x: bin_map[x]))
        axes[0].set_xlabel('Length')
        axes[0].set_ylabel('ppl values')
        axes[0].tick_params(axis='x', rotation=45)

        # Histogram
        sns.histplot(ax=axes[1], x=len_filtered, bins=bins, kde=False)
        # axes[1, i].set_title(f'Histogram of {label}')
        # axes[1, i].set_xlabel(label)
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(bins)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(f"violin/{exp_name}", exist_ok=True)
    filename = f"violin/{exp_name}/{interval}_{mode}_len.png"
    plt.savefig(filename)



def draw_violin(sim_dict, ppl, ppl_success, hard, interval, exp_name):
    sim_jaccard = sim_dict["jaccard"]
        
    sim_all_filtered, ppl_all_filtered = filter_data(ppl, sim_jaccard)
    sim_success_filtered, ppl_success_filtered = filter_data(ppl_success, sim_jaccard)

    # Setup the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(24, 16), gridspec_kw={'height_ratios': [3, 1]})

    for i, (sim_filtered, ppl_filtered) in enumerate([(sim_all_filtered, ppl_all_filtered), (sim_success_filtered, ppl_success_filtered)]):
        # Binning the data
        bins = np.linspace(min(sim_filtered), max(sim_filtered), num=9)
        bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
        sim_binned = np.digitize(sim_filtered, bins, right=True)
        sim_binned = [min(i, len(bin_labels)) for i in sim_binned]

        # Creating a DataFrame for plotting
        data = pd.DataFrame({
            'Group': [bin_labels[i-1] for i in sim_binned],
            'Value': ppl_filtered
        })

        data.sort_values('Group', inplace=True)
        
        # Violin plot
        sns.violinplot(ax=axes[0, i], x='Group', y='Value', data=data)
        # axes[0, i].set_title(f'Violin plo with outliers removed')
        axes[0, i].set_xlabel('Bins of sim (intervals)')
        axes[0, i].set_ylabel('ppl values')
        axes[0, i].tick_params(axis='x', rotation=45)

        # Histogram
        sns.histplot(ax=axes[1, i], x=sim_filtered, bins=bins, kde=False)
        # axes[1, i].set_title(f'Histogram of {label}')
        # axes[1, i].set_xlabel(label)
        axes[1, i].set_ylabel('Count')
        axes[1, i].set_xticks(bins)
        axes[1, i].tick_params(axis='x', rotation=45)

        plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(f"violin/{exp_name}", exist_ok=True)
    filename = f"violin/{exp_name}/{interval}_{'hard' if hard else 'easy'}.png"
    plt.savefig(filename)


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


def spectrum_analysis(values):
    """
    Perform linear detrending and Fourier analysis on a time-series data.

    :param values: List of floats representing the time-series data.
    :return: Plot of the frequency spectrum.
    """

    # Time parameters (assuming equal spacing)
    N = len(values)  # Number of data points
    T = 1.0 / N  # Assuming unit time interval between data points

    # Linear Detrending
    times = np.arange(N)
    detrended = values - np.poly1d(np.polyfit(times, values, 1))(times)

    # Fourier Transform
    freq_values = fftpack.fft(detrended)
    freqs = fftpack.fftfreq(N, T)
    freq_magnitudes = np.abs(freq_values) * 1 / N

    # Normalizing to make the area under the curve 1
    total_area = np.sum(freq_magnitudes) * (freqs[1] - freqs[0])  # Approximate the integral
    normalized_magnitudes = freq_magnitudes / total_area
    
    # Plotting the Frequency Spectrum
    # plt.figure(figsize=(10, 5))
    # plt.plot(freqs[:N // 2][1:], normalized_magnitudes[:N // 2][1:])  # Plot only the positive frequencies
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # plt.title('Frequency Spectrum')
    # plt.grid(True)
    # plt.show()
    # plt.savefig('spectrum_mem.png')
    return freqs[:N // 2][1:], normalized_magnitudes[:N // 2][1:]


def remove_outliers_iqr(data, multiplier=1.5, log=False):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    if log:
        print(f"{len(data)-len(filtered_data)}/{len(data)} datapoints removed")
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


def get_perturb_indices(l, max_len=500, margin=25):
    if len(l)==0:
        return []
    else:
        result = []
        for i in range(len(l)-1):
            if l[i]+margin<l[i+1]:
                result.extend(list(range(l[i]+margin,l[i+1])))
        if l[-1]<max_len-margin:
            result.extend(list(range(l[-1]+margin,max_len)))

        return result


def fit_powerlaw(raw_data, mode):
    # Fit data to a power-law distribution
    data = [abs(d) for d in raw_data]
    fit = powerlaw.Fit(data)
    alpha = fit.alpha
    xmin = fit.xmin

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # First subplot: Power-law PDF
    fit.plot_pdf(color='b', linestyle='-', ax=axs[0])
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=axs[0])

    # Compare power-law fit to an exponential distribution
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f'{mode} - Likelihood Ratio: {R}, p-value: {p}')

    axs[0].set_title(f'Power-law fit: α={alpha:.2f}, xmin={xmin}')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Probability Density Function')
    axs[0].text(0.6, 0.95, f'p-value: {p:.6f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

    # Second subplot: Histogram of data with mean and standard deviation
    mean_data = np.mean(data)
    std_data = np.std(data)
    median_data = np.median(data)
    bins = np.linspace(0, 2, 21)
    axs[1].hist(data, bins=bins, edgecolor='black')
    axs[1].set_title('Histogram of Data')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')
    # Display mean and standard deviation
    axs[1].text(0.6, 0.9, f'Mean: {mean_data:.2f}\nMedian: {median_data:.2f}\nStd: {std_data:.2f}', transform=axs[1].transAxes, fontsize=12, verticalalignment='top')

    # Third subplot: Histogram of raw data with mean and standard deviation
    mean_raw_data = np.mean(raw_data)
    std_raw_data = np.std(raw_data)
    median_raw_data = np.median(raw_data)
    bins = np.linspace(-2, 2, 41)
    axs[2].hist(raw_data, bins=bins, edgecolor='black')
    axs[2].set_title('Histogram of Raw Data')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Frequency')
    # Display mean and standard deviation
    axs[2].text(0.6, 0.9, f'Mean: {mean_raw_data:.2f}\nMedian: {median_raw_data:.2f}\nStd: {std_raw_data:.2f}', transform=axs[2].transAxes, fontsize=12, verticalalignment='top')


    # Save the figure
    plt.savefig(f'powerlaw/{args.exp_name[0]}_{mode}.png')
    np.save(f'powerlaw/raw/{args.exp_name[0]}_{mode}.npy')

    # Show plot
    plt.show()


def get_probe_measurements(ppls, learnability_per_ex, forgetting_per_ex, interval, margin=50, relative=False, absolute=False, ex_idx=-1):
    
    # Find the stabilized point
    last_train_idx=900 if ex_idx<80 else 0#Hard-coded
    for k, v in ppls.items():
        if k=='def':
            continue
        
        values=v[last_train_idx:last_train_idx+margin]
        sp=min(range(len(values)), key=values.__getitem__)+last_train_idx
        # min_ppl=min(ppls[train_idx[-1]:train_idx[-1]+margin])
        # min_ppl=mean(v[sp-10:sp+10])
        min_ppl = v[sp]
        # init_ppl=ppls[train_idx[-1]-1]
        init_ppl=v[0]
        last_ppl=v[sp+interval]
        
        if not absolute:
            learnability_per_ex[k].append((1-min_ppl/init_ppl)*100)
        else:
            learnability_per_ex[k].append(init_ppl-min_ppl)
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

    return learnability_per_ex, forgetting_per_ex


def measure_scores(result, interval=1000, skip_log_learnability=False, relative=False, absolute=False):

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
    
    mem_learnability = []
    gen_learnability = []
    gen_hard_learnability = []
    mem_forgetting = []
    gen_forgetting = []
    gen_hard_forgetting = []
    
    for ex_idx in tqdm(range(len(mem_probe_ppls['target']))):

        # print('Warning: train_idx and n_probes are hard-coded!')
        train_idx = [i*100 for i in range(10)] #Hard-coded
        n_probes = 5 #Hard-coded

        for j in range(n_probes):
            mem_ppls = {k: [d[j] for d in v[ex_idx]] for (k, v) in mem_probe_ppls.items() if k!='def'}
            gen_ppls = {k: [d[j] for d in v[ex_idx]] for (k, v) in gen_probe_ppls.items()}
            gen_hard_ppls = {k: [d[j] for d in v[ex_idx]] for (k, v) in gen_hard_probe_ppls.items()}
            
            mem_learnability_per_ex, mem_forgetting_per_ex = get_probe_measurements(mem_ppls, mem_learnability_per_ex, mem_forgetting_per_ex, interval, relative=relative, absolute=absolute, ex_idx=ex_idx)
            gen_learnability_per_ex, gen_forgetting_per_ex = get_probe_measurements(gen_ppls, gen_learnability_per_ex, gen_forgetting_per_ex, interval, relative=relative, absolute=absolute, ex_idx=ex_idx)
            gen_hard_learnability_per_ex, gen_hard_forgetting_per_ex = get_probe_measurements(gen_hard_ppls, gen_hard_learnability_per_ex, gen_hard_forgetting_per_ex, interval, absolute=absolute, relative=relative, ex_idx=ex_idx)

        if ex_idx+1 in [40, 80, 120]:
            # remove outliers
            for k in mem_learnability_per_ex.keys():
                mem_learnability_per_ex[k] = remove_outliers_iqr(mem_learnability_per_ex[k], log=False)
                gen_learnability_per_ex[k] = remove_outliers_iqr(gen_learnability_per_ex[k], log=False)
                gen_hard_learnability_per_ex[k] = remove_outliers_iqr(gen_hard_learnability_per_ex[k], log=False)
                
            # store mean values
            # mem_learnability.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_learnability.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_hard_learnability.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # mem_forgetting.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_forgetting.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})
            # gen_hard_forgetting.append({k: mean(v) for (k, v) in mem_learnability_per_ex.items()})

            if ex_idx+1==40:
                print('==========\nParaphrased\n==========')
            elif ex_idx+1==80:
                print('==========\nDuplicated\n==========')
            else:
                print('==========\nOnce\n==========')
                
            # print(f"memorizability: mean {mean(memorizability)} / {statistics.pstdev(memorizability)}")
            if not skip_log_learnability:
                print(f"mem_learnability: {mean(mem_learnability_per_ex['target']):.2f}")
                # print(f"{statistics.pstdev(mem_learnability_per_ex['target']):.2f}")
                print('-'*50)
                print(f"gen_learnability: {mean(gen_learnability_per_ex['target']):.2f}")
                print('-'*50)
                print(f"gen_hard_learnability: {mean(gen_hard_learnability_per_ex['target']):.2f}")
                print()
                print('='*50)
                print()
            print(f"mem_forgetting:\n\ttarget: {mean(mem_forgetting_per_ex['target']):.2f}")
            print('-'*50)
            print(f"gen_forgetting:\n\ttarget: {mean(gen_forgetting_per_ex['target']):.2f}")
            print('-'*50)
            print(f"gen_hard_forgetting:\n\ttarget: {mean(gen_hard_forgetting_per_ex['target']):.2f}")
            
            if ex_idx+1==120:
                break
            
            # reset values
            mem_learnability_per_ex = {'first': [], 'target': [], 'full': []}
            gen_learnability_per_ex = {'first': [], 'target': [], 'full': []}
            gen_hard_learnability_per_ex = {'first': [], 'target': [], 'full': []}
            mem_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
            gen_forgetting_per_ex = {'first': [], 'target': [], 'full': []}
            gen_hard_forgetting_per_ex = {'first': [], 'target': [], 'full': []}


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


def plot_perplexity(rows, cols, plot_number, steps, x_mem, x_gen, xlabel, ylabel, scatter_data=None):
    ax = plt.subplot(rows, cols, plot_number)
    ax.plot(steps, x_mem, color='blue', label='Memorization')
    ax.plot(steps, x_gen, color='orange', label='Generalization')
    if scatter_data:
        x_vals, y_vals, colors, sizes = scatter_data
        ax.scatter(x_vals, y_vals, color=colors, s=sizes)
    
    # Set major ticks formatter and locator
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))  # Format tick labels as 'k' units

    xlabel = xlabel.split('\n')
    new_xlabel = "\n".join([""+x for x in xlabel])
    
    ymin, ymax = ax.get_ylim()
    x_positions = [steps[i] for i in [i*100 for i in range(10)]]
    plt.vlines(x=x_positions, ymin=ymin, ymax=ymax, colors='black', linestyles='dotted', label='Injection')
    
    ax.set_xlabel(new_xlabel, loc='left')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

def plot_difference(rows, cols, plot_number, steps, x_mem, x_gen, xlabel, ylabel):
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
    x_positions = [steps[i] for i in [i*100 for i in range(10)]]
    plt.vlines(x=x_positions, ymin=ymin, ymax=ymax, colors='black', linestyles='dotted', label='Injection')

    ax.set_xlabel(xlabel, loc='left')
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_ppl_with_trained_at(result, save_dir, min_step):
    
    os.makedirs(os.path.join(save_dir, args.exp_name[:-5]), exist_ok=True)
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
    
    for ex_idx in tqdm(range(len(ppl_data['def']))):
        num_probes = 5  # Assuming all results have the same structure

        plt.figure(figsize=(60, 30))
        probes = list(zip([dataset[ex_idx]["mem_input"][i] + " " + "\""+dataset[ex_idx]["mem_target"][i]+"\"" for i in range(5)], [dataset[ex_idx]["gen_input"][i] + " " + "\""+dataset[ex_idx]["gen_target"][i]+"\"" for i in range(5)], [dataset[ex_idx]["hard_gen_input"][i] + " " + "\""+dataset[ex_idx]["hard_gen_target"][i]+"\"" for i in range(5)]))
        texts = [f"Mem probe: {m}\nGen_probe: {g}" for (m, g, h) in probes]
        hard_texts = [f"Hard_Gen_probe: {h}" for (m, g, h) in probes]
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
        plt.savefig(os.path.join(save_dir, args.exp_name[:-5], str(ex_idx)+'.png'), bbox_inches='tight')
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
        plot_ppl_with_trained_at(result, save_dir=args.save_dir, min_step=min_step)
    
    elif args.mode=='measure_scores':
        # measure_indices = list(range(len(per_ex)))
        print(f"\n\n!!!!!!!!!!!!!!!!!!\nInterval: {args.interval}\n!!!!!!!!!!!!!!!!!!\n")
        measure_scores(result, interval=args.interval, skip_log_learnability=args.skip_log_learnability, relative=args.relative, absolute=args.absolute)

    else:
        raise NotImplementedError
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # exp_name = "ft_medium_8e-6"
    # data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"


    # Add arguments
    parser.add_argument('--base_dir', type=str, default="/home/hoyeon/OLMo/")
    parser.add_argument('--save_dir', type=str, default="figs")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--mode', type=str, default="draw_figures")
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--no_take_exp', action='store_true')
    parser.add_argument('--skip_log_learnability', action='store_true')
    parser.add_argument('--relative', action='store_true')
    parser.add_argument('--absolute', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    main(args)