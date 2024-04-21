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


def draw_violin_len(ppl, mode, interval, exp_name, train_indices):
    with open('/mnt/nas/hoyeon/trl-pretrain/scratch/length_data.json', 'r') as f:
        len_data = json.load(f)
    # Setup the figure with subplots
    
    fig, axes = plt.subplots(2, 1, figsize=(24, 16), gridspec_kw={'height_ratios': [3, 1]})

    len_filtered = []
    for i, d in enumerate(len_data):
        if len(train_indices[i])>0:
            if mode == 'all':
                len_filtered.extend(d)
            elif mode == 'hard':
                len_filtered.extend(d[5:])
            else:
                len_filtered.extend(d[:5])


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
    # if log:
    #     print(f"{len(data)-len(filtered_data)}/{len(data)} datapoints removed")
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


def measure_scores(result, train_indices, premem=False, interval=10000):

    with open('similarity.json', 'r') as f:
        sim_data = json.load(f)
    with open('fictional_knowledge_contain.json', 'r') as f:
        contain_data = json.load(f)

    probe_ppls = [instance["ppl_probe"] if len(instance["ppl_probe"])>0 else [[0.0 for i in range(12)] for i in range(156)] for instance in result]
    probe_ppls = list(map(list, zip(*probe_ppls)))
    train_ppls = [instance["ppl_train"] if len(instance["ppl_train"])>0 else [0.0 for i in range(156)] for instance in result]
    train_ppls = list(map(list, zip(*train_ppls)))
    margin=50

    memorizability = []
    generalizability = []
    mem_freq_per_ex = []
    gen_freq_per_ex = []
    mem_learnability_per_ex = []
    gen_learnability_per_ex = []
    gen_learnability_all_per_ex = []
    gen_learnability_easy_per_ex = []
    gen_learnability_hard_per_ex = []
    gen_success_learnability_easy_per_ex = []
    gen_success_learnability_hard_per_ex = []
    gen_fluc_per_ex = []
    mem_fluc_per_ex = []
    similarity_easy_per_ex = {"jaccard": [], "rouge_l": []}
    similarity_hard_per_ex = {"jaccard": [], "rouge_l": []}
    freq = None
    success_count_easy=0
    success_count_hard=0
    contain = []
    hard_contain = []

    for ex_idx in tqdm(range(len(probe_ppls))):

        train_idx = train_indices[ex_idx] if not premem else []
        n_probes = len(probe_ppls[ex_idx][0])        

        # before_encounter_indices = list(range(1,train_idx[0])) if len(train_idx)>0 else list(range(1, 500))
        # perturb_indices = get_perturb_indices(train_idx)
        if len(train_idx)!=0 and not premem:
            similarity_easy_per_ex["jaccard"].extend(sim_data['normal']['jaccard'][ex_idx*5:ex_idx*5+5])
            similarity_easy_per_ex["rouge_l"].extend(sim_data['normal']['rouge_l'][ex_idx*5:ex_idx*5+5])
            similarity_hard_per_ex["jaccard"].extend(sim_data['hard']['jaccard'][ex_idx*5:ex_idx*5+5])
            similarity_hard_per_ex["rouge_l"].extend(sim_data['hard']['rouge_l'][ex_idx*5:ex_idx*5+5])

        for j in range(n_probes):
            ppls = [d[j] for d in probe_ppls[ex_idx]]
            
            if len(train_idx)!=0 and not premem:
                values=ppls[train_idx[-1]:train_idx[-1]+margin]
                sp=min(range(len(values)), key=values.__getitem__)+train_idx[-1]
                # min_ppl=min(ppls[train_idx[-1]:train_idx[-1]+margin])
                min_ppl=mean(ppls[sp-10:sp+10])
                # init_ppl=ppls[train_idx[-1]-1]
                init_ppl=ppls[train_idx[-1]-2]

                generalizability.append((1-min_ppl/init_ppl)*100)

                # Freq analysis
                # freq_x, freq_y = spectrum_analysis(ppls[sp:sp+400])
                # freq = freq_x 
                if interval>0:
                    last_ppl=ppls[round(sp+interval)]
                else:
                    last_ppl=ppls[sp+interval]
                # gen_freq_per_ex.append(freq_y)
                gen_learnability_per_ex.append((1-last_ppl/init_ppl))

                values_with_prev = ppls[train_idx[-1]-2:train_idx[-1]+margin]
                if j<5:
                    gen_learnability_easy_per_ex.append((1-last_ppl/init_ppl))
                    contain.append(contain_data[ex_idx]["contain"][j])
                    if check_success(values_with_prev):
                        success_count_easy += 1
                        gen_success_learnability_easy_per_ex.append((1-last_ppl/init_ppl))
                else:
                    gen_learnability_hard_per_ex.append((1-last_ppl/init_ppl))
                    hard_contain.append(contain_data[ex_idx]["hard_contain"][j-5])
                    if check_success(values_with_prev):
                        success_count_hard += 1
                        gen_success_learnability_hard_per_ex.append((1-last_ppl/init_ppl))
                gen_learnability_all_per_ex.append((1-last_ppl/init_ppl))
                # gen_fluc_per_ex.append(1-last_ppl/min_ppl)
                # segment = ppls[sp:sp+400]
                # gen_fluc = calculate_fluc(segment)
                # gen_fluc_per_ex.append(gen_fluc)
                gen_fluc_per_ex.append((last_ppl-min_ppl)/abs(init_ppl-min_ppl))
                # pre_gen_fluc_per_ex.append(1-ppls[99]/ppls[0])


            elif premem:
                freq_x, freq_y = spectrum_analysis(ppls[100:500])
                freq = freq_x 
                freq_x, freq_y = spectrum_analysis(ppls[100:500])
                freq = freq_x 
                gen_freq_per_ex.append(freq_y)

                values=ppls[500:500+margin]
                sp=min(range(len(values)), key=values.__getitem__)+500
                # min_ppl=min(train_ppl[train_idx[-1]:train_idx[-1]+margin])
                min_ppl=mean(ppls[sp-10:sp+10])
                
                gen_fluc_per_ex.append((ppls[round(500+interval)]-min_ppl)/abs(ppls[500]-min_ppl))

            else:
                gen_learnability_all_per_ex.append(-1)


        if len(train_idx)!=0 and not premem:
            train_ppl = train_ppls[ex_idx]
            # if n_probes>0:
            if True:
                values=train_ppl[train_idx[-1]:train_idx[-1]+margin]
                sp=min(range(len(values)), key=values.__getitem__)+train_idx[-1]
                # min_ppl=min(train_ppl[train_idx[-1]:train_idx[-1]+margin])
                min_ppl=mean(train_ppl[sp-10:sp+10])

                # last_ppl=train_ppl[-1]
                init_ppl=train_ppl[train_idx[-1]-1]

                memorizability.append((1-min_ppl/init_ppl)*100)

                # Frequency analysis
                _, freq_y = spectrum_analysis(train_ppl[sp:sp+400])
                mem_freq_per_ex.append(freq_y)
                if interval>0:
                    last_ppl=train_ppl[round(sp+interval)]
                else:
                    last_ppl=train_ppl[sp+interval]
                mem_learnability_per_ex.append((1-last_ppl/init_ppl))
                # if mem_learnability < 5:
                #     pass
                # segment = train_ppl[sp:sp+400]
                # mem_fluc = calculate_fluc(segment)
                # mem_fluc_per_ex.append(mem_fluc)
                mem_fluc_per_ex.append((last_ppl-min_ppl)/abs(init_ppl-min_ppl))
                # pre_mem_fluc_per_ex.append(1-train_ppl[99]/train_ppl[0])
                # print(f"last ppl: {last_ppl} / min ppl: {min_ppl} / init ppl: {init_ppl}")
        
            else:
                pass
                # train_ppl = train_ppls[ex_idx]
                # pre_mem_fluc_per_ex.append(abs(1-train_ppl[400]/train_ppl[0]))

        elif premem:
            train_ppl = train_ppls[ex_idx]
            freq_x, freq_y = spectrum_analysis(train_ppl[100:500])
            freq = freq_x 
            freq_x, freq_y = spectrum_analysis(train_ppl[100:500])
            freq = freq_x 
            mem_freq_per_ex.append(freq_y)

            values=train_ppl[500:500+margin]
            sp=min(range(len(values)), key=values.__getitem__)+500
            # min_ppl=min(train_ppl[train_idx[-1]:train_idx[-1]+margin])
            min_ppl=mean(train_ppl[sp-10:sp+10])

            mem_fluc_per_ex.append((train_ppl[round(500+interval)]-min_ppl)/abs(train_ppl[500]-min_ppl))

    # draw_violin_len(gen_learnability_easy_per_ex, mode='easy', interval=args.interval, exp_name=args.exp_name[0][:-5], train_indices=train_indices)
    # draw_violin_len(gen_learnability_hard_per_ex, mode='hard', interval=args.interval, exp_name=args.exp_name[0][:-5], train_indices=train_indices)
    # draw_violin_len(gen_learnability_all_per_ex, mode='all', interval=args.interval, exp_name=args.exp_name[0][:-5], train_indices=train_indices)
    # draw_violin_contain(contain, hard_contain, gen_learnability_easy_per_ex, gen_learnability_hard_per_ex, interval=args.interval, exp_name=args.exp_name[0][:-5])

    # remove outliers
    if not premem:
        # print(memorizability)
        # print(generalizability)
        memorizability = remove_outliers_iqr(memorizability)
        # train_volatility_per_ex = remove_outliers_iqr(train_volatility_per_ex)
        if len(generalizability)>0:
            generalizability = remove_outliers_iqr(generalizability)
        # volatility_per_ex = remove_outliers_iqr(volatility_per_ex)
        mem_learnability_per_ex = remove_outliers_iqr(mem_learnability_per_ex, log=True)
        mem_fluc_per_ex = remove_outliers_iqr(mem_fluc_per_ex)
        # if len(gen_learnability_per_ex)>0:
        orig_len = len(gen_learnability_easy_per_ex)
        gen_learnability_per_ex = remove_outliers_iqr(gen_learnability_per_ex, log=True)
        gen_learnability_easy_per_ex = remove_outliers_iqr(gen_learnability_easy_per_ex, log=True)
        gen_learnability_hard_per_ex = remove_outliers_iqr(gen_learnability_hard_per_ex, log=True)
        gen_success_learnability_easy_per_ex = remove_outliers_iqr(gen_success_learnability_easy_per_ex, log=True)
        gen_success_learnability_hard_per_ex = remove_outliers_iqr(gen_success_learnability_hard_per_ex, log=True)

    # Plot averqge frequency spectrum
    mem_freq = mean_of_arrays(mem_freq_per_ex)
    if len(gen_freq_per_ex)>0:
        gen_freq = mean_of_arrays(gen_freq_per_ex)
        

    
    if not premem:
        # print(f"memorizability: mean {mean(memorizability)} / {statistics.pstdev(memorizability)}")
        print(f"mem_learnability: {mean(mem_learnability_per_ex)}")
        # print(f"mem_learnability_stdev: {statistics.pstdev(mem_learnability_per_ex)}")
        # print(f"mem_fluc: {mean(mem_fluc_per_ex)}")
        # print(f"train volatility: mean {mean(train_volatility_per_ex)} / {statistics.pstdev(train_volatility_per_ex)}")
        print()
        # print(f"generalizability: mean {mean(generalizability)} / {statistics.pstdev(generalizability)}")
        # print(f"gen_learnability: {mean(gen_learnability_per_ex)}")
        # print(f"gen_learnability_stdev: {statistics.pstdev(gen_learnability_per_ex)}")
        # print()
        print(f"gen_learnability_easy: {mean(gen_learnability_easy_per_ex)}")
        # print(f"gen_success_learnability_easy: {mean(gen_success_learnability_easy_per_ex)}")
        # print(f"gen_learnability_easy_stdev: {statistics.pstdev(gen_learnability_per_ex)}")
        print()
        print(f"gen_learnability_hard: {mean(gen_learnability_hard_per_ex)}")
        # print(f"gen_success_learnability_hard: {mean(gen_success_learnability_hard_per_ex)}")
        # print()
        # print(f"easy success fraction: {success_count_easy/orig_len}")
        # print(f"hard success fraction: {success_count_hard/orig_len}")
        # print(f"gen_learnability_hard_stdev: {statistics.pstdev(gen_learnability_per_ex)}")
        # print(f"gen_fluc: {mean(gen_fluc_per_ex)}")
        # print(f"len_notrain: {len(pre_mem_fluc_per_ex)}")
    # print(f"gen volatility: mean {mean(volatility_per_ex)} / {statistics.pstdev(volatility_per_ex)}")

    print(len(gen_learnability_all_per_ex))
    print(len(gen_learnability_easy_per_ex)+len(gen_learnability_hard_per_ex))

    # Filter out -1 and get the indices
    filtered_data_with_indices = [(value, index) for index, value in enumerate(gen_learnability_all_per_ex) if value != -1]

    # Sort the list by value
    sorted_data = sorted(filtered_data_with_indices)

    # Get the indices of the lowest and highest 10 values
    lowest_10_indices = [(index, value) for value, index in sorted_data[:10]]
    top_10_indices = [(index, value) for value, index in sorted_data[-10:]]

    # Store the information in a dictionary
    result = {
        'top_10_indices': top_10_indices,
        'lowest_10_indices': lowest_10_indices
    }

    with open('indices_info.json', 'w') as f:
        json.dump(result, f, indent=4)

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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))  # Format tick labels as 'k' units

    xlabel = xlabel.split('\n')
    new_xlabel = "\n".join([""+x for x in xlabel])
    
    ymin, ymax = ax.get_ylim()
    x_positions = [steps[i] for i in [i*100 for i in range(3)]]
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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))  # Format tick labels as 'k' units

    ymin, ymax = ax.get_ylim()
    x_positions = [steps[i] for i in [i*100 for i in range(3)]]
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


def main(args):

    measure_indices = range(156)

    result=load_json(os.path.join(args.base_dir, 'analysis/results', args.exp_name))
    min_step = min([int(d["step"]) for d in result])
    print(min_step)

    if args.mode=='draw_figures':
        os.makedirs(args.save_dir, exist_ok=-True)
        plot_indices = range(156,196)
        plot_ppl_with_trained_at(result, save_dir=args.save_dir, min_step=min_step)
    
    elif args.mode=='measure_scores':
        # measure_indices = list(range(len(per_ex)))
        measure_scores(result, train_indices, interval=args.interval)

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
    parser.add_argument('--interval', type=int, default=10000)

    # Parse the arguments
    args = parser.parse_args()

    main(args)