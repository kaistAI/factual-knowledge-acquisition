import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatter


def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def logarithmic_decay(x, a, b):
    return a * np.log(x) + b

def fit_exp_linear(t, y, C=0):
    y = y - C
    valid_indices = y > 0  # Create a mask of valid indices where y is positive
    y = y[valid_indices]   # Filter y using the mask
    t = t[valid_indices]
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def fit_models(x_data, y_data, fname):
    # Fit exponential decay
    # popt_exp, pcov_exp = curve_fit(exponential_decay, x_data, y_data, maxfev=10000)
    # print('!!!!!!!!!!!!!!!!!!!!!')
    # print(popt_exp)
    # print('!!!!!!!!!!!!!!!!!!!!!')
    # y_pred_exp = exponential_decay(x_data, *popt_exp)
    # rmse_exp = np.sqrt(mean_squared_error(y_data, y_pred_exp))
    C0 = 0
    A, K = fit_exp_linear(x_data, y_data, C0)
    y_pred_exp = exponential_decay(x_data, A, -K, C0)
    print(A, K)
    rmse_exp = np.sqrt(mean_squared_error(y_data, y_pred_exp))
    
    # Fit logarithmic decay (ensure no zero or negative x-values)
    x_data_log = x_data[x_data > 0]
    y_data_log = y_data[x_data > 0]
    popt_log, pcov_log = curve_fit(logarithmic_decay, x_data_log, y_data_log, maxfev=10000)
    y_pred_log = logarithmic_decay(x_data_log, *popt_log)
    rmse_log = np.sqrt(mean_squared_error(y_data_log, y_pred_log))
    
    # Print the results
    print("Exponential Decay Fit: RMSE =", rmse_exp)
    print("Logarithmic Decay Fit: RMSE =", rmse_log)
    
    # Determine which model fits better
    if rmse_exp < rmse_log:
        print("Exponential decay provides a better fit.")
    else:
        print("Logarithmic decay provides a better fit.")
        
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, color='blue', alpha=0.05, label='Data Points')
    plt.plot(x_data, y_pred_exp, 'r-', label=f'Exponential Fit: RMSE={rmse_exp:.2f}')
    plt.plot(x_data_log, y_pred_log, 'g-', label=f'Logarithmic Fit: RMSE={rmse_log:.2f}')
    plt.title('Fit of Exponential Decay and Logarithmic Decay Models')
    plt.xlabel('t')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(f'curve_fit/curve_fit_{fname}.png')



# Function to calculate the slope
def calculate_slope(steps, y_values):
    x_values = steps
    log_x = np.log(x_values)
    slope, _ = np.polyfit(log_x, y_values, 1)
    # n = len(y_values)
    # se_slope = np.sqrt(ssr / (n-2)) / np.sqrt(np.sum((log_x - np.mean(log_x))**2))
    return slope, se_slope


def plot_trendline(ax, x, y, label, color, args):
    if args.tokens:
    # if True:
        ext_x = [1000000, 10**12]
        # ext_x = [i for i in range(1, 20000000)]
        # if 'bsize' in args.exp_name or '128' in args.exp_name:
        #     ext_x = [i*128 for i in [50, 100, 1000, 10000, 100000]]  # Adjusted to include realistic x values    
        # else:
        #     ext_x = [i*2048 for i in [50/16, 100, 1000, 10000, 100000]]  # Adjusted to include realistic x values    
    else:
        ext_x = [1, 100, 1000, 10000, 10**6]  # Adjusted to include realistic x values

    log_x = np.log10(x)
    slope, intercept = np.polyfit(log_x, y, 1)
    x_intercept = -intercept/slope

    y_pred = np.polyval([slope, intercept], log_x)
    # Calculate SSR (sum of squares of residuals)
    ssr = np.sum((y - y_pred) ** 2)
    # Calculate SST (total sum of squares)
    sst = np.sum((y - np.mean(y)) ** 2)
    # Calculate R^2
    r_squared = 1 - (ssr / sst)
    # Calculate Standard Error of the Slope
    n = len(y)
    se_slope = np.sqrt(ssr / (n-2)) / np.sqrt(np.sum((log_x - np.mean(log_x))**2))
    # Calculate the range for the slope
    slope_low = slope - se_slope
    slope_high = slope + se_slope

    trendline_y = np.log10(ext_x) * slope + intercept
    # print(ext_x, '\n', trendline_y)
    ax.plot(ext_x, trendline_y, linestyle='dotted', color=color, linewidth=2)

    print(f'Slope: {slope:.2f}, One Sigma: {se_slope:.4f}, R^2: {r_squared:.2f}')
    print(f"x-intercept: {x_intercept:.2f}")
    return slope, r_squared


def plot(x_values, mem_data, gen_data, gen_hard_data, mode, reverse, args):
    fig, ax = plt.subplots(figsize=(6,5))
    datasets = [mem_data, gen_data, gen_hard_data]
    labels = ['Memorization', 'Semantic', 'Composition']
    colors = ['blue', 'orange', 'red']
    
    for index, y in enumerate(datasets):
        
        if y[0]<0:
            y = [-i for i in y]
        
        if reverse:
            y = [(100-i)/100 for i in y]
        else:
            y = [i/100 for i in y]
        
        filtered_x_values = []
        filtered_y = []
        for i in range(len(y)):
            if y[i]!=-1.0:
                filtered_x_values.append(x_values[i])
                filtered_y.append(y[i])
        
        # print(filtered_x_values)
        # print(filtered_y)
        if args.tokens:
            if 'bsize' in args.exp_name or True:
                if '128' in args.exp_name:
                    filtered_x_values = [x*128*2048 for x in filtered_x_values]
                    print('bsize: 128')
                elif '512' in args.exp_name:
                    print('bsize: 512')
                    filtered_x_values = [x*512*2048 for x in filtered_x_values]
            else:    
                print('bsize:2048')
                filtered_x_values = [x*2048*2048 for x in filtered_x_values]
        
        slope, r_squared = plot_trendline(ax, np.array(filtered_x_values), np.array(filtered_y), labels[index], colors[index], args)
        ax.plot(filtered_x_values, filtered_y, 'o', label=f'{labels[index]} (a={-slope:.2f})', color=colors[index], alpha=0.3, markersize=3)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    if reverse:
        ax.set_ylim(bottom=0, top=1)
        # pass
    else:
        ax.set_ylim(bottom=0, top=1)
    ax.set_xscale('log')
    # ax.xaxis.set_major_formatter(LogFormatter())
    if args.tokens:
        ax.set_xlabel('Tokens', fontsize=20)
    else:
        ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylabel(r'Avg. $\mathcal{R}(q,t)$', fontsize=20)
    # ax.set_xticklabels([r'$10^{{{}}}$'.format(xi) for xi in filtered_x_values])
    ax.set_title(mode.capitalize(), fontsize=20)
    ax.legend(loc='upper right', prop={'size': 15}, markerscale=3)

    # def log_tick_formatter(val, pos=None):
    #     return f'{int(np.log10(val))}'
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
    ax.tick_params(axis='both', labelsize=14)
    # fit_models(np.array(filtered_x_values), np.array(filtered_y), mode)

    if args.tokens:
        fname = f"{save_dir}/tokens/{args.exp_name.split('/')[-1][:-5]}_{mode}_tokens.pdf"
    else:
        if reverse:
            fname = f"{save_dir}/{args.exp_name.split('/')[-1][:-5]}_{mode}_reversed.pdf"
        else:
            fname = f"{save_dir}/{args.exp_name.split('/')[-1][:-5]}_{mode}.pdf"
    plt.tight_layout()
    plt.savefig(fname)
    print(f"fig saved to {fname}")


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--tokens', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    base_dir = 'measure_data'
    save_dir = 'learnability_plots/regularized' if 'regularized' in args.exp_name else 'learnability_plots/absolute'

    with open(args.exp_name, 'r') as f:
        data = json.load(f)

    for k, v in data.items():
        print(f'\n\n#####\t{k}\t#####\n')
        
        mem_data, gen_data, gen_hard_data = v["mem"][1:], v["gen"][1:], v["gen_hard"][1:]
        data_length = len(data["duplication"]["mem"])

        steps = range(1, data_length)
        plot(steps, mem_data, gen_data, gen_hard_data, mode=k, reverse=args.reverse, args=args)

        
        
    print('\n\n\n')
    