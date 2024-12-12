import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def benchmark_plot_from_csv(csv_files, test_interval, save_dir=None, plot_name='benchmark'):
    # Load data
    all_test_returns = []
    for file in csv_files:
        data = pd.read_csv(file, header=None)
        all_test_returns.append(data[0].values)

    num_trials = len(all_test_returns)
    num_points = len(all_test_returns[0])
    all_test_returns = np.array(all_test_returns)

    # Compute stats
    mean_test_returns = all_test_returns.mean(axis=0)
    test_std = all_test_returns.std(axis=0)
    test_ci = 1.96 * test_std / np.sqrt(num_trials)
    individual_max_returns = [np.max(trial_returns) for trial_returns in all_test_returns]
    mean_max_return = np.mean(individual_max_returns)
    max_return_ci = 1.96 * np.std(individual_max_returns) / np.sqrt(num_trials)

    # Plot
    plt.figure(figsize=(12, 6))
    episodes = np.arange(0, num_points * test_interval, test_interval)
    for i in range(num_trials):
        plt.plot(episodes, all_test_returns[i], linestyle='dotted', alpha=0.5, label=f'Trial {i+1}')
    plt.plot(episodes, mean_test_returns, '-o', color='black', label='Mean Test Returns')
    plt.fill_between(episodes, mean_test_returns - test_ci, mean_test_returns + test_ci,
                     color='lightblue', alpha=0.3, label='95% CI')
    plt.xlabel('Episodes')
    plt.ylabel('Test Return')
    plt.title(f'Test Returns with 95% Confidence Interval ({plot_name})')
    plt.legend(loc='upper right')

    # Save the plot
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'{plot_name}_benchmark_plot.png')
        plt.savefig(save_path)
    plt.show()

    return mean_test_returns, test_ci, np.max(mean_test_returns), max_return_ci, individual_max_returns

def plot_mean_test_returns_comparison(mean_returns_dict, test_ci_dict, test_interval, save_dir=None):
    plt.figure(figsize=(12, 6))
    num_points = len(next(iter(mean_returns_dict.values())))
    episodes = np.arange(0, num_points * test_interval, test_interval)

    for algorithm, mean_returns in mean_returns_dict.items():
        ci = test_ci_dict[algorithm]
        plt.plot(episodes, mean_returns, '-o', label=f'{algorithm} Mean')
        plt.fill_between(episodes, mean_returns - ci, mean_returns + ci, alpha=0.2, label=f'{algorithm} 95% CI')

    plt.xlabel('Episodes')
    plt.ylabel('Mean Test Return')
    plt.title('Comparison of Mean Test Returns with 95% CI')
    plt.legend(loc='upper left')

    # Save the plot
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'comparison_plot.png')
        plt.savefig(save_path)
    plt.show()