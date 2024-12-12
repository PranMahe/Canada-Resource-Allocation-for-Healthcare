import argparse
import os

# Import envs and runners here
from Configuration.env_params import HCRAparams
from Envs.Environment import *
from Runners.maa2c_runner_po import MAA2CrunnerPO
from Runners.mappo_runner_po import MAPPOrunnerPO
from Results.Plotting.benchmark_plot import benchmark_plot_from_csv, plot_mean_test_returns_comparison

'''
To use this code you must run a script using the terminal
For example:
    <termina> python main.py --env HCRA --algo maa2c

This will run MAA2C for the HCRA Environment

To Run plotting code use the command:
    <terminal> python main.py --dir_MAA2C Results/MAA2C/random_seeds --dir_MAPPO Results/MAPPO/random_seeds --save_dir Results/Plots

NOTE:
    For specific algorithm configurations, you must look in the config file
    to set algorithm hyperparameters as well as algorithm variation parameters.

    When Plotting, there must be CSV files to get plots, only run the plotting
    script when done running required experiments
'''

def main():
    parser = argparse.ArgumentParser(description="Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, help='The environment to run. Choose from "HCRA".')
    parser.add_argument('--algo', type=str, help='The algorithm to use. Choose from "maa2c", OR "mappo".')
    parser.add_argument('--dir_MAA2C', type=str, help='Directory containing MAA2C CSV files.')
    parser.add_argument('--dir_MAPPO', type=str, help='Directory containing MAPPO CSV files.')
    parser.add_argument('--test_interval', type=int, default=1000, help='Test interval for plotting.')
    parser.add_argument('--save_dir', type=str, help='Directory to save plot images.')
    args = parser.parse_args()

    # Create the environment
    # You can set environment hyperparameters here
    # More environements can be added here
    if args.env is not None and args.algo is not None:
        if args.env == 'HCRA': # Partial Observable
            env_params = HCRAparams()
            env = HCRA(env_params.num_agents, env_params.num_patients, env_params.num_specialists, env_params.num_specialties,
                    env_params.num_hospitals, env_params.max_wait_time, env_params.episode_length, env_params.hospital_capacities)

            # Create the runner
            # More runners can be added here
            if args.algo == 'maa2c':
                runner = MAA2CrunnerPO(env, env_params)
            elif args.algo == 'mappo':
                runner = MAPPOrunnerPO(env, env_params)
            else:
                raise ValueError("Algorithm name incorrect or not found")
        else:
            raise ValueError("Environment name incorrect or not found")

        runner.run_experiment()

    # Plot Results
    elif args.dir_MAA2C is not None and args.dir_MAPPO is not None:
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)

        # Collect CSV files for MAA2C
        maa2c_files = [os.path.join(args.dir_MAA2C, f) for f in os.listdir(args.dir_MAA2C) if f.endswith('.csv')]
        mean_test_returns_MAA2C, test_ci_MAA2C, max_return_MAA2C, max_return_ci_MAA2C, individual_max_returns_MAA2C = benchmark_plot_from_csv(
            maa2c_files, test_interval=args.test_interval, save_dir=args.save_dir, plot_name='MAA2C'
        )
        print(f"MAA2C - Max Return: {max_return_MAA2C}, Max Return CI: {max_return_ci_MAA2C}")
        print(f"MAA2C - Individual Max Returns: {individual_max_returns_MAA2C}")

        # Collect CSV files for MAPPO
        mappo_files = [os.path.join(args.dir_MAPPO, f) for f in os.listdir(args.dir_MAPPO) if f.endswith('.csv')]
        mean_test_returns_MAPPO, test_ci_MAPPO, max_return_MAPPO, max_return_ci_MAPPO, individual_max_returns_MAPPO = benchmark_plot_from_csv(
            mappo_files, test_interval=args.test_interval, save_dir=args.save_dir, plot_name='MAPPO'
        )
        print(f"MAPPO - Max Return: {max_return_MAPPO}, Max Return CI: {max_return_ci_MAPPO}")
        print(f"MAPPO - Individual Max Returns: {individual_max_returns_MAPPO}")

        # Comparison plot
        mean_returns_dict = {
            'MAA2C': mean_test_returns_MAA2C,
            'MAPPO': mean_test_returns_MAPPO,
        }
        test_ci_dict = {
            'MAA2C': test_ci_MAA2C,
            'MAPPO': test_ci_MAPPO,
        }

        plot_mean_test_returns_comparison(mean_returns_dict, test_ci_dict, args.test_interval, save_dir=args.save_dir)
    else:
        raise ValueError("Must provide either --env and --algo to run experiments, or --dir_MAA2C and --dir_MAPPO to plot results.")

if __name__ == '__main__':
    main()