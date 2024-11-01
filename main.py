import argparse

# Import envs and runners here
# from envs.matrixgame import create_climbing_game, create_penalty_game, create_tnt_game
# from envs.matrixgame import *
from Envs.Environment import *
from Runners.ia2c_runner import IA2Crunner
from Runners.maa2c_runner import MAA2Crunner
from Runners.ippo_runner import IPPOrunner
from Runners.mappo_runner import MAPPOrunner

'''
To use this code you must run a script using the terminal
For example:
    <termina> python main.py --env HCRA --algo maa2c

This will run MAA2C for the HCRA Environment

NOTE:
    For specific algorithm configurations, you must look in the config file
    to set algorithm hyperparameters as well as algorithm variation parameters.
'''

def main():
    parser = argparse.ArgumentParser(description="Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "HCRA", "HCRAPO".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "maa2c", "ia2c", "mappo", or "ippo".')
    args = parser.parse_args()

    # Create the environment
    # You can set environment hyperparameters here
    # More environements can be added here
    if args.env == 'HCRA':
        env = create_hcra()
        env_name = 'HCRA'
    elif args.env == 'HCRAPO':
        env = create_hcra_po()
        env_name == "HCRAPO"
    else:
        raise ValueError("Environment name incorrect or not found")

    # Create the runner
    # More runners can be added here
    if args.algo == 'ia2c':
        runner = IA2Crunner(env, env_name)
    elif args.algo == 'maa2c':
        runner = MAA2Crunner(env, env_name)
    elif args.algo == 'ippo':
        runner = IPPOrunner(env, env_name)
    elif args.algo == 'mappo':
        runner = MAPPOrunner(env, env_name)
    else:
        raise ValueError("Algorithm name incorrect or not found")
    
    runner.run_experiment()

if __name__ == '__main__':
    main()