import argparse

# Import envs and runners here
from Configuration.env_params import HCRAparams
from Envs.Environment import *
from Runners.ia2c_runner_po import IA2CrunnerPO
from Runners.maa2c_runner_po import MAA2CrunnerPO
from Runners.ippo_runner_po import IPPOrunnerPO
from Runners.mappo_runner_po import MAPPOrunnerPO

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
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "HCRA".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "maa2c", "ia2c", "mappo", or "ippo".')
    args = parser.parse_args()

    # Create the environment
    # You can set environment hyperparameters here
    # More environements can be added here
        
    if args.env == 'HCRA': # Partial Observable
        env_params = HCRAparams()
        env = HCRA(env_params.num_agents, env_params.num_patients, env_params.num_specialists, env_params.num_specialties,
                   env_params.num_hospitals, env_params.num_patients_per_hospital, env_params.num_specialists_per_hospital,
                   env_params.max_wait_time, env_params.episode_length)

        # Create the runner
        # More runners can be added here
        if args.algo == 'ia2c':
            runner = IA2CrunnerPO(env, env_params)
        elif args.algo == 'maa2c':
            runner = MAA2CrunnerPO(env, env_params)
        elif args.algo == 'ippo':
            runner = IPPOrunnerPO(env, env_params)
        elif args.algo == 'mappo':
            runner = MAPPOrunnerPO(env, env_params)
        else:
            raise ValueError("Algorithm name incorrect or not found")
    else:
        raise ValueError("Environment name incorrect or not found")

    runner.run_experiment()

if __name__ == '__main__':
    main()