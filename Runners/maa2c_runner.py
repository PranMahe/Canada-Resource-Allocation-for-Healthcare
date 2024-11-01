from Configuration.maa2c_params import MAA2Cparameters
from Trainers.maa2c_trainer import MAA2CtrainerPS

class MAA2Crunner:
    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents
        
    def run_experiment(self, veh_pos_data, test_data_list, env_setup):
        params = MAA2Cparameters()

        train_params = {
            'env_setup': env_setup,
            'test_data_list': test_data_list,
            'veh_pos_data': veh_pos_data,
            'state_dim': self.env.stateDim,
            'action_dim': self.env.actionDim,
            'num_agents': self.num_agents,
            'gamma': params.gamma,
            'actor_hidden_dim': params.actor_hidden_dim,
            'critic_hidden_dim': params.critic_hidden_dim,
            'value_dim': params.value_dim,
            'alpha': params.alpha,
            'beta': params.beta,
            'reward_standardization': params.reward_standardization,
            't_max': params.t_max,
            'tau': params.tau,
            'test_interval':params.test_interval,
            'num_training_iteration': params.num_training_iteration,
            'num_test_episodes': params.num_test_episodes,
            'batch_size': params.num_batch_episodes,
        }

        for trial in range(params.num_trials):
            print(f"Trial: {trial+1}")
            train_rewards, test_rewards = MAA2CtrainerPS.train_MAA2C_ParameterSharing(trial, **train_params)