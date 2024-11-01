from Configuration.ia2c_params import IA2Cparameters
from Trainers.ia2c_trainer import IA2CtrainerPS

class IA2Crunner:
    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents
        
    def run_experiment(self):
        params = IA2Cparameters()

        train_params = {
            'state_dim': self.env.stateDim,
            'observation_dim': self.env.local_stateDim,
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
            'test_interval': params.test_interval,
            'num_training_iteration': params.num_training_iteration,
            'num_test_episodes': params.num_test_episodes,
            'batch_size': params.batch_size
        }

        for trial in range(params.num_trials):
            print(f"Trial: {trial+1}")
            train_rewards, test_rewards = IA2CtrainerPS.train_IA2C_ParameterSharing(trial, **train_params)
