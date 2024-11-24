from Configuration.ia2c_params import IA2Cparameters
from Trainers.ia2c_trainer_po import IA2CtrainerPO

class IA2CrunnerPO:
    def __init__(self, env, env_name):
        self.env = env
        self.env_name = env_name
        
    def run_experiment(self):
        params = IA2Cparameters()

        train_params = {
            'env': self.env,
            'env_name': self.env_name,
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
            't_max': self.env.t_max,
            'tau': params.tau,
            'test_interval': params.test_interval,
            'num_training_iteration': params.num_training_iteration,
            'num_test_episodes': params.num_test_episodes,
            'batch_size': params.batch_size
        }

        for trial in range(params.num_trials):
            print(f"Trial: {trial+1}")
            train_rewards, test_rewards = IA2CtrainerPO.train_IA2C_ParameterSharing(trial, **train_params)
