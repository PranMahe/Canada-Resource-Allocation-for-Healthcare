from Configuration.maa2c_params import MAA2Cparameters
from Trainers.maa2c_trainer_po import MAA2CtrainerPO

class MAA2CrunnerPO:
    def __init__(self, env, env_params):
        self.env = env
        self.env_params = env_params
        
    def run_experiment(self):
        params = MAA2Cparameters()

        train_params = {
            'env': self.env,
            'env_params': self.env_params,
            'state_dim': self.env.state_dim,
            'observation_dim': self.env.observation_dim,
            'action_dim': self.env.action_dim,
            'num_patients': self.env.num_patients,
            'num_specialists': self.env.num_specialists,
            'num_agents': self.env.num_agents,
            'gamma': params.gamma,
            'actor_hidden_dim': params.actor_hidden_dim,
            'critic_hidden_dim': params.critic_hidden_dim,
            'value_dim': params.value_dim,
            'alpha': params.alpha,
            'beta': params.beta,
            't_max': self.env.episode_length,
            'tau': params.tau,
            'test_interval':params.test_interval,
            'num_training_iterations': params.num_training_iterations,
            'num_test_episodes': params.num_test_episodes,
            'batch_size': params.batch_size,
            'action_mapping': self.env.action_mapping,
            'max_patients_per_hospital': self.env.num_patients_per_hospital,
            'max_specialists_per_hospital': self.env.num_specialists_per_hospital,
            'num_specialties': self.env.num_specialties
        }

        for trial in range(params.num_trials):
            print(f"Trial: {trial+1}")
            train_rewards, test_rewards = MAA2CtrainerPO.train_MAA2C_ParameterSharing(trial, **train_params)