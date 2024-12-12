from Configuration.mappo_params import MAPPOparameters
from Trainers.mappo_trainer_po import MAPPO_Trainer

class MAPPOrunnerPO:
    def __init__(self, env, env_params):
        self.env = env
        self.env_params = env_params

    def run_experiment(self):
        params = MAPPOparameters()

        train_params = {
            'env': self.env,
            'env_params': self.env_params,
            'state_dim': self.env.state_dim,
            'observation_dim': self.env.observation_dim,
            'action_dim': self.env.action_dim,
            'num_patients': self.env.num_patients,
            'num_specialists': self.env.num_specialists,
            'gamma': params.gamma,
            'actor_hidden_dim': params.actor_hidden_dim,
            'critic_hidden_dim': params.critic_hidden_dim,
            'value_dim': params.value_dim,
            'alpha': params.alpha,
            'beta': params.beta,
            'lam': params.lam,
            'entropy_coef': params.entropy_coef,
            'eps_clip': params.eps_clip,
            'num_mini_batches': params.num_mini_batches,
            'epochs': params.epochs,
            't_max': self.env.episode_length,
            'test_interval': params.test_interval,
            'num_training_iteration': params.num_training_iteration,
            'num_test_episodes': params.num_test_episodes,
            'num_agents': self.env.num_agents,
            'batch_size': params.batch_size,
            'action_mapping': self.env.action_mapping,
            'max_patients_per_hospital': self.env.max_patients_per_hospital,
            'max_specialists_per_hospital': self.env.max_specialists_per_hospital,
            'num_specialties': self.env.num_specialties
        }
        mappo_trainer = MAPPO_Trainer()
        for trial in range(params.num_trials):
            print(f"Trial: {trial+1}")
            train_rewards, test_rewards = mappo_trainer.train_MAPPO(trial, **train_params)