from Configuration.mappo_params import MAPPOparameters
from Trainers import MAPPO_Trainer

class MAPPOrunner:
    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents

    def run_experiment(self):
        params = MAPPOparameters()

        train_params = {
            'state_dim': self.env.stateDim,
            'observation_dim': self.env.local_stateDim,
            'action_dim': self.env.actionDim,
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
            't_max': params.t_max,
            'test_interval': params.test_interval,
            'num_training_iteration': params.num_training_iteration,
            'num_test_episodes': params.num_test_episodes,
            'num_agents': self.num_agents,
            'batch_size': params.batch_size,
        }
         
        mappo_trainer = MAPPO_Trainer()

        for trial in range(params.num_trials):
            print(f"Trial: {trial+1}")
            train_rewards, test_rewards = mappo_trainer.train_MAPPO(trial, **train_params)