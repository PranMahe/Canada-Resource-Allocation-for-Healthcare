import math

class IA2Cparameters:
    def __init__(self):

        self.num_trials = 5
        self.training_episodes = 100000
        self.batch_size = 8
        self.alpha = 0.0001
        self.beta = 0.0001
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.gamma = 0.99
        self.value_dim = 1
        self.reward_standardization = False
        self.t_max = 10
        self.tau = 0.01
        self.test_interval = 1000
        self.num_training_iteration = math.ceil(self.training_episodes / self.batch_size)
        self.num_test_episodes = 10