import math

class MAA2Cparameters:
    def __init__(self):

        self.num_trials = 5
        self.training_episodes = 100000
        self.batch_size = 8
        self.alpha = 0.001
        self.beta = 0.001
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.gamma = 0.99
        self.value_dim = 1
        self.tau = 0.03
        self.test_interval = 1000
        self.num_test_episodes = 10
        self.num_training_iterations = math.ceil(self.training_episodes / self.batch_size)