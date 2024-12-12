import math

class MAPPOparameters:
    def __init__(self):

        self.num_trials = 2
        self.training_episodes = 100000
        self.batch_size = 8
        self.alpha = 0.0005
        self.beta = 0.0005
        self.lam = 0.95
        self.actor_hidden_dim = 128
        self.critic_hidden_dim = 128
        self.gamma = 0.99
        self.value_dim = 1
        #self.t_max = 10
        self.tau = 0.01
        self.entropy_coef = 0.01
        self.eps_clip = 0.2
        self.num_mini_batches = 2
        self.epochs = 10
        self.test_interval = 1000
        self.num_training_iteration = math.ceil(self.training_episodes / self.batch_size)
        self.num_test_episodes = 10