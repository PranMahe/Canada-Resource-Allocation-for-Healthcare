import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_agents):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim + num_agents, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state, agent_id):
        x = th.cat([state, agent_id], dim=-1)
        x = th.tanh(self.fc1(x))
        x = th.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits


    def action_sampler(self, logits):
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, action_dist