import torch as th
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

class A2CActorShared(th.nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_agents):
        super(A2CActorShared, self).__init__()
        self.fc1 = th.nn.Linear(state_size + num_agents + action_size, hidden_size)
        self.gru = th.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc3 = th.nn.Linear(hidden_size, action_size)

    # Observation + Agent ID + Previous Action
    def forward(self, x, hidden_state):
        x = F.relu(self.fc1(x))
        x, hidden_state = self.gru(x, hidden_state)
        logits = self.fc3(x)
        return logits, hidden_state

    def action_sampler(self, logits):
        softmax_probabilities = F.softmax(logits, dim=-1)
        action_dist = Categorical(softmax_probabilities)
        action = action_dist.sample()
        return action, action_dist