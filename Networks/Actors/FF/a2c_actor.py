import torch as th
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class A2CActorShared(th.nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_agents):
        super(A2CActorShared, self).__init__()
        self.fc1 = th.nn.Linear(state_size + num_agents, hidden_size)
        self.fc2 = th.nn.Linear(hidden_size, hidden_size)
        self.fc3 = th.nn.Linear(hidden_size, action_size)

    def forward(self, state, agent_id):
        # Concatenate state with one-hot encoded agent ID
        x = th.cat([state, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def action_sampler(self, logits):
        softmax_probabilities = F.softmax(logits, dim=-1)
        action_dist = Categorical(softmax_probabilities)
        action = action_dist.sample()
        return action, action_dist