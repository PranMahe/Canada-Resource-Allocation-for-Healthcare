import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_size, hidden_size, value_dim, num_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + num_agents + action_size * num_agents, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, value_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x, hidden_state):
        x = F.tanh(self.fc1(x))
        x, hidden_state = self.gru(x, hidden_state)
        value = self.fc3(x)
        return value, hidden_state