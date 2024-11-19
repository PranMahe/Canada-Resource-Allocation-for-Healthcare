import torch as th
import torch.nn.functional as F

class A2CCentralizedCritic(th.nn.Module):
    def __init__(self, state_size, observation_size, action_size, hidden_size, value_size, num_agents):
        super(A2CCentralizedCritic, self).__init__()
        self.fc1 = th.nn.Linear(state_size + observation_size + num_agents + action_size * num_agents, hidden_size)
        self.gru = th.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc3 = th.nn.Linear(hidden_size, value_size)
    
    # Global_state + Observation + Agent ID + Previous Joint Action
    def forward(self, x, hidden_state):
        x = F.relu(self.fc1(x))
        x, hidden_state = self.gru(x, hidden_state)
        value = self.fc3(x)
        return value, hidden_state