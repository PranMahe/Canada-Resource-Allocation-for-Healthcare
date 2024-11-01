import torch as th
import torch.nn.functional as F

class SharedA2CCritic(th.nn.Module):
    def __init__(self, state_size, action_size, hidden_size, value_size, num_agents):
        super(SharedA2CCritic, self).__init__()
        self.fc1 = th.nn.Linear(state_size + num_agents + action_size, hidden_size)
        self.gru = th.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc3 = th.nn.Linear(hidden_size, value_size)
    
    # Observation + Agent ID + Previous Action
    def forward(self, x, hidden_state):
        x = F.relu(self.fc1(x))
        x, hidden_state = self.gru(x, hidden_state)
        value = self.fc3(x)
        return value, hidden_state