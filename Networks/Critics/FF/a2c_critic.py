import torch as th
import torch.nn.functional as F

class A2CCritic(th.nn.Module):
    def __init__(self, global_state_dim, hidden_size, value_dim, num_agents):
        super(A2CCritic, self).__init__()
        self.fc1 = th.nn.Linear(global_state_dim + num_agents, hidden_size)
        self.fc2 = th.nn.Linear(hidden_size, hidden_size)
        self.fc3 = th.nn.Linear(hidden_size, value_dim)

    def forward(self, global_state, agent_id):
        # Build network that maps state -> value
        x = th.cat([global_state, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value