import torch as th
import numpy as np
import torch.nn.functional as F
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class IA2CtesterPS:
    @staticmethod
    def test_IA2C_ParameterSharing(env, actor_shared, num_test_episodes, t_max, num_agents, actor_hidden_dim, action_dim):
        test_rewards = 0
        with th.no_grad():
            for i in range(num_test_episodes):

                total_rewards = 0

                hidden_state = [th.zeros(1, 1, actor_hidden_dim).to(device) for _ in range(num_agents)]
                prev_actions = [th.zeros(1, action_dim).to(device) for _ in range(num_agents)]

                for t in range(t_max):
                    actions = []

                    # Collect actions for each agent
                    for a in range(num_agents):
                        observation = env.get_observation(a, t)
                        observation = th.tensor(observation, dtype=th.float32).to(device)
                        agent_id = th.nn.functional.one_hot(th.tensor(a), num_classes=num_agents).float().unsqueeze(0).to(device)
                        prev_action_a = prev_actions[a]

                        x = th.cat([observation, agent_id, prev_action_a], dim=-1).unsqueeze(1)

                        logits, hidden_state_a = actor_shared(x, hidden_state[a])
                        hidden_state[a] = hidden_state_a

                        action, _ = actor_shared.action_sampler(logits.squeeze(1))
                        actions.append(action)
                        action_idx = action.item()
                        prev_actions[a] = F.one_hot(th.tensor(action_idx), num_classes=action_dim).float().unsqueeze(0).to(device)

                    # Execute actions in the environment
                    global_reward, individual_rewards, _ = env.step(actions)
                    total_rewards += global_reward
                
                test_rewards[i] += total_rewards

        average_reward = np.mean(test_rewards)
        return average_reward