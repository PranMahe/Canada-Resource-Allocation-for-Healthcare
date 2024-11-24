import torch as th
import numpy as np
import torch.nn.functional as F
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class MAPPOtester:

    def test_MAPPO(env, actor, num_test_episodes, num_agents, t_max, actor_hidden_dim, action_dim):

        test_rewards = 0

        for i in range(num_test_episodes):
            
            total_rewards = 0
            env.reset()

            hidden_states_actor = [th.zeros(1, 1, actor_hidden_dim).to(device) for _ in range(num_agents)]
            prev_actions = [th.zeros(1, action_dim).to(device) for _ in range(num_agents)]

            for t in range(t_max):
                actions = []

                for a in range(num_agents):
                    with th.no_grad():
                        observation = env.get_observation(a, t)
                        observation = th.tensor(observation, dtype=th.float32).to(device)
                        agent_id = F.one_hot(th.tensor(a), num_classes=num_agents).float().unsqueeze(0).to(device)
                        prev_action_a = prev_actions[a]

                        x = th.cat([observation, agent_id, prev_action_a], dim=-1).unsqueeze(1)

                        logits, hidden_state = actor(x, hidden_states_actor[a])
                        hidden_states_actor[a] = hidden_state

                        action, _, _ = actor.action_sampler(logits)
                        actions.append(action)

                        action_idx = action.item()
                        one_hot_action = F.one_hot(th.tensor(action_idx), num_classes=action_dim).float().unsqueeze(0).to(device)
                        prev_actions[a] = one_hot_action

                global_reward, individual_rewards, _ = env.step(actions)
                total_rewards += global_reward

            test_rewards[i] += total_rewards

        average_reward = np.mean(test_rewards)
        return average_reward