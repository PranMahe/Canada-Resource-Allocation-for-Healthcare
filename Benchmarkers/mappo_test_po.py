import torch as th
import numpy as np
import torch.nn.functional as F
from Envs.Environment import HCRA
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class MAPPOtester:

    def test_MAPPO(env_params, actor, num_test_episodes, num_agents, 
                   t_max, actor_hidden_dim, action_dim, action_mapping,
                   max_patients_per_hospital, max_specialists_per_hospital, num_specialties, seed=None):

        test_env = HCRA(
            num_agents=env_params.num_agents,
            num_patients=env_params.num_patients,
            num_specialists=env_params.num_specialists,
            num_specialties=env_params.num_specialties,
            num_hospitals=env_params.num_hospitals,
            max_wait_time=env_params.max_wait_time,
            episode_length=env_params.episode_length,
            hospital_capacities=env_params.hospital_capacities,
            seed=seed
        )

        test_rewards = np.zeros(num_test_episodes)

        for i in range(num_test_episodes):
            
            total_rewards = 0
            test_env.reset()

            hidden_states_actor = [th.zeros(1, 1, actor_hidden_dim).to(device) for _ in range(num_agents)]
            prev_actions = [th.zeros(1, action_dim).to(device) for _ in range(num_agents)]

            for t in range(t_max):
                actions = []

                for a in range(num_agents):
                    with th.no_grad():
                        raw_observation = test_env.get_observation(a)
                        observation = HCRA.process_observation(raw_observation, max_patients_per_hospital, max_specialists_per_hospital, num_specialties)
                        observation = th.tensor(observation, dtype=th.float32).unsqueeze(0).to(device)
                        agent_id = F.one_hot(th.tensor(a), num_classes=num_agents).float().unsqueeze(0).to(device)
                        prev_action_a = prev_actions[a]

                        x = th.cat([observation, agent_id, prev_action_a], dim=-1).unsqueeze(1)

                        logits, hidden_state = actor(x, hidden_states_actor[a])
                        hidden_states_actor[a] = hidden_state

                        action, _, _ = actor.action_sampler(logits)
                        action_idx = action.item()
                        actions.append(action_idx)

                        one_hot_action = F.one_hot(th.tensor(action_idx), num_classes=action_dim).float().unsqueeze(0).to(device)
                        prev_actions[a] = one_hot_action

                mapped_actions = [action_mapping[a_idx] for a_idx in actions]
                global_reward, individual_rewards, _ = test_env.step(mapped_actions)
                total_rewards += global_reward

            test_rewards[i] += total_rewards

        average_reward = np.mean(test_rewards)
        return average_reward