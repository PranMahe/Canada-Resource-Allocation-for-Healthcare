import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
import csv
import matplotlib.pyplot as plt

from Networks.Actors.ia2c_actor_gru import SharedA2CActor
from Networks.Critics.ia2c_critic_gru import SharedA2CCritic

from Helpers.A2C.a2c_helper import BatchTraining
from Benchmarkers.ia2c_test import IA2CtesterPS

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class IA2CtrainerPO:
    @staticmethod
    def train_IA2C_PO(trial_run, env, env_name, state_dim, action_dim, num_agents, gamma, actor_hidden_dim, critic_hidden_dim,
                                    value_dim, alpha, beta, reward_standardization, num_batch_episodes, t_max, tau,
                                    test_interval, num_training_episodes, num_test_episodes): 
        
        csv_file = open(f'IA2C_trial_{trial_run}_HCRAPO.csv', 'a', newline='')
        csv_writer = csv.writer(csv_file)

        actor_shared = SharedA2CActor(state_dim, action_dim, actor_hidden_dim, num_agents).to(device)
        critic = SharedA2CCritic(state_dim, action_dim, critic_hidden_dim, value_dim, num_agents).to(device)

        optimizer_actor_shared = th.optim.Adam(actor_shared.parameters(), lr=alpha)
        optimizer_critic = th.optim.Adam(critic.parameters(), lr=beta)

        episode_rewards = []
        test_rewards = []

        episode = 0
        for te in range(num_training_episodes):
            batch_buffer = [{'observations': [], 'actions': []} for _ in range (num_batch_episodes)]
            batch_rtrns = [[] for _ in range(num_batch_episodes)]
            for e in range(num_batch_episodes):
                
                total_rewards = 0
                done = False
                buffer = {'observations': [], 'actions': [], 'global_rewards': [], 'next_observations': []}
                hidden_state = [th.zeros(1, 1, actor_hidden_dim).to(device) for _ in range(num_agents)]
                prev_actions = [th.zeros(1, action_dim).to(device) for _ in range(num_agents)]

                for t in range(t_max):
                    

                    actions = []
                    observations = []

                    for a in range(num_agents):
                        observation = env.get_observation([a, 0], 0, t)
                        observation = th.tensor(observation, dtype=th.float32).to(device)
                        observations.append(observation.squeeze())
                        agent_id = th.nn.functional.one_hot(th.tensor(a), num_classes=num_agents).float().unsqueeze(0).to(device)
                        prev_action_a = prev_actions[a]
                        x = th.cat([observation, agent_id, prev_action_a], dim=-1).unsqueeze(0)

                        logits, hidden_state_a = actor_shared(x, hidden_state[a])
                        hidden_state[a] = hidden_state_a

                        action, _ = actor_shared.action_sampler(logits)
                        action_idx = action.item()
                        prev_actions[a] = th.nn.functional.one_hot(th.tensor(action_idx), num_classes=action_dim).float().unsqueeze(0).to(device)
                        actions.append(action_idx)
                    # Execute actions in the environment
                    global_reward, reward, done = env.step(action)
                    
                    for a in range(num_agents):
                        next_observation = env.get_observation([a, 0], 0, t + 1)
                        next_observation = th.tensor(next_observation, dtype=th.float32).squeeze().to(device)

                    buffer['observations'].append(observations)
                    buffer['actions'].append(actions)
                    buffer['global_rewards'].append(global_reward)
                    buffer['next_observations'].append(next_observation)

                    # global_state = global_next_state
                    total_rewards += global_reward
                episode_rewards.append(np.mean(total_rewards))
                episode += 1

                if reward_standardization:
                    pass

                rtrns = []
                if done:
                    R = 0
                else:
                    R = critic(buffer['observation'][-1]).item() # INCORRECT IMPLEMENTATION
                for global_reward in reversed(buffer['global_rewards']):
                    R = global_reward + gamma * R
                    rtrns.insert(0, R)
                rtrns = th.tensor(np.array(rtrns), dtype=th.float32).to(device)
                #rtrns_per_agent = np.repeat(np.array(rtrns), num_agents, axis=1)  # Shape [batch_size, num_agents, 1]

                batch_buffer[e]['observations'].extend(buffer['observations'])
                batch_buffer[e]['actions'].extend(buffer['actions'])
                batch_rtrns[e].extend(rtrns)

                if (episode) % test_interval == 0:
                    actor_shared.eval()
                    test_reward = IA2CtesterPS.test_IA2C_ParameterSharing(actor_shared, num_test_episodes, t_max, num_agents, actor_hidden_dim, action_dim)
                    test_rewards.append(test_reward)
                    csv_writer.writerow([test_reward])
                    csv_file.flush()
                    
                    plt.figure(1)
                    plt.clf()
                    plt.plot(test_rewards, label="Test Reward")
                    plt.xlabel("Test Interval")
                    plt.ylabel("Return")
                    plt.title("Test Return Over Time IA2C")
                    plt.legend()
                    plt.grid(True)
                    plt.draw()
                    plt.pause(1)
                    
                    print(f'Training reward at episode {episode}: {test_reward:.2f}')
                    actor_shared.train()

            batch_training = BatchTraining()
            batch_observations, batch_actions, batch_rtrns = batch_training.collate_batch(batch_buffer, batch_rtrns)

            #print(f"states: {batch_observations.size()}") # [batch_size, timesteps, num_agents, state_dim]
            #print(f"actions: {batch_actions.size()}") # [batch_size, timesteps, num_agents] - NOT ONE HOT ENCODED
            #print(f"hidden states: {batch_hidden_states.size()}")  # [batch_size, timesteps, num_agents, hidden_dim]
            #print(f"returns: {batch_rtrns.size()}")  # [batch_size, timesteps]

            # Update Centralized Critic and Actor
            batch_size, timesteps, _ = batch_actions.shape
            batch_prev_actions = th.zeros_like(batch_actions) 
            batch_prev_actions[:, 1:, :] = batch_actions[:, :-1, :] # Shift actions to get previous joint actions
            batch_prev_actions = F.one_hot(batch_prev_actions.long(), num_classes=action_dim)

            hidden_state_actor = th.zeros(1, batch_size, actor_hidden_dim)
            hidden_state_critic = th.zeros(1, batch_size, critic_hidden_dim)
            
            optimizer_critic.zero_grad()
            total_critic_loss = 0
            for a in range(num_agents):
                # Critic
                agent_indices = th.full((batch_size, timesteps), a, dtype=th.long).to(device) 
                agent_id = F.one_hot(agent_indices, num_classes=num_agents).float() # [batch_size, timesteps, num_agents]
                x = th.cat([batch_observations[:, :, a, :], agent_id, batch_prev_actions[:, :, a, :]], dim=-1)
                V, _ = critic(x, hidden_state_critic)
                V = V.squeeze(-1)
                
                critic_loss = (batch_rtrns - V).pow(2).mean()
                total_critic_loss += critic_loss
            
            total_critic_loss /= num_agents

            # Critic Update
            total_critic_loss.backward()
            optimizer_critic.step()

            optimizer_actor_shared.zero_grad()
            total_actor_loss = 0
            for a in range(num_agents):
                # Actor
                agent_observation = batch_observations[:, :, a, :]
                actions = batch_actions[:, :, a]
                prev_actions = batch_prev_actions[:, :, a, :]
                agent_indices = th.full((batch_size, timesteps), a, dtype=th.long).to(device) 
                agent_id = F.one_hot(agent_indices, num_classes=num_agents).float() # [batch_size, timesteps, num_agents]

                x = th.cat([agent_observation, agent_id, prev_actions], dim=-1)

                logits, _ = actor_shared(x, hidden_state_actor)
                softmax_probabilities = F.softmax(logits, dim=-1)
                action_dist = Categorical(probs=softmax_probabilities)
                
                log_probs = action_dist.log_prob(actions)
                entropies = action_dist.entropy()
                
                x_critic = th.cat([agent_observation, agent_id, prev_actions], dim=-1)
                V_values, _ = critic(x_critic, hidden_state_critic)
                V_values = V_values.squeeze(-1)

                # Calculate Advantage
                advantages = (batch_rtrns - V_values).detach() # [batch_size, time_steps]

                # Actor Loss
                actor_loss = -(log_probs * advantages).mean() - tau * entropies.mean()
                total_actor_loss += actor_loss

            # Average losses across agents
            total_actor_loss /=num_agents

            # Actor Update
            total_actor_loss.backward()
            optimizer_actor_shared.step()

        return episode_rewards, test_rewards