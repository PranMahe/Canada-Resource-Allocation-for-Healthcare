import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
import csv
import matplotlib.pyplot as plt

from Networks.Actors.maa2c_actor_gru import A2CActorShared
from Networks.Critics.maa2c_critic_gru import A2CCentralizedCritic

from Helpers.A2C.maa2c_helper import BatchTraining
from Benchmarkers.maa2c_test_po import MAA2CtesterPS

from Envs.Environment import HCRA

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class MAA2CtrainerPO:
    @staticmethod
    def train_MAA2C_ParameterSharing(trial_run, env, env_params, state_dim, observation_dim, action_dim, num_patients, num_specialists,
                                 num_agents, gamma, actor_hidden_dim, critic_hidden_dim, value_dim, alpha, beta,
                                 batch_size, t_max, tau, test_interval, num_training_iterations, num_test_episodes,
                                 action_mapping, max_patients_per_hospital, max_specialists_per_hospital, num_specialties): 
        
        csv_file = open(f'MAA2C_trial_{trial_run}_HCRA.csv', 'a', newline='')
        csv_writer = csv.writer(csv_file)

        actor_shared = A2CActorShared(observation_dim, action_dim, actor_hidden_dim, num_agents).to(device)
        critic = A2CCentralizedCritic(state_dim, observation_dim, action_dim, critic_hidden_dim, value_dim, num_agents).to(device)

        optimizer_actor_shared = th.optim.Adam(actor_shared.parameters(), lr=alpha)
        optimizer_critic = th.optim.Adam(critic.parameters(), lr=beta)

        episode_rewards = []
        test_rewards = []

        episode = 0
        for te in range(num_training_iterations):
            batch_buffer = [{'global_states': [], 'observations': [], 'joint_actions': []} for _ in range (batch_size)]
            batch_rtrns = [[] for _ in range(batch_size)]
            for e in range(batch_size):
                
                env.reset()
                total_rewards = 0
                done = False
                buffer = {'global_states':[],'observations': [], 'joint_actions': [], 'global_rewards': [], 'global_next_states':[], 'next_observations': []}
                hidden_state = [th.zeros(1, 1, actor_hidden_dim).to(device) for _ in range(num_agents)]
                prev_actions = [th.zeros(1, action_dim).to(device) for _ in range(num_agents)]

                for t in range(t_max):

                    actions = []
                    observations = []

                    # Collect actions for each agent
                    global_state_raw = env.get_global_state()
                    global_state = HCRA.process_global_state(global_state_raw, num_patients, num_specialists, num_specialties)
                    global_state = th.tensor(global_state, dtype=th.float32).squeeze().to(device)
                    for a in range(num_agents):
                        raw_observation = env.get_observation(a)
                        observation = HCRA.process_observation(raw_observation, max_patients_per_hospital, max_specialists_per_hospital, num_specialties)
                        observation = th.tensor(observation, dtype=th.float32).unsqueeze(0).to(device)
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
                    joint_action = actions
                    mapped_actions = [action_mapping[a_idx] for a_idx in actions]
                    global_reward, individual_rewards, done = env.step(mapped_actions)
                    
                    raw_global_next_state = env.get_global_state()
                    global_next_state = HCRA.process_global_state(raw_global_next_state, num_patients, num_specialists, num_specialties)
                    global_next_state = th.tensor(global_next_state, dtype=th.float32).squeeze().to(device)
                    next_observations = []
                    for a in range(num_agents):
                        raw_next_observation = env.get_observation(a)
                        next_observation = HCRA.process_observation(raw_next_observation, max_patients_per_hospital, max_specialists_per_hospital, num_specialties)
                        next_observation = th.tensor(next_observation, dtype=th.float32).squeeze().to(device)
                        next_observations.append(next_observation)

                    buffer['global_states'].append(global_state)
                    buffer['observations'].append(observations)
                    buffer['joint_actions'].append(joint_action)
                    buffer['global_rewards'].append(global_reward)
                    buffer['global_next_states'].append(global_next_state)
                    buffer['next_observations'].append(next_observations)

                    # global_state = global_next_state
                    total_rewards += global_reward
                episode_rewards.append(np.mean(total_rewards))
                episode += 1

                rtrns = []
                if done:
                    R = 0
                else:
                    last_global_state = buffer['global_next_states'][-1]
                    last_observations = buffer['next_observations'][-1]
                    # Prepare input for critic
                    agent_indices = th.arange(num_agents).to(device)
                    agent_id = F.one_hot(agent_indices, num_classes=num_agents).float()  # [num_agents, num_agents]
                    prev_joint_actions = buffer['joint_actions'][-1]
                    prev_joint_actions = th.tensor(prev_joint_actions, dtype=th.long).to(device)
                    prev_joint_actions_one_hot = F.one_hot(prev_joint_actions, num_classes=action_dim)
                    prev_joint_actions_concat = prev_joint_actions_one_hot.reshape(1, action_dim * num_agents)
                    x = th.cat([last_global_state.unsqueeze(0).repeat(num_agents, 1), th.stack(last_observations), agent_id, prev_joint_actions_concat.repeat(num_agents, 1)], dim=-1)
                    V, _ = critic(x, None)
                    R = V.mean().item()

                for global_reward in reversed(buffer['global_rewards']):
                    R = global_reward + gamma * R
                    rtrns.insert(0, R)
                rtrns = th.tensor(np.array(rtrns), dtype=th.float32).to(device)
                #rtrns_per_agent = np.repeat(np.array(rtrns), num_agents, axis=1)  # Shape [batch_size, num_agents, 1]
                
                batch_buffer[e]['global_states'].extend(buffer['global_states'])
                batch_buffer[e]['observations'].extend(buffer['observations'])
                batch_buffer[e]['joint_actions'].extend(buffer['joint_actions'])
                batch_rtrns[e].extend(rtrns)

                if (episode) % test_interval == 0:
                    actor_shared.eval()
                    test_reward = MAA2CtesterPS.test_MAA2C_ParameterSharing(env_params, actor_shared, num_test_episodes, t_max, num_agents, actor_hidden_dim,
                                                                            action_dim, action_mapping, max_patients_per_hospital, max_specialists_per_hospital,
                                                                            num_specialties, seed=69)
                    test_rewards.append(test_reward)
                    csv_writer.writerow([test_reward])
                    csv_file.flush()
                    
                    plt.figure(1)
                    plt.clf()
                    plt.plot(test_rewards, label="Test Reward")
                    plt.xlabel("Test Interval")
                    plt.ylabel("Return")
                    plt.title("Test Return Over Time MAA2C")
                    plt.legend()
                    plt.grid(True)
                    plt.draw()
                    plt.pause(1)
                    
                    print(f'Training reward at episode {episode}: {test_reward:.2f}')
                    actor_shared.train()

            batch_training = BatchTraining()
            batch_global_states, batch_observations, batch_joint_actions, batch_rtrns = batch_training.collate_batch(batch_buffer, batch_rtrns)

            #print(f"global_states: {batch_global_states.size()}") # [batch_size, timesteps, state_dim]
            #print(f"obs: {batch_observations.size()}") # [batch_size, timesteps, num_agents, obs_dim]
            #print(f"actions: {batch_actions.size()}") # [batch_size, timesteps, num_agents] - NOT ONE HOT ENCODED
            #print(f"hidden states: {batch_hidden_states.size()}")  # [batch_size, timesteps, num_agents, hidden_dim]
            #print(f"returns: {batch_rtrns.size()}")  # [batch_size, timesteps]

            # Update Centralized Critic and Actor
            # Shape [batch_size, time_steps, action_dim * num_agents] - ONE HOT ENCODED
            batch_size, timesteps, _ = batch_joint_actions.shape
            batch_prev_joint_actions = th.zeros_like(batch_joint_actions) 
            batch_prev_joint_actions[:, 1:, :] = batch_joint_actions[:, :-1, :] # Shift actions to get previous joint actions
            batch_prev_joint_actions = F.one_hot(batch_prev_joint_actions.long(), num_classes=action_dim)
            batch_prev_joint_actions_concat = batch_prev_joint_actions.reshape(batch_size, timesteps, action_dim * num_agents)
            
            hidden_state_actor = th.zeros(1, batch_size, actor_hidden_dim).to(device)
            hidden_state_critic = th.zeros(1, batch_size, critic_hidden_dim).to(device)

            optimizer_critic.zero_grad()
            total_critic_loss = 0
            for a in range(num_agents):
                # Critic
                agent_indices = th.full((batch_size, timesteps), a, dtype=th.long).to(device) 
                agent_id = F.one_hot(agent_indices, num_classes=num_agents).float() # [batch_size, timesteps, num_agents]
                x = th.cat([batch_global_states, batch_observations[:, :, a, :], agent_id, batch_prev_joint_actions_concat], dim=-1)
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
                actions = batch_joint_actions[:, :, a]
                prev_actions = batch_prev_joint_actions[:, :, a, :]
                agent_indices = th.full((batch_size, timesteps), a, dtype=th.long).to(device) 
                agent_id = F.one_hot(agent_indices, num_classes=num_agents).float() # [batch_size, timesteps, num_agents]
                
                x = th.cat([agent_observation, agent_id, prev_actions], dim=-1) # Shape: [batch_size, timesteps, input_dim]

                logits, _ = actor_shared(x, hidden_state_actor)
                softmax_probabilities = F.softmax(logits, dim=-1)
                action_dist = Categorical(probs=softmax_probabilities)

                log_probs = action_dist.log_prob(actions)
                entropies = action_dist.entropy()

                x_critic = th.cat([batch_global_states, agent_observation, agent_id, batch_prev_joint_actions_concat], dim=-1)
                V_values, _ = critic(x_critic, hidden_state_critic)
                V_values = V_values.squeeze(-1)

                # Calculate Advantage
                advantages = (batch_rtrns - V_values).detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Actor Loss
                actor_loss = -(log_probs * advantages).mean() - tau * entropies.mean()
                total_actor_loss += actor_loss

            # Average losses across agents
            total_actor_loss /= num_agents

            # Actor Update
            total_actor_loss.backward()
            optimizer_actor_shared.step()

        return episode_rewards, test_rewards