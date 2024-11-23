import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
import csv
import matplotlib.pyplot as plt

from Networks.Actors.mappo_actor_gru import Actor
from Networks.Critics.mappo_critic_gru import Critic

from Helpers.PPO.mappo_helper import Helper, BatchProcessing
from Benchmarkers.mappo_test_po import MAPPOtester

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class MAPPO_Trainer:
    def __init__(self):
        self.rtrn_running_mean = 0
        self.rtrn_running_var = 1
        self.rtrn_count = 1e-4
        self.adv_running_mean = 0
        self.adv_running_var = 1
        self.adv_count = 1e-4

    def train_MAPPO(self, trial_run, env, env_name, state_dim, observation_dim, action_dim, gamma, actor_hidden_dim, critic_hidden_dim,
                    value_dim, alpha, beta, lam, entropy_coef, eps_clip, num_mini_batches, epochs, t_max,
                    test_interval, training_iteration, num_test_episodes, num_agents, batch_size
                    ):
        
        csv_file = open(f'MAPPO_trial_{trial_run}_HCRA.csv', 'a', newline='')
        csv_writer = csv.writer(csv_file)
        
        actor_shared = Actor(observation_dim, action_dim, actor_hidden_dim, num_agents).to(device)
        centralized_critic = Critic(state_dim, action_dim, critic_hidden_dim, value_dim, num_agents).to(device)

        actor_optimizer = th.optim.Adam(actor_shared.parameters(), lr=alpha)
        critic_optimizer = th.optim.Adam(centralized_critic.parameters(), lr=beta)

        episode_rewards = []
        test_rewards = []

        episode = 0

        for it in range(training_iteration):
            batch_buffer = [{'global_states': [],
                             'observations': [],
                             'joint_actions': [],
                             'log_probs': [],
                             'values': []
                            } for _ in range (batch_size)]
            
            batch_rtrns = [[] for _ in range(batch_size)]
            batch_advantages = [[] for _ in range(batch_size)]

            for e in range(batch_size):
                buffer = {'global_states':[],
                          'observations': [],
                          'joint_actions': [],
                          'individual_rewards': [],
                          'global_next_states':[],
                          'next_observations': [],
                          'log_probs': [],
                          'values': [],
                          'dones': []}

                hidden_states_actor = [th.zeros(1, 1, actor_hidden_dim).to(device) for _ in range(num_agents)]
                hidden_states_critic = [th.zeros(1, 1, critic_hidden_dim).to(device) for _ in range(num_agents)]
                prev_actions = [th.zeros(1, action_dim).to(device) for _ in range(num_agents)]
                prev_joint_action = th.zeros(1, num_agents * action_dim).to(device)

                done = False
                total_reward = 0

                for t in range(t_max):

                    actions = []
                    old_log_probs = []
                    observations = []
                    fp_states = []
                    curr_prev_actions = []

                    # Collect actions for each agent
                    global_state = env.get_global_state(t)
                    global_state = th.tensor(global_state, dtype=th.float32).squeeze().to(device)
                    for a in range(num_agents):
                        with th.no_grad():
                            observation = env.get_observation(a, t)
                            observation = th.tensor(observation, dtype=th.float32).to(device)
                            observations.append(observation.squeeze())
                            agent_id = th.nn.functional.one_hot(th.tensor(a), num_classes=num_agents).float().unsqueeze(0).to(device)
                            fp_state = Helper.create_fp_state(global_state, observation, a, agent_id)
                            fp_states.append(fp_state)

                            prev_action_a = prev_actions[a]
                            x = th.cat([observation, agent_id, prev_action_a], dim=-1).unsqueeze(0)

                            logits, hidden_state_a = actor_shared(x, hidden_states_actor[a])
                            hidden_states_actor[a] = hidden_state_a

                            action, log_prob, _ = actor_shared.action_sampler(logits)

                            action_idx = action.item()
                            old_log_probs.append(log_prob.squeeze())
                            one_hot_action = th.nn.functional.one_hot(th.tensor(action_idx), num_classes=action_dim).float().unsqueeze(0).to(device)
                            prev_actions[a] = one_hot_action
                            curr_prev_actions.append(one_hot_action)
                            actions.append(action_idx)

                    joint_action = actions
                    prev_joint_action = th.cat(curr_prev_actions, dim=-1)

                    values = []
                    for a in range(num_agents):
                        with th.no_grad():
                            state_input = fp_states[a].unsqueeze(0)
                            prev_action_c = prev_joint_action
                            x = th.cat([state_input, prev_action_c], dim=-1).unsqueeze(0)
                            value, hidden_state_c = centralized_critic(x, hidden_states_critic[a])
                            hidden_states_critic[a] = hidden_state_c
                            values.append(value.squeeze())

                    global_reward, individual_rewards, done = env.reward_step(actions)
                    individual_rewards = [th.tensor(r, dtype=th.float32).to(device) for r in individual_rewards]
                    
                    global_next_state = env.get_state([0, 0], 0, t + 1)
                    global_next_state = th.tensor(global_next_state, dtype=th.float32).squeeze().to(device)
                    next_observations = []
                    for a in range(num_agents):
                        next_observation = env.get_state4([a, 0], 0, t + 1)
                        next_observation = th.tensor(next_observation, dtype=th.float32).squeeze().to(device)
                        next_observations.append(next_observation)                                

                    buffer['global_states'].append(fp_states)
                    buffer['observations'].append(observations)
                    buffer['joint_actions'].append(joint_action)
                    buffer['individual_rewards'].append(individual_rewards)
                    buffer['global_next_states'].append(global_next_state)
                    buffer['next_observations'].append(next_observations)
                    buffer['log_probs'].append(old_log_probs)
                    buffer['values'].append(values)
                    buffer['dones'].append(done)
                    
                    total_reward += global_reward

                    if done:
                        break

                returns, advantages = Helper.compute_GAE(buffer['individual_rewards'], buffer['values'], buffer['dones'], gamma, lam)
                
                #normalized_returns, self.rtrn_running_mean, self.rtrn_running_var, self.rtrn_count = Helper.popart_normalize(
                #    returns, self.rtrn_running_mean, self.rtrn_running_var, self.rtrn_count)

                #normalized_advantages, self.adv_running_mean, self.adv_running_var, self.adv_count = Helper.popart_normalize(
                #    advantages, self.adv_running_mean, self.adv_running_var, self.adv_count)

                episode_rewards.append(total_reward)
                episode += 1

                batch_buffer[e]['global_states'].extend(buffer['global_states'])
                batch_buffer[e]['observations'].extend(buffer['observations'])
                batch_buffer[e]['joint_actions'].extend(buffer['joint_actions'])
                batch_buffer[e]['log_probs'].extend(buffer['log_probs'])
                batch_buffer[e]['values'].extend(buffer['values'])
                batch_rtrns[e].extend(returns)
                batch_advantages[e].extend(advantages)

                if (episode) % test_interval == 0:
                    actor_shared.eval()
                    test_reward = MAPPOtester.test_MAPPO(actor_shared, num_test_episodes, num_agents, actor_hidden_dim, action_dim)
                    test_rewards.append(test_reward)
                    csv_writer.writerow([test_reward])
                    csv_file.flush()

                    plt.figure(1)
                    plt.clf()
                    plt.plot(test_rewards, label="Test Reward")
                    plt.xlabel("Test Interval")
                    plt.ylabel("Return")
                    plt.title("Test Return Over Time MAPPO")
                    plt.legend()
                    plt.grid(True)
                    plt.draw()
                    plt.pause(1)

                    print(f'Test reward at episode {episode}: {test_reward:.2f}')
                    actor_shared.train()

            batch_processing = BatchProcessing()
            batch_global_states, batch_observations, batch_joint_actions, batch_log_probs, batch_values, batch_returns, batch_advantages = batch_processing.collate_batch(batch_buffer, batch_rtrns, batch_advantages)
        
            #print("Global States: ", batch_global_states.shape)
            #print("Observations: ", batch_observations.shape)
            #print("Log Probs: ", batch_log_probs.shape)
            #print("Values: ", batch_values.shape)
            #print("Returns: ", batch_returns.shape)
            #print("Advantages: ", batch_advantages.shape)

            # After every batch, update the actor and critic using minibatching

            dataset = th.utils.data.TensorDataset(batch_global_states, batch_observations, batch_joint_actions,
                                                  batch_log_probs, batch_values, batch_returns, batch_advantages)
            mini_batch_size = len(dataset) // num_mini_batches
            dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

            for _ in range(epochs):
                for batch in dataloader:
                    global_states_mb, observations_mb, joint_actions_mb, log_probs_mb, values_mb, returns_mb, advantages_mb = batch

                    global_states_mb = global_states_mb.to(device)
                    joint_actions_mb = joint_actions_mb.to(device)
                    log_probs_mb = log_probs_mb.to(device)
                    values_mb = values_mb.to(device)
                    returns_mb = returns_mb.to(device)
                    advantages_mb = advantages_mb.to(device)

                    #print("Global States mb: ", global_states_mb.shape)
                    #print("Observations mb: ", observations_mb.shape)
                    #print("Joint Actions mb: ", joint_actions_mb.shape)
                    #print("Log Probs mb: ", log_probs_mb.shape)
                    #print("Values mb: ", values_mb.shape)
                    #print("Returns mb: ", returns_mb.shape)
                    #print("Advantages mb: ", advantages_mb.shape)
                    
                    mb_size, timesteps, _ = joint_actions_mb.shape
                    prev_joint_actions_mb = th.zeros_like(joint_actions_mb).to(device)
                    prev_joint_actions_mb[:, 1:, :] = joint_actions_mb[:, :-1, :]
                    prev_joint_actions_mb = F.one_hot(prev_joint_actions_mb.long(), num_classes=action_dim)
                    prev_joint_actions_mb_concat = prev_joint_actions_mb.reshape(mb_size, timesteps, action_dim * num_agents)

                    hidden_state_actor = [th.zeros(1, mb_size, actor_hidden_dim).to(device) for _ in range(num_agents)]
                    hidden_state_critic = [th.zeros(1, mb_size, critic_hidden_dim).to(device) for _ in range(num_agents)]

                    # Critic Update
                    critic_optimizer.zero_grad()
                    total_critic_loss = 0
                    for a in range(num_agents):
                        x = th.cat([global_states_mb[:, :, a, :], prev_joint_actions_mb_concat], dim=-1)
                        values, _ = centralized_critic(x, hidden_state_critic[a])
                        values = values.squeeze(-1)
                        critic_loss = Helper.critic_loss_fn(values, values_mb[:, :, a], returns_mb[:, :, a], eps_clip)
                        total_critic_loss += critic_loss
                    total_critic_loss /= num_agents
                    total_critic_loss.backward()
                    critic_optimizer.step()

                    # Actor Update
                    actor_optimizer.zero_grad()
                    total_actor_loss = 0
                    for a in range(num_agents):
                        agent_observation = observations_mb[:, :, a, :]
                        actions = joint_actions_mb[:, :, a]
                        prev_actions = prev_joint_actions_mb[:, :, a, :]
                        old_log_probs = log_probs_mb[:, :, a]
                        agent_indices = th.full((mb_size, timesteps), a, dtype=th.long).to(device) 
                        agent_id = F.one_hot(agent_indices, num_classes=num_agents).float() # [batch_size, timesteps, num_agents]

                        x = th.cat([agent_observation, agent_id, prev_actions], dim=-1)

                        logits, _ = actor_shared(x, hidden_state_actor[a])
                        dist = Categorical(logits=logits)
                        new_log_probs = dist.log_prob(actions)

                        actor_loss = Helper.actor_loss_fn(new_log_probs, old_log_probs, advantages_mb[:, :, a], eps_clip)

                        entropy = dist.entropy().mean()
                        actor_loss = actor_loss - entropy_coef * entropy

                        total_actor_loss += actor_loss

                    total_actor_loss /= num_agents
                    total_actor_loss.backward()
                    actor_optimizer.step()

        return episode_rewards, test_rewards