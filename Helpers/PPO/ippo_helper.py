import torch as th
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class Helper:
    def __init__(self):
        pass

    @staticmethod
    def actor_loss_fn(log_probs, old_log_probs, advantages, clip_param):
        # Calculate ratio (pi_theta / pi_theta_old)
        imp_weights = th.exp(log_probs - old_log_probs)
        # Calculate surrogate losses
        surr1 = imp_weights * advantages
        surr2 = th.clamp(imp_weights, 1.0 - clip_param, 1.0 + clip_param) * advantages
        # Return the minimum surrogate loss for clipping
        return -th.min(surr1, surr2).mean()

    @staticmethod
    def critic_loss_fn(values, old_values, returns, clips_param):
        value_clip= old_values + th.clamp(values - old_values, -clips_param, clips_param)
        loss_unclipped = (values - returns).pow(2)
        loss_clipped = (value_clip - returns).pow(2)
        value_loss = th.max(loss_unclipped, loss_clipped).mean()

        return value_loss

    @staticmethod
    def compute_GAE(rewards, values, dones, gamma, gae_lambda):
        num_agents = len(values[0])
        advantages = [[] for _ in range(num_agents)]
        returns = [[] for _ in range(num_agents)]

        values = values + [[th.zeros_like(values[0][0]) for _ in range(num_agents)]]

        gae = [th.zeros(1, dtype=th.float32).to(device) for _ in range(num_agents)]
        R = [th.zeros(1, dtype=th.float32).to(device) for _ in range(num_agents)]

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            for a in range(num_agents):
                delta = rewards[t][a] + gamma * values[t + 1][a] * next_non_terminal - values[t][a]
                gae[a] = delta + gamma * gae_lambda * next_non_terminal * gae[a]
                advantages[a].insert(0, gae[a].clone())

                # Compute the Returns
                R[a] = rewards[t][a] + gamma * R[a] * next_non_terminal
                returns[a].insert(0, R[a].clone())

        returns = [th.stack(agent_returns) for agent_returns in returns]
        advantages = [th.tensor(agent_advantages) for agent_advantages in advantages]

        returns = th.stack(returns, dim=1)
        advantages = th.stack(advantages, dim=1)

        return returns.squeeze(), advantages.squeeze()

    @staticmethod
    def popart_normalize(tensor, running_mean, running_var, count):

        batch_mean = tensor.mean()
        batch_var = tensor.var(unbiased=False)
        batch_count = len(tensor)

        # Update running statistics
        delta = batch_mean - running_mean
        new_count = count + batch_count

        new_mean = running_mean + delta * batch_count / new_count
        m_a = running_var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / new_count
        new_var = M2 / new_count

        new_var = new_var + 1e-8

        normalized_tensor = (tensor - new_mean) / th.sqrt(new_var)

        return normalized_tensor, new_mean, new_var, new_count
    
class BatchProcessing:
    def __init__(self):
        pass

    def collate_batch(self, batch_buffer, batch_rtrns, batch_advantages):
        batch_joint_actions = []
        batch_observations = []
        batch_log_probs = []
        batch_values = []
        batch_rtrns_list = []
        batch_advantages_list = []

        # Extract data from buffer
        for episode_buffer, episode_rtrns, episode_advantages in zip(batch_buffer, batch_rtrns, batch_advantages):
            # Process Observations per Agent
            episode_observations = []
            for obs_per_timestep in episode_buffer['observations']:
                # obs_per_timestep: list of [observation_dim] per agent
                obs_per_timestep_tensor = th.stack(obs_per_timestep)  # Shape: [num_agents, observation_dim]
                episode_observations.append(obs_per_timestep_tensor)
            episode_observations_tensor = th.stack(episode_observations)  # Shape: [timesteps, num_agents, observation_dim]
            batch_observations.append(episode_observations_tensor)

            # Stack Joint Actions
            episode_actions = []
            for action_per_timestep in episode_buffer['joint_actions']:
                # action_per_timestep: list of actions per agent
                action_tensor = th.tensor(action_per_timestep, dtype=th.float32)  # Shape: [num_agents]
                episode_actions.append(action_tensor)
            episode_actions_tensor = th.stack(episode_actions)  # Shape: [timesteps, num_agents]
            batch_joint_actions.append(episode_actions_tensor)

            # Process Log Probs
            episode_log_probs = []
            for log_probs_per_timestep in episode_buffer['log_probs']:
                log_probs_per_timestep_tensor = th.stack(log_probs_per_timestep)
                episode_log_probs.append(log_probs_per_timestep_tensor)
            episode_log_probs_tensor = th.stack(episode_log_probs)
            batch_log_probs.append(episode_log_probs_tensor)

            # Process Old Values
            episode_values = []
            for values_per_timestep in episode_buffer['values']:
                values_per_timestep_tensor = th.stack(values_per_timestep)
                episode_values.append(values_per_timestep_tensor)
            episode_values_tensor = th.stack(episode_values)
            batch_values.append(episode_values_tensor)

            # Stack Returns
            episode_rtrns_tensor = th.stack(episode_rtrns)  # Shape: [timesteps]
            batch_rtrns_list.append(episode_rtrns_tensor)            

            # Process Advantages
            episode_advantages_tensor = th.stack(episode_advantages)
            batch_advantages_list.append(episode_advantages_tensor)

        # Stack over episodes to create batch tensors
        batch_observations = th.stack(batch_observations).to(device)  # Shape: [batch_size, timesteps, num_agents, observation_dim]
        batch_joint_actions = th.stack(batch_joint_actions).to(device)  # Shape: [batch_size, timesteps, num_agents]
        batch_rtrns = th.stack(batch_rtrns_list).to(device)  # Shape: [batch_size, timesteps]
        batch_log_probs = th.stack(batch_log_probs).to(device)
        batch_values = th.stack(batch_values).to(device)
        batch_advantages = th.stack(batch_advantages_list).to(device)
        

        return batch_observations, batch_joint_actions, batch_log_probs, batch_values, batch_rtrns, batch_advantages