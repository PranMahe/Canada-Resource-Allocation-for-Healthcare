import torch as th
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class BatchTraining:
    def __init__ (self):
        pass

    def collate_batch(self, batch_buffer, batch_rtrns):
        batch_global_states = []
        batch_observations = []
        batch_actions = []
        batch_rtrns_list = []

        for episode_buffer, episode_rtrns in zip(batch_buffer, batch_rtrns):
            # Process Global States
            episode_states = []
            for state in episode_buffer['global_states']:
                # states_per_timestep: list of [state_dim]
                episode_states.append(state)
            episode_states_tensor = th.stack(episode_states).to(device) # Shape [timesteps, state_dim]
            batch_global_states.append(episode_states_tensor)
            
            # Process observations per agent
            episode_observations = []
            for obs_per_timestep in episode_buffer['observations']:
                # obs_per_timestep: list of [observation_dim] per agent
                obs_per_timestep_tensor = th.stack(obs_per_timestep).to(device)  # Shape: [num_agents, observation_dim]
                episode_observations.append(obs_per_timestep_tensor)
            episode_observations_tensor = th.stack(episode_observations).to(device)   # Shape: [timesteps, num_agents, observation_dim]
            batch_observations.append(episode_observations_tensor)

            # Stack joint actions: [timesteps, num_agents, action_features]
            episode_actions = []
            for action_per_timestep in episode_buffer['joint_actions']:
                # action_per_timestep: list of actions per agent
                action_tensor = th.tensor(action_per_timestep, dtype=th.float32).to(device)   # Shape: [num_agents]
                episode_actions.append(action_tensor)
            episode_actions_tensor = th.stack(episode_actions).to(device)   # Shape: [timesteps, num_agents]
            batch_actions.append(episode_actions_tensor)

            # Stack returns: [timesteps]
            episode_rtrns_tensor = th.tensor(episode_rtrns, dtype=th.float32).to(device)   # Shape: [timesteps]
            batch_rtrns_list.append(episode_rtrns_tensor)

        # Stack over episodes to create batch tensors
        batch_global_states = th.stack(batch_global_states).to(device) # Shape: [batch_size, timesteps, state_dim]
        batch_observations = th.stack(batch_observations).to(device)  # Shape: [batch_size, timesteps, num_agents, observation_dim]
        batch_actions = th.stack(batch_actions).to(device)  # Shape: [batch_size, timesteps, num_agents]
        batch_rtrns = th.stack(batch_rtrns_list).to(device)  # Shape: [batch_size, timesteps]
        
        return batch_global_states, batch_observations, batch_actions, batch_rtrns