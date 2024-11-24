import torch as th
import numpy as np
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class Helper:
    def __init__ (self):
        pass
    
    def create_action_mapping(num_patients, num_specialists):
        action_mapping = {}
        action_counter = 0
        for patient_id in range(num_patients):
            for specialist_id in range(num_specialists):
                action_mapping[action_counter] = (patient_id, specialist_id)
                action_counter += 1
        return action_mapping
    
    def process_observation(observation, max_patients_per_hospital, max_specialists_per_hospital, num_specialties):
        # Process patients
        patients = observation['patients']
        patient_features = []
        for condition_type, wait_time in patients:
            condition_one_hot = np.eye(num_specialties)[condition_type]
            patient_feature = np.concatenate([condition_one_hot, [wait_time]])
            patient_features.append(patient_feature)
        # Pad or truncate to max_patients_per_hospital
        while len(patient_features) < max_patients_per_hospital:
            patient_features.append(np.zeros(num_specialties + 1))
        patient_features = np.array(patient_features[:max_patients_per_hospital]).flatten()

        # Process specialists
        specialists = observation['specialists']
        specialist_features = []
        for specialty_type, available in specialists:
            specialty_one_hot = np.eye(num_specialties)[specialty_type]
            available_flag = [1.0 if available else 0.0]
            specialist_feature = np.concatenate([specialty_one_hot, available_flag])
            specialist_features.append(specialist_feature)
        # Pad or truncate to max_specialists_per_hospital
        while len(specialist_features) < max_specialists_per_hospital:
            specialist_features.append(np.zeros(num_specialties + 1))
        specialist_features = np.array(specialist_features[:max_specialists_per_hospital]).flatten()

        # Combine features
        observation_vector = np.concatenate([patient_features, specialist_features])
        return observation_vector
    
    def process_global_state(global_state_raw, max_patients, max_specialists, num_specialties):
        # Process patients
        patients = global_state_raw['patients']
        patient_features = []
        for condition_type, wait_time in patients:
            condition_one_hot = np.eye(num_specialties)[condition_type]
            patient_feature = np.concatenate([condition_one_hot, [wait_time]])
            patient_features.append(patient_feature)
        # Pad or truncate to max_patients
        while len(patient_features) < max_patients:
            patient_features.append(np.zeros(num_specialties + 1))
        patient_features = np.array(patient_features[:max_patients]).flatten()

        # Process specialists
        specialists = global_state_raw['specialists']
        specialist_features = []
        for specialty_type, available in specialists:
            specialty_one_hot = np.eye(num_specialties)[specialty_type]
            available_flag = [1.0 if available else 0.0]
            specialist_feature = np.concatenate([specialty_one_hot, available_flag])
            specialist_features.append(specialist_feature)
        # Pad or truncate to max_specialists
        while len(specialist_features) < max_specialists:
            specialist_features.append(np.zeros(num_specialties + 1))
        specialist_features = np.array(specialist_features[:max_specialists]).flatten()

        # Process hospital queues
        hospital_queue_lengths = np.array(list(global_state_raw['hospital_queues'].values()))
        
        # Combine all features
        global_state_vector = np.concatenate([patient_features, specialist_features, hospital_queue_lengths])
        return global_state_vector



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