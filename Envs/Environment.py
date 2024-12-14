import torch as th
import numpy as np
import random

class HCRA:
    def __init__(self, num_agents, num_patients, num_specialists, num_specialties, num_hospitals,
                 max_wait_time, episode_length, hospital_capacities, seed=None):

        # For testing consistency
        self.seed = seed
        self._initialize_random_seed()

        # General Parameters
        self.num_agents = num_agents
        self.num_patients = num_patients
        self.num_specialists = num_specialists
        self.num_specialties = num_specialties
        self.num_hospitals = num_hospitals
        self.max_wait_time = max_wait_time
        self.episode_length = episode_length

        # Initialize hospitals with capacities
        self.hospital_capacities = hospital_capacities

        # Initialize data structures
        self.patients = []
        self.specialists = []
        self.hospitals = {}
        self.current_step = 0

        self._initialize_hospitals()
        self._initialize_specialists()
        self._initialize_patients()
        self._initialize_agent_hospital_mapping()
        self._initialize_dimensions()

    # Assign agents to hospitals based on predefined mapping
    def _initialize_agent_hospital_mapping(self):
        self.agent_hospital_mapping = {
            0: 0,  # Agent 0 assigned to Hospital 0 (largest)
            1: 1,  # Agent 1 assigned to Hospital 1
            2: 2,  # Agent 2 assigned to Hospital 2
            3: 3,  # Agent 3 assigned to Hospital 3 (smallest)
        }

    # Initializing action, observation, global state dimensions for neural networks
    def _initialize_dimensions(self):
        # Initialize action mapping
        self.action_mapping = self.create_action_mapping(
            self.num_patients, self.num_specialists, self.num_hospitals)
        self.action_dim = len(self.action_mapping)

        # Compute maximum capacities
        self.max_patients_per_hospital = max(cap['max_patients'] for cap in self.hospital_capacities.values())
        self.max_specialists_per_hospital = max(cap['max_specialists'] for cap in self.hospital_capacities.values())

        # Compute observation_dim
        sample_observation_raw = self.get_observation(agent_id=0)
        sample_observation_vector = self.process_observation(
            sample_observation_raw,
            max_patients_per_hospital=self.max_patients_per_hospital,
            max_specialists_per_hospital=self.max_specialists_per_hospital,
            num_specialties=self.num_specialties
        )
        self.observation_dim = len(sample_observation_vector)

        # Compute state_dim
        sample_global_state_raw = self.get_global_state()
        sample_global_state_vector = self.process_global_state(
            sample_global_state_raw,
            max_patients=self.num_patients,
            max_specialists=self.num_specialists,
            num_specialties=self.num_specialties
        )
        self.state_dim = len(sample_global_state_vector)

    # Setting up hospital information for agents
    def _initialize_hospitals(self):
        for h_id in range(self.num_hospitals):
            self.hospitals[h_id] = {
                'hospital_id': h_id,
                'specialists': [],
                'queue': []
            }

    # Assigning specialist information using the number of available specialists
    def _initialize_specialists(self):
        specialist_ids = list(range(self.num_specialists))
        random.shuffle(specialist_ids)
        specialist_index = 0

        for h_id, capacities in self.hospital_capacities.items():
            max_specialists = capacities['max_specialists']
            for _ in range(max_specialists):
                if specialist_index >= self.num_specialists:
                    break  # All specialists have been assigned
                s_id = specialist_ids[specialist_index]
                specialty_type = random.randint(0, self.num_specialties - 1)
                specialist = {
                    'specialist_id': s_id,
                    'specialty_type': specialty_type,
                    'hospital_id': h_id,
                    'available': True,
                    'assignments': 0  # Track number of assignments
                }
                self.specialists.append(specialist)
                self.hospitals[h_id]['specialists'].append(specialist)
                specialist_index += 1

    # Assigning patient information using the number of available patients
    def _initialize_patients(self):
        patient_ids = list(range(self.num_patients))
        random.shuffle(patient_ids)
        patient_index = 0

        for h_id, capacities in self.hospital_capacities.items():
            max_patients = capacities['max_patients']
            for _ in range(max_patients):
                if patient_index >= self.num_patients:
                    break  # All patients have been assigned
                p_id = patient_ids[patient_index]
                condition_type = random.randint(0, self.num_specialties - 1)
                patient = {
                    'patient_id': p_id,
                    'condition_type': condition_type,
                    'wait_time': random.randint(1, self.max_wait_time),
                    'hospital_id': h_id  # Track which hospital the patient is in
                }
                self.patients.append(patient)
                self.hospitals[h_id]['queue'].append(patient)
                patient_index += 1

    # Randomization of the environment
    def _initialize_random_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            th.manual_seed(self.seed)

    # Maps actions from raw action form to environment freindly mapped form
    @staticmethod
    def create_action_mapping(num_patients, num_specialists, num_hospitals):
        action_mapping = {}
        action_counter = 0
        # Assignment actions
        for patient_id in range(num_patients):
            for specialist_id in range(num_specialists):
                action_mapping[action_counter] = ('assign', patient_id, specialist_id)
                action_counter += 1
        # Transfer actions
        for patient_id in range(num_patients):
            for hospital_id in range(num_hospitals):
                action_mapping[action_counter] = ('transfer', patient_id, hospital_id)
                action_counter += 1
        return action_mapping

    # Helper for converting raw observation information into readable vector
    @staticmethod
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

        # Process summary information
        summary_info = observation['summary_info']
        total_patients = summary_info['total_patients']
        average_wait_time = summary_info['average_wait_time']
        total_capacity = summary_info['total_capacity']
        average_utilization = summary_info['average_utilization']

        # Check for Nan and inf
        if not np.isfinite(average_wait_time):
            average_wait_time = 0.0
        if not np.isfinite(average_utilization):
            average_utilization = 0.0

        summary_features = np.array([
            total_patients,
            average_wait_time,
            total_capacity,
            average_utilization
        ])

        # Combine features
        observation_vector = np.concatenate([patient_features, specialist_features, summary_features])

        # Replace any NaN or inf values with 0.0
        observation_vector = np.nan_to_num(observation_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return observation_vector

    # Helper for converting raw global state information into readable vector 
    @staticmethod
    def process_global_state(global_state_raw, max_patients, max_specialists, num_specialties):
        # Process patients
        patients = global_state_raw['patients']
        patient_features = []
        for condition_type, wait_time, hospital_id in patients:
            condition_one_hot = np.eye(num_specialties)[condition_type]
            hospital_one_hot = np.eye(global_state_raw['num_hospitals'])[hospital_id]
            patient_feature = np.concatenate([condition_one_hot, [wait_time], hospital_one_hot])
            patient_features.append(patient_feature)
        
        # Pad or truncate to max_patients
        while len(patient_features) < max_patients:
            patient_features.append(np.zeros(num_specialties + 1 + global_state_raw['num_hospitals']))
        patient_features = np.array(patient_features[:max_patients]).flatten()

        # Process specialists
        specialists = global_state_raw['specialists']
        specialist_features = []
        for specialty_type, available, hospital_id in specialists:
            specialty_one_hot = np.eye(num_specialties)[specialty_type]
            available_flag = [1.0 if available else 0.0]
            hospital_one_hot = np.eye(global_state_raw['num_hospitals'])[hospital_id]
            specialist_feature = np.concatenate([specialty_one_hot, available_flag, hospital_one_hot])
            specialist_features.append(specialist_feature)
        
        # Pad or truncate to max_specialists
        while len(specialist_features) < max_specialists:
            specialist_features.append(np.zeros(num_specialties + 1 + global_state_raw['num_hospitals']))
        specialist_features = np.array(specialist_features[:max_specialists]).flatten()

        # Process hospital queues
        hospital_queue_lengths = np.array(list(global_state_raw['hospital_queues'].values()))
        
        # Check for NaN and inf
        if np.isnan(hospital_queue_lengths).any():
            hospital_queue_lengths = np.nan_to_num(hospital_queue_lengths, nan=0.0, posinf=0.0, neginf=0.0)

        # Combine all features
        global_state_vector = np.concatenate([patient_features, specialist_features, hospital_queue_lengths])

        # Replace any NaNs or inf values with 0.0
        global_state_vector = np.nan_to_num(global_state_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return global_state_vector

    # Reset the HCRA environment
    def reset(self, seed=None):
        self.current_step = 0
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            th.manual_seed(self.seed)

        self.patients = []
        self.specialists = []
        for h_id in self.hospitals:
            self.hospitals[h_id]['specialists'] = []
            self.hospitals[h_id]['queue'] = []
        self._initialize_specialists()
        self._initialize_patients()
        return self.get_global_state()

    # Main environment step function
    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        done = False

        # Process each agent's action
        for agent_id, action in enumerate(actions):
            if action is None:
                continue

            action_type, *action_details = action

            if action_type == 'assign':
                patient_id, specialist_id = action_details
                if not (0 <= patient_id < self.num_patients) or not (0 <= specialist_id < self.num_specialists):
                    continue

                patient = self._get_patient_by_id(patient_id)
                specialist = self._get_specialist_by_id(specialist_id)

                if patient is None or specialist is None:
                    continue  # Invalid patient or specialist

                agent_hospital_id = self.agent_hospital_mapping[agent_id]

                # Agents can only assign patients currently in their hospital
                queue_patient_ids = [p['patient_id'] for p in self.hospitals[agent_hospital_id]['queue']]
                if patient_id not in queue_patient_ids:
                    rewards[agent_id] -= 0.5  # Penalty for invalid assignment
                    continue

                # Find the patient in the queue
                queue = self.hospitals[agent_hospital_id]['queue']
                patient_in_queue = next((p for p in queue if p['patient_id'] == patient_id), None)

                if specialist['available'] and specialist['specialty_type'] == patient['condition_type']:
                    # Assign patient to specialist
                    specialist['available'] = False
                    patient_in_queue['wait_time'] = 0
                    specialist['assignments'] += 1

                    # Remove patient from hospital queue
                    queue.remove(patient_in_queue)
                    # Patient is no longer in any hospital queue

                    # Base reward
                    base_reward = 1.0

                    # Additional reward for reducing wait time
                    wait_time_reward = 1 - (patient_in_queue['wait_time'] / self.max_wait_time)

                    # Total reward
                    rewards[agent_id] += base_reward + wait_time_reward
                else:
                    rewards[agent_id] -= 1.0  # Penalty for mismatched specialty or other

            elif action_type == 'transfer':
                patient_id, target_hospital_id = action_details
                if not (0 <= patient_id < self.num_patients) or not (0 <= target_hospital_id < self.num_hospitals):
                    continue

                patient = self._get_patient_by_id(patient_id)
                agent_hospital_id = self.agent_hospital_mapping[agent_id]

                # Check if patient is in agent's hospital queue by patient_id
                queue_patient_ids = [p['patient_id'] for p in self.hospitals[agent_hospital_id]['queue']]
                if patient_id not in queue_patient_ids:
                    rewards[agent_id] -= 0.5  # Penalty for invalid transfer
                    continue

                # Find the patient in the queue
                queue = self.hospitals[agent_hospital_id]['queue']
                patient_in_queue = next((p for p in queue if p['patient_id'] == patient_id), None)

                if patient_in_queue is not None:
                    # Check if target hospital has capacity
                    if len(self.hospitals[target_hospital_id]['queue']) >= self.hospital_capacities[target_hospital_id]['max_patients']:
                        rewards[agent_id] -= 0.5  # Penalty for attempting to overfill the hospital
                        continue

                    # Transfer patient
                    queue.remove(patient_in_queue)
                    patient_in_queue['hospital_id'] = target_hospital_id  # Update hospital_id
                    self.hospitals[target_hospital_id]['queue'].append(patient_in_queue)

                    # Reward for successful transfer (e.g., balancing load)
                    rewards[agent_id] += 1.0  # Encourage transfers that help balance loads
                else:
                    # Patient not found in queue
                    rewards[agent_id] -= 0.5
                    continue

            else:
                # Invalid action type
                rewards[agent_id] -= 0.5  # Penalty for invalid action
                continue

        # Update patient wait times
        for patient in self.patients:
            patient['wait_time'] += 1

        # Specialists become available again in the next time step
        for specialist in self.specialists:
            specialist['available'] = True

        self.current_step += 1
        if self.current_step >= self.episode_length:
            done = True

        # Global reward is the sum of all agents individual rewards
        global_reward = np.sum(rewards)
        return global_reward, rewards, done

    # Get raw observation information
    def get_observation(self, agent_id):
        # Local observations
        hospital_id = self.agent_hospital_mapping[agent_id]
        hospital = self.hospitals[hospital_id]

        # Patients in the agent's hospital queue
        patient_observations = [
            (patient['condition_type'], patient['wait_time']) for patient in hospital['queue']
        ]

        # Specialists in the agent's hospital
        specialist_observations = [
            (specialist['specialty_type'], specialist['available']) for specialist in hospital['specialists']
        ]

        # Summary statistics (limited global information to adhere to partial observability)
        all_queues = [patient for hosp in self.hospitals.values() for patient in hosp['queue']]
        wait_times = [patient['wait_time'] for patient in all_queues]

        if len(wait_times) > 0:
            average_wait_time = np.mean(wait_times)
        else:
            average_wait_time = 0.0  # Default value if no patients are waiting

        total_patients = len(all_queues)
        total_capacity = sum(self.hospital_capacities[h_id]['max_patients'] for h_id in self.hospitals)

        if total_capacity > 0:
            average_utilization = total_patients / total_capacity
        else:
            average_utilization = 0.0  # Avoid division by zero if total_capacity is 0

        summary_info = {
            'total_patients': total_patients,
            'average_wait_time': average_wait_time,
            'total_capacity': total_capacity,
            'average_utilization': average_utilization
        }

        # Combine observations
        observation = {
            'patients': patient_observations,
            'specialists': specialist_observations,
            'hospital_id': hospital_id,
            'summary_info': summary_info
        }

        return observation

    # Get raw global state information
    def get_global_state(self):
        state = {
            'patients': [(patient['condition_type'], patient['wait_time'], patient['hospital_id']) for patient in self.patients],
            'specialists': [(specialist['specialty_type'], specialist['available'], specialist['hospital_id']) for specialist in self.specialists],
            'hospital_queues': {h_id: len(hospital['queue']) for h_id, hospital in self.hospitals.items()},
            'num_hospitals': self.num_hospitals
        }
        return state

    # Helper for patient ID
    def _get_patient_by_id(self, patient_id):
        return next((patient for patient in self.patients if patient['patient_id'] == patient_id), None)

    # Helper for specialist ID
    def _get_specialist_by_id(self, specialist_id):
        return next((specialist for specialist in self.specialists if specialist['specialist_id'] == specialist_id), None)

    # Rendering info for debugging
    def render(self):
        print(f"Step: {self.current_step}")
        print("Patients:")
        for patient in self.patients:
            print(f"ID: {patient['patient_id']}, Condition: {patient['condition_type']}, Wait Time: {patient['wait_time']}, Hospital: {patient['hospital_id']}")
        print("Specialists:")
        for specialist in self.specialists:
            print(f"ID: {specialist['specialist_id']}, Specialty: {specialist['specialty_type']}, Available: {specialist['available']}, Hospital: {specialist['hospital_id']}")
        print("Hospitals:")
        for h_id, hospital in self.hospitals.items():
            print(f"Hospital {h_id}: Queue Length: {len(hospital['queue'])}, Specialists: {len(hospital['specialists'])}")           