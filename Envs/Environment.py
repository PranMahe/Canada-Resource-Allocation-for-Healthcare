import torch as th
import numpy as np
import random

class HCRA:
    def __init__(self, num_agents, num_patients, num_specialists, num_specialties, num_hospitals, 
                 num_patients_per_hospital, num_specialists_per_hospital,  max_wait_time, episode_length, seed=None):
        
        # For testing consistency
        self.seed = seed 
        self._initialize_random_seed()

        # General
        self.num_agents = num_agents
        self.num_patients = num_patients
        self.num_specialists = num_specialists
        self.num_specialties = num_specialties
        self.num_hospitals = num_hospitals
        self.num_patients_per_hospital = num_patients_per_hospital
        self.num_specialists_per_hospital = num_specialists_per_hospital
        self.max_wait_time = max_wait_time
        self.episode_length = episode_length

        # Initialize patients, specialists, and hospitals
        self.patients = []
        self.specialists = []
        self.hospitals = {}
        self.current_step = 0

        self._initialize_hospitals()
        self._initialize_specialists()
        self._initialize_patients()
        self._initialize_dimensions()
        

    def _initialize_dimensions(self):
        # Initialize action mapping
        self.action_mapping = self.create_action_mapping(self.num_patients, self.num_specialists)
        self.action_dim = len(self.action_mapping)

        # Compute observation_dim
        sample_observation_raw = self.get_observation(agent_id=0)
        sample_observation_vector = self.process_observation(
            sample_observation_raw,
            self.num_patients_per_hospital,
            self.num_specialists_per_hospital,
            self.num_specialties
        )
        self.observation_dim = len(sample_observation_vector)

        # Compute state_dim
        sample_global_state_raw = self.get_global_state()
        sample_global_state_vector = self.process_global_state(
            sample_global_state_raw,
            self.num_patients,
            self.num_specialists,
            self.num_specialties
        )
        self.state_dim = len(sample_global_state_vector)

    def _initialize_hospitals(self):
        for h_id in range(self.num_hospitals):
            self.hospitals[h_id] = {
                'hospital_id': h_id,
                'specialists': [],
                'queue': []
            }

    def _initialize_specialists(self):
        for s_id in range(self.num_specialists):
            specialty_type = random.randint(0, self.num_specialties - 1)
            hospital_id = random.randint(0, self.num_hospitals - 1)
            specialist = {
                'specialist_id': s_id,
                'specialty_type': specialty_type,
                'hospital_id': hospital_id,
                'available': True
            }
            self.specialists.append(specialist)
            self.hospitals[hospital_id]['specialists'].append(specialist)

    def _initialize_patients(self):
        for p_id in range(self.num_patients):
            condition_type = random.randint(0, self.num_specialties - 1)
            patient = {
                'patient_id': p_id,
                'condition_type': condition_type,
                'wait_time': random.randint(1, self.max_wait_time)
            }
            self.patients.append(patient)
            # Assign patients to a random hospital queue initially
            hospital_id = random.randint(0, self.num_hospitals - 1)
            self.hospitals[hospital_id]['queue'].append(patient)

    def _initialize_random_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            th.manual_seed(self.seed)

    @staticmethod
    def create_action_mapping(num_patients, num_specialists):
        action_mapping = {}
        action_counter = 0
        for patient_id in range(num_patients):
            for specialist_id in range(num_specialists):
                action_mapping[action_counter] = (patient_id, specialist_id)
                action_counter += 1
        return action_mapping
    
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

        # Combine features
        observation_vector = np.concatenate([patient_features, specialist_features])
        return observation_vector
    
    @staticmethod
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

    def reset(self):
        self.current_step = 0
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

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        done = False

        # Process each agent's action
        for agent_id, action in enumerate(actions):
            if action is None:
                continue

            patient_id, specialist_id = action
            if not (0 <= patient_id < self.num_patients) or not (0 <= specialist_id < self.num_specialists):
                continue

            patient = self._get_patient_by_id(patient_id)
            specialist = self._get_specialist_by_id(specialist_id)

            if specialist['available'] and specialist['specialty_type'] == patient['condition_type']:
                # Assign patient to specialist
                specialist['available'] = False
                patient['wait_time'] = 0

                # Remove patient from hospital queue
                hospital_id = specialist['hospital_id']
                self.hospitals[hospital_id]['queue'] = [
                    p for p in self.hospitals[hospital_id]['queue'] if p['patient_id'] != patient_id
                ]

                # Base reward
                base_reward = 1

                # Additional reward based on patient's wait time
                wait_time_reward = (self.max_wait_time - patient['wait_time']) / self.max_wait_time

                # Additional reward for underutilized specialists
                specialist_utilization = self._calculate_specialist_utilization(specialist_id)
                utilization_reward = 1 - specialist_utilization

                # Total reward
                rewards[agent_id] += base_reward + wait_time_reward + utilization_reward
            else:
                # penalty for invalid assignment
                rewards[agent_id] -= 0.5

        # Update patient wait times
        for patient in self.patients:
            patient['wait_time'] += 1

        # Specialists become available again in the next time step
        for specialist in self.specialists:
            specialist['available'] = True

        self.current_step += 1
        if self.current_step >= self.episode_length:
            done = True

        global_reward = np.sum(rewards)
        return global_reward, rewards, done

    def _calculate_specialist_utilization(self, specialist_id):
        specialist = self._get_specialist_by_id(specialist_id)
        total_assignments = self.current_step  # Assuming one assignment per time step
        utilization = specialist.get('assignments', 0) / total_assignments if total_assignments > 0 else 0
        return utilization

    def get_observation(self, agent_id):
        """
        Returns the observation for the given agent.
        Agents are assigned to hospitals, and they observe:
        - Patients in their hospital's queue (condition types and wait times)
        - Specialists in their hospital (specialty types and availability)
        """
        # Assign each agent to a hospital (can be adjusted as needed)
        hospital_id = agent_id % self.num_hospitals
        hospital = self.hospitals[hospital_id]

        # Patients in the hospital queue
        patient_observations = [
            (patient['condition_type'], patient['wait_time']) for patient in hospital['queue']
        ]

        # Specialists in the hospital
        specialist_observations = [
            (specialist['specialty_type'], specialist['available']) for specialist in hospital['specialists']
        ]

        # Combine observations
        observation = {
            'patients': patient_observations,
            'specialists': specialist_observations,
            'hospital_id': hospital_id
        }
        return observation

    def get_global_state(self):
        """
        Returns the global state of the environment.
        """
        state = {
            'patients': [(patient['condition_type'], patient['wait_time']) for patient in self.patients],
            'specialists': [(specialist['specialty_type'], specialist['available']) for specialist in self.specialists],
            'hospital_queues': {h_id: len(hospital['queue']) for h_id, hospital in self.hospitals.items()}
        }
        return state

    def _get_patient_by_id(self, patient_id):
        return next((patient for patient in self.patients if patient['patient_id'] == patient_id), None)

    def _get_specialist_by_id(self, specialist_id):
        return next((specialist for specialist in self.specialists if specialist['specialist_id'] == specialist_id), None)

    def render(self):
        print(f"Step: {self.current_step}")
        print("Patients:")
        for patient in self.patients:
            print(f"ID: {patient['patient_id']}, Condition: {patient['condition_type']}, Wait Time: {patient['wait_time']}")
        print("Specialists:")
        for specialist in self.specialists:
            print(f"ID: {specialist['specialist_id']}, Specialty: {specialist['specialty_type']}, Available: {specialist['available']}")
        print("Hospitals:")
        for h_id, hospital in self.hospitals.items():
            print(f"Hospital {h_id}: Queue Length: {len(hospital['queue'])}")

            