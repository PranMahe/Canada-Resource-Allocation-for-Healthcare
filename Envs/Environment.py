import numpy as np
import random

class HCRA:
    def __init__(self, num_agents, num_patients, num_specialists, num_specialties, num_hospitals, 
                 num_patients_per_hospital, num_specialists_per_hospital,  max_wait_time, episode_length, t_max):
        self.num_agents = num_agents
        self.num_patients = num_patients
        self.num_specialists = num_specialists
        self.num_specialties = num_specialties
        self.num_hospitals = num_hospitals
        self.num_patients_per_hospital = num_patients_per_hospital
        self.num_specialists_per_hospital = num_specialists_per_hospital
        self.max_wait_time = max_wait_time
        self.episode_length = episode_length
        self.t_max = t_max

        # Initialize patients, specialists, and hospitals
        self.patients = []
        self.specialists = []
        self.hospitals = {}
        self.current_step = 0

        self._initialize_hospitals()
        self._initialize_specialists()
        self._initialize_patients()

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

    def reset(self):
        self.current_step = 0
        self.patients = []
        self.specialists = []
        for h_id in self.hospitals:
            self.hospitals[h_id]['specialists'] = []
            self.hospitals[h_id]['queue'] = []
        self._initialize_specialists()
        self._initialize_patients()
        return self.get_global_state()

    def step(self, actions):
        """
        actions: list of actions from each agent.
        Each action is a tuple (patient_id, specialist_id)
        """
        rewards = np.zeros(self.num_agents)
        done = False

        # Process each agent's action
        for agent_id, action in enumerate(actions):
            if action is None:
                # No action taken
                continue

            patient_id, specialist_id = action
            # Validate patient and specialist IDs
            if not (0 <= patient_id < self.num_patients) or not (0 <= specialist_id < self.num_specialists):
                continue

            patient = self._get_patient_by_id(patient_id)
            specialist = self._get_specialist_by_id(specialist_id)

            # Check if specialist is available and matches the patient's condition
            if specialist['available'] and specialist['specialty_type'] == patient['condition_type']:
                # Assign patient to specialist
                specialist['available'] = False
                patient['wait_time'] = 0  # Reset wait time since patient is being attended to

                # Remove patient from hospital queue
                hospital_id = specialist['hospital_id']
                self.hospitals[hospital_id]['queue'] = [
                    p for p in self.hospitals[hospital_id]['queue'] if p['patient_id'] != patient_id
                ]

                # Reward for successful assignment
                rewards[agent_id] += 1
            else:
                # Penalty for invalid assignment
                rewards[agent_id] -= 0.1

        # Update patient wait times
        for patient in self.patients:
            patient['wait_time'] += 1

        # Specialists become available again in the next time step (simple model)
        for specialist in self.specialists:
            specialist['available'] = True

        self.current_step += 1
        if self.current_step >= self.episode_length:
            done = True

        global_reward = np.sum(rewards)
        return global_reward, rewards, done

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

            