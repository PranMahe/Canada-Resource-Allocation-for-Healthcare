class HCRAparams:
    def __init__(self):
        self.num_agents = 4
        self.num_patients = 10 # Number of patients needing treatment
        self.num_specialists = 8 # Total number of specialists
        self.num_specialties = 2 # A service a certain specialist provides. e.g. cancer screening
        self.num_hospitals = 4 # Total number of hospitals
        self.max_wait_time = 5
        self.episode_length = 10
        self.hospital_capacities = {
            0: {'max_patients': 5, 'max_specialists': 3},  # Largest hospital
            1: {'max_patients': 2, 'max_specialists': 2},
            2: {'max_patients': 3, 'max_specialists': 1},
            3: {'max_patients': 1,  'max_specialists': 2},   # Smallest hospital
        }
