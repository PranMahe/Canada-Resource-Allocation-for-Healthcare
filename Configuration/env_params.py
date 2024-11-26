class HCRAparams:
    def __init__(self):
        self.num_agents = 4
        self.num_patients = 50 # Number of patients needing treatment
        self.num_specialists = 25 # Total number of specialists
        self.num_specialties = 3 # A service a certain specialist provides. e.g. cancer screening
        self.num_hospitals = 4 # Total number of hospitals
        self.max_wait_time = 5
        self.episode_length = 10