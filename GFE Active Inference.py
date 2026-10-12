import torch
import torch.nn.functional as F
import random
from no_running_in_the_class_room import *

amount_of_states = 2
amount_of_observations = 2
amount_of_actions = 2
short_term_memory = 200
planning_horizon = 3

environment = Environment2()
agent = ActInfAgent(amount_of_states, amount_of_observations, amount_of_actions, short_term_memory, planning_horizon)
agent.initialise_preferences(0, 5)
agent.initialise_preferences(1, -5)

while True:
    agent.commit_to_memory(environment.play_turn(agent.action))
    agent.infer_present(rounds=20)
    agent.infer_future()