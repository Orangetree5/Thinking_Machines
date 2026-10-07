import torch
import torch.nn.functional as F
import random
from no_running_in_the_class_room import *

amount_of_states = 8
amount_of_observations = 4
amount_of_actions = 2
short_term_memory = 100
planning_horizon = 3

environment = Environment1()
agent = ActInfAgent(amount_of_states, amount_of_observations, amount_of_actions, short_term_memory, planning_horizon)
agent.initialise_preferences(0, 5)

while True:
    agent.commit_to_memory(environment.play_turn(agent.action))
    agent.infer_past()
    agent.infer_parameters()
    agent.infer_future()