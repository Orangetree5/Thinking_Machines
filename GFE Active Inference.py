from no_running_in_the_class_room import *

amount_of_states = 100
amount_of_observations = 24
amount_of_actions = 4
short_term_memory = 500
planning_horizon = 3

environment = Environment3(amount_of_observations//2)
agent = ActInfAgent_02(amount_of_states, amount_of_observations, amount_of_actions, short_term_memory, planning_horizon)

for index in range(amount_of_observations):
    agent.initialise_preferences(index, 5-(10*index/(amount_of_observations-1)))
agent.initialise_preferences(internalise=True)

while True:
    agent.commit_to_memory(environment.play_turn(agent.action))
    agent.infer_past()
    agent.infer_future()