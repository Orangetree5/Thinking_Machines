from thinking_machines import BoardGame, Gui
import threading
from no_running_in_the_class_room import *
from time import sleep


def play_game(sender, game: BoardGame, agent):
    while True:
        action = agent.action
        observation = game.play_turn(action)
 
        agent.commit_to_memory(observation)
        sender.send(game)

        agent.infer_past_faster()
        agent.infer_future_faster()
            
def main():
    ### AGENT INITIALISATION ###
    amount_of_states = 8
    amount_of_observations = 20
    amount_of_actions = 4
    short_term_memory = 1500
    planning_horizon = 3
    forgetfulness = 0

    agent = ActInfAgent_02(amount_of_states, amount_of_observations, amount_of_actions, short_term_memory,
                           planning_horizon, forgetfulness)

    agent.initialise_preferences(0, -6)
    agent.initialise_preferences(1, -5)
    agent.initialise_preferences(2, 5)
    for index in range(3, amount_of_observations):
        agent.initialise_preferences(index, 4 - 8 * ((index - 3) / amount_of_observations - 4))
    agent.initialise_preferences(internalise=True)
    ############################

    gui = Gui()

    game_sender = gui.game_sender

    game = BoardGame(width=amount_of_observations-2, height=amount_of_observations-2, number_of_coins=3)
    print(str(game))

    # This executes the play_game function in parallel, if you are lost let LLM explain
    t = threading.Thread(target=play_game, args=(game_sender, game, agent))

    t.start()

    gui.run_gui()

if __name__ == "__main__":
    main()
