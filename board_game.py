from time import sleep
import traceback
from thinking_machines import BoardGame, Gui
import threading
from no_running_in_the_class_room import *


def play_game(sender, game: BoardGame, agent):
    while True:
        # Noah put your action logic here
        agent.commit_to_memory(2+game.play_turn(agent.action))
        sender.send(game)
        agent.infer_past()
        agent.infer_future()
        # 0: Up, 1: Right, 2: Down, 3: Left

        # result = 0: "found coin", -1: "No coin in this direction", -2: "Invalid moving direction", n: "Distance to coin in moved direction"

        #sleep(0.3) # Stop for 0.3 second
            
def main():
    ### AGENT INITIALISATION ###
    amount_of_states = 10
    amount_of_observations = 20
    amount_of_actions = 4
    short_term_memory = 500
    planning_horizon = 3

    agent = ActInfAgent_02(amount_of_states, amount_of_observations, amount_of_actions, short_term_memory,
                           planning_horizon)

    agent.initialise_preferences(0, -5)
    agent.initialise_preferences(1, -5)
    agent.initialise_preferences(2, 5)
    for index in range(3, amount_of_observations):
        agent.initialise_preferences(index, 4-8*((index-3)/amount_of_observations-1))
    agent.initialise_preferences(internalise=True)
    ############################

    gui = Gui()

    board_sender = gui.board_sender

    game = BoardGame(width=amount_of_observations-2, height=amount_of_observations-2, number_of_coins=3)
    print(str(game))

    # This executes the play_game function in parallel, if you are lost let LLM explain
    t = threading.Thread(target=play_game, args=(board_sender, game, agent))

    t.start()

    gui.run_gui()

if __name__ == "__main__":
    main()
