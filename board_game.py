from thinking_machines import BoardGame, Gui, board_info_to_grid
from time import sleep
import threading

def play_game(sender, game: BoardGame):
    while True:
        # Noah put your action logic here

        # 0: Up, 1: Right, 2: Down, 3: Left
        move = 2

        # result = 0: "found coin", -1: "No coin in this direction", -2: "Invalid moving direction", n: "Distance to coin in moved direction"
        result = game.play_turn(move)

        print(result)

        cells = board_info_to_grid(game) 
        sender.send(cells)

        sleep(0.5) # Stop for 0.5 second
        

def main():
    gui = Gui(width=10, height=10)

    board_sender = gui.board_sender

    game = BoardGame(width=10, height=10, number_of_coins=3)
    print(str(game))

    # This executes the play_game function in parallel, if you are lost let LLM explain
    t = threading.Thread(target=play_game, args=(board_sender, game))

    t.start()

    gui.run_gui()

if __name__ == "__main__":
    main()
