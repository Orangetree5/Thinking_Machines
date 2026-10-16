from thinking_machines import BoardGame, Gui, board_info_to_grid
from time import sleep
import threading

def play_game(sender, game: BoardGame, move: int):
    while True:
        game.play_turn(1)

        cells = board_info_to_grid(game) 
        sender.send(cells)

        sleep(1)
        

if __name__ == "__main__":
    gui = Gui(width=10, height=10)

    board_sender = gui.board_sender
    game = BoardGame(width=10, height=10, number_of_coins=3)
    print(str(game))

    cells = board_info_to_grid(game) 

    t = threading.Thread(target=play_game, args=(board_sender, game, 1))

    t.start()

    gui.run_gui()
