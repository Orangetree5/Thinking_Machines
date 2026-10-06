from thinking_machines import BoardGame

game = BoardGame(width=5, height=5, number_of_coins=3)

print(str(game))

result = game.play_turn(2)

print(str(game))
