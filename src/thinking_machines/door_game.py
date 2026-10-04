import random


class DoorGame:
    move: int
    light_on: bool

    def __init__(self) -> None:
        self.move = 0
        self.light_on = True
 
    def play_turn(self, should_light_be_on_action: bool) -> int:
        self.light_on = should_light_be_on_action

        print()
        print()

        current_move = self.move % 4
        if self.light_on:
            print("Clara is on move", current_move, "and lights are on.") 

        else:
            print("Clara is on move", current_move, "and lights are off.")

        if current_move == 0:
            is_moving_this_turn = bool(random.getrandbits(1))

            if is_moving_this_turn:
                self.move += 1
                if not self.light_on:
                    print("Door moves.")
                    return 2
                else:
                    # The model doesnt know that the door moved in this case?
                    print("Door moves. Clara is unhappy :(")
                    return 1
            else:
                return 3

        if current_move == 3:
            self.move += 1
            if not self.light_on:
                print("Clara is happy :)")
                return 0 
            else:
                print("Clara is unhappy :(")
                return 1

        else:
            self.move += 1
            if self.light_on:
                print("Clara is happy :)")
                return 0 
            else:
                print("Clara is unhappy :(")
                return 1
