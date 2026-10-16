import unittest
from unittest.mock import patch

from thinking_machines.door_game import DoorGame


class TestPlayTurnBetter(unittest.TestCase):
    def setUp(self):
        """Runs before every individual test method."""
        self.game = DoorGame()

    @patch("random.getrandbits")
    def test_move_zero_light_off_door_moves(self, mock_randbits):
        """When move % 4 == 0, lights off, and door moves -> returns 2."""
        self.game.move = 0
        mock_randbits.return_value = 1  # Forces is_moving_this_turn = True

        result = self.game.play_turn(should_light_be_on_action=False)

        self.assertEqual(result, 2)
        self.assertEqual(self.game.move, 1)
        self.assertFalse(self.game.light_on)

    @patch("random.getrandbits")
    def test_move_zero_door_does_not_move_light_on(self, mock_randbits):
        """When move % 4 == 0 and door does not move -> returns 3."""
        self.game.move = 0
        mock_randbits.return_value = 0  # Forces is_moving_this_turn = False

        result = self.game.play_turn(should_light_be_on_action=True)

        self.assertEqual(result, 3) # Clara is unhappy
        self.assertEqual(self.game.move, 0)  # Move shouldn't increment

    def test_move_greater_than_zero_light_on(self):
        """When move % 4 > 0 and light is on -> returns 1."""
        self.game.move = 1  # 1 % 4 == 1 > 0

        result = self.game.play_turn(should_light_be_on_action=True)

        self.assertEqual(result, 0) # Clara is happy
        self.assertEqual(self.game.move, 2)

    def test_move_greater_than_zero_light_off(self):
        """When move % 4 > 0 and light is on -> returns 1."""
        self.game.move = 1  # 1 % 4 == 1 > 0

        result = self.game.play_turn(should_light_be_on_action=False)

        self.assertEqual(result, 1) # Clara is unhappy
        self.assertEqual(self.game.move, 2)

    def test_move_3_than_zero_light_off(self):
        """When move % 4 > 0 and light is off -> returns 0."""
        self.game.move = 3  # 2 % 4 == 2 > 0

        result = self.game.play_turn(should_light_be_on_action=False)

        self.assertEqual(result, 0) # Clara is happy
        self.assertEqual(self.game.move, 4)

    def test_move_3_than_zero_light_on(self):
        """When move % 4 > 0 and light is off -> returns 0."""
        self.game.move = 3  # 2 % 4 == 2 > 0

        result = self.game.play_turn(should_light_be_on_action=True)

        self.assertEqual(result, 1) # Clara is unhappy
        self.assertEqual(self.game.move, 4)


if __name__ == "__main__":
    unittest.main()
