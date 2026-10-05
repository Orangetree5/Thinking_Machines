pub struct BoardGame {
    width: u32,
    height: u32,

    coin_positions: Vec<Position>,
    player_position: Position,
}

impl BoardGame {
    pub fn new(
        width: u32,
        height: u32,
        coin_positions: Vec<Position>,
        player_position: Position,
    ) -> Self {
        Self {
            width,
            height,
            coin_positions,
            player_position,
        }
    }

    pub fn play_turn(&mut self, player_move: PlayerMove) -> Option<TurnResult> {
        self.player_position = self.move_player(&player_move)?;

        let mut index_coin_to_remove = None;
        for (i, coin_position) in self.coin_positions.iter().copied().enumerate() {
            if self.player_position == coin_position {
                index_coin_to_remove = Some(i);
            }
        }

        if let Some(index) = index_coin_to_remove {
            self.coin_positions.remove(index);

            return Some(TurnResult::CoinFound);
        }

        let distance: Option<u32> = self.distance_to_coin_in_moved_direction(player_move);

        Some(TurnResult::DistanceToCoinInMovingDirection(distance))
    }

    fn distance_to_coin_in_moved_direction(&self, player_move: PlayerMove) -> Option<u32> {
        for coin_position in self.coin_positions.iter().copied() {
            if self.player_position.x == coin_position.x {
                match player_move {
                    PlayerMove::Up => {
                        for (distance, y) in (self.player_position.y..self.height).enumerate() {
                            if y == coin_position.y {
                                return Some(distance as u32);
                            }
                        }
                    }
                    PlayerMove::Down => {
                        for (distance, y) in (0..=self.player_position.y).rev().enumerate() {
                            println!("{distance}");
                            if y == coin_position.y {
                                return Some(distance as u32);
                            }
                        }
                    }
                    _ => {}
                }
            } else if self.player_position.y == coin_position.y {
                match player_move {
                    PlayerMove::Right => {
                        for (distance, x) in (self.player_position.x..self.width).enumerate() {
                            if x == coin_position.x {
                                return Some(distance as u32);
                            }
                        }
                    }
                    PlayerMove::Left => {
                        for (distance, x) in (0..=self.player_position.x).rev().enumerate() {
                            if x == coin_position.x {
                                return Some(distance as u32);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }

    fn move_player(&self, player_move: &PlayerMove) -> Option<Position> {
        match player_move {
            PlayerMove::Up => {
                if self.player_position.y >= self.height - 1 {
                    None
                } else {
                    Some(Position {
                        x: self.player_position.x,
                        y: self.player_position.y + 1,
                    })
                }
            }
            PlayerMove::Right => {
                if self.player_position.x >= self.width - 1 {
                    None
                } else {
                    Some(Position {
                        x: self.player_position.x + 1,
                        y: self.player_position.y,
                    })
                }
            }
            PlayerMove::Down => {
                if self.player_position.y == 0 {
                    None
                } else {
                    Some(Position {
                        x: self.player_position.x,
                        y: self.player_position.y - 1,
                    })
                }
            }
            PlayerMove::Left => {
                if self.player_position.x == 0 {
                    None
                } else {
                    Some(Position {
                        x: self.player_position.x - 1,
                        y: self.player_position.y,
                    })
                }
            }
        }
    }
}

impl std::fmt::Display for BoardGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                let current_pos = Position { x, y };

                if self.coin_positions.contains(&current_pos) {
                    write!(f, "C")?;
                } else if self.player_position == current_pos {
                    write!(f, "P")?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    x: u32,
    y: u32,
}

pub enum PlayerMove {
    Up,
    Right,
    Down,
    Left,
}

pub enum TurnResult {
    CoinFound,
    DistanceToCoinInMovingDirection(Option<u32>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_nothing_found() {
        let mut board_game = BoardGame::new(
            5,
            5,
            vec![Position { x: 3, y: 2 }, Position { x: 1, y: 1 }],
            Position { x: 1, y: 2 },
        );
        println!("{}", board_game);

        let result = board_game.play_turn(PlayerMove::Up).unwrap();
        println!("{}", board_game);

        match result {
            TurnResult::CoinFound => panic!("Shouldnt have found coin"),
            TurnResult::DistanceToCoinInMovingDirection(distance) => {
                assert!(distance.is_none())
            }
        }
    }

    #[test]
    fn test_game_find_coin() {
        let mut board_game =
            BoardGame::new(5, 5, vec![Position { x: 2, y: 2 }], Position { x: 1, y: 2 });

        let result = board_game.play_turn(PlayerMove::Right).unwrap();

        match result {
            TurnResult::CoinFound => {}
            _ => panic!("Should have found coin"),
        }

        println!("{}", board_game);
    }

    #[test]
    fn test_game_distance_right() {
        let mut board_game =
            BoardGame::new(5, 5, vec![Position { x: 4, y: 2 }], Position { x: 1, y: 2 });

        let result = board_game.play_turn(PlayerMove::Right).unwrap();

        match result {
            TurnResult::DistanceToCoinInMovingDirection(distance) => {
                let distance = distance.unwrap();

                assert_eq!(distance, 2);
            }
            _ => panic!("Should got distance"),
        }

        println!("{}", board_game);
    }

    #[test]
    fn test_game_distance_down() {
        let mut board_game =
            BoardGame::new(5, 5, vec![Position { x: 4, y: 0 }], Position { x: 4, y: 4 });

        let result = board_game.play_turn(PlayerMove::Down).unwrap();
        println!("{}", board_game);

        match result {
            TurnResult::DistanceToCoinInMovingDirection(distance) => {
                let distance = distance.unwrap();

                assert_eq!(distance, 3);
            }
            _ => panic!("Should got distance"),
        }
    }

    #[test]
    fn test_game_distance_up() {
        let mut board_game =
            BoardGame::new(5, 5, vec![Position { x: 4, y: 4 }], Position { x: 4, y: 0 });

        let result = board_game.play_turn(PlayerMove::Up).unwrap();
        println!("{}", board_game);

        match result {
            TurnResult::DistanceToCoinInMovingDirection(distance) => {
                let distance = distance.unwrap();

                assert_eq!(distance, 3);
            }
            _ => panic!("Should got distance"),
        }
    }

    #[test]
    fn test_game_distance_left() {
        let mut board_game =
            BoardGame::new(5, 5, vec![Position { x: 0, y: 0 }], Position { x: 4, y: 0 });

        let result = board_game.play_turn(PlayerMove::Left).unwrap();
        println!("{}", board_game);

        match result {
            TurnResult::DistanceToCoinInMovingDirection(distance) => {
                let distance = distance.unwrap();

                assert_eq!(distance, 3);
            }
            _ => panic!("Should got distance"),
        }
    }
}
