use rand::{RngExt, rngs::ThreadRng};

use crate::tps::Tps;

mod tps;

pub struct BoardGame {
    width: u32,
    height: u32,

    coin_positions: Vec<Position>,
    view_positions: Vec<(Position, u32)>,
    player_position: Position,

    tps: Tps,
    coins_found: u64,
}

impl BoardGame {
    fn new(
        width: u32,
        height: u32,
        coin_positions: Vec<Position>,
        player_position: Position,
    ) -> Self {
        let cap = width.max(height);

        Self {
            view_positions: Vec::with_capacity(cap as usize - 1),
            width,
            height,
            coin_positions,
            player_position,
            tps: Tps::new(),
            coins_found: 0,
        }
    }

    pub fn new_random(width: u32, height: u32, number_of_coins: u32) -> Self {
        let mut rng = rand::rng();

        let mut positions = Vec::new();
        for _ in 0..number_of_coins + 1 {
            let mut position = Self::generate_pot_position(&mut rng, width, height);

            while positions.contains(&position) {
                position = Self::generate_pot_position(&mut rng, width, height);
            }

            positions.push(position);
        }

        let player_position = positions.pop().unwrap();

        Self::new(width, height, positions, player_position)
    }

    pub fn play_turn(&mut self, player_move: PlayerMove) -> Option<TurnResult> {
        self.tps.update();

        self.player_position = self.move_player(&player_move)?;

        let mut index_coin_to_remove = None;
        for (i, coin_position) in self.coin_positions.iter().copied().enumerate() {
            if self.player_position == coin_position {
                index_coin_to_remove = Some(i);
            }
        }

        if let Some(index) = index_coin_to_remove {
            self.coin_positions.remove(index);
            self.coin_positions.push(self.unique_random_position());
            self.view_positions.clear();
            self.coins_found += 1;

            return Some(TurnResult::CoinFound);
        }

        let distance: Option<u32> = self.distance_to_coin_in_moved_direction(&player_move);

        if let Some(distance) = distance {
            self.update_viewing_positions(&player_move, distance);
        } else {
            self.view_positions.clear();
        }

        Some(TurnResult::DistanceToCoinInMovingDirection(distance))
    }

    fn unique_random_position(&self) -> Position {
        let mut rng = rand::rng();

        let mut pot_position = Self::generate_pot_position(&mut rng, self.width, self.height);
        while self.coin_positions.contains(&pot_position) || pot_position == self.player_position {
            pot_position = Self::generate_pot_position(&mut rng, self.width, self.height);
        }

        pot_position
    }

    fn generate_pot_position(rng: &mut ThreadRng, width: u32, height: u32) -> Position {
        let rand_x = rng.random_range(0..width);
        let rand_y = rng.random_range(0..height);

        Position {
            x: rand_x,
            y: rand_y,
        }
    }

    fn update_viewing_positions(&mut self, player_move: &PlayerMove, distance: u32) {
        self.view_positions.clear();

        for (current_pos, current_distance) in
            Self::direction_walker(self.width, self.height, self.player_position, player_move)
                .take(distance as usize)
        {
            let view_distance = distance - current_distance;

            self.view_positions.push((current_pos, view_distance));
        }
    }

    fn distance_to_coin_in_moved_direction(&mut self, player_move: &PlayerMove) -> Option<u32> {
        for (current_pos, distance) in
            Self::direction_walker(self.width, self.height, self.player_position, player_move)
        {
            if self.coin_positions.contains(&current_pos) {
                return Some(distance);
            }
        }

        None
    }

    fn direction_walker(
        width: u32,
        height: u32,
        player_position: Position,
        player_move: &PlayerMove,
    ) -> Box<dyn Iterator<Item = (Position, u32)> + '_> {
        match player_move {
            PlayerMove::Up => Box::new((player_position.y..height).map(move |y| {
                (
                    Position {
                        x: player_position.x,
                        y,
                    },
                    y - player_position.y,
                )
            })),
            PlayerMove::Down => Box::new((0..=player_position.y).rev().map(move |y| {
                (
                    Position {
                        x: player_position.x,
                        y,
                    },
                    player_position.y - y,
                )
            })),

            PlayerMove::Right => Box::new((player_position.x..width).map(move |x| {
                (
                    Position {
                        x,
                        y: player_position.y,
                    },
                    x - player_position.x,
                )
            })),
            PlayerMove::Left => Box::new((0..=player_position.x).rev().map(move |x| {
                (
                    Position {
                        x,
                        y: player_position.y,
                    },
                    player_position.x - x,
                )
            })),
        }
    }

    pub fn get_board_info(&self) -> BoardInfo {
        fn format_duration(d: std::time::Duration) -> String {
            let total_secs = d.as_secs();
            let minutes = total_secs / 60;
            let seconds = total_secs % 60;

            format!("{:02}:{:02}", minutes, seconds)
        }

        let current_tps = self.tps.current_tps();
        let average_tps = self.tps.average_tps();

        print!(
            "\rCurrent TPS: {:.2}, Average TPS: {:.2}, Total Turns: {}, Total time elapsed: {}",
            current_tps,
            average_tps,
            self.tps.total_turns(),
            format_duration(self.tps.total_time_elapsed())
        );

        BoardInfo {
            coin_positions: self.coin_positions.to_vec(),
            view_positions: self.view_positions.to_vec(),
            player_position: self.player_position,
            current_tps,
            average_tps,
            width: self.width,
            height: self.height,
        }
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

#[derive(Clone)]
pub struct BoardInfo {
    pub coin_positions: Vec<Position>,
    pub view_positions: Vec<(Position, u32)>,
    pub player_position: Position,
    pub current_tps: f32,
    pub average_tps: f32,
    pub height: u32,
    pub width: u32,
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
    pub x: u32,
    pub y: u32,
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
            TurnResult::DistanceToCoinInMovingDirection(distance) => {
                assert!(distance.is_none())
            }
            _ => panic!("Shouldnt have found coin"),
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
    fn test_game_double_distance_right() {
        let mut board_game = BoardGame::new(
            5,
            5,
            vec![Position { x: 4, y: 2 }, Position { x: 3, y: 2 }],
            Position { x: 1, y: 2 },
        );

        let result = board_game.play_turn(PlayerMove::Right).unwrap();

        match result {
            TurnResult::DistanceToCoinInMovingDirection(distance) => {
                let distance = distance.unwrap();

                assert_eq!(distance, 1);
            }
            _ => panic!("Should got distance"),
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
