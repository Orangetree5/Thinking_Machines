use board_game_engine::{PlayerMove, TurnResult};
use pyo3::prelude::*;

use pyo3_stub_gen::{
    define_stub_info_gatherer, derive::gen_stub_pyfunction, derive::gen_stub_pymethods,
};
use pyo3_stub_gen_derive::gen_stub_pyclass;


#[gen_stub_pyclass]
#[pyclass]
pub struct BoardGame {
    game: board_game_engine::BoardGame,
}

#[gen_stub_pymethods]
#[pymethods]
impl BoardGame {
    #[new]
    fn new(width: u32, height: u32, number_of_coins: u32) -> Self {
        let game = board_game_engine::BoardGame::new_random(width, height, number_of_coins);

        println!("lool");

        Self { game }
    }

    fn play_turn(&mut self, player_move: u32) -> i32 {
        let player_move = encode_player_move(player_move);

        decode_turn_result(self.game.play_turn(player_move).expect("Invalid move!"))
    }

    fn __str__(&self) -> String {
        self.game.to_string()
    }
}

fn encode_player_move(number: u32) -> PlayerMove {
    match number {
        0 => PlayerMove::Up,
        1 => PlayerMove::Right,
        2 => PlayerMove::Down,
        3 => PlayerMove::Right,
        _ => panic!("Invalid moving direction!")
    } 
}

fn decode_turn_result(turn_result: TurnResult) -> i32 {
    match turn_result {
        TurnResult::CoinFound => 0,
        TurnResult::DistanceToCoinInMovingDirection(distance) => {
            match distance {
                None => -1,
                Some(distance) => distance as i32,
            }
        }
    }
}

#[pymodule]
#[pyo3(name = "_rust")]
pub mod board_game_py {
    use pyo3::prelude::*;


    #[pymodule_export]
    use super::{BoardGame};
}


use std::path::Path;
use pyo3_stub_gen::StubInfo;

pub fn stub_info() -> pyo3_stub_gen::Result<StubInfo> {
    let manifest_dir: &Path = env!("CARGO_MANIFEST_DIR").as_ref();
    // Point to root pyproject.toml from inside crates/board_game_pyo3_bindings/
    let root_pyproject = manifest_dir.join("../../pyproject.toml");
    StubInfo::from_pyproject_toml(root_pyproject)
}
