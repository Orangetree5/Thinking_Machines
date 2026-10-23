use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pymethods;
use pyo3_stub_gen_derive::gen_stub_pyclass;

use crate::{
    BoardGamePy,
    gui_integration::{
        BoardInfo, BoardReceiver, BoardSender, CoinCell, N, StatsReceiver, StatsSender, Stats,
        board_controller, stats_controller,
    },
};

#[gen_stub_pyclass]
#[pyclass(name = "Gui")]
pub struct GuiPy {
    gui:
        Mutex<Option<generic_board_gui::Gui<N, CoinCell, BoardReceiver, BoardInfo, StatsReceiver>>>,

    #[pyo3(get)]
    game_sender: GameSenderPy,
}

#[gen_stub_pymethods]
#[pymethods]
impl GuiPy {
    #[new]
    pub fn new() -> Self {
        let (board_sender, board_receiver) = board_controller();
        let (stats_sender, stats_receiver) = stats_controller();

        let gui = generic_board_gui::Gui::new(board_receiver, stats_receiver);

        let gui = Mutex::new(Some(gui));

        let game_sender = GameSenderPy {
            board_sender,
            stats_sender,
        };

        Self { gui, game_sender }
    }

    pub fn run_gui(&mut self, py: Python<'_>) {
        let mut gui = self.gui.lock().unwrap();

        let gui = gui.take().expect("Gui has already been run!");

        // Release the GIL here so other Python threads can run!
        py.detach(move || {
            gui.run_gui();
        });
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "BoardSender")]
#[derive(Clone)]
pub struct GameSenderPy {
    board_sender: self::BoardSender,
    stats_sender: self::StatsSender,
}

#[gen_stub_pymethods]
#[pymethods]
impl GameSenderPy {
    fn send(&mut self, game: &BoardGamePy) {
        let game_info = game.game.get_game_info();

        let board_info = BoardInfo::new(game_info.board_info);
        let stats = Stats::new(game_info.stats);

 
        self.board_sender.sender.blocking_send(board_info).unwrap();
        println!("Sender board");
        self.stats_sender.sender.blocking_send(stats).unwrap();
        println!("Send Stats");
    }
}
