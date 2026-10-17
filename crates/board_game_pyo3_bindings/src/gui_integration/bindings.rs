use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pymethods;
use pyo3_stub_gen_derive::gen_stub_pyclass;

use crate::{BoardGamePy, gui_integration::{BoardController, BoardReceiver, BoardSender, CoinCell, Info}};

#[gen_stub_pyclass]
#[pyclass(name = "Gui")]
pub struct GuiPy {
    gui: Mutex<Option<generic_board_gui::Gui<CoinCell, BoardReceiver, BoardSender, Info>>>,

    #[pyo3(get)]
    board_sender: BoardSenderPy,
}

#[gen_stub_pymethods]
#[pymethods]
impl GuiPy {
    #[new]
    pub fn new() -> Self {
        let board_controllers = BoardController::new();

        let (gui, sender) = generic_board_gui::Gui::new(board_controllers);

        let gui = Mutex::new(Some(gui));

        let board_sender = BoardSenderPy { sender };

        Self {gui, board_sender}
    }

    pub fn run_gui(&mut self) {
        let mut gui = self.gui.lock().unwrap();

        let gui = gui.take().expect("Gui has already been run!");

        gui.run_gui();
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "BoardSender")]
#[derive(Clone)]
pub struct BoardSenderPy {
    sender: self::BoardSender,
}

#[gen_stub_pymethods]
#[pymethods]
impl BoardSenderPy {
    fn send(&mut self, game: &BoardGamePy) {
        let info = Info::new(game.game.get_board_info());

        self.sender.sender.blocking_send(info).unwrap();
    }
}
