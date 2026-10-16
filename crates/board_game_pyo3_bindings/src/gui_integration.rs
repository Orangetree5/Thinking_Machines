use board_game_engine::{BoardGame, BoardInfo, Position};
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::mpsc::{Sender, Receiver};
use generic_board_gui::{BoardControllers, Cell};
use iced::Theme;
use iced::widget::button;
use iced::widget::button::{Status, Style};

use pyo3::prelude::*;

use pyo3_stub_gen::derive::gen_stub_pymethods;
use pyo3_stub_gen_derive::gen_stub_pyclass;


#[derive(Clone, Debug)]
#[pyclass]
pub enum CoinCell {
    Empty,
    Coin,
    Player,
}

impl Cell for CoinCell {
    fn style(&self) -> fn(&Theme, Status) -> Style {
        match self {
            CoinCell::Coin => {
                (|_, _| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(
                        0.8, 0.7, 0.0,
                    ))),
                    ..Default::default()
                }) as fn(&Theme, Status) -> Style
            }
            CoinCell::Empty => {
                (|_, _| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(
                        0.1, 0.1, 0.1,
                    ))),
                    ..Default::default()
                }) as fn(&Theme, Status) -> Style
            }

            CoinCell::Player => {
                (|_, _| button::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgb(
                        1.0, 0.0, 0.0,
                    ))),
                    ..Default::default()
                }) as fn(&Theme, Status) -> Style
            }
        }
    }
}

#[pyfunction]
pub fn board_info_to_grid(game: &crate::BoardGame) -> Vec<Vec<CoinCell>> {
    let BoardInfo {
        coin_positions,
        player_position,
        height,
        width,
    } = game.game.get_board_info();

    let mut grid = vec![vec![CoinCell::Empty; width as usize]; height as usize];

    for coin_pos in coin_positions.iter().cloned() {
        let Position { x, y } = coin_pos;

        grid[y as usize][x as usize] = CoinCell::Coin;
    }

    let Position { x, y } = *player_position;
    grid[y as usize][x as usize] = CoinCell::Player;

    grid
}

#[derive(Clone)]
pub struct BoardReceiver {
    receiver: Arc<Mutex<Receiver<Vec<Vec<CoinCell>>>>>,
}

impl generic_board_gui::BoardReceiver<CoinCell> for BoardReceiver {
    fn recv_board_state_blocking(&self) -> Vec<Vec<CoinCell>> {
        let lock = self.receiver.lock().unwrap();
        lock.recv().unwrap()
    }
}


#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass]
pub struct BoardSender {
    sender: Sender<Vec<Vec<CoinCell>>>,
}

#[pymethods]
impl BoardSender {
    pub fn send(&self, cells: Vec<Vec<CoinCell>>) {
        self.sender.send(cells).unwrap(); // FIX better error handling
    }
}

impl generic_board_gui::BoardSender<CoinCell> for BoardSender {}

struct Controllers {
    sender: BoardSender,
    receiver: BoardReceiver,
}

impl Controllers {
    fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();

        let sender = BoardSender { sender };
        let receiver = BoardReceiver { receiver: Arc::new(Mutex::new(receiver)) };

        Self { sender, receiver }
    }
}

impl BoardControllers<
        CoinCell,
        BoardReceiver,
        BoardSender,
    > for Controllers
{
    fn board_controllers(
        self,
    ) -> (
        BoardSender,
        BoardReceiver
    ) {
        (self.sender, self.receiver)
    }
}

#[gen_stub_pyclass]
#[pyclass]
pub struct Gui {
    pub gui: generic_board_gui::Gui<CoinCell, BoardReceiver, BoardSender>,

    #[pyo3(get)]
    pub board_sender: BoardSender,
}

#[gen_stub_pymethods]
#[pymethods]
impl Gui {
    #[new]
    pub fn new(width: u32, height: u32) -> Self {
        let board_controllers = Controllers::new();

        let (gui, sender) = generic_board_gui::Gui::new(board_controllers, width, height);

        Self {gui, board_sender: sender}
    }

    pub fn run_gui(&mut self) {
        self.gui.run_gui();
    }
}
