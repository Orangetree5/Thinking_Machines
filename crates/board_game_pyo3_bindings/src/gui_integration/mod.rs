use board_game_engine::{BoardGame, Position};
use generic_board_gui::{BoardControllers, Cell, Gui};
use tokio::sync::Mutex;

pub mod bindings;

#[derive(Clone, Debug)]
pub enum CoinCell {
    Empty,
    CoinDistance(u32),
    Coin,
    Player,
}

impl Cell for CoinCell {
    fn cell_style(&self) -> generic_board_gui::CellStyle {
        match self {
            CoinCell::Empty => generic_board_gui::CellStyle {
                r: 0.1,
                g: 0.1,
                b: 0.1,
                text: "".to_string(),
            },
            CoinCell::CoinDistance(distance) => generic_board_gui::CellStyle {
                r: 0.2,
                g: 0.2,
                b: 0.2,
                text: distance.to_string(),
            },

            CoinCell::Coin => generic_board_gui::CellStyle {
                r: 0.8,
                g: 0.7,
                b: 0.1,
                text: "C".to_string(),
            },

            CoinCell::Player => generic_board_gui::CellStyle {
                r: 1.0,
                g: 0.1,
                b: 0.1,
                text: "P".to_string(),
            },
        }
    }
}

#[derive(Clone)]
pub struct Info {
    info: board_game_engine::BoardInfo,
}

impl Info {
    fn new(info: board_game_engine::BoardInfo) -> Self {
        Self { info }
    }
}

impl generic_board_gui::BoardInfo<CoinCell> for Info {
    fn update_grid_inplace(&self, grid: &mut Vec<Vec<CoinCell>>) {
        if grid.len() != self.info.height as usize {
            for row in grid.iter_mut() {
                row.resize(self.info.width as usize, CoinCell::Empty);
            }
        }
        grid.resize(
            self.info.height as usize,
            vec![CoinCell::Empty; self.info.width as usize],
        );

        for cell in grid.iter_mut().flatten() {
            *cell = CoinCell::Empty;
        }

        for coin_pos in &self.info.coin_positions {
            let Position { x, y } = *coin_pos;

            grid[y as usize][x as usize] = CoinCell::Coin;
        }

        for (view_pos, distance) in &self.info.view_positions {
            let Position { x, y } = *view_pos;

            grid[y as usize][x as usize] = CoinCell::CoinDistance(*distance);
        }

        let Position { x, y } = self.info.player_position;
        grid[y as usize][x as usize] = CoinCell::Player;
    }
}

pub struct BoardReceiver {
    receiver: Mutex<tokio::sync::mpsc::Receiver<Info>>,
}

impl generic_board_gui::BoardReceiver<CoinCell> for BoardReceiver {
    type BoardInfo = Info;

    async fn recv_board_state(&self) -> Option<Info> {
        let mut lock = self.receiver.lock().await;

        lock.recv().await
    }
}

#[derive(Clone)]
pub struct BoardSender {
    sender: tokio::sync::mpsc::Sender<Info>,
}

impl generic_board_gui::BoardSender<CoinCell> for BoardSender {}

pub struct BoardController {
    receiver: BoardReceiver,
    sender: BoardSender,
}

impl BoardController {
    fn new() -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);

        let sender = BoardSender { sender };
        let receiver = BoardReceiver {
            receiver: Mutex::new(receiver),
        };

        Self { receiver, sender }
    }
}

impl BoardControllers<CoinCell, BoardReceiver, BoardSender> for BoardController {
    fn board_controllers(self) -> (BoardSender, BoardReceiver) {
        (self.sender, self.receiver)
    }
}
