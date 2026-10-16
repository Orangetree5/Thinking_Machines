use board_game_engine::{BoardGame, Position};
use tokio::sync::Mutex;
use std::sync::Arc;
use std::sync::mpsc::{Sender, Receiver};
use generic_board_gui::{BoardControllers, Cell, Gui};
use iced::Theme;
use iced::widget::button;
use iced::widget::button::{Status, Style};


#[derive(Clone, Debug)]
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

#[derive(Clone)]
pub struct Info {
    info: board_game_engine::BoardInfo,
}

impl generic_board_gui::BoardInfo<CoinCell> for Info {
    fn update_grid_inplace(&self, grid: &mut Vec<Vec<CoinCell>>) {
        if grid.len() != self.info.height as usize {
            for row in grid.iter_mut() {
                row.resize(self.info.width as usize, CoinCell::Empty);
            }
        }
        grid.resize(self.info.height as usize, vec![CoinCell::Empty; self.info.width as usize]);

        for cell in grid.iter_mut().flatten() {
            *cell = CoinCell::Empty;
        }
    
        for coin_pos in &self.info.coin_positions {
            let Position { x, y } = *coin_pos;

            grid[y as usize][x as usize] = CoinCell::Coin;
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

    fn recv_board_state(
        &self,
    ) -> impl std::future::Future<Output = Option<Info>> + Send
    {
        async move {
            let mut lock = self.receiver.lock().await;

            lock.recv().await
        }
    }
}

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
        let receiver = BoardReceiver { receiver: Mutex::new(receiver)};

        Self { receiver, sender }
    }
}

impl BoardControllers<CoinCell, BoardReceiver, BoardSender> for BoardController {
    fn board_controllers(self) -> (BoardSender, BoardReceiver) {
        (self.sender, self.receiver)
    }
}

fn main() {
    let (gui, sender) = Gui::new(BoardController::new());

    let board_sender = sender.sender;

    std::thread::spawn(move || {
        let mut game = board_game_engine::BoardGame::new_random(15, 15, 4);

        let board_sender = board_sender;

        loop {
            let result = game.play_turn(board_game_engine::PlayerMove::Up);

            let board_info = Info { info: game.get_board_info()};

            board_sender.blocking_send(board_info).unwrap();

            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    });

    gui.run_gui();

}
