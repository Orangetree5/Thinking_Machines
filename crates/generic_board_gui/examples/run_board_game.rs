use std::cell::RefCell;

use board_game_engine::{BoardGame, BoardInfo, Position};
use generic_board_gui::{BoardControllers, BoardReceiver, BoardSender, Cell, Gui};
use iced::Theme;
use iced::widget::button;
use iced::widget::button::{Status, Style};

#[derive(Clone, Debug)]
enum CoinCell {
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

impl BoardReceiver<CoinCell> for std::sync::mpsc::Receiver<Vec<Vec<CoinCell>>> {
    fn width(&self) -> u32 {
        // FIXME how should i handle this? maybe dimensions as constants in trait or a
        // constructor in trait
        5
    }

    fn height(&self) -> u32 {
        5
    }

    fn recv_board_state_blocking(&self) -> Vec<Vec<CoinCell>> {
        self.recv().unwrap()
    }
}

fn board_info_to_grid(game: &BoardGame) -> Vec<Vec<CoinCell>> {
    let BoardInfo {
        coin_positions,
        player_position,
        height,
        width,
    } = game.get_board_info();

    let mut grid = vec![vec![CoinCell::Empty; width as usize]; height as usize];

    for coin_pos in coin_positions.iter().cloned() {
        let Position { x, y } = coin_pos;

        grid[y as usize][x as usize] = CoinCell::Coin;
    }

    let Position { x, y } = *player_position;
    grid[y as usize][x as usize] = CoinCell::Player;

    grid
}

struct Controllers {
    sender: std::sync::mpsc::Sender<Vec<Vec<CoinCell>>>,
    receiver: std::sync::mpsc::Receiver<Vec<Vec<CoinCell>>>,
}

impl BoardSender<CoinCell> for std::sync::mpsc::Sender<Vec<Vec<CoinCell>>> {

}

impl BoardControllers<CoinCell,  std::sync::mpsc::Receiver<Vec<Vec<CoinCell>>>, std::sync::mpsc::Sender<Vec<Vec<CoinCell>>>> for Controllers {
    fn board_controllers(self) -> (std::sync::mpsc::Sender<Vec<Vec<CoinCell>>>, std::sync::mpsc::Receiver<Vec<Vec<CoinCell>>>) {
        (self.sender, self.receiver)
    }
}

fn main() {
    let (board_sender, board_receiver): (
        std::sync::mpsc::Sender<Vec<Vec<CoinCell>>>,
        std::sync::mpsc::Receiver<Vec<Vec<CoinCell>>>,
    ) = std::sync::mpsc::channel();
    
    let board_controllers = Controllers { sender: board_sender, receiver: board_receiver};

    let (gui, board_sender) = Gui::new(board_controllers);

    std::thread::spawn(move || {
        let mut game = BoardGame::new_random(5, 5, 2);

        println!("{game}");

        let board_sender = board_sender;

        loop {
            let grid = board_info_to_grid(&game);

            board_sender.send(grid).unwrap();

            game.play_turn(board_game_engine::PlayerMove::Up);

            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    gui.run_gui();

}
