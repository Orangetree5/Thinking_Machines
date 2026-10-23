use std::time::Duration;

use board_game_engine::Position;
use generic_board_gui::board::{Cell, CellStyle};
use tokio::sync::Mutex;
use tokio::sync::mpsc::channel;

pub mod bindings;

#[derive(Clone, Debug)]
pub enum CoinCell {
    Empty,
    CoinDistance(u32),
    Coin,
    Player,
}

impl Cell for CoinCell {
    fn cell_style(&self) -> generic_board_gui::board::CellStyle {
        match self {
            CoinCell::Empty => CellStyle {
                r: 0.1,
                g: 0.1,
                b: 0.1,
                text: "".to_string(),
            },
            CoinCell::CoinDistance(distance) => CellStyle {
                r: 0.2,
                g: 0.2,
                b: 0.2,
                text: distance.to_string(),
            },

            CoinCell::Coin => CellStyle {
                r: 0.8,
                g: 0.7,
                b: 0.1,
                text: "".to_string(),
            },

            CoinCell::Player => CellStyle {
                r: 1.0,
                g: 0.1,
                b: 0.1,
                text: "".to_string(),
            },
        }
    }
}

#[derive(Clone)]
pub struct BoardInfo {
    info: board_game_engine::BoardInfo,
}

impl BoardInfo {
    fn new(info: board_game_engine::BoardInfo) -> Self {
        Self { info }
    }
}

impl generic_board_gui::board::BoardInfo<CoinCell> for BoardInfo {
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

#[derive(Clone)]
pub struct Stats {
    current_tps: f32,
    average_tps: f32,
    turns_played: u64,
    coins_found: u64,
    time_elapsed: Duration,
}

impl Stats {
    fn new(game_stats: board_game_engine::Stats) -> Self {
        let board_game_engine::Stats {
            coins_found,
            current_tps,
            average_tps,
            time_elapsed,
            turns_played,
        } = game_stats;

        Self {
            current_tps,
            average_tps,
            turns_played,
            coins_found,
            time_elapsed,
        }
    }
}

pub const N: usize = 5;

impl generic_board_gui::stats::TextInfo<N> for Stats {
    fn content(self) -> [String; N] {
        fn format_duration(d: std::time::Duration) -> String {
            let total_secs = d.as_secs();
            let minutes = total_secs / 60;
            let seconds = total_secs % 60;

            format!("{:02}:{:02}", minutes, seconds)
        }

        [
            format!("Current TPS: {}", self.current_tps),
            format!("Average TPS: {}", self.average_tps),
            format!("Coins found: {}", self.coins_found),
            format!("Total number of turns played: {}", self.turns_played),
            format!("Total minutes played: {}", format_duration(self.time_elapsed))
        ]
    }
}

pub struct StatsReceiver {
    receiver: Mutex<tokio::sync::mpsc::Receiver<Stats>>,
}

impl generic_board_gui::stats::TextContentReceiver<N> for StatsReceiver {
    type TextContent = Stats;

    async fn recv_text_content(&self) -> Option<Stats> {
        let mut lock = self.receiver.lock().await;

        lock.recv().await
    }
}

#[derive(Clone)]
pub struct StatsSender {
    sender: tokio::sync::mpsc::Sender<Stats>,
}

fn stats_controller() -> (StatsSender, StatsReceiver) {
    let (sender, receiver) = channel(1);
    let receiver = Mutex::new(receiver);

    (StatsSender { sender }, StatsReceiver { receiver })
}

pub struct BoardReceiver {
    receiver: Mutex<tokio::sync::mpsc::Receiver<BoardInfo>>,
}

impl generic_board_gui::board::BoardReceiver<CoinCell> for BoardReceiver {
    type BoardInfo = BoardInfo;

    async fn recv_board_state(&self) -> Option<BoardInfo> {
        let mut lock = self.receiver.lock().await;

        lock.recv().await
    }
}

#[derive(Clone)]
pub struct BoardSender {
    sender: tokio::sync::mpsc::Sender<BoardInfo>,
}

fn board_controller() -> (BoardSender, BoardReceiver) {
    let (sender, receiver) = tokio::sync::mpsc::channel(1);

    let sender = BoardSender { sender };
    let receiver = BoardReceiver {
        receiver: Mutex::new(receiver),
    };

    (sender, receiver)
}
