use std::sync::Arc;

use iced::futures::Stream;
use iced::futures::channel::mpsc;
use iced::futures::channel::mpsc::{Receiver, Sender};
use iced::futures::sink::SinkExt;
use iced::stream;
use iced::widget::button;
use iced::widget::{
    Container,
    button::{Status, Style},
    column, container, row,
};
use iced::{Alignment, Length, Subscription, Theme};

pub trait BoardSupplier<C: Cell>: Send + 'static {
    fn recv_board_state_blocking(&self) -> Vec<Vec<C>>;
    fn height(&self) -> u32;
    fn width(&self) -> u32;
}

pub trait Cell: Clone + Send + Sync + 'static {
    fn style(&self) -> fn(&Theme, Status) -> Style;
}

pub struct Gui<C: Cell, S: BoardSupplier<C>> {
    board_supplier: Option<S>,

    currently_rendering_cells: Arc<Vec<Vec<C>>>,
    width: u32,
    height: u32,
}

impl<C: Cell, S: BoardSupplier<C>> Gui<C, S> {
    pub fn new(board_supplier: S) -> Self {
        Self {
            width: board_supplier.width(),
            height: board_supplier.height(),
            board_supplier: Some(board_supplier),
            currently_rendering_cells: Arc::new(Vec::new()),
        }
    }

    pub fn update(&mut self, message: Message<C, S>) {
        match message {
            Message::NewBoardState(new_cells) => {
                self.currently_rendering_cells = new_cells;
            }
            Message::TxReady(mut sender) => {
                sender
                    .try_send(
                        self.board_supplier
                            .take()
                            .expect("supplier should have been initialized"),
                    )
                    .expect("Couldnt send");

            }
        }
    }

    pub fn subscription(&self) -> Subscription<Message<C, S>> {
        fn sub<C: Cell, S: BoardSupplier<C>>() -> impl Stream<Item = Message<C, S>> {
            stream::channel(1, async |mut output| {
                let (sender, mut receiver) = mpsc::channel(1);

                output
                    .send(Message::TxReady(sender))
                    .await
                    .expect("couldnt send TxReady message");

                let (board_tx, mut board_rx) = tokio::sync::mpsc::channel(1);

                let board_supplier: S = receiver.recv().await.unwrap();
                std::thread::spawn(move || {
                    loop {
                        let new_cells = board_supplier.recv_board_state_blocking();

                        if board_tx.blocking_send(new_cells).is_err() {
                            return;
                        }
                    }
                });

                while let Some(new_cells) = board_rx.recv().await {
                    output
                        .send(Message::NewBoardState(Arc::new(new_cells)))
                        .await
                        .unwrap();
                }
            })
        }

        Subscription::run(sub)
    }

    pub fn view(&self) -> iced::Element<'_, Message<C, S>> {
        let board = self.board_container();

        board.into()
    }

    fn board_container(&self) -> Container<'_, Message<C, S>> {
        let mut board_layout = column![].spacing(1);

        for y in (0..self.height).rev() {
            let mut row_layout = row![].spacing(1);

            for x in 0..self.width {
                let cell = if let Some(row) = self.currently_rendering_cells.get(y as usize)
                    && let Some(cell) = row.get(x as usize)
                {
                    cell
                } else {
                    break;
                };

                let style = cell.style();

                let tile_button = button("")
                    .style(style)
                    .width(Length::Fixed(40.0))
                    .height(Length::Fixed(40.0));

                row_layout = row_layout.push(tile_button);
            }

            board_layout = board_layout.push(row_layout);
        }

        container(board_layout)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_y(Alignment::Center)
            .align_x(Alignment::Center)
    }
}

#[derive(Debug)]
pub enum Message<C: Cell, S: BoardSupplier<C>> {
    TxReady(Sender<S>),
    NewBoardState(Arc<Vec<Vec<C>>>),
}

impl<C: Cell, S: BoardSupplier<C>> Clone for Message<C, S> {
    fn clone(&self) -> Self {
        match self {
            Message::NewBoardState(cells) => Self::NewBoardState(Arc::clone(cells)),
            // FIXME What the fux
            Message::TxReady(_sender) => panic!("Cant clone BoardSupplier"),
        }
    }
}
