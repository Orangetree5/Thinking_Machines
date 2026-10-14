use std::cell::RefCell;
use std::marker::PhantomData;
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

pub trait BoardReceiver<C: Cell>: Send + 'static {
    fn recv_board_state_blocking(&self) -> Vec<Vec<C>>;
    fn height(&self) -> u32;
    fn width(&self) -> u32;
}

pub trait Cell: Clone + Send + Sync + 'static {
    fn style(&self) -> fn(&Theme, Status) -> Style;
}

pub trait BoardSender<C: Cell>: Send + Sync + 'static {

}

pub trait BoardControllers<C: Cell, R: BoardReceiver<C>, S: BoardSender<C>> {
    fn board_controllers(self) -> (S, R);
}

pub struct Gui<C: Cell, R: BoardReceiver<C>, S: BoardSender<C>> {
    board_receiver: Option<R>,
    _sender: PhantomData<S>,

    currently_rendering_cells: Arc<Vec<Vec<C>>>,
    width: u32,
    height: u32,
}

impl<C: Cell, R: BoardReceiver<C>, S: BoardSender<C>> Gui<C, R, S> {
    pub fn run_gui(self) -> ! {
        let s = RefCell::new(Some(self));

        let gui_new = move || {
            s
                .borrow_mut()
                .take()
                .expect("Gui initialized more than once")
        };

        iced::application(gui_new, Gui::update, Gui::view)
            .subscription(Gui::subscription)
            .run()
            .unwrap();

        unreachable!()
    }

    pub fn new(board_controllers: impl BoardControllers<C, R, S>) -> (Self, S) {
        let (sender, board_receiver) = board_controllers.board_controllers();

        (Self {
            width: board_receiver.width(),
            height: board_receiver.height(),
            board_receiver: Some(board_receiver),
            currently_rendering_cells: Arc::new(Vec::new()),
            _sender: PhantomData,
        }, sender)
    }

    pub fn update(&mut self, message: Message<C, R>) {
        match message {
            Message::NewBoardState(new_cells) => {
                self.currently_rendering_cells = new_cells;
            }
            Message::TxReady(mut sender) => {
                sender
                    .try_send(
                        self.board_receiver
                            .take()
                            .expect("supplier should have been initialized"),
                    )
                    .expect("Couldnt send");
            }
        }
    }

    pub fn subscription(&self) -> Subscription<Message<C, R>> {
        fn sub<C: Cell, R: BoardReceiver<C>>() -> impl Stream<Item = Message<C, R>> {
            stream::channel(1, async |mut output| {
                let (sender, mut receiver) = mpsc::channel(1);

                output
                    .send(Message::TxReady(sender))
                    .await
                    .expect("couldnt send TxReady message");

                let (board_tx, mut board_rx) = tokio::sync::mpsc::channel(1);

                let board_receiver: R = receiver.recv().await.unwrap();
                std::thread::spawn(move || {
                    loop {
                        let new_cells = board_receiver.recv_board_state_blocking();

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

    pub fn view(&self) -> iced::Element<'_, Message<C, R>> {
        let board = self.board_container();

        board.into()
    }

    fn board_container(&self) -> Container<'_, Message<C, R>> {
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

pub enum Message<C: Cell, R: BoardReceiver<C>> {
    TxReady(Sender<R>),
    NewBoardState(Arc<Vec<Vec<C>>>),
}

impl<C: Cell, R: BoardReceiver<C>> Clone for Message<C, R> {
    fn clone(&self) -> Self {
        match self {
            Message::NewBoardState(cells) => Self::NewBoardState(Arc::clone(cells)),
            // FIXME What the fux
            Message::TxReady(_sender) => panic!("Cant clone BoardReceiver"),
        }
    }
}
