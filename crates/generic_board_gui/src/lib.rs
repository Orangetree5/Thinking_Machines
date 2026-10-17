use iced::widget::button;
use iced::widget::{
    Container,
    button::{Status, Style},
    column, container, row,
};
use iced::{Alignment, Length, Subscription, Theme};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

pub trait BoardInfo<C: Cell>: Clone + Send + 'static {
    fn update_grid_inplace(&self, grid: &mut Vec<Vec<C>>);
}

pub trait BoardReceiver<C: Cell>: Send + Sync + 'static {
    type BoardInfo: BoardInfo<C>;

    fn recv_board_state(&self)
    -> impl std::future::Future<Output = Option<Self::BoardInfo>> + Send;
}

pub trait Cell: Clone + Send + Sync + 'static {
    fn style(&self) -> fn(&Theme, Status) -> Style;
}

pub trait BoardSender<C: Cell>: Send + Sync + 'static {}

pub trait BoardControllers<C: Cell, R: BoardReceiver<C>, S: BoardSender<C>> {
    fn board_controllers(self) -> (S, R);
}

pub struct Gui<C: Cell, R: BoardReceiver<C, BoardInfo = I>, S: BoardSender<C>, I: BoardInfo<C>> {
    board_receiver: Arc<R>,
    _sender: PhantomData<S>,

    currently_rendering_cells: Vec<Vec<C>>,
}

impl<C: Cell, R: BoardReceiver<C, BoardInfo = I>, S: BoardSender<C>, I: BoardInfo<C>>
    Gui<C, R, S, I>
{
    pub fn run_gui(self) -> ! {
        let s = RefCell::new(Some(self));

        let gui_new = move || {
            s.borrow_mut()
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

        (
            Self {
                board_receiver: Arc::new(board_receiver),
                currently_rendering_cells: Vec::new(),
                _sender: PhantomData,
            },
            sender,
        )
    }

    pub fn update(&mut self, message: Message<C, I>) {
        match message {
            Message::NewBoardInfo(new_cells) => {
                let old_cells = &mut self.currently_rendering_cells;

                new_cells.update_grid_inplace(old_cells);
            }

            Message::_Cell(_) => unreachable!(),
        }
    }

    pub fn subscription(&self) -> Subscription<Message<C, I>> {
        pub struct BoardReceiverHashable<C: Cell, R: BoardReceiver<C>>(Arc<R>, PhantomData<C>);
        impl<C: Cell, R: BoardReceiver<C>> std::hash::Hash for BoardReceiverHashable<C, R> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                std::ptr::hash(Arc::as_ptr(&self.0), state);
            }
        }

        let board_receiver = BoardReceiverHashable(Arc::clone(&self.board_receiver), PhantomData);

        Subscription::run_with(board_receiver, |board_receiver| {
            let BoardReceiverHashable(board_receiver, _) = board_receiver;
            let board_receiver = Arc::clone(board_receiver);

            async_stream::stream! {
                loop {
                    if let Some(board_info) = board_receiver.recv_board_state().await {
                        yield Message::NewBoardInfo(board_info);
                    }
                }
            }
        })
    }

    pub fn view(&self) -> iced::Element<'_, Message<C, I>> {
        let board = self.board_container();

        board.into()
    }

    fn board_container(&self) -> Container<'_, Message<C, I>> {
        let height = self.currently_rendering_cells.len();
        let width = self.currently_rendering_cells.first().map(|row| row.len()).unwrap_or(0);

        let mut board_layout = column![].spacing(1);

        for y in (0..height).rev() {
            let mut row_layout = row![].spacing(1);

            for x in 0..width {
                let cell = if let Some(row) = self.currently_rendering_cells.get(y)
                    && let Some(cell) = row.get(x)
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

#[derive(Debug, Clone)]
pub enum Message<C: Cell, I: BoardInfo<C>> {
    NewBoardInfo(I),

    _Cell(PhantomData<C>),
}
