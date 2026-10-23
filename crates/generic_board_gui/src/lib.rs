use iced::widget::button;
use iced::widget::text;
use iced::widget::{Container, column, container, row};
use iced::{Alignment, Length, Subscription};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::board::BoardInfo;
use crate::board::BoardReceiver;
use crate::board::Cell;
use crate::stats::TextContentReceiver;
use crate::stats::TextInfo;

pub mod board;
pub mod stats;

pub struct Gui<
    const N: usize,
    C: Cell,
    R: BoardReceiver<C, BoardInfo = I>,
    I: BoardInfo<C>,
    TC: TextContentReceiver<N>,
> {
    board_receiver: Arc<R>,
    text_receiver: Arc<TC>,

    currently_rendering_cells: Vec<Vec<C>>,
    text_info: [String; N],
}

impl<
    const N: usize,
    C: Cell,
    R: BoardReceiver<C, BoardInfo = I>,
    I: BoardInfo<C>,
    TC: TextContentReceiver<N, TextContent = T>,
    T: TextInfo<N>,
> Gui<N, C, R, I, TC>
{
    pub fn new(board_receiver: R, text_receiver: TC) -> Self {
        Self {
            board_receiver: Arc::new(board_receiver),
            text_receiver: Arc::new(text_receiver),
            currently_rendering_cells: Vec::new(),
            text_info: [const {String::new()}; N],
        }
    }

    pub fn run_gui(self) -> ! {
        let s = RefCell::new(Some(self));

        let gui_new = move || {
            s.borrow_mut()
                .take()
                .expect("Gui initialized more than once")
        };

        iced::application(gui_new, Gui::update, Gui::view)
            .subscription(Gui::all_subscriptions)
            .run()
            .unwrap();

        unreachable!()
    }

    pub fn update(&mut self, message: Message<N, C, I, T>) {
        match message {
            Message::NewBoardInfo(new_cells) => {
                let old_cells = &mut self.currently_rendering_cells;

                new_cells.update_grid_inplace(old_cells);
            }

            Message::NewTextInfo(new_text) => {
                println!("Got new text!");
                self.text_info = new_text.content();
            }

            Message::_Cell(_) => unreachable!(),
        }
    }

    fn all_subscriptions(&self) -> Subscription<Message<N, C, I, T>> {
        Subscription::batch([self.board_subscription(), self.text_content_subscription()])
    }

    fn board_subscription(&self) -> Subscription<Message<N, C, I, T>> {
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

    fn text_content_subscription(&self) -> Subscription<Message<N, C, I, T>> {
        pub struct TextContentReceiverHashable<const N: usize, TC: TextContentReceiver<N>>(Arc<TC>);
        impl<const N: usize, TC: TextContentReceiver<N>> std::hash::Hash
            for TextContentReceiverHashable<N, TC>
        {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                std::ptr::hash(Arc::as_ptr(&self.0), state);
            }
        }

        let text_receiver = TextContentReceiverHashable(Arc::clone(&self.text_receiver));

        Subscription::run_with(text_receiver, |text_receiver| {
            let TextContentReceiverHashable(text_receiver) = text_receiver;
            let text_receiver = Arc::clone(text_receiver);

            async_stream::stream! {
                loop {
                    if let Some(text_info) = text_receiver.recv_text_content().await {
                        yield Message::NewTextInfo(text_info)
                    }
                }
            }
        })
    }

    pub fn view(&self) -> iced::Element<'_, Message<N, C, I, T>> {
        let board = self.board_container();
        let text_row = self.text_row();

        row![text_row, board].into()
    }

    fn text_row(&self) -> Container<'_, Message<N, C, I, T>> {
        let mut text_column = column![].spacing(1);

        println!("{:?}", self.text_info);

        for t in &self.text_info {
            text_column = text_column.push(text(t));
        }

        container(text_column)
    }

    fn board_container(&self) -> Container<'_, Message<N, C, I, T>> {
        let height = self.currently_rendering_cells.len();
        let width = self
            .currently_rendering_cells
            .first()
            .map(|row| row.len())
            .unwrap_or(0);

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

                let (style, content) = cell.cell_style().into_style_and_text();

                let tile_button = button(text(content))
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
pub enum Message<const N: usize, C: Cell, I: BoardInfo<C>, T: TextInfo<N>> {
    NewBoardInfo(I),
    NewTextInfo(T),

    _Cell(PhantomData<C>),
}
