use iced::Theme;
use iced::widget::button;
use iced::widget::button::{Status, Style};

pub trait BoardInfo<C: Cell>: Clone + Send + 'static {
    fn update_grid_inplace(&self, grid: &mut Vec<Vec<C>>);
}

pub trait BoardReceiver<C: Cell>: Send + Sync + 'static {
    type BoardInfo: BoardInfo<C>;

    fn recv_board_state(&self)
    -> impl std::future::Future<Output = Option<Self::BoardInfo>> + Send;
}

pub trait Cell: Clone + Send + Sync + 'static {
    fn cell_style(&self) -> CellStyle;
}

pub struct CellStyle {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub text: String,
}

impl CellStyle {
    pub(crate) fn into_style_and_text(self) -> (impl Fn(&Theme, Status) -> Style, String) {
        (
            (move |_, _| button::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgb(
                    self.r, self.g, self.b,
                ))),
                ..Default::default()
            }),
            self.text,
        )
    }
}
