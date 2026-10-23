pub trait TextContentReceiver<const N: usize>: Send + Sync + 'static {
    type TextContent: TextInfo<N>;

    fn recv_text_content(
        &self,
    ) -> impl std::future::Future<Output = Option<Self::TextContent>> + Send;
}

pub trait TextInfo<const N: usize>: Clone + Send + 'static {
    fn content(self) -> [String; N];
}
