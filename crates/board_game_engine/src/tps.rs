use std::time::Instant;

pub(crate) struct Tps {
    time_since_last_turn: Instant,
    current_time: Option<Instant>,

    start_time: Instant,
    total_number_of_turns: u64,
}

impl Tps {
    pub fn new() -> Self {
        Self {
            time_since_last_turn: Instant::now(),
            current_time: None,
            start_time: Instant::now(),
            total_number_of_turns: 0,
        }
    }

    pub fn update(&mut self) {
        if let Some(last_time) = self.current_time {
            self.time_since_last_turn = last_time;
        }

        self.current_time = Some(Instant::now());
        self.total_number_of_turns += 1;
    }

    pub fn current_tps(&self) -> f32 {
        let elapsed_time = self
            .current_time
            .unwrap()
            .duration_since(self.time_since_last_turn);

        1.0 / elapsed_time.as_secs_f32()
    }

    pub fn average_tps(&self) -> f32 {
        let total_elapsed_time = self.current_time.unwrap().duration_since(self.start_time);

        self.total_number_of_turns as f32 / total_elapsed_time.as_secs_f32()
    }

    pub fn total_time_elapsed(&self) -> std::time::Duration {
        self.current_time.unwrap().duration_since(self.start_time)
    }
}
