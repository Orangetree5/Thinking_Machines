use _rust::{BoardGame, gui_integration::{Gui, board_info_to_grid}};

fn main() {
    let mut gui = Gui::new(5, 5);

    let board_sender= gui.board_sender.clone();


    std::thread::spawn(move || {
        let mut game = BoardGame::new(5, 5, 3);

        let board_sender = board_sender;

        loop {
            game.game.play_turn(board_game_engine::PlayerMove::Up);

            let grid = board_info_to_grid(&game);

            board_sender.send(grid);

            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    });

    gui.run_gui();
}

