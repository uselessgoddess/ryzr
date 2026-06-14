use ryzr_gui::{Board, CellKind, EditorRuntime};

#[test]
fn matrix_gate_reads_west_and_north_neighbors() {
    let mut board = Board::new(4, 3);
    board.set_kind(0, 0, CellKind::Input);
    board.set_kind(1, 0, CellKind::Wire);
    board.set_kind(2, 0, CellKind::Wire);
    board.set_kind(0, 1, CellKind::Input);
    board.set_kind(1, 1, CellKind::Wire);
    board.set_kind(2, 1, CellKind::And);
    board.set_kind(3, 1, CellKind::Led);

    board.set_input_value(0, 0, true);
    board.set_input_value(0, 1, true);

    let mut runtime = EditorRuntime::new(board).expect("matrix circuit builds");
    runtime.step();
    assert!(runtime.cell_value(3, 1));

    runtime.set_input_value(0, 1, false).expect("input exists");
    runtime.step();
    assert!(!runtime.cell_value(3, 1));
}

#[test]
fn register_cell_advances_only_on_explicit_steps() {
    let mut board = Board::new(4, 1);
    board.set_kind(0, 0, CellKind::Input);
    board.set_kind(1, 0, CellKind::Register);
    board.set_kind(2, 0, CellKind::Led);
    board.set_input_value(0, 0, true);

    let mut runtime = EditorRuntime::new(board).expect("matrix circuit builds");
    assert!(!runtime.cell_value(2, 0));

    runtime.step();
    assert!(!runtime.cell_value(2, 0));

    runtime.step();
    assert!(runtime.cell_value(2, 0));
}

#[test]
fn demo_board_contains_editable_vcb_like_components() {
    let board = Board::demo();
    assert!(board.count_kind(CellKind::Input) >= 2);
    assert!(board.count_kind(CellKind::And) >= 1);
    assert!(board.count_kind(CellKind::Register) >= 1);
    assert!(board.count_kind(CellKind::Led) >= 2);

    let runtime = EditorRuntime::new(board).expect("demo circuit builds");
    assert!(runtime.circuit_stats().gates > 0);
    assert!(runtime.circuit_stats().registers > 0);
    assert_eq!(runtime.tick_count(), 0);
}
