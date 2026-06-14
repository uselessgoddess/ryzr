use std::fmt;

use ryzr_backend::{Engine, HybridEngine};
use ryzr_core::{Circuit, CircuitBuilder, GateOp, InstData, Signal};

pub const DEFAULT_BOARD_WIDTH: usize = 16;
pub const DEFAULT_BOARD_HEIGHT: usize = 10;

const PALETTE: [CellKind; 11] = [
    CellKind::Empty,
    CellKind::Input,
    CellKind::Clock,
    CellKind::Wire,
    CellKind::Not,
    CellKind::And,
    CellKind::Or,
    CellKind::Xor,
    CellKind::Nand,
    CellKind::Register,
    CellKind::Led,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellKind {
    Empty,
    Input,
    Clock,
    Wire,
    Not,
    And,
    Or,
    Xor,
    Nand,
    Register,
    Led,
}

impl CellKind {
    pub fn palette() -> &'static [Self] {
        &PALETTE
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Empty => "",
            Self::Input => "IN",
            Self::Clock => "CLK",
            Self::Wire => "WIRE",
            Self::Not => "NOT",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Xor => "XOR",
            Self::Nand => "NAND",
            Self::Register => "DFF",
            Self::Led => "LED",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Empty => "Erase",
            Self::Input => "Input source",
            Self::Clock => "Clock source",
            Self::Wire => "Wire",
            Self::Not => "Inverter",
            Self::And => "AND gate",
            Self::Or => "OR gate",
            Self::Xor => "XOR gate",
            Self::Nand => "NAND gate",
            Self::Register => "D flip-flop",
            Self::Led => "Output LED",
        }
    }

    pub const fn is_source(self) -> bool {
        matches!(self, Self::Input | Self::Clock)
    }

    const fn gate_op(self) -> Option<GateOp> {
        match self {
            Self::And => Some(GateOp::And),
            Self::Or => Some(GateOp::Or),
            Self::Xor => Some(GateOp::Xor),
            Self::Nand => Some(GateOp::Nand),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellPos {
    pub x: usize,
    pub y: usize,
}

impl CellPos {
    pub const fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub kind: CellKind,
    pub input_value: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self { kind: CellKind::Empty, input_value: false }
    }
}

#[derive(Debug, Clone)]
pub struct Board {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0, "board width must be non-zero");
        assert!(height > 0, "board height must be non-zero");
        Self { width, height, cells: vec![Cell::default(); width * height] }
    }

    pub fn demo() -> Self {
        let mut board = Self::new(DEFAULT_BOARD_WIDTH, DEFAULT_BOARD_HEIGHT);

        board.set_kind(0, 1, CellKind::Input);
        board.set_kind(1, 1, CellKind::Wire);
        board.set_kind(2, 1, CellKind::Wire);
        board.set_kind(3, 1, CellKind::Wire);
        board.set_input_value(0, 1, true);

        board.set_kind(0, 2, CellKind::Input);
        board.set_kind(1, 2, CellKind::Wire);
        board.set_kind(2, 2, CellKind::And);
        board.set_kind(3, 2, CellKind::Led);
        board.set_input_value(0, 2, true);

        board.set_kind(5, 1, CellKind::Input);
        board.set_kind(6, 1, CellKind::Wire);
        board.set_kind(7, 1, CellKind::Wire);
        board.set_input_value(5, 1, true);

        board.set_kind(5, 2, CellKind::Clock);
        board.set_kind(6, 2, CellKind::Wire);
        board.set_kind(7, 2, CellKind::Xor);
        board.set_kind(8, 2, CellKind::Register);
        board.set_kind(9, 2, CellKind::Led);

        board.set_kind(0, 5, CellKind::Input);
        board.set_kind(1, 5, CellKind::Not);
        board.set_kind(2, 5, CellKind::Led);

        board.set_kind(5, 5, CellKind::Input);
        board.set_kind(6, 5, CellKind::Wire);
        board.set_kind(7, 5, CellKind::Wire);
        board.set_kind(5, 6, CellKind::Input);
        board.set_kind(6, 6, CellKind::Wire);
        board.set_kind(7, 6, CellKind::Or);
        board.set_kind(8, 6, CellKind::Led);
        board.set_input_value(5, 6, true);

        board
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn cell(&self, x: usize, y: usize) -> Option<&Cell> {
        self.index(x, y).map(|i| &self.cells[i])
    }

    pub fn set_kind(&mut self, x: usize, y: usize, kind: CellKind) {
        let idx = self.checked_index(x, y);
        let previous_value = self.cells[idx].input_value;
        self.cells[idx] =
            Cell { kind, input_value: if kind.is_source() { previous_value } else { false } };
    }

    pub fn set_input_value(&mut self, x: usize, y: usize, value: bool) {
        let idx = self.checked_index(x, y);
        if self.cells[idx].kind.is_source() {
            self.cells[idx].input_value = value;
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(Cell::default());
    }

    pub fn count_kind(&self, kind: CellKind) -> usize {
        self.cells.iter().filter(|cell| cell.kind == kind).count()
    }

    pub fn compile(&self) -> Result<BoardCircuit, BoardError> {
        let mut builder = CircuitBuilder::new();
        let ground = builder.const_val(false);
        builder.output("__ground", ground);

        let mut signals: Vec<Option<Signal>> = vec![None; self.cells.len()];
        let mut cell_outputs = vec![None; self.cells.len()];
        let mut input_bindings = Vec::new();
        let mut output_index = 1;

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.checked_index(x, y);
                let cell = self.cells[idx];
                let west = x.checked_sub(1).and_then(|wx| signals[self.checked_index(wx, y)]);
                let north = y.checked_sub(1).and_then(|ny| signals[self.checked_index(x, ny)]);
                let west_input = west.unwrap_or(ground);
                let north_input = north.unwrap_or(ground);
                let first_input = west.or(north).unwrap_or(ground);

                let signal = match cell.kind {
                    CellKind::Empty => None,
                    CellKind::Input | CellKind::Clock => {
                        let signal = builder.input(format!("{}_{}_{}", cell.kind.label(), x, y));
                        input_bindings.push(InputBinding {
                            pos: CellPos::new(x, y),
                            input_index: input_bindings.len(),
                            value: cell.input_value,
                            clock: cell.kind == CellKind::Clock,
                        });
                        Some(signal)
                    }
                    CellKind::Wire | CellKind::Led => Some(builder.buf(first_input)),
                    CellKind::Not => Some(builder.not(first_input)),
                    CellKind::Register => {
                        Some(builder.register(format!("dff_{x}_{y}"), first_input, false))
                    }
                    kind => {
                        let op = kind.gate_op().expect("binary gate kind");
                        Some(builder.binary(op, west_input, north_input))
                    }
                };

                if let Some(signal) = signal {
                    signals[idx] = Some(signal);
                    cell_outputs[idx] = Some(output_index);
                    output_index += 1;
                    builder.output(format!("cell_{}_{}_{}", x, y, cell.kind.label()), signal);
                }
            }
        }

        let circuit = builder.finish().map_err(|err| BoardError::Circuit(err.to_string()))?;
        let stats = CircuitStats::from_circuit(&circuit, cell_outputs.iter().flatten().count());
        Ok(BoardCircuit { circuit, cell_outputs, input_bindings, stats })
    }

    fn index(&self, x: usize, y: usize) -> Option<usize> {
        (x < self.width && y < self.height).then_some(y * self.width + x)
    }

    fn checked_index(&self, x: usize, y: usize) -> usize {
        self.index(x, y).expect("cell coordinate is inside the board")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputBinding {
    pub pos: CellPos,
    pub input_index: usize,
    pub value: bool,
    pub clock: bool,
}

#[derive(Debug, Clone)]
pub struct BoardCircuit {
    pub circuit: Circuit,
    pub cell_outputs: Vec<Option<usize>>,
    pub input_bindings: Vec<InputBinding>,
    pub stats: CircuitStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitStats {
    pub signals: usize,
    pub gates: usize,
    pub registers: usize,
    pub inputs: usize,
    pub outputs: usize,
}

impl CircuitStats {
    fn from_circuit(circuit: &Circuit, outputs: usize) -> Self {
        let gates = circuit
            .insts
            .iter()
            .filter(|(_, inst)| matches!(inst.data, InstData::Gate { .. }))
            .count();
        Self {
            signals: circuit.insts.len(),
            gates,
            registers: circuit.regs.len(),
            inputs: circuit.input_count as usize,
            outputs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoardError {
    Circuit(String),
    NoInputAt(CellPos),
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Circuit(err) => f.write_str(err),
            Self::NoInputAt(pos) => write!(f, "cell {},{} is not an input source", pos.x, pos.y),
        }
    }
}

impl std::error::Error for BoardError {}

pub struct EditorRuntime {
    board: Board,
    circuit: Circuit,
    engine: Box<dyn Engine>,
    cell_outputs: Vec<Option<usize>>,
    input_bindings: Vec<InputBinding>,
    stats: CircuitStats,
    tick_count: u64,
}

impl EditorRuntime {
    pub fn new(board: Board) -> Result<Self, BoardError> {
        let compiled = board.compile()?;
        let engine = Box::new(HybridEngine::new(&compiled.circuit));
        let mut runtime = Self {
            board,
            circuit: compiled.circuit,
            engine,
            cell_outputs: compiled.cell_outputs,
            input_bindings: compiled.input_bindings,
            stats: compiled.stats,
            tick_count: 0,
        };
        runtime.apply_inputs();
        Ok(runtime)
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    pub fn circuit_stats(&self) -> CircuitStats {
        self.stats
    }

    pub fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn cell_value(&self, x: usize, y: usize) -> bool {
        self.board
            .index(x, y)
            .and_then(|idx| self.cell_outputs[idx])
            .is_some_and(|output| self.engine.output(output))
    }

    pub fn set_kind(&mut self, x: usize, y: usize, kind: CellKind) -> Result<(), BoardError> {
        self.board.set_kind(x, y, kind);
        self.rebuild()
    }

    pub fn set_input_value(&mut self, x: usize, y: usize, value: bool) -> Result<(), BoardError> {
        let pos = CellPos::new(x, y);
        let input_index = if let Some(binding) =
            self.input_bindings.iter_mut().find(|binding| binding.pos == pos)
        {
            binding.value = value;
            binding.input_index
        } else {
            return Err(BoardError::NoInputAt(pos));
        };
        self.board.set_input_value(x, y, value);
        self.engine.set_input(input_index, value);
        Ok(())
    }

    pub fn toggle_input(&mut self, x: usize, y: usize) -> Result<(), BoardError> {
        self.set_input_value(x, y, !self.cell_input_value(x, y).unwrap_or(false))
    }

    pub fn cell_input_value(&self, x: usize, y: usize) -> Option<bool> {
        self.board.cell(x, y).and_then(|cell| cell.kind.is_source().then_some(cell.input_value))
    }

    pub fn clear(&mut self) -> Result<(), BoardError> {
        self.board.clear();
        self.rebuild()
    }

    pub fn load_demo(&mut self) -> Result<(), BoardError> {
        self.board = Board::demo();
        self.rebuild()
    }

    pub fn reset(&mut self) -> Result<(), BoardError> {
        self.rebuild()
    }

    pub fn step(&mut self) {
        self.advance_clocks();
        self.engine.tick();
        self.tick_count += 1;
    }

    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }

    fn rebuild(&mut self) -> Result<(), BoardError> {
        let compiled = self.board.compile()?;
        self.circuit = compiled.circuit;
        self.engine = Box::new(HybridEngine::new(&self.circuit));
        self.cell_outputs = compiled.cell_outputs;
        self.input_bindings = compiled.input_bindings;
        self.stats = compiled.stats;
        self.tick_count = 0;
        self.apply_inputs();
        Ok(())
    }

    fn apply_inputs(&mut self) {
        for binding in &self.input_bindings {
            self.engine.set_input(binding.input_index, binding.value);
        }
    }

    fn advance_clocks(&mut self) {
        let clocks: Vec<CellPos> = self
            .input_bindings
            .iter()
            .filter_map(|binding| binding.clock.then_some(binding.pos))
            .collect();
        for pos in clocks {
            let value = !self.cell_input_value(pos.x, pos.y).unwrap_or(false);
            self.set_input_value(pos.x, pos.y, value)
                .expect("clock binding remains valid during a step");
        }
    }
}
