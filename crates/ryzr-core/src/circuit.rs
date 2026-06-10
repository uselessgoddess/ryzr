use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use cranelift_entity::{EntityList, EntityRef, ListPool, PrimaryMap, entity_impl};

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Signal(u32);
entity_impl!(Signal, "s");

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(u32);
entity_impl!(Reg, "r");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateOp {
    And,
    Or,
    Xor,
    Not,
    Nand,
    Nor,
    Xnor,
    Buf,
    Mux,
}

#[derive(Debug, Clone)]
pub enum InstData {
    Const { value: bool },
    Input { index: u32 },
    Gate { op: GateOp, inputs: EntityList<Signal> },
    RegisterOutput { reg: Reg },
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub data: InstData,
    pub debug_name_index: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Register {
    pub data_input: Signal,
    pub initial: bool,
    pub name_index: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Circuit {
    pub insts: PrimaryMap<Signal, Instruction>,
    pub regs: PrimaryMap<Reg, Register>,

    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_signals: Vec<Signal>,
    #[allow(unused)]
    debug_names: Vec<String>,

    pub list_pool: ListPool<Signal>,

    pub input_count: u32,
    pub register_count: u32,
    pub output_count: u32,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[non_exhaustive]
pub enum Error {
    CycleDetected,
}

impl core::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::CycleDetected => write!(f, "Cycle detected in combinational logic"),
        }
    }
}

pub struct CircuitBuilder {
    insts: PrimaryMap<Signal, Instruction>,
    regs: PrimaryMap<Reg, Register>,

    input_names: Vec<String>,
    output_names: Vec<String>,
    output_signals: Vec<Signal>,
    debug_names: Vec<String>,

    list_pool: ListPool<Signal>,

    next_input_index: u32,
}

impl Default for CircuitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBuilder {
    pub fn new() -> Self {
        Self {
            insts: PrimaryMap::new(),
            regs: PrimaryMap::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            output_signals: Vec::new(),
            debug_names: Vec::new(),
            list_pool: ListPool::new(),
            next_input_index: 0,
        }
    }

    pub fn input(&mut self, name: impl Into<String>) -> Signal {
        let name = name.into();
        let index = self.next_input_index;
        self.next_input_index += 1;

        self.input_names.push(name);
        self.insts.push(Instruction { data: InstData::Input { index }, debug_name_index: None })
    }

    pub fn const_val(&mut self, value: bool) -> Signal {
        self.insts.push(Instruction { data: InstData::Const { value }, debug_name_index: None })
    }

    pub fn unary(&mut self, op: GateOp, input: Signal) -> Signal {
        assert!(matches!(op, GateOp::Not | GateOp::Buf));
        let mut inputs = EntityList::new();
        inputs.push(input, &mut self.list_pool);

        self.insts.push(Instruction { data: InstData::Gate { op, inputs }, debug_name_index: None })
    }

    pub fn binary(&mut self, op: GateOp, a: Signal, b: Signal) -> Signal {
        assert!(matches!(
            op,
            GateOp::And | GateOp::Or | GateOp::Xor | GateOp::Nand | GateOp::Nor | GateOp::Xnor
        ));
        let mut inputs = EntityList::new();
        inputs.push(a, &mut self.list_pool);
        inputs.push(b, &mut self.list_pool);

        self.insts.push(Instruction { data: InstData::Gate { op, inputs }, debug_name_index: None })
    }

    pub fn mux(&mut self, sel: Signal, then_val: Signal, else_val: Signal) -> Signal {
        let mut inputs = EntityList::new();
        inputs.push(sel, &mut self.list_pool);
        inputs.push(then_val, &mut self.list_pool);
        inputs.push(else_val, &mut self.list_pool);

        self.insts.push(Instruction {
            data: InstData::Gate { op: GateOp::Mux, inputs },
            debug_name_index: None,
        })
    }

    pub fn and(&mut self, a: Signal, b: Signal) -> Signal {
        self.binary(GateOp::And, a, b)
    }
    pub fn or(&mut self, a: Signal, b: Signal) -> Signal {
        self.binary(GateOp::Or, a, b)
    }
    pub fn xor(&mut self, a: Signal, b: Signal) -> Signal {
        self.binary(GateOp::Xor, a, b)
    }
    pub fn not(&mut self, a: Signal) -> Signal {
        self.unary(GateOp::Not, a)
    }
    pub fn nand(&mut self, a: Signal, b: Signal) -> Signal {
        self.binary(GateOp::Nand, a, b)
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        data_input: Signal,
        initial: bool,
    ) -> Signal {
        let name = name.into();
        let name_index = self.debug_names.len() as u32;
        self.debug_names.push(name);

        let reg_id = self.regs.push(Register { data_input, initial, name_index: Some(name_index) });

        self.insts.push(Instruction {
            data: InstData::RegisterOutput { reg: reg_id },
            debug_name_index: Some(name_index),
        })
    }

    pub fn output(&mut self, name: impl Into<String>, signal: Signal) {
        self.output_names.push(name.into());
        self.output_signals.push(signal);
    }

    pub fn finish(self) -> Result<Circuit, Error> {
        let order = self.topo_order()?;
        let register_count = self.regs.len() as u32;
        let output_count = self.output_names.len() as u32;

        // old signal id -> position in the topo schedule (= new signal id)
        let mut remap = vec![Signal::new(0); order.len()];
        for (new_index, &old) in order.iter().enumerate() {
            remap[old.index()] = Signal::new(new_index);
        }

        let mut list_pool = ListPool::new();
        let mut insts = PrimaryMap::with_capacity(order.len());
        for &old in &order {
            let inst = &self.insts[old];
            let data = match &inst.data {
                InstData::Gate { op, inputs } => {
                    let mut remapped = EntityList::new();
                    for &input in inputs.as_slice(&self.list_pool) {
                        remapped.push(remap[input.index()], &mut list_pool);
                    }
                    InstData::Gate { op: *op, inputs: remapped }
                }
                other => other.clone(),
            };
            insts.push(Instruction { data, debug_name_index: inst.debug_name_index });
        }

        let mut regs = PrimaryMap::with_capacity(self.regs.len());
        for (_, reg) in self.regs.iter() {
            regs.push(Register { data_input: remap[reg.data_input.index()], ..reg.clone() });
        }

        let output_signals = self.output_signals.iter().map(|s| remap[s.index()]).collect();

        Ok(Circuit {
            insts,
            regs,
            input_names: self.input_names,
            output_names: self.output_names,
            output_signals,
            debug_names: self.debug_names,
            list_pool,
            input_count: self.next_input_index,
            register_count,
            output_count,
        })
    }

    /// Kahn's algorithm over the combinational graph, deterministic by
    /// processing signals in creation order. Register outputs are sources
    /// (they read the previous tick's state), so cycles through registers
    /// are fine; only purely combinational cycles are rejected.
    fn topo_order(&self) -> Result<Vec<Signal>, Error> {
        let n = self.insts.len();
        let mut in_degree = vec![0u32; n];
        let mut dependents = vec![Vec::new(); n];

        for (sig, inst) in self.insts.iter() {
            if let InstData::Gate { inputs, .. } = &inst.data {
                let inputs = inputs.as_slice(&self.list_pool);
                in_degree[sig.index()] = inputs.len() as u32;
                for &input in inputs {
                    dependents[input.index()].push(sig);
                }
            }
        }

        let mut queue: VecDeque<Signal> =
            (0..n).map(Signal::new).filter(|s| in_degree[s.index()] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(sig) = queue.pop_front() {
            order.push(sig);
            for &dep in &dependents[sig.index()] {
                in_degree[dep.index()] -= 1;
                if in_degree[dep.index()] == 0 {
                    queue.push_back(dep);
                }
            }
        }

        if order.len() != n {
            return Err(Error::CycleDetected);
        }

        Ok(order)
    }
}
