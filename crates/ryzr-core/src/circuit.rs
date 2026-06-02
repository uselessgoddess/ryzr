use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use cranelift_entity::{EntityList, ListPool, PrimaryMap, entity_impl};

use crate::HashMap;

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
    next_register_index: u32,
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
            next_register_index: 0,
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

    pub fn register(&mut self, name: impl Into<String>, data: Signal, initial: bool) -> Signal {
        let name = name.into();
        let name_index = self.debug_names.len() as u32;
        self.debug_names.push(name);

        self.regs.push(Register { data_input: data, initial, name_index: Some(name_index) });
        self.insts.push(Instruction {
            data: InstData::Const { value: initial },
            debug_name_index: Some(name_index),
        })
    }

    pub fn output(&mut self, name: impl Into<String>, signal: Signal) {
        self.output_names.push(name.into());
        self.output_signals.push(signal);
    }

    pub fn finish(self) -> Result<Circuit, Error> {
        let sorted = self.topo_sort()?;
        let output_count = self.output_names.len() as u32;

        Ok(Circuit {
            insts: sorted,
            regs: self.regs,
            input_names: self.input_names,
            output_names: self.output_names,
            output_signals: self.output_signals,
            debug_names: self.debug_names,
            list_pool: self.list_pool,
            input_count: self.next_input_index,
            register_count: self.next_register_index,
            output_count,
        })
    }

    fn topo_sort(&self) -> Result<PrimaryMap<Signal, Instruction>, Error> {
        let mut in_degree: HashMap<_, _> = HashMap::with_capacity(self.insts.len());
        let mut dependents: HashMap<_, Vec<_>> = HashMap::with_capacity(self.insts.len());

        for (sig, inst) in self.insts.iter() {
            in_degree.entry(sig).or_insert(0);
            if let InstData::Gate { inputs, .. } = &inst.data {
                for &input in inputs.as_slice(&self.list_pool) {
                    *in_degree.entry(input).or_insert(0) += 1;
                    dependents.entry(input).or_default().push(sig);
                }
            }
        }

        let mut queue: VecDeque<_> =
            in_degree.iter().filter(|(_, deg)| **deg == 0).map(|(sig, _)| *sig).collect();

        let mut sorted = PrimaryMap::with_capacity(self.insts.len());

        while let Some(sig) = queue.pop_front() {
            if let Some(inst) = self.insts.get(sig) {
                sorted.push(inst.clone());
            }
            if let Some(deps) = dependents.get(&sig) {
                for &dep in deps {
                    let deg = in_degree.get_mut(&dep).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }

        if sorted.len() != self.insts.len() {
            return Err(Error::CycleDetected);
        }

        Ok(sorted)
    }
}
