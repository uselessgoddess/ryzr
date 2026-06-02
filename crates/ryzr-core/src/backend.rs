use alloc::vec;
use alloc::vec::Vec;

use cranelift_entity::EntityRef;

use crate::{Circuit, GateOp, InstData};

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    fn tick(&self, circuit: &Circuit, inputs: &[bool], state: &mut [bool], outputs: &mut [bool]);
}

#[derive(Default)]
pub struct Interpreter;

impl Backend for Interpreter {
    fn name(&self) -> &str {
        "interpreter"
    }

    fn tick(&self, circuit: &Circuit, inputs: &[bool], state: &mut [bool], outputs: &mut [bool]) {
        let mut values: Vec<_> = vec![false; circuit.insts.len()];
        let mut reg_outputs: Vec<Option<usize>> = vec![None; circuit.insts.len()];

        for (reg_idx, _) in circuit.regs.iter() {
            if let Some((sig_idx, _)) = circuit.insts.iter().enumerate().nth(reg_idx.index()) {
                reg_outputs[sig_idx] = Some(reg_idx.index());
                values[sig_idx] = state[reg_idx.index()];
            }
        }

        for (sig, inst) in circuit.insts.iter() {
            let idx = sig.index();

            let result = match &inst.data {
                InstData::Const { value } => *value,
                InstData::Input { index } => inputs[*index as usize],
                InstData::Gate { op, inputs } => {
                    let input_vals: [bool; 3] = inputs
                        .as_slice(&circuit.list_pool)
                        .iter()
                        .map(|&s| values[s.index()])
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap_or([false; 3]);

                    match op {
                        GateOp::And => input_vals[0] && input_vals[1],
                        GateOp::Or => input_vals[0] || input_vals[1],
                        GateOp::Xor => input_vals[0] ^ input_vals[1],
                        GateOp::Not => !input_vals[0],
                        GateOp::Nand => !(input_vals[0] && input_vals[1]),
                        GateOp::Nor => !(input_vals[0] || input_vals[1]),
                        GateOp::Xnor => !(input_vals[0] ^ input_vals[1]),
                        GateOp::Buf => input_vals[0],
                        GateOp::Mux => {
                            if input_vals[0] {
                                input_vals[1]
                            } else {
                                input_vals[2]
                            }
                        }
                    }
                }
            };

            values[idx] = result;
        }

        let mut next_state = state.to_vec();
        for (reg_idx, reg) in circuit.regs.iter() {
            let next_val = values[reg.data_input.index()];
            next_state[reg_idx.index()] = next_val;
        }
        state.copy_from_slice(&next_state);

        for (i, &sig) in circuit.output_signals.iter().enumerate() {
            if i < outputs.len() {
                outputs[i] = values[sig.index()];
            }
        }
    }
}
