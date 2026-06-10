//! Level-parallel engine.
//!
//! Gates within one level are independent by construction, so each level's
//! slot range can be evaluated by multiple threads with no synchronization
//! beyond the per-level barrier. The borrow split is fully safe: gates of
//! the current level only read slots `< level.start` (strictly lower
//! levels) and write their own contiguous `[level.start, level.end)` range,
//! which `split_at_mut` hands out disjointly.
//!
//! Parallelism pays off only when a level is *wide* — the per-level rayon
//! barrier costs microseconds, which at >1M ticks/s budgets is enormous.
//! Narrow levels therefore fall back to the scalar run loop; the width
//! threshold is tunable.

use std::sync::Arc;

use rayon::prelude::*;

use crate::Engine;
use crate::compile::{Compiled, Op};
use crate::scalar::{apply_edge, capture_next, eval_runs};

pub struct ThreadedEngine {
    tape: Arc<Compiled>,
    values: Vec<u8>,
    reg_scratch: Vec<u8>,
    /// Minimum level width before fanning out to rayon.
    parallel_threshold: usize,
}

impl ThreadedEngine {
    pub fn new(circuit: &ryzr_core::Circuit) -> Self {
        Self::with_tape(Arc::new(Compiled::new(circuit)))
    }

    pub fn with_tape(tape: Arc<Compiled>) -> Self {
        let values = tape.initial_values();
        let reg_scratch = tape.reg_initial.clone();
        Self { tape, values, reg_scratch, parallel_threshold: 1 << 15 }
    }

    /// Tune the width at which a level is worth parallelizing. Levels
    /// narrower than this run on the calling thread.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold.max(1);
        self
    }

    /// Restore power-on state: constants, register initials, inputs low.
    pub(crate) fn reset(&mut self) {
        self.values = self.tape.initial_values();
        self.reg_scratch.copy_from_slice(&self.tape.reg_initial);
    }
}

#[inline(always)]
fn eval_gate(tape: &Compiled, lower: &[u8], i: usize) -> u8 {
    // SAFETY: operands of a gate live at strictly lower levels, hence at
    // slots `< level.start == lower.len()`; validated in `Compiled::new`.
    let (a, b, c) = unsafe {
        (
            *lower.get_unchecked(tape.a[i] as usize),
            *lower.get_unchecked(tape.b[i] as usize),
            *lower.get_unchecked(tape.c[i] as usize),
        )
    };
    match tape.ops[i] {
        Op::And => a & b,
        Op::Or => a | b,
        Op::Xor => a ^ b,
        Op::Nand => (a & b) ^ 1,
        Op::Nor => (a | b) ^ 1,
        Op::Xnor => a ^ b ^ 1,
        Op::Not => a ^ 1,
        Op::Buf => a,
        Op::Mux => (a & b) | ((a ^ 1) & c),
    }
}

impl Engine for ThreadedEngine {
    fn name(&self) -> &'static str {
        "threaded"
    }

    fn input_count(&self) -> usize {
        self.tape.input_count()
    }

    fn output_count(&self) -> usize {
        self.tape.output_count()
    }

    fn set_input(&mut self, index: usize, value: bool) {
        self.values[self.tape.input_slots[index] as usize] = u8::from(value);
    }

    fn output(&self, index: usize) -> bool {
        self.values[self.tape.output_slots[index] as usize] != 0
    }

    fn tick(&mut self) {
        let tape = &self.tape;
        apply_edge(tape, &mut self.values, &self.reg_scratch);

        for level in &tape.levels {
            let width = (level.end - level.start) as usize;
            if width < self.parallel_threshold {
                eval_runs(tape, &mut self.values, level.run_start as usize..level.run_end as usize);
                continue;
            }

            let (lower, rest) = self.values.split_at_mut(level.start as usize);
            let (current, _) = rest.split_at_mut(width);
            let lower = &*lower;
            let base = level.start as usize;

            current.par_iter_mut().with_min_len(4096).enumerate().for_each(|(k, out)| {
                *out = eval_gate(tape, lower, base + k);
            });
        }

        capture_next(tape, &self.values, &mut self.reg_scratch);
    }
}
