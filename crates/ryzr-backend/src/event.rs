//! Event-driven (activity-based) engine.
//!
//! Real circuits are mostly idle: in a CPU-like design only a few percent
//! of gates change value on any given clock. This engine exploits that by
//! recomputing *only* gates whose inputs actually changed, walking the
//!
//! dirty set level by level (a gate is evaluated at most once per tick).
//! Results are bit-for-bit identical to the dense engines — the cone of
//! influence of any change is always fully recomputed; everything outside
//! it provably cannot change.
//!
//! This is the same idea that lets openVCB-style simulators reach millions
//! of ticks per second on large circuits, applied to synchronous
//! semantics: per-tick cost is O(activity), not O(circuit).

use std::sync::Arc;

use crate::Engine;
use crate::compile::{Compiled, Op, arity};

pub struct EventEngine {
    tape: Arc<Compiled>,
    values: Vec<u8>,
    /// Next state captured at the end of the previous tick.
    reg_scratch: Vec<u8>,
    /// Per-level worklists of dirty gate slots.
    buckets: Vec<Vec<u32>>,
    /// Dedup bitmask: slot already queued this tick.
    queued: Vec<u64>,
}

impl EventEngine {
    pub fn new(circuit: &ryzr_core::Circuit) -> Self {
        Self::with_tape(Arc::new(Compiled::new(circuit)))
    }

    pub fn with_tape(tape: Arc<Compiled>) -> Self {
        let reg_scratch = vec![0; tape.register_count()];
        let max_level = tape.slot_level.iter().copied().max().unwrap_or(0) as usize;
        let buckets = vec![Vec::new(); max_level + 1];
        let queued = vec![0u64; tape.slot_count().div_ceil(64)];
        let mut engine = Self { tape, values: Vec::new(), reg_scratch, buckets, queued };
        engine.reset();
        engine
    }

    /// Restore power-on state: constants, register initials, inputs low.
    pub(crate) fn reset(&mut self) {
        self.values = self.tape.initial_values();
        self.reg_scratch.copy_from_slice(&self.tape.reg_initial);
        self.queued.fill(0);
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        // The value buffer is all-zero apart from sources; gate slots have
        // never been evaluated. Seed every gate once so the first tick
        // settles the whole circuit.
        for slot in self.tape.gate_start..self.tape.slot_count() as u32 {
            self.enqueue(slot);
        }
    }

    #[inline]
    fn enqueue(&mut self, slot: u32) {
        let (word, bit) = (slot as usize / 64, slot as usize % 64);
        if self.queued[word] >> bit & 1 == 0 {
            self.queued[word] |= 1 << bit;
            self.buckets[self.tape.slot_level[slot as usize] as usize].push(slot);
        }
    }

    /// Write a source slot; if the value changed, make dependents dirty.
    #[inline]
    fn write_source(&mut self, slot: u32, value: u8) {
        if self.values[slot as usize] != value {
            self.values[slot as usize] = value;
            let (lo, hi) = (
                self.tape.succ_offsets[slot as usize] as usize,
                self.tape.succ_offsets[slot as usize + 1] as usize,
            );
            for i in lo..hi {
                let succ = self.tape.succ[i];
                self.enqueue(succ);
            }
        }
    }

    #[inline]
    fn eval_gate(&self, slot: usize) -> u8 {
        let t = &self.tape;
        let a = self.values[t.a[slot] as usize];
        let op = t.ops[slot];
        let b = if arity(op) > 1 { self.values[t.b[slot] as usize] } else { 0 };
        let c = if arity(op) > 2 { self.values[t.c[slot] as usize] } else { 0 };
        match op {
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
}

impl Engine for EventEngine {
    fn name(&self) -> &'static str {
        "event"
    }

    fn input_count(&self) -> usize {
        self.tape.input_count()
    }

    fn output_count(&self) -> usize {
        self.tape.output_count()
    }

    fn set_input(&mut self, index: usize, value: bool) {
        let slot = self.tape.input_slots[index];
        self.write_source(slot, u8::from(value));
    }

    fn output(&self, index: usize) -> bool {
        self.values[self.tape.output_slots[index] as usize] != 0
    }

    fn tick(&mut self) {
        // 1. Apply the clock edge: changed register outputs dirty their cones.
        for r in 0..self.tape.register_count() {
            let slot = self.tape.reg_out_slots[r];
            if slot != u32::MAX {
                let next = self.reg_scratch[r];
                self.write_source(slot, next);
            }
        }

        // 2. Settle: process dirty gates level by level. A changed gate can
        //    only dirty strictly higher levels, so one ascending sweep is a
        //    complete settle.
        for level in 0..self.buckets.len() {
            // A gate's successors live at strictly higher levels, so this
            // bucket cannot grow while it is being drained.
            let mut bucket = std::mem::take(&mut self.buckets[level]);
            for &slot in &bucket {
                let (word, bit) = (slot as usize / 64, slot as usize % 64);
                self.queued[word] &= !(1 << bit);

                let value = self.eval_gate(slot as usize);
                if self.values[slot as usize] != value {
                    self.values[slot as usize] = value;
                    let (lo, hi) = (
                        self.tape.succ_offsets[slot as usize] as usize,
                        self.tape.succ_offsets[slot as usize + 1] as usize,
                    );
                    for i in lo..hi {
                        let succ = self.tape.succ[i];
                        self.enqueue(succ);
                    }
                }
            }
            debug_assert!(self.buckets[level].is_empty());
            // Hand the allocation back so steady-state ticks never allocate.
            bucket.clear();
            self.buckets[level] = bucket;
        }

        // 3. Capture next state for the coming edge.
        for r in 0..self.tape.register_count() {
            self.reg_scratch[r] = self.values[self.tape.reg_in_slots[r] as usize];
        }
    }
}
