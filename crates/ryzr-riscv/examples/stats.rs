//! Print compiled-tape statistics for the gate-level RV32I core — circuit
//! size, level structure, and run counts. Useful for reasoning about which
//! engine should win on it.

use ryzr_backend::Compiled;
use ryzr_riscv::{build_cpu, programs};

fn main() {
    let circuit = build_cpu(&programs::fib_forever(), 256);
    let tape = Compiled::new(&circuit);

    let gates = tape.slot_count() - tape.gate_start as usize;
    println!("slots:      {}", tape.slot_count());
    println!("gates:      {gates}");
    println!("registers:  {}", tape.register_count());
    println!("levels:     {}", tape.levels.len());
    println!("runs:       {}", tape.runs.len());

    let widest =
        tape.levels.iter().map(|level| (level.end - level.start) as usize).max().unwrap_or(0);
    println!("widest level: {widest} gates");
}
