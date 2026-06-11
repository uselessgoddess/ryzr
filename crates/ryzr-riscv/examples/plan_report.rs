//! Per-tick work census of the packed plan for the gate-level RV32I core.
//! Temporary profiling aid: shows what dominates a tick (word ops by kind,
//! gather segment counts, capture and memory traffic).

use ryzr_backend::{Compiled, plan_report};
use ryzr_riscv::{build_cpu, programs};

fn main() {
    let circuit = build_cpu(&programs::fib_forever(), 256);
    let tape = Compiled::new(&circuit);
    print!("{}", plan_report(&tape));
}
