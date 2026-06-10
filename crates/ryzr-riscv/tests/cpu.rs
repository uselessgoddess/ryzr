//! The gate-level core against the instruction-level emulator, in lockstep:
//! after every retired instruction the full architectural state (pc and all
//! 32 registers) must match bit for bit, on every engine.

use std::sync::Arc;

use ryzr_backend::{
    BatchEngine, Compiled, Engine, EventEngine, HybridEngine, JitEngine, PackedEngine,
    ScalarEngine, Strategy, ThreadedEngine,
};
use ryzr_core::Circuit;
use ryzr_riscv::{Emulator, build_cpu, programs};

fn engines(circuit: &Circuit) -> Vec<Box<dyn Engine>> {
    vec![
        Box::new(ScalarEngine::new(circuit)),
        Box::new(EventEngine::new(circuit)),
        Box::new(BatchEngine::new(circuit)),
        Box::new(PackedEngine::new(circuit)),
        Box::new(ThreadedEngine::new(circuit).with_threshold(64)),
        Box::new(JitEngine::new(circuit)),
        // The single-instance racer; threshold 64 lets the level-parallel
        // candidate exercise its parallel path on the CPU's wide levels.
        Box::new(HybridEngine::with_parallel_threshold(circuit, 64)),
        // The wide mode, observed through its lane-0 view.
        Box::new(HybridEngine::with_config(Arc::new(Compiled::new(circuit)), 64, Strategy::Auto)),
    ]
}

fn read_word(engine: &dyn Engine, base: usize) -> u32 {
    (0..32).map(|i| u32::from(engine.output(base + i)) << i).sum()
}

/// Outputs show the settled pre-edge values, so after tick `t` the circuit
/// exposes the state the emulator has after `t - 1` steps.
fn lockstep(program: &[u32], ram_words: usize, ticks: usize) {
    let circuit = build_cpu(program, ram_words);
    for mut engine in engines(&circuit) {
        let mut emu = Emulator::new(program, ram_words);
        for tick in 0..ticks {
            engine.tick();
            assert_eq!(
                read_word(engine.as_ref(), 0),
                emu.pc,
                "{}: pc diverged at tick {tick}",
                engine.name()
            );
            for r in 0..32 {
                assert_eq!(
                    read_word(engine.as_ref(), 32 * (r + 1)),
                    emu.regs[r],
                    "{}: x{r} diverged at tick {tick}",
                    engine.name()
                );
            }
            emu.step();
        }
    }
}

#[test]
fn fib_lockstep() {
    lockstep(&programs::fib_forever(), 4, 300);
}

#[test]
fn alu_lockstep() {
    lockstep(&programs::alu_exercise(), 4, 40);
}

#[test]
fn memory_lockstep() {
    lockstep(&programs::memory_exercise(), 16, 40);
}

#[test]
fn branch_lockstep() {
    lockstep(&programs::branch_exercise(), 4, 40);
}

/// Architecturally known result computed by the actual gates: fib(20).
#[test]
fn fib_20_computes_6765_in_hardware() {
    let circuit = build_cpu(&programs::fib_terminating(20), 4);
    let mut engine = ScalarEngine::new(&circuit);
    // 3 init + 20 iterations x 6 + halt margin; the spin loop holds state.
    engine.run(200);
    assert_eq!(read_word(&engine, 32 * 11), 6765, "a0 must hold fib(20)");
}
