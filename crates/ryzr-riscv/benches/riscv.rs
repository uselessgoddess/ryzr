//! The headline benchmark: instructions per second on the gate-level RV32I
//! core. One engine tick retires one instruction, so the throughput numbers
//! are directly in instructions/sec.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ryzr_backend::{
    BatchEngine, Engine, EventEngine, HybridEngine, JitEngine, PackedEngine, ScalarEngine,
    ThreadedEngine,
};
use ryzr_core::Circuit;
use ryzr_riscv::{build_cpu, programs};
use std::hint::black_box;

fn engines(circuit: &Circuit) -> Vec<Box<dyn Engine>> {
    vec![
        Box::new(ScalarEngine::new(circuit)),
        Box::new(EventEngine::new(circuit)),
        Box::new(BatchEngine::new(circuit)),
        Box::new(PackedEngine::new(circuit)),
        Box::new(ThreadedEngine::new(circuit)),
        Box::new(JitEngine::new(circuit)),
        Box::new(HybridEngine::new(circuit)),
        Box::new(HybridEngine::wide(circuit)),
    ]
}

fn bench_riscv(c: &mut Criterion) {
    let circuit = build_cpu(&programs::fib_forever(), 256);
    let mut group = c.benchmark_group("riscv");
    for mut engine in engines(&circuit) {
        // 64-instance SWAR engines retire one instruction per lane per
        // tick, so scale the element count to keep elem/s = instructions/s
        // everywhere.
        let lanes = match engine.name() {
            "batch64" | "hybrid64" => 64,
            _ => 1,
        };
        group.throughput(Throughput::Elements(lanes));
        group.bench_function(engine.name(), |b| b.iter(|| black_box(&mut engine).tick()));
    }
    group.finish();
}

criterion_group!(benches, bench_riscv);
criterion_main!(benches);
