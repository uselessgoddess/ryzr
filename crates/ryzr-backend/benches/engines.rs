use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(all(feature = "jit", feature = "rayon"))]
use ryzr_backend::HybridEngine;
#[cfg(feature = "jit")]
use ryzr_backend::JitEngine;
#[cfg(feature = "rayon")]
use ryzr_backend::ThreadedEngine;
use ryzr_backend::{BatchEngine, Engine, EventEngine, PackedEngine, ScalarEngine};
use ryzr_core::{Circuit, CircuitBuilder};

/// N-bit register counter with real feedback: bit[i] <= bit[i] ^ carry[i].
/// Wide and shallow; activity is low (a counter flips ~2 bits per tick on
/// average), which is the event engine's best case.
fn build_counter(n: u32) -> Circuit {
    let mut b = CircuitBuilder::new();

    let regs: Vec<_> = (0..n).map(|i| b.reg(format!("BIT{i}"), false)).collect();

    let mut carry = b.const_val(true);
    for &(reg, bit) in &regs {
        let next = b.xor(bit, carry);
        b.drive(reg, next);
        carry = b.and(carry, bit);
    }

    for (i, &(_, bit)) in regs.iter().enumerate() {
        b.output(format!("OUT[{i}]"), bit);
    }

    b.finish().unwrap()
}

/// W parallel N-bit linear feedback shift registers (Fibonacci LFSR,
/// taps n-1 and 0). Every register flips constantly — worst case for the
/// event engine, level-parallel friendly for the threaded one.
fn build_lfsr_array(width: u32, n: u32) -> Circuit {
    let mut b = CircuitBuilder::new();

    for w in 0..width {
        let regs: Vec<_> = (0..n).map(|i| b.reg(format!("R{w}_{i}"), i == 0)).collect();

        let feedback = b.xor(regs[(n - 1) as usize].1, regs[0].1);
        for i in 0..n as usize {
            let next = if i == 0 { feedback } else { regs[i - 1].1 };
            b.drive(regs[i].0, next);
        }
        b.output(format!("OUT{w}"), regs[(n - 1) as usize].1);
    }

    b.finish().unwrap()
}

fn engines(circuit: &Circuit) -> Vec<Box<dyn Engine>> {
    vec![
        Box::new(ScalarEngine::new(circuit)),
        Box::new(EventEngine::new(circuit)),
        Box::new(BatchEngine::new(circuit)),
        Box::new(PackedEngine::new(circuit)),
        #[cfg(feature = "rayon")]
        Box::new(ThreadedEngine::new(circuit)),
        #[cfg(feature = "jit")]
        Box::new(JitEngine::new(circuit)),
        #[cfg(all(feature = "jit", feature = "rayon"))]
        Box::new(HybridEngine::new(circuit)),
        #[cfg(all(feature = "jit", feature = "rayon"))]
        Box::new(HybridEngine::wide(circuit)),
    ]
}

/// 64-instance SWAR engines simulate 64 instances per tick; scale the
/// element count so elem/s means instance-ticks/s for every engine.
fn lanes_of(engine: &dyn Engine) -> u64 {
    match engine.name() {
        "batch64" | "hybrid64" => 64,
        _ => 1,
    }
}

fn bench_counter(c: &mut Criterion) {
    let mut group = c.benchmark_group("counter");

    for size in [64u32, 1024, 16384] {
        let circuit = build_counter(size);
        for mut engine in engines(&circuit) {
            group.throughput(Throughput::Elements(lanes_of(engine.as_ref())));
            group.bench_with_input(BenchmarkId::new(engine.name(), size), &size, |b, _| {
                b.iter(|| black_box(&mut engine).tick())
            });
        }
    }
    group.finish();
}

fn bench_lfsr(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfsr");

    for (width, n) in [(64u32, 64u32), (512, 64), (4096, 64)] {
        let circuit = build_lfsr_array(width, n);
        let gates = width * n;
        for mut engine in engines(&circuit) {
            group.throughput(Throughput::Elements(lanes_of(engine.as_ref())));
            group.bench_with_input(BenchmarkId::new(engine.name(), gates), &gates, |b, _| {
                b.iter(|| black_box(&mut engine).tick())
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_counter, bench_lfsr);
criterion_main!(benches);
