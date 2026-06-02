use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ryzr_core::{Backend, Circuit, CircuitBuilder};

/// N-bit ripple-carry adder
fn build_adder(n: u32) -> Circuit {
    let mut b = CircuitBuilder::new();

    let mut carry = b.const_val(false);
    let mut sum_signals = Vec::new();

    for i in 0..n {
        let a = b.input(format!("A{i}"));
        let b_in = b.input(format!("B{i}"));

        let axb = b.xor(a, b_in);
        let sum = b.xor(axb, carry);
        sum_signals.push(sum);

        let a_and_b = b.and(a, b_in);
        let axb_and_c = b.and(axb, carry);
        carry = b.or(a_and_b, axb_and_c);
    }

    for (i, &sig) in sum_signals.iter().enumerate() {
        b.output(format!("SUM{i}"), sig);
    }
    b.output("CARRY_OUT", carry);

    b.finish().unwrap()
}

/// N-bit register counter
fn build_counter(n: u32) -> Circuit {
    let mut b = CircuitBuilder::new();

    let mut bits = Vec::new();
    for i in 0..n {
        let bit = b.const_val(false);
        bits.push(b.register(format!("BIT[{i}]"), bit, false));
    }

    //  BIT[i+1] = BIT[i] XOR (BIT[i-1] AND BIT[i-2] AND ... AND BIT[0])
    let mut carry = b.const_val(true);
    let mut next_bits = Vec::new();

    for i in 0..n as usize {
        let next = b.xor(bits[i], carry);
        next_bits.push(next);
        carry = b.and(carry, bits[i]);
    }

    for (i, &next) in next_bits.iter().enumerate() {
        let _ = b.register(format!("NEXT[{i}]"), next, false);
    }

    for (i, &bit) in bits.iter().enumerate() {
        b.output(format!("OUT[{i}]"), bit);
    }

    b.finish().unwrap()
}

/// Цепочка AND вентилей (проверка глубины графа)
fn build_chain(n: u32) -> ryzr_core::Circuit {
    let mut b = CircuitBuilder::new();
    let mut sig = b.input("IN");

    for i in 0..n {
        let other = b.input(format!("X{}", i));
        sig = b.and(sig, other);
    }

    b.output("OUT", sig);
    b.finish().unwrap()
}

fn bench_adder(c: &mut Criterion) {
    let backend = ryzr_core::Interpreter;
    let mut group = c.benchmark_group("adder");

    for size in [8, 16, 32, 64].iter() {
        let circuit = build_adder(*size);
        let inputs = vec![false; circuit.input_count as usize];
        let mut state = vec![false; circuit.register_count as usize];
        let mut outputs = vec![false; circuit.output_count as usize];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("tick", size), size, |b, _| {
            b.iter(|| {
                backend.tick(
                    black_box(&circuit),
                    black_box(&inputs),
                    black_box(&mut state),
                    black_box(&mut outputs),
                )
            })
        });
    }
    group.finish();
}

fn bench_counter(c: &mut Criterion) {
    let backend = ryzr_core::Interpreter;
    let mut group = c.benchmark_group("counter");

    for size in [8, 16, 32].iter() {
        let circuit = build_counter(*size);
        let inputs = vec![false; circuit.input_count as usize];
        let mut state = vec![false; circuit.register_count as usize];
        let mut outputs = vec![false; circuit.output_count as usize];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("tick", size), size, |b, _| {
            b.iter(|| {
                backend.tick(
                    black_box(&circuit),
                    black_box(&inputs),
                    black_box(&mut state),
                    black_box(&mut outputs),
                )
            })
        });
    }
    group.finish();
}

fn bench_chain(c: &mut Criterion) {
    let backend = ryzr_core::Interpreter;
    let mut group = c.benchmark_group("chain");

    for size in [100, 1000, 10000].iter() {
        let circuit = build_chain(*size);
        let inputs = vec![true; circuit.input_count as usize];
        let mut state = vec![false; circuit.register_count as usize];
        let mut outputs = vec![false; circuit.output_count as usize];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("tick", size), size, |b, _| {
            b.iter(|| {
                backend.tick(
                    black_box(&circuit),
                    black_box(&inputs),
                    black_box(&mut state),
                    black_box(&mut outputs),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_adder, bench_counter, bench_chain);
criterion_main!(benches);
