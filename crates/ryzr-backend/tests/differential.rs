//! Differential testing: every engine must produce bit-for-bit identical
//! outputs to the `ryzr-core` reference interpreter, on every tick, for
//! randomly generated sequential circuits and random input sequences.

use ryzr_backend::{BatchEngine, Engine, EventEngine, PackedEngine, ScalarEngine};
use ryzr_core::{Backend, Circuit, CircuitBuilder, Interpreter, Signal};

#[cfg(feature = "jit")]
use ryzr_backend::JitEngine;
#[cfg(feature = "rayon")]
use ryzr_backend::ThreadedEngine;
#[cfg(all(feature = "jit", feature = "rayon"))]
use ryzr_backend::{Compiled, HybridEngine, Strategy};

/// Deterministic xorshift64* PRNG — no rand dependency, reproducible cases.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bool(&mut self) -> bool {
        self.next() & 1 != 0
    }
}

/// Random sequential circuit: a mix of every gate kind over a growing pool
/// of signals, with registers fed back into the pool so state actually
/// evolves over time.
fn random_circuit(rng: &mut Rng, inputs: usize, gates: usize, registers: usize) -> Circuit {
    let mut b = CircuitBuilder::new();
    let mut pool: Vec<Signal> = Vec::new();

    pool.push(b.const_val(false));
    pool.push(b.const_val(true));
    for i in 0..inputs {
        let s = b.input(format!("IN{i}"));
        pool.push(s);
    }
    let regs: Vec<_> = (0..registers)
        .map(|i| {
            let (reg, out) = b.reg(format!("R{i}"), rng.bool());
            pool.push(out);
            reg
        })
        .collect();

    for _ in 0..gates {
        let a = pool[rng.below(pool.len())];
        let x = pool[rng.below(pool.len())];
        let y = pool[rng.below(pool.len())];
        let s = match rng.below(9) {
            0 => b.and(a, x),
            1 => b.or(a, x),
            2 => b.xor(a, x),
            3 => b.nand(a, x),
            4 => b.nor(a, x),
            5 => b.xnor(a, x),
            6 => b.not(a),
            7 => b.buf(a),
            _ => b.mux(a, x, y),
        };
        pool.push(s);
    }

    for &reg in &regs {
        let s = pool[rng.below(pool.len())];
        b.drive(reg, s);
    }
    for i in 0..8.min(pool.len()) {
        b.output(format!("OUT{i}"), pool[pool.len() - 1 - i]);
    }

    b.finish().unwrap()
}

/// Drive the reference interpreter and one engine in lockstep, comparing
/// all outputs after every tick.
fn check_engine(circuit: &Circuit, engine: &mut dyn Engine, rng: &mut Rng, ticks: usize) {
    let oracle = Interpreter;
    let mut state = circuit.initial_state();
    let mut outputs = vec![false; circuit.output_count as usize];
    let mut inputs = vec![false; circuit.input_count as usize];

    for tick in 0..ticks {
        for (i, value) in inputs.iter_mut().enumerate() {
            *value = rng.bool();
            engine.set_input(i, *value);
        }

        oracle.tick(circuit, &inputs, &mut state, &mut outputs);
        engine.tick();

        for (i, &expected) in outputs.iter().enumerate() {
            assert_eq!(
                engine.output(i),
                expected,
                "{} diverged from oracle at tick {tick}, output {i}",
                engine.name()
            );
        }
    }
}

/// Wide-mode hybrid engine with a forced plan; threshold 4 forces the
/// parallel path even on tiny circuits.
#[cfg(all(feature = "jit", feature = "rayon"))]
fn hybrid(circuit: &Circuit, threshold: usize, strategy: Strategy) -> HybridEngine {
    HybridEngine::with_config(std::sync::Arc::new(Compiled::new(circuit)), threshold, strategy)
}

fn engines(circuit: &Circuit) -> Vec<Box<dyn Engine>> {
    vec![
        Box::new(ScalarEngine::new(circuit)),
        Box::new(EventEngine::new(circuit)),
        Box::new(BatchEngine::new(circuit)),
        Box::new(PackedEngine::new(circuit)),
        #[cfg(feature = "rayon")]
        Box::new(ThreadedEngine::new(circuit).with_threshold(4)),
        #[cfg(feature = "jit")]
        Box::new(JitEngine::new(circuit)),
        // The single-instance racer: whichever candidate wins must match.
        #[cfg(all(feature = "jit", feature = "rayon"))]
        Box::new(HybridEngine::new(circuit)),
        // Both wide plans must match the oracle, so pin each explicitly
        // instead of letting auto-tuning pick one.
        #[cfg(all(feature = "jit", feature = "rayon"))]
        Box::new(hybrid(circuit, 4, Strategy::Jit)),
        #[cfg(all(feature = "jit", feature = "rayon"))]
        Box::new(hybrid(circuit, 4, Strategy::Interpret)),
    ]
}

#[test]
fn random_circuits_match_oracle() {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    for case in 0..20 {
        let inputs = 1 + rng.below(8);
        let gates = 10 + rng.below(400);
        let registers = rng.below(16);
        let circuit = random_circuit(&mut rng, inputs, gates, registers);

        for mut engine in engines(&circuit) {
            let mut io_rng = Rng(0xFACE_0000 + case);
            check_engine(&circuit, engine.as_mut(), &mut io_rng, 64);
        }
    }
}

#[test]
fn large_random_circuit_matches_oracle() {
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let circuit = random_circuit(&mut rng, 16, 20_000, 256);

    for mut engine in engines(&circuit) {
        let mut io_rng = Rng(7);
        check_engine(&circuit, engine.as_mut(), &mut io_rng, 32);
    }
}

#[test]
fn counter_semantics_all_engines() {
    // 6-bit counter with feedback; value after t ticks must equal t mod 64.
    let mut b = CircuitBuilder::new();
    let regs: Vec<_> = (0..6).map(|i| b.reg(format!("BIT{i}"), false)).collect();
    let mut carry = b.const_val(true);
    for &(reg, bit) in &regs {
        let next = b.xor(bit, carry);
        b.drive(reg, next);
        carry = b.and(carry, bit);
    }
    for (i, &(_, bit)) in regs.iter().enumerate() {
        b.output(format!("OUT{i}"), bit);
    }
    let circuit = b.finish().unwrap();

    // Outputs observe the settled pre-edge values: after the t-th tick the
    // counter reads t-1 (same convention as the ryzr-core oracle tests).
    for mut engine in engines(&circuit) {
        for t in 0u64..200 {
            engine.tick();
            let value: u64 = (0..6).map(|i| u64::from(engine.output(i)) << i).sum();
            assert_eq!(value, t % 64, "{} wrong counter value at tick {t}", engine.name());
        }
    }
}

#[test]
fn batch_lanes_are_independent() {
    // An adder-like circuit; each lane gets different inputs and must
    // produce that lane's correct result.
    let mut b = CircuitBuilder::new();
    let x = b.input("X");
    let y = b.input("Y");
    let z = b.input("Z");
    let s1 = b.xor(x, y);
    let sum = b.xor(s1, z);
    let c1 = b.and(x, y);
    let c2 = b.and(s1, z);
    let carry = b.or(c1, c2);
    b.output("SUM", sum);
    b.output("CARRY", carry);
    let circuit = b.finish().unwrap();

    // Lane k carries the k-th bit of these masks.
    let (mx, my, mz) = (0x0123_4567_89AB_CDEF_u64, 0xFEDC_BA98_7654_3210, 0xAAAA_5555_F0F0_0F0F);
    let expected_sum = mx ^ my ^ mz;
    let expected_carry = (mx & my) | ((mx ^ my) & mz);

    let mut batch = BatchEngine::new(&circuit);
    batch.set_input_mask(0, mx);
    batch.set_input_mask(1, my);
    batch.set_input_mask(2, mz);
    batch.tick();
    assert_eq!(batch.output_mask(0), expected_sum);
    assert_eq!(batch.output_mask(1), expected_carry);

    #[cfg(all(feature = "jit", feature = "rayon"))]
    for strategy in [Strategy::Jit, Strategy::Interpret] {
        let mut hy = hybrid(&circuit, 2, strategy);
        hy.set_input_mask(0, mx);
        hy.set_input_mask(1, my);
        hy.set_input_mask(2, mz);
        hy.tick();
        assert_eq!(hy.output_mask(0), expected_sum, "{strategy:?}");
        assert_eq!(hy.output_mask(1), expected_carry, "{strategy:?}");
    }
}
