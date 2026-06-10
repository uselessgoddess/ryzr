//! Differential testing: every engine must produce bit-for-bit identical
//! outputs to the `ryzr-core` reference interpreter, on every tick, for
//! randomly generated sequential circuits and random input sequences.

#[cfg(feature = "rayon")]
use ryzr_backend::ThreadedEngine;
use ryzr_backend::{BatchEngine, Engine, EventEngine, PackedEngine, ScalarEngine};
#[cfg(all(feature = "jit", feature = "rayon"))]
use ryzr_backend::{Compiled, HybridEngine, Strategy};
#[cfg(feature = "jit")]
use ryzr_backend::{JitEngine, PackedJitEngine};
use ryzr_core::{Backend, Circuit, CircuitBuilder, Interpreter, Signal};

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
        #[cfg(feature = "jit")]
        Box::new(PackedJitEngine::new(circuit)),
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

/// Gate-level ripple-carry adder bit: returns the sum, updates the carry.
fn ripple_bit(b: &mut CircuitBuilder, p: Signal, q: Signal, carry: &mut Signal) -> Signal {
    let pxq = b.xor(p, q);
    let sum = b.xor(pxq, *carry);
    let g = b.and(p, q);
    let prop = b.and(pxq, *carry);
    *carry = b.or(g, prop);
    sum
}

#[test]
fn fused_ripple_accumulator_matches_oracle() {
    // 32-bit `acc += in` — the canonical full-adder chain the packed
    // planner fuses into a native word add. Every output bit and the
    // carry-out are observed, so any fusion bug is visible.
    let mut b = CircuitBuilder::new();
    let inputs: Vec<Signal> = (0..32).map(|i| b.input(format!("IN{i}"))).collect();
    let regs: Vec<_> = (0..32).map(|i| b.reg(format!("ACC{i}"), false)).collect();
    let mut carry = b.const_val(false);
    for i in 0..32 {
        let sum = ripple_bit(&mut b, regs[i].1, inputs[i], &mut carry);
        b.drive(regs[i].0, sum);
        b.output(format!("OUT{i}"), regs[i].1);
    }
    b.output("COUT", carry);
    let circuit = b.finish().unwrap();

    for mut engine in engines(&circuit) {
        let mut io_rng = Rng(0x0ADD_0ADD);
        check_engine(&circuit, engine.as_mut(), &mut io_rng, 256);
    }
}

#[test]
fn fused_subtractor_matches_oracle() {
    // `acc -= in` as `acc + !in + 1`: carry-in is constant one, which the
    // optimizer folds into bit 0 — the chain head it leaves behind starts
    // with a non-trivial carry expression.
    let mut b = CircuitBuilder::new();
    let inputs: Vec<Signal> = (0..32).map(|i| b.input(format!("IN{i}"))).collect();
    let regs: Vec<_> = (0..32).map(|i| b.reg(format!("ACC{i}"), true)).collect();
    let mut carry = b.const_val(true);
    for i in 0..32 {
        let nq = b.not(inputs[i]);
        let sum = ripple_bit(&mut b, regs[i].1, nq, &mut carry);
        b.drive(regs[i].0, sum);
        b.output(format!("OUT{i}"), regs[i].1);
    }
    b.output("BORROW", carry);
    let circuit = b.finish().unwrap();

    for mut engine in engines(&circuit) {
        let mut io_rng = Rng(0x050B_050B);
        check_engine(&circuit, engine.as_mut(), &mut io_rng, 256);
    }
}

#[test]
fn fused_wide_counter_matches_oracle() {
    // 24-bit incrementer chain (`x + 1` folded shape) — wide enough to
    // fuse, with an interior sum tapped as an extra observed output.
    let mut b = CircuitBuilder::new();
    let regs: Vec<_> = (0..24).map(|i| b.reg(format!("BIT{i}"), false)).collect();
    let mut carry = b.const_val(true);
    for (i, &(reg, bit)) in regs.iter().enumerate() {
        let next = b.xor(bit, carry);
        b.drive(reg, next);
        b.output(format!("OUT{i}"), bit);
        if i == 12 {
            b.output("TAP", next);
        }
        carry = b.and(carry, bit);
    }
    b.output("WRAP", carry);
    let circuit = b.finish().unwrap();

    // Output order: OUT0..OUT11, TAP wedged in after OUT12, rest, WRAP.
    let bit_out = |i: usize| if i <= 12 { i } else { i + 1 };

    for mut engine in engines(&circuit) {
        for t in 1u64..=4000 {
            engine.tick();
            let value: u64 = (0..24).map(|i| u64::from(engine.output(bit_out(i))) << i).sum();
            // Outputs observe settled pre-edge values: after tick t the
            // counter reads t - 1, and TAP shows the incremented bit 12.
            let expected = (t - 1) % (1 << 24);
            assert_eq!(value, expected, "{} wrong at tick {t}", engine.name());
            assert_eq!(engine.output(13), (t >> 12) & 1 != 0, "TAP at tick {t}");
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
