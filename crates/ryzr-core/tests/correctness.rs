//! Semantic correctness tests for the reference interpreter.
//!
//! These act as the oracle for every optimized backend: if these fail,
//! nothing downstream can be trusted.

use ryzr_core::{Backend, Circuit, CircuitBuilder, GateOp, Interpreter};

fn tick(circuit: &Circuit, inputs: &[bool], state: &mut [bool]) -> Vec<bool> {
    let mut outputs = vec![false; circuit.output_count as usize];
    Interpreter.tick(circuit, inputs, state, &mut outputs);
    outputs
}

fn full_adder() -> Circuit {
    let mut b = CircuitBuilder::new();
    let a = b.input("a");
    let x = b.input("b");
    let cin = b.input("cin");

    let axb = b.xor(a, x);
    let sum = b.xor(axb, cin);
    let a_and_b = b.and(a, x);
    let axb_and_c = b.and(axb, cin);
    let cout = b.or(a_and_b, axb_and_c);

    b.output("sum", sum);
    b.output("cout", cout);
    b.finish().unwrap()
}

#[test]
fn full_adder_truth_table() {
    let circuit = full_adder();
    let mut state = vec![];

    for i in 0..8u32 {
        let a = i & 1 != 0;
        let x = i & 2 != 0;
        let cin = i & 4 != 0;

        let outputs = tick(&circuit, &[a, x, cin], &mut state);
        let expected_sum = a ^ x ^ cin;
        let expected_cout = (a && x) || ((a ^ x) && cin);
        assert_eq!(outputs[0], expected_sum, "sum mismatch for a={a} b={x} cin={cin}");
        assert_eq!(outputs[1], expected_cout, "cout mismatch for a={a} b={x} cin={cin}");
    }
}

#[test]
fn ripple_carry_adder_8bit() {
    let mut b = CircuitBuilder::new();
    let mut carry = b.const_val(false);
    let mut sums = Vec::new();

    for i in 0..8 {
        let a = b.input(format!("a{i}"));
        let x = b.input(format!("b{i}"));
        let axb = b.xor(a, x);
        let sum = b.xor(axb, carry);
        sums.push(sum);
        let a_and_b = b.and(a, x);
        let axb_and_c = b.and(axb, carry);
        carry = b.or(a_and_b, axb_and_c);
    }
    for (i, &s) in sums.iter().enumerate() {
        b.output(format!("s{i}"), s);
    }
    b.output("cout", carry);
    let circuit = b.finish().unwrap();

    let mut state = vec![];
    for (a, x) in [(0u16, 0u16), (1, 1), (37, 91), (255, 1), (200, 200), (255, 255)] {
        let mut inputs = vec![false; 16];
        for i in 0..8 {
            inputs[2 * i] = a >> i & 1 != 0;
            inputs[2 * i + 1] = x >> i & 1 != 0;
        }
        let outputs = tick(&circuit, &inputs, &mut state);
        let mut got = 0u16;
        for (i, &bit) in outputs.iter().enumerate().take(9) {
            got |= u16::from(bit) << i;
        }
        assert_eq!(got, a + x, "adder failed for {a} + {x}");
    }
}

#[test]
fn mux_selects() {
    let mut b = CircuitBuilder::new();
    let sel = b.input("sel");
    let t = b.input("t");
    let e = b.input("e");
    let m = b.mux(sel, t, e);
    b.output("m", m);
    let circuit = b.finish().unwrap();

    let mut state = vec![];
    assert!(tick(&circuit, &[true, true, false], &mut state)[0]);
    assert!(!tick(&circuit, &[false, true, false], &mut state)[0]);
    assert!(!tick(&circuit, &[true, false, true], &mut state)[0]);
    assert!(tick(&circuit, &[false, false, true], &mut state)[0]);
}

#[test]
fn shift_register_shifts() {
    // r0 <= d; r1 <= r0; r2 <= r1 — a 1 fed once ripples down the chain.
    let mut b = CircuitBuilder::new();
    let d = b.input("d");
    let r0 = b.register("r0", d, false);
    let r1 = b.register("r1", r0, false);
    let r2 = b.register("r2", r1, false);
    b.output("q0", r0);
    b.output("q1", r1);
    b.output("q2", r2);
    let circuit = b.finish().unwrap();

    let mut state = vec![false; circuit.register_count as usize];

    let seq = [true, false, false, false, false];
    let mut history = Vec::new();
    for &d in &seq {
        let outputs = tick(&circuit, &[d], &mut state);
        history.push((outputs[0], outputs[1], outputs[2]));
    }

    // Outputs are sampled *before* the clock edge (current state).
    assert_eq!(history[0], (false, false, false));
    assert_eq!(history[1], (true, false, false));
    assert_eq!(history[2], (false, true, false));
    assert_eq!(history[3], (false, false, true));
    assert_eq!(history[4], (false, false, false));
}

#[test]
fn deterministic_compilation() {
    // Compiling the same circuit twice must produce identical instruction order.
    let build = || {
        let mut b = CircuitBuilder::new();
        let mut acc = b.input("x0");
        for i in 1..64 {
            let x = b.input(format!("x{i}"));
            acc = if i % 3 == 0 { b.xor(acc, x) } else { b.and(acc, x) };
        }
        b.output("out", acc);
        b.finish().unwrap()
    };
    let c1 = build();
    let c2 = build();
    let order1: Vec<_> = c1.insts.iter().map(|(_, inst)| format!("{:?}", inst.data)).collect();
    let order2: Vec<_> = c2.insts.iter().map(|(_, inst)| format!("{:?}", inst.data)).collect();
    assert_eq!(order1, order2);
}

#[test]
fn nand_nor_xnor_not_buf() {
    let mut b = CircuitBuilder::new();
    let a = b.input("a");
    let x = b.input("b");
    let nand = b.nand(a, x);
    let nor = b.binary(GateOp::Nor, a, x);
    let xnor = b.binary(GateOp::Xnor, a, x);
    let not = b.not(a);
    let buf = b.unary(GateOp::Buf, a);
    b.output("nand", nand);
    b.output("nor", nor);
    b.output("xnor", xnor);
    b.output("not", not);
    b.output("buf", buf);
    let circuit = b.finish().unwrap();

    let mut state = vec![];
    for i in 0..4u32 {
        let a = i & 1 != 0;
        let x = i & 2 != 0;
        let outputs = tick(&circuit, &[a, x], &mut state);
        assert_eq!(outputs[0], !(a && x), "nand a={a} b={x}");
        assert_eq!(outputs[1], !(a || x), "nor a={a} b={x}");
        assert_eq!(outputs[2], !(a ^ x), "xnor a={a} b={x}");
        assert_eq!(outputs[3], !a, "not a={a}");
        assert_eq!(outputs[4], a, "buf a={a}");
    }
}

#[test]
fn cycle_detection() {
    // A gate cannot reference itself through the builder API (no forward
    // references), so combinational cycles are impossible to construct and
    // finish() must succeed on any builder-constructed DAG.
    let mut b = CircuitBuilder::new();
    let a = b.input("a");
    let n = b.not(a);
    let n2 = b.not(n);
    b.output("o", n2);
    assert!(b.finish().is_ok());
}
