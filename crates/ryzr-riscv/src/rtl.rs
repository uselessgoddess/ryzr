//! Word-level RTL helpers over the bit-level [`CircuitBuilder`]: multi-bit
//! constants, muxes, decoders, ripple-carry adders, comparators and barrel
//! shifters. A word is a vector of wires, least significant bit first.

use ryzr_core::{CircuitBuilder, Signal};

pub const XLEN: usize = 32;

/// A machine word as individual wires, least significant bit first.
pub type Word = Vec<Signal>;

/// Constant word, one constant cell per bit.
pub fn constant(b: &mut CircuitBuilder, value: u32) -> Word {
    (0..XLEN).map(|i| b.const_val(value >> i & 1 == 1)).collect()
}

/// Per-bit 2:1 mux: `sel ? then_w : else_w`.
pub fn mux_word(b: &mut CircuitBuilder, sel: Signal, then_w: &[Signal], else_w: &[Signal]) -> Word {
    debug_assert_eq!(then_w.len(), else_w.len());
    then_w.iter().zip(else_w).map(|(&t, &e)| b.mux(sel, t, e)).collect()
}

/// `items[sel]` for the runtime value of `sel` (LSB first);
/// `items.len()` must equal `1 << sel.len()`.
pub fn mux_tree(b: &mut CircuitBuilder, sel: &[Signal], items: &[Word]) -> Word {
    assert_eq!(items.len(), 1 << sel.len());
    if sel.is_empty() {
        return items[0].clone();
    }
    let (rest, msb) = sel.split_at(sel.len() - 1);
    let half = items.len() / 2;
    let lo = mux_tree(b, rest, &items[..half]);
    let hi = mux_tree(b, rest, &items[half..]);
    mux_word(b, msb[0], &hi, &lo)
}

/// One-hot decode: line `k` is true iff `sel` (LSB first) reads `k`.
pub fn decoder(b: &mut CircuitBuilder, sel: &[Signal]) -> Vec<Signal> {
    let mut lines = vec![b.const_val(true)];
    for &s in sel {
        let ns = b.not(s);
        let mut next = Vec::with_capacity(lines.len() * 2);
        for &line in &lines {
            next.push(b.and(ns, line));
        }
        for &line in &lines {
            next.push(b.and(s, line));
        }
        lines = next;
    }
    lines
}

/// True iff `bits` (LSB first) equals the constant `value`.
pub fn match_bits(b: &mut CircuitBuilder, bits: &[Signal], value: u32) -> Signal {
    let mut acc: Option<Signal> = None;
    for (i, &bit) in bits.iter().enumerate() {
        let term = if value >> i & 1 == 1 { bit } else { b.not(bit) };
        acc = Some(match acc {
            Some(prev) => b.and(prev, term),
            None => term,
        });
    }
    acc.expect("match_bits needs at least one bit")
}

pub fn word_not(b: &mut CircuitBuilder, a: &[Signal]) -> Word {
    a.iter().map(|&s| b.not(s)).collect()
}

pub fn word_and(b: &mut CircuitBuilder, a: &[Signal], x: &[Signal]) -> Word {
    a.iter().zip(x).map(|(&p, &q)| b.and(p, q)).collect()
}

pub fn word_or(b: &mut CircuitBuilder, a: &[Signal], x: &[Signal]) -> Word {
    a.iter().zip(x).map(|(&p, &q)| b.or(p, q)).collect()
}

pub fn word_xor(b: &mut CircuitBuilder, a: &[Signal], x: &[Signal]) -> Word {
    a.iter().zip(x).map(|(&p, &q)| b.xor(p, q)).collect()
}

/// Ripple-carry adder; returns `(sum, carry_out)`.
pub fn adder(
    b: &mut CircuitBuilder,
    a: &[Signal],
    x: &[Signal],
    carry_in: Signal,
) -> (Word, Signal) {
    let mut carry = carry_in;
    let mut sum = Vec::with_capacity(a.len());
    for (&p, &q) in a.iter().zip(x) {
        let pxq = b.xor(p, q);
        sum.push(b.xor(pxq, carry));
        let generate = b.and(p, q);
        let propagate = b.and(pxq, carry);
        carry = b.or(generate, propagate);
    }
    (sum, carry)
}

/// `a + x`, carry discarded.
pub fn add(b: &mut CircuitBuilder, a: &[Signal], x: &[Signal]) -> Word {
    let zero = b.const_val(false);
    adder(b, a, x, zero).0
}

/// OR of all bits, as a balanced tree.
pub fn reduce_or(b: &mut CircuitBuilder, bits: &[Signal]) -> Signal {
    assert!(!bits.is_empty());
    let mut layer = bits.to_vec();
    while layer.len() > 1 {
        let next: Vec<Signal> = layer
            .chunks(2)
            .map(|pair| if pair.len() == 2 { b.or(pair[0], pair[1]) } else { pair[0] })
            .collect();
        layer = next;
    }
    layer[0]
}

/// True iff the two words are bit-for-bit equal.
pub fn equal(b: &mut CircuitBuilder, a: &[Signal], x: &[Signal]) -> Signal {
    let diff = word_xor(b, a, x);
    let any = reduce_or(b, &diff);
    b.not(any)
}

/// Logical left barrel shifter, zero fill; `amount` is LSB first.
pub fn shift_left(b: &mut CircuitBuilder, a: &[Signal], amount: &[Signal]) -> Word {
    let zero = b.const_val(false);
    let mut cur = a.to_vec();
    for (stage, &s) in amount.iter().enumerate() {
        let by = 1usize << stage;
        let shifted: Word =
            (0..cur.len()).map(|i| if i >= by { cur[i - by] } else { zero }).collect();
        cur = mux_word(b, s, &shifted, &cur);
    }
    cur
}

/// Right barrel shifter; vacated bits take `fill` (zero for SRL, the sign
/// bit for SRA).
pub fn shift_right(b: &mut CircuitBuilder, a: &[Signal], amount: &[Signal], fill: Signal) -> Word {
    let mut cur = a.to_vec();
    for (stage, &s) in amount.iter().enumerate() {
        let by = 1usize << stage;
        let n = cur.len();
        let shifted: Word = (0..n).map(|i| if i + by < n { cur[i + by] } else { fill }).collect();
        cur = mux_word(b, s, &shifted, &cur);
    }
    cur
}
