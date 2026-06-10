//! Operand-pattern census for the gate-level RV32I core: how much of the
//! tape a packed single-instance engine could cover with word-wide segment
//! moves (contiguous funnel-shift extracts and single-bit splats) versus
//! bit-by-bit gathering — and how much an operand-aware reorder of gates
//! within each run would improve it. Run this before touching the packed
//! planner.

use ryzr_backend::Compiled;
use ryzr_backend::compile::arity;
use ryzr_riscv::{build_cpu, programs};

#[derive(Default)]
struct Census {
    funnel_segs: usize,
    splat_vals: usize,
    const_bits: usize,
    funnel_bits: usize,
    splat_bits: usize,
    streams: usize,
}

/// Plan one operand stream within one 64-gate window under the masked
/// segment model:
///   - bits sourced from constant slots fold into a compile-time immediate;
///   - contiguous source runs of >= MIN become funnel-shift extracts;
///   - every remaining distinct source bit becomes one masked splat
///     (destinations need not be contiguous), regardless of how many
///     window positions it feeds.
fn scan(stream: &[u32], is_const: &[bool], census: &mut Census, min: usize) {
    census.streams += 1;
    let mut covered = vec![false; stream.len()];

    for (i, &s) in stream.iter().enumerate() {
        if is_const[s as usize] {
            covered[i] = true;
            census.const_bits += 1;
        }
    }

    let mut i = 0;
    while i < stream.len() {
        if covered[i] {
            i += 1;
            continue;
        }
        let mut contig = 1;
        while i + contig < stream.len()
            && !covered[i + contig]
            && stream[i + contig] == stream[i] + contig as u32
        {
            contig += 1;
        }
        if contig >= min {
            census.funnel_segs += 1;
            census.funnel_bits += contig;
            covered[i..i + contig].iter_mut().for_each(|c| *c = true);
        }
        i += contig;
    }

    let mut leftovers: Vec<u32> =
        stream.iter().zip(&covered).filter(|&(_, &c)| !c).map(|(&s, _)| s).collect();
    census.splat_bits += leftovers.len();
    leftovers.sort_unstable();
    leftovers.dedup();
    census.splat_vals += leftovers.len();
}

fn census_tape(tape: &Compiled, reorder: bool, min: usize) -> (Census, usize, usize) {
    let mut is_const = vec![false; tape.slot_count()];
    for &(s, _) in &tape.const_slots {
        is_const[s as usize] = true;
    }

    let mut census = Census::default();
    let mut windows = 0usize;
    let mut gate_bits = 0usize;

    for run in &tape.runs {
        let (start, end) = (run.start as usize, run.end as usize);
        let ar = arity(run.op);

        // Optional operand-aware reorder: any permutation within a level is
        // semantically valid; cluster gates so windows see fewer distinct
        // source values and longer contiguous runs.
        let mut order: Vec<usize> = (start..end).collect();
        if reorder {
            order.sort_by_key(|&i| (tape.a[i], tape.b[i], tape.c[i]));
        }

        let mut s = 0;
        while s < order.len() {
            let e = usize::min(s + 64, order.len());
            windows += 1;
            gate_bits += e - s;
            let gather = |src: &[u32]| order[s..e].iter().map(|&i| src[i]).collect::<Vec<_>>();
            scan(&gather(&tape.a), &is_const, &mut census, min);
            if ar > 1 {
                scan(&gather(&tape.b), &is_const, &mut census, min);
            }
            if ar > 2 {
                scan(&gather(&tape.c), &is_const, &mut census, min);
            }
            s = e;
        }
    }
    (census, windows, gate_bits)
}

fn report(label: &str, census: &Census, windows: usize, gate_bits: usize) {
    let total = census.const_bits + census.funnel_bits + census.splat_bits;
    println!("== {label} ==");
    println!("gates: {gate_bits}, output words: {windows}, operand bits: {total}");
    println!(
        "  const-folded: {:5.1}%   funnel: {:5.1}%   splat: {:5.1}%",
        100.0 * census.const_bits as f64 / total as f64,
        100.0 * census.funnel_bits as f64 / total as f64,
        100.0 * census.splat_bits as f64 / total as f64
    );
    println!(
        "  funnel segments: {} ({:.2}/stream)   splat values: {} ({:.2}/stream)",
        census.funnel_segs,
        census.funnel_segs as f64 / census.streams as f64,
        census.splat_vals,
        census.splat_vals as f64 / census.streams as f64
    );
    // Rough op costs: funnel extract ~6, masked splat ~5, per-stream
    // setup ~2, per-window eval+store ~5; scalar gate ~6.
    let packed_ops =
        census.funnel_segs * 6 + census.splat_vals * 5 + census.streams * 2 + windows * 5;
    let scalar_ops = gate_bits * 6;
    println!(
        "  op budget: packed ~{packed_ops}, scalar ~{scalar_ops} ({:.2}x)\n",
        scalar_ops as f64 / packed_ops as f64
    );
}

fn main() {
    let circuit = build_cpu(&programs::fib_forever(), 256);
    let tape = Compiled::new(&circuit);

    for min in [2, 3, 4, 6, 8, 16] {
        let (census, windows, gate_bits) = census_tape(&tape, false, min);
        report(&format!("creation order, FUNNEL_MIN={min}"), &census, windows, gate_bits);
    }
    let (census, windows, gate_bits) = census_tape(&tape, true, 8);
    report("operand-sorted within runs, FUNNEL_MIN=8", &census, windows, gate_bits);
}
