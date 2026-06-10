//! Splat census by circuit region: which levels/ops/run-sizes produce the
//! splat-heavy gather programs in the packed plan. Run this to decide where
//! placement or task specialization pays off.

use ryzr_backend::Compiled;
use ryzr_backend::compile::arity;
use ryzr_riscv::{build_cpu, programs};

const FUNNEL_MIN: usize = 3;

#[derive(Default, Clone)]
struct Bucket {
    windows: usize,
    gates: usize,
    funnels: usize,
    splats: usize,
    sel_uniform: usize,
}

fn scan(stream: &[u32], is_const: &[bool]) -> (usize, usize) {
    let mut covered = vec![false; stream.len()];
    for (i, &s) in stream.iter().enumerate() {
        if is_const[s as usize] {
            covered[i] = true;
        }
    }
    let mut funnels = 0;
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
        if contig >= FUNNEL_MIN {
            funnels += 1;
            covered[i..i + contig].iter_mut().for_each(|c| *c = true);
        }
        i += contig;
    }
    let mut leftovers: Vec<u32> =
        stream.iter().zip(&covered).filter(|&(_, &c)| !c).map(|(&s, _)| s).collect();
    leftovers.sort_unstable();
    leftovers.dedup();
    (funnels, leftovers.len())
}

fn main() {
    let circuit = build_cpu(&programs::fib_forever(), 256);
    let tape = Compiled::new(&circuit);
    let mut is_const = vec![false; tape.slot_count()];
    for &(s, _) in &tape.const_slots {
        is_const[s as usize] = true;
    }

    // Bucket key: (run size class, op).
    let mut by_class: std::collections::BTreeMap<(&'static str, u8), Bucket> =
        std::collections::BTreeMap::new();
    let mut level_hist = vec![0usize; 200];

    for run in &tape.runs {
        let size = (run.end - run.start) as usize;
        let class = match size {
            0..=8 => "a:1-8",
            9..=63 => "b:9-63",
            64..=255 => "c:64-255",
            _ => "d:256+",
        };
        let ar = arity(run.op);
        let lvl = tape.slot_level[run.start as usize] as usize;
        level_hist[lvl.min(199)] += size;

        let bucket = by_class.entry((class, run.op as u8)).or_default();
        let mut s = run.start as usize;
        while s < run.end as usize {
            let e = usize::min(s + 64, run.end as usize);
            bucket.windows += 1;
            bucket.gates += e - s;
            let (f, sp) = scan(&tape.a[s..e], &is_const);
            bucket.funnels += f;
            bucket.splats += sp;
            if ar > 1 {
                let (f, sp) = scan(&tape.b[s..e], &is_const);
                bucket.funnels += f;
                bucket.splats += sp;
            }
            if ar > 2 {
                let (f, sp) = scan(&tape.c[s..e], &is_const);
                bucket.funnels += f;
                bucket.splats += sp;
            }
            // uniform select: every a-operand identical (mux only)
            if ar > 2 && tape.a[s..e].iter().all(|&x| x == tape.a[s]) {
                bucket.sel_uniform += 1;
            }
            s = e;
        }
    }

    println!("{:>10} {:>4} {:>8} {:>7} {:>8} {:>8} {:>8}", "class", "op", "windows", "gates", "funnels", "splats", "selU");
    let mut tot = Bucket::default();
    for ((class, op), b) in &by_class {
        println!(
            "{:>10} {:>4} {:>8} {:>7} {:>8} {:>8} {:>8}",
            class, op, b.windows, b.gates, b.funnels, b.splats, b.sel_uniform
        );
        tot.windows += b.windows;
        tot.gates += b.gates;
        tot.funnels += b.funnels;
        tot.splats += b.splats;
        tot.sel_uniform += b.sel_uniform;
    }
    println!(
        "{:>10} {:>4} {:>8} {:>7} {:>8} {:>8} {:>8}",
        "TOTAL", "", tot.windows, tot.gates, tot.funnels, tot.splats, tot.sel_uniform
    );

    println!("\ngates per level (first 100):");
    for (lvl, &g) in level_hist.iter().enumerate().take(100) {
        if g > 0 {
            println!("  level {lvl:>3}: {g}");
        }
    }
}
