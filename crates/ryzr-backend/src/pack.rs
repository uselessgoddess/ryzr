//! Bit-packed single-instance engine: SWAR *within* one circuit instance.
//!
//! [`BatchEngine`](crate::BatchEngine) reaches word-level parallelism by
//! running 64 independent instances; this engine reaches it for a *single*
//! instance by packing every signal into one bit of a dense `u64` arena and
//! evaluating up to 64 same-op gates per word operation. One tick still
//! computes every gate — the honesty contract is untouched — but the unit
//! of work is a word, not a gate.
//!
//! The hard part is operand gathering: the 64 gates of an output word read
//! 64 arbitrary source bits per operand stream. Doing that bit by bit would
//! cost as much as the scalar engine, so a compile-time pass analyzes the
//! execution graph and lowers every operand stream to a short *gather
//! program* built from three primitives:
//!
//! - **immediate** — bits sourced from constants fold into a compile-time
//!   word; they cost nothing at runtime.
//! - **funnel** — a run of bit positions that is contiguous in the arena
//!   becomes one funnel shift (two loads, shift, or, mask), moving up to 64
//!   bits in ~6 ops. Word-level structure (buses, register files, mux
//!   trees) shows up as exactly this pattern.
//! - **splat** — one source bit broadcast to an arbitrary destination mask
//!   (`0u64.wrapping_sub(bit) & mask`). Costs ~5 ops per *distinct source
//!   bit* in the window, no matter how many destinations it feeds — this is
//!   what makes select lines and fan-out cheap.
//!
//! The arena layout is chosen by the planner: inputs first, then live
//! register outputs (contiguous, in register order, so the clock edge is a
//! straight word copy), then each gate run word-aligned so every store is a
//! full-word write. Pad bits hold garbage; nothing ever reads them, because
//! gather masks only cover real positions. A trailing pad word keeps the
//! funnel's second load in bounds.
//!
//! On the gate-level RV32I core this lowers ~136K scalar gate-ops per tick
//! to ~51K word-ops — and the whole arena is ~8 KB, so the working set
//! drops from L2 to L1.

use crate::Engine;
use crate::compile::{Compiled, Op, arity};

/// Minimum contiguous run length worth a funnel shift; shorter runs join
/// the splat groups (measured on the RV32I core: 2..=8 are within noise,
/// 3 is the optimum).
const FUNNEL_MIN: usize = 3;

/// One gather step. `kind` is implicit in the pool layout: each stream
/// stores its funnel segments first, then its splat segments.
#[derive(Clone, Copy)]
struct Seg {
    /// Source bit position in the arena.
    src: u32,
    /// Destination shift (funnel only; splat destinations live in `mask`).
    shift: u8,
    /// Destination bits this segment is allowed to write.
    mask: u64,
}

/// Per-stream slice of the segment pool.
#[derive(Clone, Copy, Default)]
struct StreamRef {
    funnels: u8,
    splats: u8,
}

/// One output word: up to 64 same-op gates of one run.
#[derive(Clone, Copy)]
struct Task {
    op: Op,
    /// Destination word index in the arena.
    dst: u32,
    streams: [StreamRef; 3],
    /// Constant-folded bits per operand stream.
    imm: [u64; 3],
}

/// Where an output reads from.
#[derive(Clone, Copy)]
enum OutSrc {
    Bit(u32),
    Const(bool),
}

pub struct PackedEngine {
    /// The packed value arena; one bit per live signal plus padding.
    bits: Vec<u64>,
    /// Register next-state captured at tick end, applied at next tick start.
    staging: Vec<u64>,
    tasks: Vec<Task>,
    /// Shared segment pool, consumed in task/stream order.
    segs: Vec<Seg>,
    /// Gather program for the register capture: one entry per staging word.
    capture: Vec<(u64, StreamRef)>,
    cap_segs: Vec<Seg>,
    /// Word range of the register-output region (edge = word copy).
    reg_word: usize,
    /// Initial staging content, kept for reset.
    reg_init: Vec<u64>,
    outputs: Vec<OutSrc>,
    input_count: usize,
}

fn ones(len: usize) -> u64 {
    if len >= 64 { !0 } else { (1u64 << len) - 1 }
}

/// Lower one operand stream of one window to a gather program. `window`
/// holds the source slot per destination bit; returns the constant-folded
/// immediate and appends funnel segments then splat segments to `segs`.
fn plan_stream(window: &[u32], pos: &[u32], cval: &[u8], segs: &mut Vec<Seg>) -> (u64, StreamRef) {
    debug_assert!(window.len() <= 64);
    let mut imm = 0u64;
    let mut covered = 0u64;

    for (i, &slot) in window.iter().enumerate() {
        let v = cval[slot as usize];
        if v != u8::MAX {
            covered |= 1 << i;
            imm |= u64::from(v) << i;
        }
    }

    let funnel_at = segs.len();
    let mut i = 0;
    while i < window.len() {
        if covered >> i & 1 != 0 {
            i += 1;
            continue;
        }
        let p = pos[window[i] as usize];
        let mut len = 1;
        while i + len < window.len()
            && covered >> (i + len) & 1 == 0
            && pos[window[i + len] as usize] == p + len as u32
        {
            len += 1;
        }
        if len >= FUNNEL_MIN {
            segs.push(Seg { src: p, shift: i as u8, mask: ones(len) << i });
            covered |= ones(len) << i;
        }
        i += len;
    }
    let funnels = segs.len() - funnel_at;

    // Group the remaining positions by source bit: one masked splat per
    // distinct source, regardless of destination count.
    let mut rest: Vec<(u32, usize)> = window
        .iter()
        .enumerate()
        .filter(|&(i, _)| covered >> i & 1 == 0)
        .map(|(i, &slot)| (pos[slot as usize], i))
        .collect();
    rest.sort_unstable();
    let splat_at = segs.len();
    for (p, i) in rest {
        if segs.len() > splat_at && segs.last().is_some_and(|seg| seg.src == p) {
            segs.last_mut().expect("just checked").mask |= 1 << i;
        } else {
            segs.push(Seg { src: p, shift: 0, mask: 1 << i });
        }
    }
    let splats = segs.len() - splat_at;

    (imm, StreamRef { funnels: funnels as u8, splats: splats as u8 })
}

/// Run one gather program over the arena.
///
/// SAFETY contract (established by the planner): every `src` position lies
/// within the real arena, and the arena carries one trailing pad word, so
/// `w` and `w + 1` are always in bounds.
#[inline(always)]
fn gather(bits: &[u64], segs: &[Seg], funnels: usize, imm: u64) -> u64 {
    let mut acc = imm;
    for seg in &segs[..funnels] {
        let w = (seg.src >> 6) as usize;
        let o = seg.src & 63;
        // SAFETY: see function-level contract.
        let (lo, hi) = unsafe { (*bits.get_unchecked(w), *bits.get_unchecked(w + 1)) };
        // Branchless funnel: source bit `src + k` lands at bit `shift + k`.
        // At o == 0 the high half shifts out entirely.
        let word = (lo >> o) | ((hi << (63 - o)) << 1);
        acc |= (word << seg.shift) & seg.mask;
    }
    for seg in &segs[funnels..] {
        let w = (seg.src >> 6) as usize;
        let o = seg.src & 63;
        // SAFETY: see function-level contract.
        let bit = unsafe { *bits.get_unchecked(w) } >> o & 1;
        acc |= 0u64.wrapping_sub(bit) & seg.mask;
    }
    acc
}

impl PackedEngine {
    pub fn new(circuit: &ryzr_core::Circuit) -> Self {
        Self::with_tape(&Compiled::new(circuit))
    }

    pub fn with_tape(tape: &Compiled) -> Self {
        let n = tape.slot_count();

        // Constant values per slot; u8::MAX = not a constant.
        let mut cval = vec![u8::MAX; n];
        for &(slot, v) in &tape.const_slots {
            cval[slot as usize] = v;
        }

        // Arena layout: [inputs][align][live register outputs][align]
        // [gate runs, each word-aligned][pad word]. Constants get no
        // position — every read of them folds into an immediate.
        let mut pos = vec![u32::MAX; n];
        let mut bit = tape.input_slots.len() as u32;
        for (i, &slot) in tape.input_slots.iter().enumerate() {
            pos[slot as usize] = i as u32;
        }

        bit = bit.next_multiple_of(64);
        let reg_word = (bit / 64) as usize;
        let live: Vec<usize> =
            (0..tape.register_count()).filter(|&r| tape.reg_out_slots[r] != u32::MAX).collect();
        for (k, &r) in live.iter().enumerate() {
            pos[tape.reg_out_slots[r] as usize] = bit + k as u32;
        }
        bit += live.len() as u32;

        for run in &tape.runs {
            bit = bit.next_multiple_of(64);
            for slot in run.start..run.end {
                pos[slot as usize] = bit + (slot - run.start);
            }
            bit += run.end - run.start;
        }
        let words = bit.div_ceil(64) as usize + 1;

        // Lower every 64-gate window of every run to a task.
        let mut tasks = Vec::new();
        let mut segs = Vec::new();
        for run in &tape.runs {
            let ar = arity(run.op);
            let mut s = run.start as usize;
            while s < run.end as usize {
                let e = usize::min(s + 64, run.end as usize);
                let mut imm = [0u64; 3];
                let mut streams = [StreamRef::default(); 3];
                (imm[0], streams[0]) = plan_stream(&tape.a[s..e], &pos, &cval, &mut segs);
                if ar > 1 {
                    (imm[1], streams[1]) = plan_stream(&tape.b[s..e], &pos, &cval, &mut segs);
                }
                if ar > 2 {
                    (imm[2], streams[2]) = plan_stream(&tape.c[s..e], &pos, &cval, &mut segs);
                }
                tasks.push(Task { op: run.op, dst: pos[s] / 64, streams, imm });
                s = e;
            }
        }

        // Gather program for the register capture: live registers' next
        // states, packed in the same order as the register-output region.
        let mut capture = Vec::new();
        let mut cap_segs = Vec::new();
        for chunk in live.chunks(64) {
            let window: Vec<u32> = chunk.iter().map(|&r| tape.reg_in_slots[r]).collect();
            let (imm, sr) = plan_stream(&window, &pos, &cval, &mut cap_segs);
            capture.push((imm, sr));
        }

        // Validate the gather SAFETY contract once, here.
        let limit = (words as u32 - 1) * 64;
        for seg in segs.iter().chain(&cap_segs) {
            assert!(seg.src < limit, "gather source out of the arena");
        }

        let mut reg_init = vec![0u64; live.len().div_ceil(64)];
        for (k, &r) in live.iter().enumerate() {
            reg_init[k / 64] |= u64::from(tape.reg_initial[r]) << (k % 64);
        }

        let outputs = tape
            .output_slots
            .iter()
            .map(|&slot| match cval[slot as usize] {
                u8::MAX => OutSrc::Bit(pos[slot as usize]),
                v => OutSrc::Const(v != 0),
            })
            .collect();

        let mut engine = Self {
            bits: vec![0; words],
            staging: vec![0; reg_init.len()],
            tasks,
            segs,
            capture,
            cap_segs,
            reg_word,
            reg_init,
            outputs,
            input_count: tape.input_slots.len(),
        };
        engine.reset();
        engine
    }

    /// Restore power-on state: register initials latched, inputs low.
    fn reset(&mut self) {
        self.bits.fill(0);
        self.staging.copy_from_slice(&self.reg_init);
        self.bits[self.reg_word..self.reg_word + self.staging.len()].copy_from_slice(&self.staging);
    }
}

impl Engine for PackedEngine {
    fn name(&self) -> &'static str {
        "packed"
    }

    fn input_count(&self) -> usize {
        self.input_count
    }

    fn output_count(&self) -> usize {
        self.outputs.len()
    }

    fn set_input(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.input_count);
        let mask = 1u64 << (index % 64);
        if value {
            self.bits[index / 64] |= mask;
        } else {
            self.bits[index / 64] &= !mask;
        }
    }

    fn output(&self, index: usize) -> bool {
        match self.outputs[index] {
            OutSrc::Bit(p) => self.bits[(p >> 6) as usize] >> (p & 63) & 1 != 0,
            OutSrc::Const(v) => v,
        }
    }

    fn tick(&mut self) {
        // Clock edge: the register region is contiguous and word-aligned,
        // so applying the captured next-state is a straight word copy.
        self.bits[self.reg_word..self.reg_word + self.staging.len()].copy_from_slice(&self.staging);

        // Combinational settle: tasks are in (level, op) tape order, so
        // every gather reads only already-settled words.
        let mut cursor = self.segs.as_slice();
        for task in &self.tasks {
            let mut take = |sr: StreamRef| {
                let n = sr.funnels as usize + sr.splats as usize;
                let (head, rest) = cursor.split_at(n);
                cursor = rest;
                (head, sr.funnels as usize)
            };
            let (sa, fa) = take(task.streams[0]);
            let (sb, fb) = take(task.streams[1]);
            let (sc, fc) = take(task.streams[2]);

            let a = gather(&self.bits, sa, fa, task.imm[0]);
            let word = match task.op {
                Op::Not => !a,
                Op::Buf => a,
                Op::And => a & gather(&self.bits, sb, fb, task.imm[1]),
                Op::Or => a | gather(&self.bits, sb, fb, task.imm[1]),
                Op::Xor => a ^ gather(&self.bits, sb, fb, task.imm[1]),
                Op::Nand => !(a & gather(&self.bits, sb, fb, task.imm[1])),
                Op::Nor => !(a | gather(&self.bits, sb, fb, task.imm[1])),
                Op::Xnor => !(a ^ gather(&self.bits, sb, fb, task.imm[1])),
                Op::Mux => {
                    let b = gather(&self.bits, sb, fb, task.imm[1]);
                    let c = gather(&self.bits, sc, fc, task.imm[2]);
                    c ^ (a & (b ^ c))
                }
            };
            self.bits[task.dst as usize] = word;
        }

        // Capture every live register's next state into staging; the edge
        // at the start of the next tick applies it, so output() observes
        // settled pre-edge values.
        let mut cursor = self.cap_segs.as_slice();
        for (k, &(imm, sr)) in self.capture.iter().enumerate() {
            let n = sr.funnels as usize + sr.splats as usize;
            let (head, rest) = cursor.split_at(n);
            cursor = rest;
            self.staging[k] = gather(&self.bits, head, sr.funnels as usize, imm);
        }
    }
}
