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
use crate::fuse::{Chain, find_chains};
use crate::mem::find_banks;

/// Minimum contiguous run length worth a funnel shift; shorter runs join
/// the splat groups (measured on the RV32I core: 2..=8 are within noise,
/// 3 is the optimum).
const FUNNEL_MIN: usize = 3;

/// One gather step. `kind` is implicit in the pool layout: each stream
/// stores its funnel segments first, then its splat segments.
#[derive(Clone, Copy)]
pub(crate) struct Seg {
    /// Source bit position in the arena.
    pub(crate) src: u32,
    /// Destination shift (funnel only; splat destinations live in `mask`).
    pub(crate) shift: u8,
    /// Destination bits this segment is allowed to write.
    pub(crate) mask: u64,
}

/// Per-stream slice of the segment pool.
#[derive(Clone, Copy, Default)]
pub(crate) struct StreamRef {
    pub(crate) funnels: u8,
    pub(crate) splats: u8,
}

/// Word-level operation of a task: an ordinary boolean gate word, or a
/// fused ripple-carry chain evaluated as one native add (see [`crate::fuse`]).
#[derive(Clone, Copy)]
pub(crate) enum TaskOp {
    Gate(Op),
    /// `dst = a + b + c`: sums in bits `0..len`, carry-out at bit `len`.
    Add,
    /// Fused RAM read: gather `bank[addr]` into `dst` (a fresh word). Indexes
    /// [`Plan::mem_reads`]; streams are empty.
    MemRead(u32),
}

/// A fused single-port RAM read: `dst = bank[addr]`, evaluated as one
/// dynamic-index funnel load over the bank's register-output region. The
/// address is assembled from the (already settled) address bits.
pub(crate) struct MemRead {
    /// Arena bit positions of the address bits, least significant first.
    pub(crate) addr_pos: Vec<u32>,
    /// Arena bit of bank cell `(word 0, bit 0)`.
    pub(crate) base_bit: u32,
    /// Bits per word.
    pub(crate) width: u32,
    /// Destination arena word for the read result (bit `i` = `bank[addr][i]`).
    pub(crate) dst: u32,
}

/// A fused single-port RAM write: `if E { bank[addr] = data }`, applied to
/// the register staging after the ordinary capture has copied the bank's
/// current contents forward (the unwritten "hold").
pub(crate) struct MemWrite {
    /// Arena bit positions of the address bits, least significant first.
    pub(crate) addr_pos: Vec<u32>,
    /// Staging bit of bank cell `(word 0, bit 0)`.
    pub(crate) base_bit: u32,
    pub(crate) width: u32,
    /// Arena bit of the shared store-enable.
    pub(crate) enable_pos: u32,
    /// Gather program (in [`Plan::mem_segs`]) producing the `width`-bit data
    /// word, plus its constant-folded immediate.
    pub(crate) data: StreamRef,
    pub(crate) data_imm: u64,
}

/// One output word: up to 64 same-op gates of one run.
#[derive(Clone, Copy)]
pub(crate) struct Task {
    pub(crate) op: TaskOp,
    /// Destination word index in the arena.
    pub(crate) dst: u32,
    pub(crate) streams: [StreamRef; 3],
    /// Constant-folded bits per operand stream.
    pub(crate) imm: [u64; 3],
}

/// Where an output reads from.
#[derive(Clone, Copy)]
pub(crate) enum OutSrc {
    Bit(u32),
    Const(bool),
}

/// Destination words of one fused chain: the sum word (sum bit i of link
/// i, carry-out at bit `len`), and the whole `P ^ Q` / `P & Q` words when
/// CSE let some pxq / g escape the chain (`u32::MAX` when not needed).
struct ChainDst {
    sum: u32,
    pxq: u32,
    g: u32,
}

/// The complete word-level execution plan: arena layout, gather programs
/// and register capture. [`PackedEngine`] interprets it; the packed JIT
/// engine compiles it to native code. Both execute the exact same plan.
pub(crate) struct Plan {
    /// Arena size in words (including the trailing pad word).
    pub(crate) words: usize,
    pub(crate) tasks: Vec<Task>,
    /// Shared segment pool, consumed in task/stream order.
    pub(crate) segs: Vec<Seg>,
    /// Gather program for the register capture: one entry per staging word.
    pub(crate) capture: Vec<(u64, StreamRef)>,
    pub(crate) cap_segs: Vec<Seg>,
    /// Word range of the register-output region (edge = word copy).
    pub(crate) reg_word: usize,
    /// Initial staging content, kept for reset.
    pub(crate) reg_init: Vec<u64>,
    pub(crate) outputs: Vec<OutSrc>,
    pub(crate) input_count: usize,
    /// Fused RAM reads, indexed by [`TaskOp::MemRead`].
    pub(crate) mem_reads: Vec<MemRead>,
    /// Fused RAM writes, applied after the capture in plan order.
    pub(crate) mem_writes: Vec<MemWrite>,
    /// Segment pool for the write data gathers, consumed in write order.
    pub(crate) mem_segs: Vec<Seg>,
    /// Staging length in words, including the pad word the write patch may
    /// touch when a bank word straddles a 64-bit boundary. The clock edge
    /// only copies the first `reg_init.len()` words back into the arena.
    pub(crate) staging_words: usize,
}

/// Write a `width`-bit value `val` (low bits) at bit offset `off` of a u64
/// slice, but only where `cond` is set (`!0` to write, `0` to leave intact).
/// The field may straddle two words; the higher word is touched only when it
/// genuinely overlaps, so a non-straddling write at the last live word never
/// reads past it.
fn patch(stage: &mut [u64], off: usize, width: usize, val: u64, cond: u64) {
    let lo = off / 64;
    let sh = off % 64;
    let vmask = ones(width) as u128;
    let placed = ((val as u128) & vmask) << sh;
    let pmask = vmask << sh;
    let lo_pm = (pmask as u64) & cond;
    stage[lo] = (stage[lo] & !lo_pm) | (placed as u64 & lo_pm);
    let pmask_hi = (pmask >> 64) as u64;
    if pmask_hi != 0 {
        let hi_pm = pmask_hi & cond;
        stage[lo + 1] = (stage[lo + 1] & !hi_pm) | ((placed >> 64) as u64 & hi_pm);
    }
}

pub struct PackedEngine {
    /// The packed value arena; one bit per live signal plus padding.
    bits: Vec<u64>,
    /// Register next-state captured at tick end, applied at next tick start.
    staging: Vec<u64>,
    plan: Plan,
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
        // `u32::MAX` marks an absent operand (an incrementer link's zero
        // q-bit inside a fused chain): constant zero, nothing to gather.
        if slot == u32::MAX {
            covered |= 1 << i;
            continue;
        }
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

impl Plan {
    pub(crate) fn new(tape: &Compiled) -> Self {
        let n = tape.slot_count();

        // Recognize ripple-carry adders first: their gates leave the window
        // pipeline entirely and settle as native word adds instead.
        let (chains, mut fused) = find_chains(tape);

        // Recognize single-port RAM banks: the whole read mux-tree and write
        // mux fabric leaves the pipeline, replaced by a dynamic-index gather
        // (read) and a copy-and-patch (write). Their muxes are marked fused
        // so they never occupy a run; the read roots are relocated below.
        let banks = find_banks(tape);
        for bank in &banks {
            for &slot in &bank.fused {
                fused[slot as usize] = true;
            }
        }

        // Constant values per slot; u8::MAX = not a constant.
        let mut cval = vec![u8::MAX; n];
        for &(slot, v) in &tape.const_slots {
            cval[slot as usize] = v;
        }

        // Arena layout: [inputs][align][live register outputs][align]
        // [gate runs, each word-aligned][chain words][pad word]. Constants
        // get no position — every read of them folds into an immediate.
        // Fused gates are compacted out of their runs; the survivors of a
        // run stay contiguous so funnels keep working across them.
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

        let run_slots: Vec<Vec<u32>> = tape
            .runs
            .iter()
            .map(|run| (run.start..run.end).filter(|&s| !fused[s as usize]).collect())
            .collect();
        for slots in &run_slots {
            if slots.is_empty() {
                continue;
            }
            bit = bit.next_multiple_of(64);
            for (i, &slot) in slots.iter().enumerate() {
                pos[slot as usize] = bit + i as u32;
            }
            bit += slots.len() as u32;
        }

        // Chain destinations: the add leaves the carry-out at bit `len`
        // for free, so it needs no gate of its own.
        let chain_dst: Vec<ChainDst> = chains
            .iter()
            .map(|chain| {
                let len = chain.links.len() as u32;
                bit = bit.next_multiple_of(64);
                let sum = bit / 64;
                for (i, link) in chain.links.iter().enumerate() {
                    if link.sum != u32::MAX {
                        pos[link.sum as usize] = bit + i as u32;
                    }
                }
                pos[chain.links.last().expect("chains are non-empty").carry as usize] = bit + len;
                bit += len + 1;
                let mut side = |on: bool, field: fn(&crate::fuse::Link) -> u32| {
                    if !on {
                        return u32::MAX;
                    }
                    bit = bit.next_multiple_of(64);
                    let word = bit / 64;
                    for (i, link) in chain.links.iter().enumerate() {
                        if field(link) != u32::MAX {
                            pos[field(link) as usize] = bit + i as u32;
                        }
                    }
                    bit += len;
                    word
                };
                let pxq = side(chain.ext_pxq, |l| l.pxq);
                let g = side(chain.ext_g, |l| l.g);
                ChainDst { sum, pxq, g }
            })
            .collect();

        // Fused-read destinations: one fresh word per bank holds `bank[addr]`.
        // The read roots are relocated here so downstream gathers read the
        // selected word instead of the (now absent) mux-tree tops. By this
        // point every address-bit and bank-cell slot already has a position.
        let mem_reads: Vec<MemRead> = banks
            .iter()
            .map(|bank| {
                bit = bit.next_multiple_of(64);
                let dst = bit / 64;
                for (i, &top) in bank.read_tops.iter().enumerate() {
                    pos[top as usize] = bit + i as u32;
                }
                bit += bank.width;
                MemRead {
                    addr_pos: bank.addr_bits.iter().map(|&s| pos[s as usize]).collect(),
                    base_bit: pos[tape.reg_out_slots[bank.base_reg as usize] as usize],
                    width: bank.width,
                    dst,
                }
            })
            .collect();

        let words = bit.div_ceil(64) as usize + 1;

        // Lower every 64-gate window of every run to a task, interleaving
        // each chain right before the windows of its `ready` level — by
        // then all of its operands have settled, and everything that reads
        // a chain output originally lived at level >= ready, so it still
        // executes after (validated by the chain finder).
        let mut tasks = Vec::new();
        let mut segs = Vec::new();

        let mut chain_order: Vec<usize> = (0..chains.len()).collect();
        chain_order.sort_by_key(|&i| chains[i].ready);
        let mut next_chain = 0usize;

        // Fused reads interleave the same way: emit each at its `read_ready`
        // level, by which point every address bit has settled, and well
        // below the levels of anything that consumes the selected word.
        let mut mem_order: Vec<usize> = (0..banks.len()).collect();
        mem_order.sort_by_key(|&i| banks[i].read_ready);
        let mut next_mem = 0usize;
        let emit_read = |m: usize, tasks: &mut Vec<Task>| {
            tasks.push(Task {
                op: TaskOp::MemRead(m as u32),
                dst: mem_reads[m].dst,
                streams: [StreamRef::default(); 3],
                imm: [0; 3],
            });
        };
        let emit_chain =
            |chain: &Chain, dst: &ChainDst, tasks: &mut Vec<Task>, segs: &mut Vec<Seg>| {
                let p: Vec<u32> = chain.links.iter().map(|l| l.p).collect();
                let q: Vec<u32> = chain.links.iter().map(|l| l.q).collect();
                let mut word_task = |op: TaskOp, dst: u32, windows: [&[u32]; 3]| {
                    let mut imm = [0u64; 3];
                    let mut streams = [StreamRef::default(); 3];
                    for (k, window) in windows.iter().enumerate() {
                        if !window.is_empty() {
                            (imm[k], streams[k]) = plan_stream(window, &pos, &cval, segs);
                        }
                    }
                    tasks.push(Task { op, dst, streams, imm });
                };
                if dst.pxq != u32::MAX {
                    word_task(TaskOp::Gate(Op::Xor), dst.pxq, [&p, &q, &[]]);
                }
                if dst.g != u32::MAX {
                    word_task(TaskOp::Gate(Op::And), dst.g, [&p, &q, &[]]);
                }
                word_task(TaskOp::Add, dst.sum, [&p, &q, &[chain.cin]]);
            };

        for (lvl, level) in tape.levels.iter().enumerate() {
            while next_chain < chain_order.len()
                && chains[chain_order[next_chain]].ready <= lvl as u32
            {
                let c = chain_order[next_chain];
                emit_chain(&chains[c], &chain_dst[c], &mut tasks, &mut segs);
                next_chain += 1;
            }
            while next_mem < mem_order.len() && banks[mem_order[next_mem]].read_ready <= lvl as u32
            {
                emit_read(mem_order[next_mem], &mut tasks);
                next_mem += 1;
            }
            let range = level.run_start as usize..level.run_end as usize;
            for (slots, run) in run_slots[range.clone()].iter().zip(&tape.runs[range]) {
                let op = run.op;
                let ar = arity(op);
                for chunk in slots.chunks(64) {
                    let window =
                        |sel: &[u32]| chunk.iter().map(|&s| sel[s as usize]).collect::<Vec<u32>>();
                    let mut imm = [0u64; 3];
                    let mut streams = [StreamRef::default(); 3];
                    (imm[0], streams[0]) = plan_stream(&window(&tape.a), &pos, &cval, &mut segs);
                    if ar > 1 {
                        (imm[1], streams[1]) =
                            plan_stream(&window(&tape.b), &pos, &cval, &mut segs);
                    }
                    if ar > 2 {
                        (imm[2], streams[2]) =
                            plan_stream(&window(&tape.c), &pos, &cval, &mut segs);
                    }
                    tasks.push(Task {
                        op: TaskOp::Gate(op),
                        dst: pos[chunk[0] as usize] / 64,
                        streams,
                        imm,
                    });
                }
            }
        }
        while next_chain < chain_order.len() {
            let c = chain_order[next_chain];
            emit_chain(&chains[c], &chain_dst[c], &mut tasks, &mut segs);
            next_chain += 1;
        }
        while next_mem < mem_order.len() {
            emit_read(mem_order[next_mem], &mut tasks);
            next_mem += 1;
        }

        // Capture sources: each live register's next state, except bank cells
        // hold their current value (their write muxes are fused away). The
        // fused write then patches just the addressed word into the staging.
        let mut cap_src: Vec<u32> = tape.reg_in_slots.clone();
        for bank in &banks {
            for cell in 0..bank.words * bank.width {
                let r = (bank.base_reg + cell) as usize;
                cap_src[r] = tape.reg_out_slots[r];
            }
        }

        // Gather program for the register capture: live registers' next
        // states, packed in the same order as the register-output region.
        let mut capture = Vec::new();
        let mut cap_segs = Vec::new();
        for chunk in live.chunks(64) {
            let window: Vec<u32> = chunk.iter().map(|&r| cap_src[r]).collect();
            let (imm, sr) = plan_stream(&window, &pos, &cval, &mut cap_segs);
            capture.push((imm, sr));
        }

        // Fused writes: gather each bank's data bus into a `width`-bit word;
        // at run time it patches the addressed staging word when the shared
        // store-enable is high.
        let mut mem_segs = Vec::new();
        let mem_writes: Vec<MemWrite> = banks
            .iter()
            .map(|bank| {
                let (data_imm, data) = plan_stream(&bank.data, &pos, &cval, &mut mem_segs);
                MemWrite {
                    addr_pos: bank.addr_bits.iter().map(|&s| pos[s as usize]).collect(),
                    base_bit: pos[tape.reg_out_slots[bank.base_reg as usize] as usize]
                        - (reg_word * 64) as u32,
                    width: bank.width,
                    enable_pos: pos[bank.enable as usize],
                    data,
                    data_imm,
                }
            })
            .collect();
        // One pad word: the JIT's write patch stores the (possibly straddled)
        // high word unconditionally, so the addressed bank word at the very
        // top of the staging needs a scratch word above it.
        let staging_words = live.len().div_ceil(64) + usize::from(!banks.is_empty());

        // Diagnostics, default off: RYZR_PACK_STATS=1 prints plan shape.
        if std::env::var_os("RYZR_PACK_STATS").is_some() {
            let absorbed = fused.iter().filter(|&&f| f).count();
            let links: usize = chains.iter().map(|c| c.links.len()).sum();
            eprintln!(
                "pack plan: {} tasks, {} segs, {} words; {} chains ({} links, {} gates absorbed)",
                tasks.len(),
                segs.len(),
                words,
                chains.len(),
                links,
                absorbed,
            );
        }

        // Validate the gather SAFETY contract once, here.
        let limit = (words as u32 - 1) * 64;
        for seg in segs.iter().chain(&cap_segs).chain(&mem_segs) {
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

        Self {
            words,
            tasks,
            segs,
            capture,
            cap_segs,
            reg_word,
            reg_init,
            outputs,
            input_count: tape.input_slots.len(),
            mem_reads,
            mem_writes,
            mem_segs,
            staging_words,
        }
    }
}

impl PackedEngine {
    pub fn new(circuit: &ryzr_core::Circuit) -> Self {
        Self::with_tape(&Compiled::new(circuit))
    }

    pub fn with_tape(tape: &Compiled) -> Self {
        let plan = Plan::new(tape);
        let mut engine =
            Self { bits: vec![0; plan.words], staging: vec![0; plan.staging_words], plan };
        engine.reset();
        engine
    }

    /// Restore power-on state: register initials latched, inputs low.
    pub(crate) fn reset(&mut self) {
        self.bits.fill(0);
        self.staging.fill(0);
        let regs = self.plan.reg_init.len();
        self.staging[..regs].copy_from_slice(&self.plan.reg_init);
        self.bits[self.plan.reg_word..self.plan.reg_word + regs]
            .copy_from_slice(&self.plan.reg_init);
    }
}

impl Engine for PackedEngine {
    fn name(&self) -> &'static str {
        "packed"
    }

    fn input_count(&self) -> usize {
        self.plan.input_count
    }

    fn output_count(&self) -> usize {
        self.plan.outputs.len()
    }

    fn set_input(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.plan.input_count);
        let mask = 1u64 << (index % 64);
        if value {
            self.bits[index / 64] |= mask;
        } else {
            self.bits[index / 64] &= !mask;
        }
    }

    fn output(&self, index: usize) -> bool {
        match self.plan.outputs[index] {
            OutSrc::Bit(p) => self.bits[(p >> 6) as usize] >> (p & 63) & 1 != 0,
            OutSrc::Const(v) => v,
        }
    }

    fn tick(&mut self) {
        // Clock edge: the register region is contiguous and word-aligned, so
        // applying the captured next-state is a straight word copy. Only the
        // live-register words copy back; any staging pad word the write patch
        // touched is scratch and stays behind.
        let regs = self.plan.reg_init.len();
        self.bits[self.plan.reg_word..self.plan.reg_word + regs]
            .copy_from_slice(&self.staging[..regs]);

        // Combinational settle: tasks are in (level, op) tape order, so
        // every gather reads only already-settled words.
        let mut cursor = self.plan.segs.as_slice();
        for task in &self.plan.tasks {
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
                TaskOp::Gate(Op::Not) => !a,
                TaskOp::Gate(Op::Buf) => a,
                TaskOp::Gate(Op::And) => a & gather(&self.bits, sb, fb, task.imm[1]),
                TaskOp::Gate(Op::Or) => a | gather(&self.bits, sb, fb, task.imm[1]),
                TaskOp::Gate(Op::Xor) => a ^ gather(&self.bits, sb, fb, task.imm[1]),
                TaskOp::Gate(Op::Nand) => !(a & gather(&self.bits, sb, fb, task.imm[1])),
                TaskOp::Gate(Op::Nor) => !(a | gather(&self.bits, sb, fb, task.imm[1])),
                TaskOp::Gate(Op::Xnor) => !(a ^ gather(&self.bits, sb, fb, task.imm[1])),
                TaskOp::Gate(Op::Mux) => {
                    let b = gather(&self.bits, sb, fb, task.imm[1]);
                    let c = gather(&self.bits, sc, fc, task.imm[2]);
                    c ^ (a & (b ^ c))
                }
                // Fused ripple chain: the native add propagates the carry
                // through all lanes; bit `len` is the chain's carry-out.
                TaskOp::Add => {
                    let b = gather(&self.bits, sb, fb, task.imm[1]);
                    let c = gather(&self.bits, sc, fc, task.imm[2]);
                    a.wrapping_add(b).wrapping_add(c)
                }
                // Fused RAM read: assemble the address from the settled
                // address bits, then a single dynamic-index funnel load pulls
                // `bank[addr]` out of the register-output region.
                TaskOp::MemRead(m) => {
                    let mr = &self.plan.mem_reads[m as usize];
                    let mut addr = 0usize;
                    for (j, &p) in mr.addr_pos.iter().enumerate() {
                        addr |= ((self.bits[(p >> 6) as usize] >> (p & 63) & 1) as usize) << j;
                    }
                    let src = mr.base_bit as usize + addr * mr.width as usize;
                    let w = src / 64;
                    let o = src % 64;
                    let lo = self.bits[w];
                    let word = if o == 0 { lo } else { (lo >> o) | (self.bits[w + 1] << (64 - o)) };
                    word & ones(mr.width as usize)
                }
            };
            self.bits[task.dst as usize] = word;
        }

        // Capture every live register's next state into staging; the edge
        // at the start of the next tick applies it, so output() observes
        // settled pre-edge values.
        let mut cursor = self.plan.cap_segs.as_slice();
        for (k, &(imm, sr)) in self.plan.capture.iter().enumerate() {
            let n = sr.funnels as usize + sr.splats as usize;
            let (head, rest) = cursor.split_at(n);
            cursor = rest;
            self.staging[k] = gather(&self.bits, head, sr.funnels as usize, imm);
        }

        // Fused RAM writes: the capture already held each bank's contents
        // forward (cells captured their own outputs), so a single guarded
        // patch overwrites just the addressed word when the store-enable is
        // high — exactly `mux(we_r, data, self)` for word `addr`, none else.
        let mut mcursor = self.plan.mem_segs.as_slice();
        for mw in &self.plan.mem_writes {
            let n = mw.data.funnels as usize + mw.data.splats as usize;
            let (head, rest) = mcursor.split_at(n);
            mcursor = rest;
            let mut addr = 0usize;
            for (j, &p) in mw.addr_pos.iter().enumerate() {
                addr |= ((self.bits[(p >> 6) as usize] >> (p & 63) & 1) as usize) << j;
            }
            let e = self.bits[(mw.enable_pos >> 6) as usize] >> (mw.enable_pos & 63) & 1;
            let cond = 0u64.wrapping_sub(e);
            let data = gather(&self.bits, head, mw.data.funnels as usize, mw.data_imm);
            let off = mw.base_bit as usize + addr * mw.width as usize;
            patch(&mut self.staging, off, mw.width as usize, data, cond);
        }
    }
}
