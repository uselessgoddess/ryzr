//! Single-port RAM-bank fusion: recognize a register array addressed by a
//! balanced mux-tree read port and a one-hot-decoded write port, and replace
//! the whole structure with a dynamic-index gather (read) and a copy-and-
//! patch (write).
//!
//! A gate-level RAM of `W` words is the packed engines' worst case. The read
//! port is a `W`-way mux tree — `(W - 1) * B` muxes that recompute the entire
//! array every tick to surface one selected word. The write port is another
//! `W * B` muxes, one per cell, each forwarding the cell's own value unless
//! its word is selected. On the RV32I core's 256-word RAM that is ~16K mux
//! gates — more than half the per-tick work — yet the function it computes is
//! just `read = bank[addr]` and `if store { bank[addr] = data }`.
//!
//! The detector finds the idiom the netlist optimizer leaves behind:
//!
//! - **write cells** — every bank register `r` has next-state
//!   `mux(we_r, data_i, r)`: selected-word-writes-data, else hold. Cells of
//!   one word share `we_r`; all words share the `data` bus; `we_r =
//!   and(E, dec_r)` shares a common store-enable `E`.
//! - **read tree** — a balanced `mux_tree(addr, words)`: `read_i` is the root
//!   of a `W`-leaf tree whose leaves are the bank cells' outputs for bit `i`
//!   and whose per-level selects are the address bits.
//!
//! Correctness is established structurally and exhaustively, not by pattern
//! trust: the read tree is *reconstructed* from the bank cells bottom-up (so
//! the matched roots provably compute `bank[addr]`), and the write decoder is
//! *simulated over its entire address space* to confirm `dec_r` is exactly
//! the one-hot `addr == r` over the very address bits the read tree uses. So
//! the fused read and write agree on the index, bit-for-bit, and the honesty
//! contract (every engine matches the naive interpreter on every declared
//! output) is untouched. Interior nodes (the mux trees themselves) are
//! unobservable and may be rewritten away.

use std::collections::HashMap;

use crate::compile::{Compiled, Op};

/// Smallest bank worth fusing; below this the mux tree is already cheap and
/// the dynamic-index overhead is not amortized.
const MIN_WORDS: usize = 16;

/// A fusable single-port RAM bank.
pub(crate) struct MemBank {
    /// Register index of cell `(word 0, bit 0)`; cells are contiguous in
    /// register order, word-major: cell `(w, i)` is `base_reg + w * width + i`.
    pub(crate) base_reg: u32,
    /// Number of words (a power of two).
    pub(crate) words: u32,
    /// Bits per word.
    pub(crate) width: u32,
    /// Address bit slots, least significant first.
    pub(crate) addr_bits: Vec<u32>,
    /// Shared store-enable slot.
    pub(crate) enable: u32,
    /// Data-bus slots, least significant first (`width` of them).
    pub(crate) data: Vec<u32>,
    /// Read-port output slots, least significant first (`width` of them).
    /// These are relocated to a fresh word produced by the fused read.
    pub(crate) read_tops: Vec<u32>,
    /// Every mux removed from the schedule: the whole read tree (internal
    /// nodes and roots) plus every write cell mux.
    pub(crate) fused: Vec<u32>,
    /// Schedule level at which the fused read may run: one past the latest
    /// address bit, so every address bit has settled.
    pub(crate) read_ready: u32,
}

/// Evaluate `slot`'s cone under an address-bit assignment. Returns `None` if
/// the cone reaches any non-constant signal that is not an address bit — i.e.
/// the candidate decoder depends on something other than the address.
fn eval(
    tape: &Compiled,
    slot: u32,
    assign: &HashMap<u32, bool>,
    cval: &[u8],
    memo: &mut HashMap<u32, Option<bool>>,
) -> Option<bool> {
    if let Some(&v) = assign.get(&slot) {
        return Some(v);
    }
    let cv = cval[slot as usize];
    if cv != u8::MAX {
        return Some(cv != 0);
    }
    if (slot as usize) < tape.gate_start as usize {
        return None;
    }
    if let Some(&v) = memo.get(&slot) {
        return v;
    }
    let s = slot as usize;
    let op = tape.ops[s];
    let a = eval(tape, tape.a[s], assign, cval, memo)?;
    let r = match op {
        Op::Not => Some(!a),
        Op::Buf => Some(a),
        _ => {
            let b = eval(tape, tape.b[s], assign, cval, memo)?;
            match op {
                Op::And => Some(a & b),
                Op::Or => Some(a | b),
                Op::Xor => Some(a ^ b),
                Op::Nand => Some(!(a & b)),
                Op::Nor => Some(!(a | b)),
                Op::Xnor => Some(!(a ^ b)),
                Op::Mux => Some(if a { b } else { eval(tape, tape.c[s], assign, cval, memo)? }),
                Op::Not | Op::Buf => unreachable!(),
            }
        }
    };
    memo.insert(slot, r);
    r
}

/// Rebuild the read mux tree for bit `i` over word range `[lo, hi)`, a power
/// of two. Returns the slot that computes `bank[addr][i]` for that range, or
/// `None` if any expected mux is missing. Records every touched mux in
/// `seen`; extracts and checks the per-level address bit into `addr_bits`.
#[allow(clippy::too_many_arguments)]
fn match_tree(
    i: usize,
    lo: usize,
    hi: usize,
    depth: usize,
    leaf: &dyn Fn(usize, usize) -> u32,
    mux_by_bc: &HashMap<(u32, u32), (u32, u32)>,
    addr_bits: &mut [u32],
    seen: &mut Vec<u32>,
) -> Option<u32> {
    if hi - lo == 1 {
        return Some(leaf(i, lo));
    }
    let half = (hi - lo) / 2;
    let lo_s = match_tree(i, lo, lo + half, depth - 1, leaf, mux_by_bc, addr_bits, seen)?;
    let hi_s = match_tree(i, lo + half, hi, depth - 1, leaf, mux_by_bc, addr_bits, seen)?;
    let &(sel, slot) = mux_by_bc.get(&(hi_s, lo_s))?;
    match addr_bits[depth - 1] {
        u32::MAX => addr_bits[depth - 1] = sel,
        prev if prev != sel => return None,
        _ => {}
    }
    seen.push(slot);
    Some(slot)
}

/// A fusable register file: a dual-read, one-write array whose logical word 0
/// is hardwired to zero (RISC-V `x0`, no storage). Only the *read* ports are
/// fused here — the writes stay ordinary gates, since the write decoder skips
/// the absent word 0 and the capture handles them as usual. One [`RegRead`] is
/// produced per read port: same region, same address bits, different roots.
pub(crate) struct RegRead {
    /// Register index of stored cell `(logical word 1, bit 0)`. Logical word 0
    /// is the constant zero (no storage); logical word `w >= 1` is the stored
    /// register at `base_reg + (w - 1) * width + bit`.
    pub(crate) base_reg: u32,
    /// Bits per word.
    pub(crate) width: u32,
    /// Address bit slots, least significant first (`log2(logical words)`).
    pub(crate) addr_bits: Vec<u32>,
    /// Read-port output slots, least significant first (`width` of them);
    /// relocated to the fused read's fresh word.
    pub(crate) read_tops: Vec<u32>,
    /// Every gate removed: the read-tree muxes (roots included) plus the
    /// leftmost `and(sel, x1)` of each bit (mux strength reduction of
    /// `mux(sel, x1, 0)`).
    pub(crate) fused: Vec<u32>,
    /// Schedule level at which the fused read may run.
    pub(crate) read_ready: u32,
}

/// Reconstruct one read port's mux tree for bit `i` over logical words
/// `[lo, hi)`. Logical word 0 is the constant-zero leaf, so the leftmost
/// bottom node `[0, 2)` is `and(sel, reg(word 1))` — the netlist optimizer
/// strength-reduces `mux(sel, x1, 0)` to `and(sel, x1)`. Every other internal
/// node is a balanced mux. The hi (right) half is reconstructed first so the
/// lower address bits are fixed before the zero spine needs them; with the LSB
/// select pre-seeded into `addr_bits[0]`, the two ports' shared bottom muxes
/// disambiguate by select.
#[allow(clippy::too_many_arguments)]
fn match_reg_tree(
    i: usize,
    lo: usize,
    hi: usize,
    depth: usize,
    mux_mm: &HashMap<(u32, u32), Vec<(u32, u32)>>,
    and_mm: &HashMap<(u32, u32), u32>,
    reg_out: &dyn Fn(usize, usize) -> u32,
    addr_bits: &mut [u32],
    seen: &mut Vec<u32>,
) -> Option<u32> {
    if hi - lo == 1 {
        // Single stored-register leaf (word >= 1; word 0 is never reached —
        // the `[0, 2)` collapsed node handles the zero spine directly).
        return Some(reg_out(i, lo));
    }
    if lo == 0 && hi - lo == 2 {
        let x1 = reg_out(i, 1);
        let sel = addr_bits[depth - 1];
        if sel == u32::MAX {
            return None;
        }
        // `and` is commutative-canonicalized, so try both operand orders.
        let &slot = and_mm.get(&(sel, x1)).or_else(|| and_mm.get(&(x1, sel)))?;
        seen.push(slot);
        return Some(slot);
    }
    let half = (hi - lo) / 2;
    let hi_s =
        match_reg_tree(i, lo + half, hi, depth - 1, mux_mm, and_mm, reg_out, addr_bits, seen)?;
    let lo_s =
        match_reg_tree(i, lo, lo + half, depth - 1, mux_mm, and_mm, reg_out, addr_bits, seen)?;
    let cands = mux_mm.get(&(hi_s, lo_s))?;
    let want = addr_bits[depth - 1];
    let mut chosen = None;
    for &(sel, slot) in cands {
        if want == u32::MAX || want == sel {
            if chosen.is_some() {
                return None; // ambiguous select at an unfixed level
            }
            chosen = Some((sel, slot));
        }
    }
    let (sel, slot) = chosen?;
    match addr_bits[depth - 1] {
        u32::MAX => addr_bits[depth - 1] = sel,
        prev if prev != sel => return None,
        _ => {}
    }
    seen.push(slot);
    Some(slot)
}

/// Detect every fusable register file (dual-read array with a hardwired-zero
/// word 0) and return one [`RegRead`] per read port.
pub(crate) fn find_regfiles(tape: &Compiled) -> Vec<RegRead> {
    let n = tape.slot_count();

    // Multimap (then, else) -> [(sel, mux slot)]: the two read ports share
    // their bottom-level operands, so this is genuinely multi-valued. The
    // leftmost spine `mux(sel, x1, 0)` is strength-reduced to `and(sel, x1)`.
    let mut mux_mm: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::new();
    let mut and_mm: HashMap<(u32, u32), u32> = HashMap::new();
    for s in tape.gate_start as usize..n {
        match tape.ops[s] {
            Op::Mux => {
                mux_mm.entry((tape.b[s], tape.c[s])).or_default().push((tape.a[s], s as u32))
            }
            Op::And => {
                and_mm.insert((tape.a[s], tape.b[s]), s as u32);
            }
            _ => {}
        }
    }

    // Register arrays: write cells `mux(we, data, self)` grouped by `we`, then
    // by shared data bus; contiguous word-major. (Same idiom `find_banks`
    // finds, but here we keep the arrays whose stored-word count is one short
    // of a power of two — the missing word is the hardwired-zero x0.)
    let mut by_select: HashMap<u32, Vec<u32>> = HashMap::new();
    for r in 0..tape.register_count() {
        let out = tape.reg_out_slots[r];
        if out == u32::MAX {
            continue;
        }
        let m = tape.reg_in_slots[r] as usize;
        if tape.ops[m] == Op::Mux && tape.c[m] == out {
            by_select.entry(tape.a[m]).or_default().push(r as u32);
        }
    }
    let data_of = |regs: &[u32]| -> Vec<u32> {
        regs.iter().map(|&r| tape.b[tape.reg_in_slots[r as usize] as usize]).collect()
    };
    let mut arrays: HashMap<Vec<u32>, Vec<Vec<u32>>> = HashMap::new();
    for regs in by_select.values() {
        let mut regs = regs.clone();
        regs.sort_unstable();
        if regs.windows(2).all(|w| w[1] == w[0] + 1) {
            arrays.entry(data_of(&regs)).or_default().push(regs);
        }
    }

    let reg_in: HashMap<u32, u32> = {
        let mut m = HashMap::new();
        for r in 0..tape.register_count() {
            if tape.reg_out_slots[r] != u32::MAX {
                m.insert(tape.reg_in_slots[r], r as u32);
            }
        }
        m
    };
    let is_output: Vec<bool> = {
        let mut v = vec![false; n];
        for &s in &tape.output_slots {
            v[s as usize] = true;
        }
        v
    };

    // Seeds: each contiguous write array gives a `(base, width)` of stored
    // cells. The write side fragments under optimization (some words' write
    // muxes are rewritten so their hold-branch is no longer the cell's own
    // output), so the *number* of stored words it reports is unreliable — but
    // the base register and word width of any surviving fragment are not. The
    // true logical word count is recovered from the read tree below.
    let mut seeds: Vec<(u32, u32)> = Vec::new();
    for groups in arrays.values() {
        for regs in groups {
            let width = regs.len() as u32;
            // Contiguous bit-cells of one word: `regs` is one word's cells.
            if regs.windows(2).all(|w| w[1] == w[0] + 1) {
                let seed = (regs[0], width);
                if !seeds.contains(&seed) {
                    seeds.push(seed);
                }
            }
        }
    }

    // Try one `(base, width, logical)` shape: reconstruct every read port's
    // tree and return one `RegRead` per port that passes every safety check.
    let try_shape = |base: u32, width: u32, logical: usize| -> Vec<RegRead> {
        let mut found = Vec::new();
        if logical < MIN_WORDS || !logical.is_power_of_two() {
            return found;
        }
        let kbits = logical.trailing_zeros() as usize;
        // Largest stored cell the tree reads must stay in range.
        let top_reg = base as usize + (logical - 1) * width as usize + width as usize;
        if top_reg > tape.register_count() {
            return found;
        }
        let reg_out = |i: usize, word: usize| -> u32 {
            tape.reg_out_slots[base as usize + (word - 1) * width as usize + i]
        };
        if (0..width as usize)
            .any(|i| reg_out(i, 1) == u32::MAX || reg_out(i, logical - 1) == u32::MAX)
        {
            return found;
        }

        // LSB select candidates: the logical (2, 3) bottom mux at bit 0 is
        // `mux(sel0, reg3, reg2)`, present once per read port.
        let Some(cands) = mux_mm.get(&(reg_out(0, 3), reg_out(0, 2))) else { return found };
        let lsb_candidates: Vec<u32> = cands.iter().map(|&(sel, _)| sel).collect();

        for s0 in lsb_candidates {
            let mut addr_bits = vec![u32::MAX; kbits];
            addr_bits[0] = s0;
            let mut tree_muxes = Vec::new();
            let mut read_tops = Vec::with_capacity(width as usize);
            let mut ok = true;
            for i in 0..width as usize {
                match match_reg_tree(
                    i,
                    0,
                    logical,
                    kbits,
                    &mux_mm,
                    &and_mm,
                    &reg_out,
                    &mut addr_bits,
                    &mut tree_muxes,
                ) {
                    Some(top) => read_tops.push(top),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || addr_bits.contains(&u32::MAX) {
                continue;
            }

            // Removal safety: the read tree (interior nodes and the leftmost
            // `and`s) vanishes; the roots survive, relocated. Every consumer of
            // a vanishing gate must itself vanish — nothing outside the tree,
            // no register, no declared output may read it.
            let removed: std::collections::HashSet<u32> = tree_muxes.iter().copied().collect();
            let roots: std::collections::HashSet<u32> = read_tops.iter().copied().collect();
            let safe = removed.iter().all(|&m| {
                if roots.contains(&m) {
                    return true;
                }
                if is_output[m as usize] || reg_in.contains_key(&m) {
                    return false;
                }
                tape.successors(m).iter().all(|&s| removed.contains(&s))
            });
            if !safe {
                continue;
            }

            // The fused read runs one level past the last address bit; every
            // consumer of a read root sits strictly above (the roots were the
            // tree's top, read only by deeper logic).
            let read_ready =
                addr_bits.iter().map(|&a| tape.slot_level[a as usize]).max().unwrap_or(0) + 1;
            let consumers_above = read_tops.iter().all(|&t| {
                tape.successors(t).iter().all(|&s| tape.slot_level[s as usize] >= read_ready)
            });
            if !consumers_above {
                continue;
            }

            found.push(RegRead {
                base_reg: base,
                width,
                addr_bits,
                read_tops,
                fused: tree_muxes,
                read_ready,
            });
        }
        found
    };

    let mut out = Vec::new();
    let mut used: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for &(base, width) in &seeds {
        // Recover the logical word count from the read tree: try the largest
        // power-of-two tree first and take the first that reconstructs fully.
        // A subtree of the true tree would also match, so largest-first is
        // essential — it fuses the whole register file, not a left half.
        let max_words = (tape.register_count() - base as usize) / width as usize + 1;
        let mut logical = max_words.next_power_of_two();
        if logical > max_words {
            logical /= 2;
        }
        while logical >= MIN_WORDS {
            let found = try_shape(base, width, logical);
            // Only accept a shape whose roots have not already been fused by a
            // larger seed (guards against re-fusing an inner subtree).
            if !found.is_empty()
                && found.iter().all(|rf| rf.read_tops.iter().all(|t| !used.contains(t)))
            {
                for rf in &found {
                    used.extend(rf.fused.iter().copied());
                    used.extend(rf.read_tops.iter().copied());
                }
                out.extend(found);
                break;
            }
            logical /= 2;
        }
    }
    out
}

/// Detect every fusable single-port RAM bank on the tape.
pub(crate) fn find_banks(tape: &Compiled) -> Vec<MemBank> {
    let n = tape.slot_count();

    let mut cval = vec![u8::MAX; n];
    for &(s, v) in &tape.const_slots {
        cval[s as usize] = v;
    }

    // Reverse mux map: (then, else) -> (sel, slot). A balanced read tree has
    // distinct (then, else) per node, so this is unambiguous.
    let mut mux_by_bc: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for s in tape.gate_start as usize..n {
        if tape.ops[s] == Op::Mux {
            mux_by_bc.insert((tape.b[s], tape.c[s]), (tape.a[s], s as u32));
        }
    }

    // Bank cells: registers whose next-state is `mux(sel, data, self)`.
    // Group by `sel` (the per-word write-enable).
    let mut by_select: HashMap<u32, Vec<u32>> = HashMap::new();
    for r in 0..tape.register_count() {
        let out = tape.reg_out_slots[r];
        if out == u32::MAX {
            continue;
        }
        let m = tape.reg_in_slots[r] as usize;
        if tape.ops[m] == Op::Mux && tape.c[m] == out {
            by_select.entry(tape.a[m]).or_default().push(r as u32);
        }
    }

    // A "word" is a write-enable group of cells at consecutive register
    // indices. Group words into banks by their shared data-bus vector.
    let data_of = |regs: &[u32]| -> Vec<u32> {
        regs.iter().map(|&r| tape.b[tape.reg_in_slots[r as usize] as usize]).collect()
    };
    let mut banks: HashMap<Vec<u32>, Vec<(u32, Vec<u32>)>> = HashMap::new();
    for (&we, regs) in &by_select {
        let mut regs = regs.clone();
        regs.sort_unstable();
        if regs.windows(2).all(|w| w[1] == w[0] + 1) {
            banks.entry(data_of(&regs)).or_default().push((we, regs));
        }
    }

    // Consumer map for the removal-safety check: which gate slots read a
    // given slot, and whether a register's next-state reads it.
    let reg_in: HashMap<u32, u32> = {
        let mut m = HashMap::new();
        for r in 0..tape.register_count() {
            if tape.reg_out_slots[r] != u32::MAX {
                m.insert(tape.reg_in_slots[r], r as u32);
            }
        }
        m
    };
    let is_output: Vec<bool> = {
        let mut v = vec![false; n];
        for &s in &tape.output_slots {
            v[s as usize] = true;
        }
        v
    };

    let mut out = Vec::new();
    for (_, bank) in banks {
        let w = bank.len();
        if w < MIN_WORDS || !w.is_power_of_two() {
            continue;
        }
        let width = bank[0].1.len() as u32;
        // Words ordered by base register; require a clean contiguous,
        // word-major layout: word `k` at `base + k * width`.
        let mut words: Vec<&(u32, Vec<u32>)> = bank.iter().collect();
        words.sort_by_key(|(_, regs)| regs[0]);
        let base = words[0].1[0];
        let layout_ok = words
            .iter()
            .enumerate()
            .all(|(k, (_, regs))| regs.len() as u32 == width && regs[0] == base + k as u32 * width);
        if !layout_ok {
            continue;
        }
        let kbits = w.trailing_zeros() as usize;

        let leaf = |i: usize, word: usize| -> u32 {
            tape.reg_out_slots[base as usize + word * width as usize + i]
        };

        // Reconstruct the read tree for every bit; collect roots and the
        // address bits.
        let mut addr_bits = vec![u32::MAX; kbits];
        let mut tree_muxes = Vec::new();
        let mut read_tops = Vec::with_capacity(width as usize);
        let mut matched = true;
        for i in 0..width as usize {
            match match_tree(i, 0, w, kbits, &leaf, &mux_by_bc, &mut addr_bits, &mut tree_muxes) {
                Some(top) => read_tops.push(top),
                None => {
                    matched = false;
                    break;
                }
            }
        }
        if !matched || addr_bits.contains(&u32::MAX) {
            continue;
        }

        // Common store-enable across every write-enable: `we = and(E, dec)`.
        let we0 = words[0].0 as usize;
        if tape.ops[we0] != Op::And {
            continue;
        }
        let enable = [tape.a[we0], tape.b[we0]].into_iter().find(|&c| {
            words.iter().all(|(we, _)| {
                let s = *we as usize;
                tape.ops[s] == Op::And && (tape.a[s] == c || tape.b[s] == c)
            })
        });
        let Some(enable) = enable else { continue };

        // Exhaustively verify the write decoder: `dec_r == (addr == r)` over
        // the whole address space, using the read tree's address bits.
        let mut decoder_ok = true;
        'verify: for v in 0..w {
            let assign: HashMap<u32, bool> =
                addr_bits.iter().enumerate().map(|(j, &ab)| (ab, (v >> j) & 1 == 1)).collect();
            let mut memo = HashMap::new();
            for (wi, (we, _)) in words.iter().enumerate() {
                let s = *we as usize;
                let dec = if tape.a[s] == enable { tape.b[s] } else { tape.a[s] };
                match eval(tape, dec, &assign, &cval, &mut memo) {
                    Some(bit) if bit == (wi == v) => {}
                    _ => {
                        decoder_ok = false;
                        break 'verify;
                    }
                }
            }
        }
        if !decoder_ok {
            continue;
        }

        // The write cell muxes (each bank cell's next-state).
        let write_muxes: Vec<u32> = words
            .iter()
            .flat_map(|(_, regs)| regs.iter().map(|&r| tape.reg_in_slots[r as usize]))
            .collect();

        // Removal safety: the read tree's interior plus the write muxes
        // vanish. Every consumer of a vanishing mux must itself vanish (no
        // gate outside the tree, no other register, no declared output may
        // read it). The read roots survive — relocated — so they are exempt.
        let removed: std::collections::HashSet<u32> =
            tree_muxes.iter().chain(&write_muxes).copied().collect();
        let roots: std::collections::HashSet<u32> = read_tops.iter().copied().collect();
        let safe = removed.iter().all(|&m| {
            // Read roots survive (relocated to the fused read's word), so
            // their external consumers are fine; only the vanishing nodes —
            // interior read muxes and write cells — must take all consumers
            // with them.
            if roots.contains(&m) {
                return true;
            }
            if is_output[m as usize] {
                return false;
            }
            // A write mux may be read only by its own bank cell; an interior
            // read mux by no register at all.
            if let Some(&r) = reg_in.get(&m) {
                let owns = write_muxes.contains(&m) && (base..base + w as u32 * width).contains(&r);
                if !owns {
                    return false;
                }
            }
            tape.successors(m).iter().all(|&s| removed.contains(&s))
        });
        if !safe {
            continue;
        }

        // The fused read runs one level past the last address bit. Every
        // consumer of a read root must sit strictly above that (it does: the
        // roots were the tree's top, read only by deeper logic), so the
        // relocation preserves scheduling.
        let read_ready =
            addr_bits.iter().map(|&a| tape.slot_level[a as usize]).max().unwrap_or(0) + 1;
        let consumers_above = read_tops.iter().all(|&t| {
            tape.successors(t).iter().all(|&s| tape.slot_level[s as usize] >= read_ready)
                && reg_in.get(&t).is_none_or(|_| true)
        });
        if !consumers_above {
            continue;
        }

        let mut fused = tree_muxes;
        fused.extend(write_muxes);
        out.push(MemBank {
            base_reg: base,
            words: w as u32,
            width,
            addr_bits,
            enable,
            data: bank[0]
                .1
                .iter()
                .map(|&r| tape.b[tape.reg_in_slots[r as usize] as usize])
                .collect(),
            read_tops,
            fused,
            read_ready,
        });
    }
    out
}
