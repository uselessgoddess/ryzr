//! Carry-chain fusion: recognize ripple-carry adders on the tape and
//! collapse each one into a single word-level add.
//!
//! Gate-level adders dominate the RV32I core's critical path: a 32-bit
//! ripple adder is ~160 gates spread over ~64 levels, and the packed
//! engines pay for every level with tiny one-bit splat windows — the carry
//! recurrence has no word-level parallelism to pack. But the function the
//! chain computes is just `S = P + Q + cin`, and the host ALU evaluates
//! that in one instruction, carry propagation included.
//!
//! The detector finds the two shapes the netlist optimizer leaves behind:
//!
//! - **full-adder links** — `pxq = p ^ q`, `g = p & q`, `prop = pxq & c`,
//!   `c' = g | prop`, `sum = pxq ^ c`: the canonical ripple bit;
//! - **incrementer links** — `c' = p & c`, `sum = p ^ c`: what constant
//!   folding leaves of an `x + const` adder (e.g. `pc + 4`).
//!
//! A run of links connected through their carries becomes one fused task:
//! gather the P and Q operand words with the ordinary gather machinery,
//! then `S = P + Q + cin`. Sums land at bits `0..len` and bit `len` is the
//! carry-out — for free, because the add already propagated it. Interior
//! nets (`pxq`, `g`, `prop`, intermediate carries) usually die with the
//! chain; when CSE merged them with gates that escape it (the ALU's
//! XOR/AND results share gates with its adder), the planner materializes
//! whole `P ^ Q` / `P & Q` words alongside — two word ops instead of two
//! per bit.
//!
//! Fusion is exact, not approximate: the fused add computes bit-for-bit
//! the same boolean functions as the gates it replaces, so the honesty
//! contract (every engine matches the naive interpreter on every declared
//! output, every tick) is untouched.

use std::collections::HashMap;

use crate::compile::{Compiled, Op};

/// Minimum links worth fusing; below this the window machinery is fine.
const MIN_CHAIN: usize = 8;
/// Sums occupy bits `0..len` and the carry-out lands at bit `len`, so a
/// chain (plus its carry) must fit in one word.
const MAX_CHAIN: usize = 63;

/// One ripple bit: `c' = maj(p, q, c)`, `sum = p ^ q ^ c`. Full-adder
/// links carry the whole five-gate structure; incrementer links (`q`
/// absent, i.e. zero) are just the carry `and` plus the sum.
#[derive(Clone, Copy)]
pub(crate) struct Link {
    pub(crate) p: u32,
    /// `u32::MAX` for incrementer links (q = 0); same for the three
    /// full-adder-only gates below.
    pub(crate) q: u32,
    /// `xor(p, q)`.
    pub(crate) pxq: u32,
    /// `and(p, q)`.
    pub(crate) g: u32,
    /// `and(pxq, c)`.
    prop: u32,
    /// The carry-out gate (`or` for full adders, `and` for incrementers).
    pub(crate) carry: u32,
    /// `xor(pxq | p, c)`; `u32::MAX` when the sum is never used.
    pub(crate) sum: u32,
    /// Carry-in; for every link but a chain's head this is the previous
    /// link's carry.
    cin: u32,
    /// Whether pxq / g escape the chain and must be materialized.
    pub(crate) ext_pxq: bool,
    pub(crate) ext_g: bool,
}

fn link_gates(link: &Link) -> impl Iterator<Item = u32> {
    [link.pxq, link.g, link.prop, link.carry, link.sum].into_iter().filter(|&g| g != u32::MAX)
}

pub(crate) struct Chain {
    /// Carry-in slot of the first link.
    pub(crate) cin: u32,
    pub(crate) links: Vec<Link>,
    /// Some link's pxq / g escapes: materialize the whole P^Q / P&Q word.
    pub(crate) ext_pxq: bool,
    pub(crate) ext_g: bool,
    /// Emission boundary: the fused tasks run right before the windows of
    /// this level, by which point every gathered operand has settled.
    pub(crate) ready: u32,
}

/// Find fusable carry chains. Returns the chains plus a per-slot flag for
/// every gate a chain replaces (excluded from window planning).
pub(crate) fn find_chains(tape: &Compiled) -> (Vec<Chain>, Vec<bool>) {
    Finder::new(tape).run()
}

fn key(x: u32, y: u32) -> (u32, u32) {
    if x <= y { (x, y) } else { (y, x) }
}

/// Match one full-adder link `carry = or(g, prop)` under the given operand
/// assignment.
fn full_link(
    tape: &Compiled,
    xor_of: &HashMap<(u32, u32), u32>,
    g: u32,
    prop: u32,
    carry: u32,
) -> Option<Link> {
    if tape.ops[g as usize] != Op::And || tape.ops[prop as usize] != Op::And || g == prop {
        return None;
    }
    let (p, q) = (tape.a[g as usize], tape.b[g as usize]);
    let pxq = *xor_of.get(&key(p, q))?;
    let (x, y) = (tape.a[prop as usize], tape.b[prop as usize]);
    let cin = if x == pxq {
        y
    } else if y == pxq {
        x
    } else {
        return None;
    };
    let sum = xor_of.get(&key(pxq, cin)).copied().unwrap_or(u32::MAX);
    // The matched gates must be five distinct slots (sum may be absent).
    let gates = [pxq, g, prop, carry, sum];
    for (i, &gi) in gates.iter().enumerate() {
        if gi != u32::MAX && gates[i + 1..].contains(&gi) {
            return None;
        }
    }
    Some(Link { p, q, pxq, g, prop, carry, sum, cin, ext_pxq: false, ext_g: false })
}

struct Finder<'t> {
    tape: &'t Compiled,
    /// Xor gate by canonical operand pair (CSE guarantees uniqueness).
    xor_of: HashMap<(u32, u32), u32>,
    /// And gates by operand, for incrementer-link lookups.
    and_with: HashMap<u32, Vec<u32>>,
    /// Full-adder links by carry-in, for chain extension.
    full_by_cin: HashMap<u32, Vec<Link>>,
    /// Slots observed outside the gate graph (declared outputs and live
    /// registers' next-state taps); never interior to a chain.
    io: Vec<bool>,
    /// Gates already claimed by a committed chain.
    used: Vec<bool>,
}

impl<'t> Finder<'t> {
    fn new(tape: &'t Compiled) -> Self {
        let n = tape.slot_count();
        let mut xor_of = HashMap::new();
        let mut and_with: HashMap<u32, Vec<u32>> = HashMap::new();
        for s in tape.gate_start as usize..n {
            let (a, b) = (tape.a[s], tape.b[s]);
            match tape.ops[s] {
                Op::Xor => {
                    xor_of.entry(key(a, b)).or_insert(s as u32);
                }
                Op::And => {
                    and_with.entry(a).or_default().push(s as u32);
                    if b != a {
                        and_with.entry(b).or_default().push(s as u32);
                    }
                }
                _ => {}
            }
        }

        let mut full_by_cin: HashMap<u32, Vec<Link>> = HashMap::new();
        for s in tape.gate_start as usize..n {
            if tape.ops[s] != Op::Or {
                continue;
            }
            let (a, b) = (tape.a[s], tape.b[s]);
            for (g, prop) in [(a, b), (b, a)] {
                if let Some(link) = full_link(tape, &xor_of, g, prop, s as u32) {
                    full_by_cin.entry(link.cin).or_default().push(link);
                }
            }
        }

        let mut io = vec![false; n];
        for &s in &tape.output_slots {
            io[s as usize] = true;
        }
        for (r, &out) in tape.reg_out_slots.iter().enumerate() {
            if out != u32::MAX {
                io[tape.reg_in_slots[r] as usize] = true;
            }
        }

        Finder { tape, xor_of, and_with, full_by_cin, io, used: vec![false; n] }
    }

    fn run(mut self) -> (Vec<Chain>, Vec<bool>) {
        let mut chains = Vec::new();

        // Full-adder heads first (they reclaim five gates per bit), in tape
        // order so a chain is tried from its true head before any mid-link.
        let mut heads: Vec<Link> = self.full_by_cin.values().flatten().copied().collect();
        heads.sort_by_key(|link| link.carry);
        for head in heads {
            self.try_chain(head, &mut chains);
        }

        // Incrementer heads: any and-gate, either operand as the carry-in.
        for s in self.tape.gate_start as usize..self.tape.slot_count() {
            if self.tape.ops[s] != Op::And {
                continue;
            }
            let (a, b) = (self.tape.a[s], self.tape.b[s]);
            for (p, cin) in [(a, b), (b, a)] {
                let head = self.inc_link(p, cin, s as u32);
                self.try_chain(head, &mut chains);
            }
        }

        (chains, self.used)
    }

    /// Build an incrementer link `carry = and(p, cin)`.
    fn inc_link(&self, p: u32, cin: u32, carry: u32) -> Link {
        let sum = self.xor_of.get(&key(p, cin)).copied().unwrap_or(u32::MAX);
        let nil = u32::MAX;
        Link {
            p,
            q: nil,
            pxq: nil,
            g: nil,
            prop: nil,
            carry,
            sum,
            cin,
            ext_pxq: false,
            ext_g: false,
        }
    }

    fn only_feeds(&self, slot: u32, allowed: &[u32]) -> bool {
        self.tape.successors(slot).iter().all(|c| allowed.contains(c))
    }

    /// Per-link admission: gates unclaimed, interior nets interior. Fills
    /// the escape flags for pxq / g.
    fn admit(&self, mut link: Link, local: &[u32]) -> Option<Link> {
        for g in link_gates(&link) {
            if self.used[g as usize] || local.contains(&g) {
                return None;
            }
        }
        if link.prop != u32::MAX {
            // prop has no cheap word form; it must die with the chain.
            if self.io[link.prop as usize] || !self.only_feeds(link.prop, &[link.carry]) {
                return None;
            }
            link.ext_pxq =
                self.io[link.pxq as usize] || !self.only_feeds(link.pxq, &[link.prop, link.sum]);
            link.ext_g = self.io[link.g as usize] || !self.only_feeds(link.g, &[link.carry]);
        }
        Some(link)
    }

    /// Next link after `carry`, if the carry can stay interior: consumed
    /// only by that link's gates and never observed directly.
    fn continuation(&self, carry: u32, local: &[u32]) -> Option<Link> {
        if self.io[carry as usize] {
            return None;
        }
        for cand in self.full_by_cin.get(&carry).map_or(&[][..], |v| v) {
            if self.only_feeds(carry, &[cand.prop, cand.sum])
                && let Some(link) = self.admit(*cand, local)
            {
                return Some(link);
            }
        }
        for &and in self.and_with.get(&carry).map_or(&[][..], |v| v) {
            let (a, b) = (self.tape.a[and as usize], self.tape.b[and as usize]);
            let cand = self.inc_link(if a == carry { b } else { a }, carry, and);
            if self.only_feeds(carry, &[cand.carry, cand.sum])
                && let Some(link) = self.admit(cand, local)
            {
                return Some(link);
            }
        }
        None
    }

    /// Check a chain against the emission rule and return its boundary.
    ///
    /// The fused tasks run right before the windows of level `ready =
    /// 1 + max(operand levels)`. That is sound only if every materialized
    /// output originally settled at `ready` or later — then each external
    /// consumer (a window at a strictly higher level, or another chain
    /// whose own boundary is strictly higher) still runs after the fused
    /// task. Chain inputs must also not be the chain's own gates, or the
    /// gathers would read the destination words being written.
    fn consistent(&self, links: &[Link]) -> Option<u32> {
        let lvl = |s: u32| self.tape.slot_level[s as usize];
        let mut ready = lvl(links[0].cin) + 1;
        for link in links {
            ready = ready.max(lvl(link.p) + 1);
            if link.q != u32::MAX {
                ready = ready.max(lvl(link.q) + 1);
            }
        }

        let gates: Vec<u32> = links.iter().flat_map(link_gates).collect();
        let inputs = links.iter().flat_map(|l| [l.p, l.q]).chain([links[0].cin]);
        for s in inputs {
            if s != u32::MAX && gates.contains(&s) {
                return None;
            }
        }

        let ext_pxq = links.iter().any(|l| l.ext_pxq);
        let ext_g = links.iter().any(|l| l.ext_g);
        let last = links.len() - 1;
        for (i, link) in links.iter().enumerate() {
            let materialized = (link.sum != u32::MAX)
                .then_some(link.sum)
                .into_iter()
                .chain((i == last).then_some(link.carry))
                .chain((ext_pxq && link.ext_pxq).then_some(link.pxq))
                .chain((ext_g && link.ext_g).then_some(link.g));
            for m in materialized {
                if lvl(m) < ready {
                    return None;
                }
            }
        }
        Some(ready)
    }

    /// Grow the longest admissible chain from `head`, trim it until it is
    /// consistent, and commit it if it is still worth fusing.
    fn try_chain(&mut self, head: Link, chains: &mut Vec<Chain>) {
        let Some(head) = self.admit(head, &[]) else { return };
        let mut links = vec![head];
        let mut local: Vec<u32> = link_gates(&head).collect();
        while links.len() < MAX_CHAIN {
            let carry = links.last().expect("chain is never empty").carry;
            let Some(next) = self.continuation(carry, &local) else { break };
            local.extend(link_gates(&next));
            links.push(next);
        }
        loop {
            if links.len() < MIN_CHAIN {
                return;
            }
            if let Some(ready) = self.consistent(&links) {
                for link in &links {
                    for g in link_gates(link) {
                        self.used[g as usize] = true;
                    }
                }
                chains.push(Chain {
                    cin: links[0].cin,
                    ext_pxq: links.iter().any(|l| l.ext_pxq),
                    ext_g: links.iter().any(|l| l.ext_g),
                    ready,
                    links,
                });
                return;
            }
            links.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use ryzr_core::{CircuitBuilder, Signal};

    use super::*;

    /// 32-bit ripple accumulator `acc += in`, the canonical full-adder
    /// chain (bit 0 is folded away by the optimizer, so 31 links remain).
    fn accumulator() -> ryzr_core::Circuit {
        let mut b = CircuitBuilder::new();
        let inputs: Vec<Signal> = (0..32).map(|i| b.input(format!("IN{i}"))).collect();
        let regs: Vec<_> = (0..32).map(|i| b.reg(format!("ACC{i}"), false)).collect();
        let mut carry = b.const_val(false);
        for i in 0..32 {
            let (p, q) = (regs[i].1, inputs[i]);
            let pxq = b.xor(p, q);
            let sum = b.xor(pxq, carry);
            let g = b.and(p, q);
            let prop = b.and(pxq, carry);
            carry = b.or(g, prop);
            b.drive(regs[i].0, sum);
        }
        b.output("COUT", carry);
        b.finish().unwrap()
    }

    #[test]
    fn finds_full_adder_chain() {
        let tape = Compiled::new(&accumulator());
        let (chains, dead) = find_chains(&tape);
        assert_eq!(chains.len(), 1);
        let chain = &chains[0];
        assert_eq!(chain.links.len(), 31);
        assert!(!chain.ext_pxq && !chain.ext_g, "interior nets must not escape");
        // Sums for bits 1..31 plus bit 0's folded share: 30 sums present
        // (bit 31's carry feeds COUT, its sum drives ACC31).
        assert!(chain.links.iter().filter(|l| l.sum != u32::MAX).count() >= 30);
        assert!(dead.iter().filter(|&&d| d).count() >= 31 * 4);
    }

    #[test]
    fn finds_incrementer_chain() {
        // 24-bit counter: `bit' = bit ^ carry`, `carry' = carry & bit`.
        let mut b = CircuitBuilder::new();
        let regs: Vec<_> = (0..24).map(|i| b.reg(format!("BIT{i}"), false)).collect();
        let mut carry = b.const_val(true);
        for &(reg, bit) in &regs {
            let next = b.xor(bit, carry);
            b.drive(reg, next);
            carry = b.and(carry, bit);
        }
        b.output("WRAP", carry);
        let tape = Compiled::new(&b.finish().unwrap());
        let (chains, _) = find_chains(&tape);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].links.len(), 23, "bit 0 folds, bits 1..24 fuse");
    }

    #[test]
    fn escaped_pxq_is_flagged() {
        // Same accumulator, but tap one interior xor as an output — CSE
        // merges it with the adder's pxq, which must then be materialized.
        let mut b = CircuitBuilder::new();
        let inputs: Vec<Signal> = (0..16).map(|i| b.input(format!("IN{i}"))).collect();
        let regs: Vec<_> = (0..16).map(|i| b.reg(format!("ACC{i}"), false)).collect();
        let mut carry = b.const_val(false);
        for i in 0..16 {
            let (p, q) = (regs[i].1, inputs[i]);
            let pxq = b.xor(p, q);
            let sum = b.xor(pxq, carry);
            let g = b.and(p, q);
            let prop = b.and(pxq, carry);
            carry = b.or(g, prop);
            b.drive(regs[i].0, sum);
            if i == 8 {
                b.output("TAP", pxq);
            }
        }
        b.output("COUT", carry);
        let tape = Compiled::new(&b.finish().unwrap());
        let (chains, _) = find_chains(&tape);
        assert_eq!(chains.len(), 1);
        assert!(chains[0].ext_pxq);
        assert!(!chains[0].ext_g);
    }
}
