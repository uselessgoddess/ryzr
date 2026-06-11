# Performance analysis: how close can an honest gate simulator get to Verilog?

This document is the step-by-step profiling and feasibility study behind the
single-instance engines. The question it answers: the gate-level RV32I core
runs at **~1.34 M instructions/s** on the packed JIT (one simulated CPU on a
6-core desktop); Verilator-class RTL simulators reach **~5 M instructions/s**
on a comparable single-cycle core. Is that gap closable while keeping the
honesty contract — *every gate computed every tick* — and if so, how?

All numbers below come from the `plan_report` and `where_splats` examples in
`ryzr-riscv` (run them yourself: `cargo run -p ryzr-riscv --release --example
plan_report`).

## 1. Where the time goes

The packed JIT lowers the circuit to a straight-line *word program*: each task
evaluates up to 64 same-op gates as one 64-bit word op, after gathering its
operands from scattered arena bits. On the RV32I core (256-word RAM), one tick
is:

| metric | count |
|---|---|
| word tasks | 220 |
| — of which muxes | 110 (50%) |
| settle gather segments | 328 funnels + **1219 splats** |
| capture gather segments | 167 funnels + 330 splats |
| fused carry chains | 4 |
| fused RAM banks | 1 (256×32) |

A **funnel** is a contiguous run of source bits moved with one shift (~6 ops
for up to 64 bits). A **splat** broadcasts one scattered source bit to a
destination mask (~5 ops). Splats are the expensive primitive, and they are
dominated by muxes:

| op | tasks | splats |
|---|---|---|
| **mux** | **110** | **874** |
| and | 66 | 213 |
| or | 13 | 82 |
| xor | 9 | 21 |
| not | 17 | 20 |
| add (fused) | 4 | 9 |
| memread (fused) | 1 | 0 |

**Muxes alone produce 874 of the 1219 settle splats — 72% of the gather cost.**
So the headline is: *the per-tick cost is dominated by mux-tree gathers.*

## 2. What the muxes are

Carry-chain fusion already turns the ALU adders into 4 native adds, and RAM
fusion already replaces the 256-word RAM read mux-tree (~16K gate-muxes) with a
single dynamic-index gather. What is left? Counting the unfused muxes by
schedule level:

```
unfused muxes (gate level): 3176
  level   4: 960   level   5: 512   level   6: 256   level   7: 128   level 8: 76
  level  10..17: ~460   (ALU/branch/immediate selection)
  level  33..37: ~250   (writeback / next-state selection)
  level  41..81: 16 each (barrel-shifter stages, ~320 total)
```

The single biggest structure is **levels 4–8: a 32→16→8→4→2 reduction**
(960+512+256+128+76 ≈ 1932 muxes). That is the **register file's two read ports**
(`rs1`, `rs2`): each is a 32-way mux tree over 32-bit words selecting
`regs[rs]`. The register file is the RAM's twin — yet it is *not* fused, for
three structural reasons:

1. **31 stored words, not 32.** `x0` is hardwired to zero (no storage), so the
   write-cell array is X1–X31 — not a power of two, which the bank detector
   requires.
2. **A constant leaf.** The read tree's leaf 0 is a constant-zero word, not a
   register output, so the bottom-up tree reconstruction in `find_banks` does
   not match.
3. **Two read ports share the leaves.** The bottom-level muxes of `rs1` and
   `rs2` have identical `(then, else)` operands (`(X1, zero)`), so they collide
   in the reverse-mux map and only one port is ever reconstructed.

Fusing the register file read ports the same way RAM is fused would remove
~1932 muxes and the bulk of the 874 mux splats — the clearest available win,
and the one this branch implements (see §4).

## 3. Is 5 MIPS reachable, and what does it take?

**The gap is structural, not constant-factor.** Verilator does not simulate
gates: it lowers the *RTL* — `regs[rs1]` is an array index, `a + b` is a machine
add — and lets a C++ compiler optimise the datapath. `ryzr` is contractually
honest: it computes all 22,679 gates every tick. The only way an honest
simulator approaches RTL speed is to *recognise the structures the gates spell
out and execute them as their RTL equivalent*, bit-for-bit. That is exactly
what carry-chain fusion (ripple adder → native add) and RAM fusion (mux-tree →
indexed gather) already do. The path to 5 MIPS is **more of the same fusion,
until the residue of genuinely irregular gates is small enough that raw SWAR
covers it cheaply.**

Concretely, in descending order of expected payoff:

1. **Register-file read & write fusion** (this branch). Removes the ~1932 read
   muxes and most mux splats. Expected: the largest single step.
2. **ROM / instruction-fetch fusion.** The fetch is `rom[pc]` — a mux-tree over
   constants. Fusing it to an indexed load over a constant table removes the
   fetch tree entirely.
3. **Barrel-shifter recognition.** Levels 41–81 (~320 muxes) are the SLL/SRL/SRA
   stages: `mux(sh[k], x << 2^k, x)`. A recognised barrel shifter becomes a
   handful of native shifts + selects instead of 5 levels of per-bit muxes.
4. **Operand placement to convert splats → funnels.** Even unfused logic gets
   cheaper if a gate's operands are laid out contiguously. A placement pass that
   numbers slots to maximise contiguity directly trades the 5-op splat for the
   amortised funnel.
5. **Word-level SIMD (AVX-512).** The arena is bit-packed `u64`s; the gather and
   boolean ops are embarrassingly vectorisable across the ~365-word arena.

**Verdict.** 5 MIPS on this core *while staying honest* is plausible but not a
single change — it is the sum of fusing every remaining regular structure
(register file, ROM, shifters) plus better placement, each verified against the
oracle. Steps 1–3 alone should roughly double-to-triple throughput; closing the
final gap to 5 MIPS likely needs step 5 as well. None of it abandons the
contract: every fused structure computes the exact boolean function its gates
declare, and the differential + lockstep suites prove it on every tick.

## 4. What this branch changes

See the commit history: profiling tooling (`plan_report`), then register-file
read-port fusion. Each fusion step is gated behind the same exhaustive
structural verification as RAM fusion and checked by the differential and
RISC-V lockstep suites (full architectural state compared against an
instruction-level emulator after every retired instruction).
