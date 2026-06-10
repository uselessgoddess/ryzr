# ryzr

Blazingly fast digital logic simulation engines for VCB-like games.

The rule that everything here obeys: **simulation is honest**. If a user
builds an adder out of gates, every bit of that adder is actually computed,
every tick. Engines are free to be clever about *how* values are computed —
never about *whether* they are. All engines produce bit-for-bit identical
results to a naive reference interpreter, and the test suite enforces it.

## Workspace

| crate | purpose |
|---|---|
| `ryzr-core` | circuit IR, builder, topological sort, reference interpreter (the oracle) |
| `ryzr-backend` | the engines — eight of them, one compiled tape |
| `ryzr-riscv` | gate-level RV32I core: the honesty benchmark |

## Engines

Every engine consumes the same compiled tape: the circuit levelized,
sorted by `(level, op)` into homogeneous runs, and laid out as
struct-of-arrays with pre-validated operand indices (no bounds checks in
the hot loop, no `unsafe` without a compile-time-established contract).

| engine | strategy |
|---|---|
| `ScalarEngine` | dense forward pass, one dispatch per run instead of per gate |
| `EventEngine` | recomputes only the cone affected by actual changes |
| `BatchEngine` | 64 independent instances bit-packed per `u64` (SWAR across instances) |
| `PackedEngine` | one instance bit-packed: up to 64 same-op gates per word op (SWAR within the circuit), ripple-carry chains fused into native adds |
| `PackedJitEngine` | the packed plan compiled to native code via Cranelift — the fastest single-instance engine |
| `ThreadedEngine` | wide levels fanned out across cores via rayon |
| `JitEngine` | per-gate settle pass compiled to native code via Cranelift |
| `HybridEngine` | the winning set behind one type — the one that rules them all |

### Packed: SWAR for a single circuit

`BatchEngine` gets its 64× from running 64 *copies* — useless when you
care about one machine. `PackedEngine` turns the same trick inward: every
signal of one instance occupies one bit of a `u64` arena, and because the
tape sorts each level into homogeneous op runs, one word op evaluates up
to 64 *different gates of the same kind* at once.

The catch is that those 64 gates read from 64 scattered bit positions, and
a gather per operand would eat the win. So gathers are compiled, not
interpreted: at construction, the execution graph is analyzed per output
word and each one gets the cheapest program that assembles its operands —
constants fold to an immediate (free), operands that sit contiguously in
source order stream through a funnel shift (~6 ops for up to 64 bits), and
the scattered remainder is filled by masked splats. A tick then replays
straight-line word ops with no per-gate branching at all.

The plan also recognizes structure the user built out of gates. Ripple
adder and incrementer chains — the textbook xor/and/or full-adder lattice —
are detected in the gate graph and fused: an entire carry chain of up to
63 bits becomes one native 64-bit add whose sum bits *are* the chain's sum
gates and whose top bit is the carry-out. This is not an abstraction
shortcut — the add computes the exact same boolean functions the gates
declare, bit for bit, and the differential suite proves it. On the RV32I
core, fusion absorbs ~480 gates into 4 adds.

`PackedJitEngine` then takes the same plan and compiles it to native code
with Cranelift: every gather offset and mask baked in as an immediate,
word values flowing through registers instead of the arena. It is the
fastest way here to simulate one machine.

### The hybrid engine

`HybridEngine` is the answer to "which engine should I use?" — it doesn't
guess, it measures. At construction it builds every plan that can serve
the request, times each for a fraction of a millisecond *on the live
circuit*, keeps the winner, and resets it to power-on state. Either way
the results are bit-for-bit identical; only the speed differs.

- **`HybridEngine::new`** accelerates a single instance: it races
  `PackedJitEngine`, `PackedEngine`, `EventEngine`, and `ThreadedEngine`.
  The winner depends on real circuit properties — the packed JIT wins on
  dense always-active logic, event on mostly-idle circuits, threaded on
  very wide levels. (The per-gate `JitEngine` is not raced: the packed
  JIT executes the same circuit in strictly fewer instructions.)
- **`HybridEngine::wide`** runs 64 independent instances and multiplies
  SWAR × rayon × JIT, racing the jitted settle against the SWAR
  interpreter for the same reason: past a few thousand gates straight-line
  native code stops fitting in icache, and the interpreter's tiny resident
  loop — whose "program" is index arrays flowing through the data
  prefetcher — takes over.

## The honesty benchmark: a RISC-V processor made of gates

`ryzr-riscv` builds a single-cycle RV32I core from nothing but `ryzr`
gate primitives — ripple-carry ALU, barrel shifters, register file and
RAM as D flip-flops behind mux trees, ROM as combinational mux trees.
With 256 words of RAM that is **22,679 gates and 9,216 flip-flops across
88 logic levels**, and one engine tick retires exactly one instruction.

Correctness is not asserted, it is *proven in lockstep*: tests run the
gate-level circuit against an instruction-level emulator and compare the
full architectural state — pc and all 32 registers — after every retired
instruction, across arithmetic, branch, and memory test programs, plus
end-to-end results (`fib(20) = 6765` computed by actual gates). CI runs
the same lockstep suite in release mode on the exact binaries it then
benchmarks.

Representative numbers from a 6-core desktop (`fib` loop;
1 tick = 1 retired instruction):

| engine | throughput | what it simulates |
|---|---|---|
| event | ~14 K instr/s | one CPU |
| scalar | ~45 K instr/s | one CPU |
| jit | ~54 K instr/s | one CPU |
| packed | ~183 K instr/s | one CPU |
| packed-jit | ~614 K instr/s | one CPU |
| **hybrid** | **~609 K instr/s** | **one CPU** |
| hybrid64 | ~3.3 M instr/s | 64 independent CPUs |

The single-CPU number is the honest headline: the hybrid engine retires
~610 K instructions/s on one simulated machine — 13× the scalar pass —
with the packed JIT winning the race on this circuit (carry-chain fusion
plus SWAR packing plus native code, compounding). The wide row is real
throughput too, but it is *aggregate* throughput over 64 independent
processors, and the table says so.

A note on comparing with Virtual Circuit Board numbers:
[vcb-riscv](https://github.com/WildDude7/VCB-RISCV) reaches ~1.1 M
*ticks*/s in VCB, but a VCB tick is a single signal-propagation step, not
a clock cycle — signals cross roughly one gate per tick, so one
instruction takes many ticks (an ALU adder alone costs about 7 ticks per
stage of carry). In `ryzr`, one tick settles the entire 88-level
combinational cone and latches every flip-flop: one tick = one full clock
cycle = one retired instruction. The two rates measure different things
and dividing VCB's tick rate by its ticks-per-instruction is the only
fair conversion. What `ryzr` keeps from VCB is the honesty: every gate is
computed every tick, nothing is abstracted away.

## Running it

```sh
cargo test --workspace            # oracle + differential + RISC-V lockstep
cargo bench -p ryzr-riscv         # instructions/sec on the gate-level core
cargo bench -p ryzr-backend       # synthetic microbenchmarks
cargo run -p ryzr-riscv --release --example stats   # circuit statistics
```

`ryzr-backend` features: `jit` and `rayon` are on by default; the crate
builds and passes its tests with `--no-default-features` (scalar, event,
and SWAR engines only).
