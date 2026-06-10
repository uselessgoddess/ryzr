Solves #1.

## What this is

A workspace of honest, blazingly fast digital-logic simulation engines for VCB-like games, benchmarked the only way that counts: by running a real gate-level RISC-V processor — and reporting **single-CPU** acceleration as the headline, not aggregate lane throughput.

The non-negotiable rule everywhere: **simulation is honest**. If a user builds an adder, every bit of it is actually computed every tick. Engines may be clever about *how* values are computed, never about *whether*. Every engine is differential-tested bit-for-bit against a naive reference interpreter.

## Workspace

| crate | purpose |
|---|---|
| `ryzr-core` | circuit IR, builder, topological sort, reference interpreter (the oracle) |
| `ryzr-backend` | seven engines over one compiled tape |
| `ryzr-riscv` | gate-level RV32I core — the honesty benchmark |

## One compiled tape, seven engines

All engines consume the same `Compiled` tape: the circuit levelized, sorted by `(level, op)` into homogeneous runs, laid out as struct-of-arrays with operand indices pre-validated at compile time (`operand < slot`, strictly lower level) — so hot loops index unchecked and parallel writes are provably disjoint, with the safety contract established once, at tape construction.

| engine | strategy |
|---|---|
| `ScalarEngine` | dense forward pass, one dispatch per homogeneous run |
| `EventEngine` | recomputes only the cone affected by actual changes |
| `BatchEngine` | SWAR across instances — 64 independent copies bit-packed per `u64` |
| `PackedEngine` | **SWAR within one instance** — up to 64 same-op gates of one circuit per word op |
| `ThreadedEngine` | wide levels fanned out across cores via rayon |
| `JitEngine` | settle pass compiled to native code via Cranelift |
| **`HybridEngine`** | **every technique above behind one type — the one that rules them all** |

### Packed: instruction batching for a single CPU

Per review feedback: SWAR here means accelerating **one** simulated machine by batching gate operations, not running 64 copies. `PackedEngine` packs every signal of one instance into one bit of a `u64` arena; because the tape sorts each level into homogeneous op runs, a single word op evaluates up to 64 *different gates of the same kind* at once.

The hard part is operand gathering — 64 gates read 64 scattered bit positions, and a naive per-bit gather would eat the win. So gathers are **compiled at construction via execution-graph analysis**: for each output word, the engine picks the cheapest program that assembles its operands —
- constants fold into an immediate (free at runtime),
- operands contiguous in source order stream through a funnel shift (~6 ops per up-to-64 bits),
- the scattered remainder is filled with masked splats (`0u64.wrapping_sub(bit) & mask`).

A tick then replays straight-line word ops with zero per-gate branching. On the RV32I core this packs 22.7 K gates into ~2.3 K word tasks.

### The hybrid engine (the "one engine to rule them all")

`HybridEngine` doesn't guess which plan wins — it **measures on the live circuit** at construction, keeps the winner, and resets it to power-on state. Both modes are differential- and lockstep-tested; only speed differs.

- **`HybridEngine::new`** (default) accelerates a single CPU: it races `PackedEngine`, `EventEngine`, `ThreadedEngine`, and `JitEngine` for a fraction of a millisecond each and keeps the fastest. Packed wins on wide homogeneous levels, event wins on mostly-idle circuits, JIT wins on small hot circuits whose native settle fits in instruction cache.
- **`HybridEngine::wide`** keeps the 64-instance throughput mode: SWAR × rayon × JIT with bitwise-select mux lowering, SSA value reuse, store elision driven by successor analysis, and the same measured JIT-vs-interpreter calibration (past a few thousand gates the jitted settle stops fitting in icache and the SWAR interpreter's tiny resident loop wins).

## The honesty benchmark: RISC-V made of gates

`ryzr-riscv` builds a single-cycle RV32I core from nothing but gate primitives — ripple-carry ALU, barrel shifters, register file and RAM as D flip-flops behind mux trees. With 256 words of RAM: **22,679 gates, 9,216 flip-flops, 88 logic levels**, and one engine tick retires exactly one instruction.

Execution results are verified, not assumed: lockstep tests run the gate-level circuit against an instruction-level emulator and compare **pc plus all 32 registers after every retired instruction**, across arithmetic, branch, and memory programs, plus end-to-end results (`fib(20) = 6765` computed by actual gates). The bench workflow runs this lockstep suite in release mode on the exact binaries it then times.

### Numbers

6-core desktop, criterion, `fib` loop, 1 element = 1 retired instruction:

| engine | throughput | simulates |
|---|---|---|
| scalar | ~51 K instr/s | one CPU |
| event | ~16 K instr/s | one CPU |
| threaded | ~45 K instr/s | one CPU |
| jit | ~45 K instr/s | one CPU |
| packed | ~89 K instr/s | one CPU |
| **hybrid** | **~95 K instr/s** | **one CPU** |
| batch64 | ~2.2 M instr/s | 64 independent CPUs |
| hybrid64 | ~2.9 M instr/s | 64 independent CPUs |

The honest headline is the single-CPU row: **~95 K retired instructions/s on one simulated machine** — 1.8× the scalar pass and 2× the JIT — with the wide rows clearly labeled as aggregate throughput over 64 independent processors.

### On comparing with VCB ticks/s

Per review feedback, the earlier comparison against vcb-riscv's ~1.1 M ticks/s was misleading: a VCB tick is one signal-propagation step, not a clock cycle — one instruction takes many ticks (an ALU adder alone costs ~7 ticks per carry stage). In `ryzr`, one tick settles the entire 88-level combinational cone and latches every flip-flop, so one tick = one full clock cycle = one retired instruction. The README and the bench summary now state these units explicitly instead of implying ticks/s ≈ instructions/s.

## CI

- **`ci`** — fmt, clippy (`-D warnings`), full test suite (oracle + differential + RISC-V lockstep), plus a `--no-default-features` build/test (no jit, no rayon).
- **`bench`** — first re-verifies execution results with `cargo test --workspace --release` (same binaries), then runs the RISC-V benchmark and engine microbenchmarks, publishing throughput to the job summary with the single-CPU/64-CPU distinction spelled out. Shared runners are noisy, so results are order-of-magnitude tracking, not a regression gate.

## Reproduce

```sh
cargo test --workspace            # oracle + differential + RISC-V lockstep
cargo bench -p ryzr-riscv         # instructions/sec on the gate-level core
cargo bench -p ryzr-backend       # synthetic microbenchmarks
cargo run -p ryzr-riscv --release --example stats   # circuit statistics
```

Fixes uselessgoddess/ryzr#1
