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
| `ryzr-backend` | the engines — six of them, one compiled tape |
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
| `BatchEngine` | 64 independent instances bit-packed per `u64` (SWAR) |
| `ThreadedEngine` | wide levels fanned out across cores via rayon |
| `JitEngine` | settle pass compiled to native code via Cranelift |
| `HybridEngine` | SWAR × rayon × JIT — the one that rules them all |

### The hybrid engine

`HybridEngine` composes all three multipliers: 64 SWAR lanes per word,
rayon fan-out for wide levels, and a Cranelift-jitted settle pass. But JIT
is a trade, not a free win: straight-line code spends instruction bytes on
every gate, so past a few thousand gates the settle stops fitting in
instruction cache and every tick streams the whole program from memory —
at which point the SWAR interpreter's tiny resident loop wins, because its
"program" (the tape's index arrays) flows through the data prefetcher
instead of the CPU front end.

Where the crossover sits depends on the circuit and the host, so the
hybrid engine doesn't guess: at construction it builds both plans, times
each on the live circuit for a fraction of a millisecond, and keeps the
winner. Either plan produces identical results; only the speed differs.

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

Representative numbers from a 6-core desktop (criterion, `fib` loop;
1 element = 1 retired instruction):

| engine | throughput |
|---|---|
| scalar | ~44 K instr/s |
| event | ~12 K instr/s |
| jit | ~41 K instr/s |
| batch64 | ~2.2 M instr/s (64 CPUs in parallel) |
| **hybrid** | **~2.4 M instr/s** (64 CPUs in parallel) |

For scale: [vcb-riscv](https://github.com/WildDude7/VCB-RISCV) reaches
~1.1 M ticks/s inside Virtual Circuit Board on comparable hardware.
Different machines and different circuits, so treat it as an
order-of-magnitude comparison — but `ryzr` crosses that scale while
honestly computing every gate, and a tick here is a *full clock cycle*
(one retired instruction), not a single simulation step.

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
