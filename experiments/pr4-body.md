Solves #3.

## What this does

The issue asks for a deep runtime analysis and maximal acceleration of single-instance simulation — "Verilator does 1–5 MIPS, I have barely 100 K" — with explicit permission to drop dominated engines and keep only the set that wins.

Result on the gate-level RV32I core (22.7 K gates, 88 levels, one tick = one retired instruction), same 6-core dev box, before → after:

| engine | baseline | this PR |
|---|---|---|
| scalar | ~44 K instr/s | ~45 K instr/s |
| packed | ~95 K instr/s | ~183 K instr/s |
| packed-jit | — | **~614 K instr/s** |
| **hybrid (the product)** | **~92 K instr/s** | **~609 K instr/s** |

**6.6× single-CPU**, from ~0.1 to ~0.61 MIPS — inside striking distance of Verilator's quoted 1–5 MIPS range, on a machine where the naive pass does 45 K. Every step preserves the honesty contract: all engines stay bit-for-bit identical to the reference interpreter on declared outputs, enforced by the differential and RISC-V lockstep suites.

## How the 6.6× compounds

Each step is a separate commit, measured on the live core before landing.

### 1. Observable-equivalence netlist optimizer (`perf(core)`)

Before tape compile: constant propagation, dead-cone elimination, and common-subexpression merging — *only* on nodes that cannot be observed (not an output, not a register tap). A user's adder is still computed bit for bit; gates whose values nothing can ever read are as-if optimized, which the honesty contract explicitly permits. Plus creation-order signal scheduling instead of Kahn BFS, which keeps related signals adjacent for the packed engine's gathers.

### 2. Packed-plan gather compilation (`perf(backend)`, prior PR groundwork extended)

The packed engine's per-word gather programs (immediates, funnel shifts, masked splats) got placement-aware layout; on the core the plan compresses 22.7 K gates into 474 word tasks.

### 3. Carry-chain fusion (`fuse.rs` — the big structural win)

The plan now recognizes the textbook full-adder lattice (`pxq = p⊕q`, `g = p∧q`, `prop = pxq∧c`, `carry = g∨prop`, `sum = pxq⊕c`) and incrementer chains (`carry = p∧c`, `sum = p⊕c`) in the gate graph. A ripple chain of up to 63 bits becomes **one native 64-bit add**: `S = P + Q + cin`, sum bits landing exactly where the chain's sum gates live, carry-out free in the top bit. This is exact — the add computes the same boolean functions the gates declare, even under operand aliasing — and ordering safety is enforced per chain (every materialized output must sit at or above the chain's ready level; the chain trims itself otherwise). CSE-escaped intermediates (`pxq`/`g` with external consumers) are materialized as whole `P⊕Q` / `P∧Q` words before the add.

On the RV32I core: **4 chains, 118 links, 479 gates absorbed into 4 adds** (the ALU adder, the PC incrementers). New differential tests cover a 32-bit accumulator, a subtractor (inverted operand + carry-in), and a wide counter with an interior sum tapped as an observed output.

### 4. Packed JIT (`pack_jit.rs`)

The packed plan — gathers, word ops, fused adds — compiled to straight-line native code via Cranelift, 2048 tasks per function. Every offset and mask is an immediate; each arena word is written by exactly one task in topo order, so words flow through a register-level SSA cache instead of memory. This is the ~614 K engine.

### 5. Engine consolidation (`perf(hybrid)`)

Per the issue's directive to drop unneeded engines: the hybrid race now runs **packed-jit, packed, event, threaded** — the per-gate `JitEngine` is out of the race because the packed JIT executes the same circuit in strictly fewer instructions in every measurement (~600 K vs ~50 K). `JitEngine` stays a public type (its compiler powers the hybrid wide mode, and the test suite uses it as an independent oracle), it just never wins and is no longer constructed by `HybridEngine::new`.

## Bonus: cargo profiles from the issue

```toml
[profile.micro]   # inherits release: codegen-units = 1, opt-level = 3, lto = "thin", strip = true
[profile.nano]    # inherits micro:  opt-level = "z", lto = "fat", panic = "abort"
```

Both verified to build the workspace.

## Testing

- `fuse.rs` unit tests: chain discovery on accumulator/counter netlists, escape flagging.
- New differential tests: fused accumulator, subtractor, and tapped wide counter vs the oracle over thousands of ticks.
- Existing suites all green: oracle, differential (7 tests), RISC-V lockstep (pc + all 32 registers after every retired instruction), `--no-default-features` build.
- `clippy -D warnings` clean, nightly rustfmt applied.
- `RYZR_PACK_STATS=1` (default off) prints plan diagnostics for future tuning.

## Reproduce

```sh
cargo test --workspace --all-features
cargo run --release -p ryzr-riscv --example perf --all-features
RYZR_PACK_STATS=1 cargo run --release -p ryzr-riscv --example perf --all-features
```
