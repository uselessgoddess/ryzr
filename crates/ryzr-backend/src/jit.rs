//! Cranelift JIT engine: the settle pass compiled to native machine code.
//!
//! Interpreters pay a tax on every gate — operand fetch through the tape's
//! index arrays, dispatch, bounds bookkeeping. Here the tape itself becomes
//! the program: every gate is emitted as a couple of native instructions
//! with its operand offsets baked in as immediates, and Cranelift's
//! register allocator keeps hot intermediate values out of memory entirely
//! (within a chunk, a gate consumed by a later gate flows through a
//! register, not the value buffer).
//!
//! The gate region is split into fixed-size chunks, one function per chunk:
//! register allocation is superlinear in function size, and chunking keeps
//! compile times flat for million-gate circuits. Values cross chunk
//! boundaries through the value buffer, which every gate stores to anyway —
//! the buffer must stay observable for `output()` and the register capture.
//!
//! Sequential semantics (clock-edge scatter, next-state gather) stay in
//! Rust; only the combinational settle is jitted.

use std::sync::Arc;

use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::Engine;
use crate::compile::{Compiled, Op};
use crate::scalar::{apply_edge, capture_next};

/// Gates per jitted function. Large enough to amortize the call, small
/// enough to keep register allocation linear in practice.
pub(crate) const CHUNK: usize = 8192;

pub(crate) type TickFn = unsafe extern "C" fn(*mut u8);

/// Compile one straight-line settle function per slot range.
///
/// Shared by [`JitEngine`] (`swar = false`, one byte per slot) and the
/// hybrid engine (`swar = true`, one `u64` of 64 packed instances per
/// slot). The returned module owns the executable memory behind the
/// function pointers; it must outlive every call and be freed on drop.
pub(crate) fn compile_ranges(
    tape: &Compiled,
    ranges: &[core::ops::Range<usize>],
    swar: bool,
) -> (JITModule, Vec<TickFn>) {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .expect("host architecture unsupported by cranelift")
        .finish(settings::Flags::new(flags))
        .expect("failed to construct native isa");
    let mut module =
        JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));

    let pointer = module.target_config().pointer_type();
    let mut signature = module.make_signature();
    signature.params.push(AbiParam::new(pointer));

    let mut ctx = module.make_context();
    let mut fb_ctx = FunctionBuilderContext::new();

    // Slots that must stay observable in the value buffer no matter what:
    // declared outputs and register next-state taps. Everything else only
    // needs a store if some gate *outside* the chunk reads it.
    let mut pinned = vec![false; tape.slot_count()];
    for &slot in &tape.output_slots {
        pinned[slot as usize] = true;
    }
    for &slot in &tape.reg_in_slots {
        pinned[slot as usize] = true;
    }

    // SSA cache: slot -> value already materialized in this chunk.
    let mut cache: Vec<Option<Value>> = vec![None; tape.slot_count()];
    let mut ids = Vec::new();

    for range in ranges {
        cache.fill(None);
        build_chunk(
            &mut ctx.func,
            &mut fb_ctx,
            &signature,
            tape,
            &pinned,
            &mut cache,
            range.clone(),
            swar,
        );

        let name = format!("settle{}", ids.len());
        let id = module
            .declare_function(&name, Linkage::Export, &ctx.func.signature)
            .expect("declare jit chunk");
        module.define_function(id, &mut ctx).expect("compile jit chunk");
        module.clear_context(&mut ctx);
        ids.push(id);
    }

    module.finalize_definitions().expect("finalize jit module");
    let fns = ids
        .into_iter()
        .map(|id| {
            let ptr = module.get_finalized_function(id);
            // SAFETY: the function was compiled with exactly the
            // `fn(*mut u8)` signature built above.
            unsafe { core::mem::transmute::<*const u8, TickFn>(ptr) }
        })
        .collect();
    (module, fns)
}

pub struct JitEngine {
    tape: Arc<Compiled>,
    values: Vec<u8>,
    reg_scratch: Vec<u8>,
    /// Chunk functions in schedule order; together they perform one settle.
    fns: Vec<TickFn>,
    /// Owns the executable memory behind `fns`. `Some` until drop.
    module: Option<JITModule>,
}

impl JitEngine {
    pub fn new(circuit: &ryzr_core::Circuit) -> Self {
        Self::with_tape(Arc::new(Compiled::new(circuit)))
    }

    pub fn with_tape(tape: Arc<Compiled>) -> Self {
        let n = tape.slot_count();
        assert!(
            i32::try_from(n).is_ok(),
            "circuit too large for jit engine (slot offsets exceed i32)"
        );

        let mut ranges = Vec::new();
        let mut start = tape.gate_start as usize;
        while start < n {
            let end = usize::min(start + CHUNK, n);
            ranges.push(start..end);
            start = end;
        }
        let (module, fns) = compile_ranges(&tape, &ranges, false);

        let values = tape.initial_values();
        let reg_scratch = tape.reg_initial.clone();
        Self { tape, values, reg_scratch, fns, module: Some(module) }
    }

    /// Restore power-on state: constants, register initials, inputs low.
    pub(crate) fn reset(&mut self) {
        self.values = self.tape.initial_values();
        self.reg_scratch.copy_from_slice(&self.tape.reg_initial);
    }
}

/// Emit one chunk: a straight-line function evaluating gate slots
/// `range` against the value buffer passed as its only argument.
///
/// `swar = false`: one `I8` per slot holding a canonical 0/1, inversion is
/// `xor 1`, mux is a native `select`. `swar = true`: one `I64` of 64 packed
/// instances per slot, every bit significant, so inversion is a full `bnot`
/// and mux must be the bitwise `bitselect` — `select` would collapse all
/// lanes to one condition.
///
/// A gate's value is stored back to the buffer only if something outside
/// the chunk can read it: a `pinned` slot (output / register tap) or a
/// successor gate past `range.end` (successors are always later in topo
/// order, so in-chunk consumers hit the SSA cache instead). Skipping the
/// rest removes most of the store traffic and shrinks the emitted code.
#[expect(clippy::too_many_arguments, reason = "internal helper with one call site per engine")]
fn build_chunk(
    func: &mut cranelift_codegen::ir::Function,
    fb_ctx: &mut FunctionBuilderContext,
    signature: &Signature,
    tape: &Compiled,
    pinned: &[bool],
    cache: &mut [Option<Value>],
    range: core::ops::Range<usize>,
    swar: bool,
) {
    func.signature = signature.clone();
    let mut fb = FunctionBuilder::new(func, fb_ctx);
    let block = fb.create_block();
    fb.append_block_params_for_function_params(block);
    fb.switch_to_block(block);
    fb.seal_block(block);
    let base = fb.block_params(block)[0];

    let (ty, scale) = if swar { (types::I64, 8) } else { (types::I8, 1) };
    let offset = |slot: usize| (slot * scale) as i32;

    let operand = |fb: &mut FunctionBuilder, cache: &mut [Option<Value>], slot: u32| {
        cache[slot as usize].unwrap_or_else(|| {
            let v = fb.ins().load(ty, MemFlags::trusted(), base, offset(slot as usize));
            cache[slot as usize] = Some(v);
            v
        })
    };
    let invert = |fb: &mut FunctionBuilder, v: Value| {
        if swar { fb.ins().bnot(v) } else { fb.ins().bxor_imm(v, 1) }
    };

    for slot in range.clone() {
        let op = tape.ops[slot];
        let a = operand(&mut fb, cache, tape.a[slot]);
        let value = match op {
            Op::And | Op::Or | Op::Xor | Op::Nand | Op::Nor | Op::Xnor => {
                let b = operand(&mut fb, cache, tape.b[slot]);
                let raw = match op {
                    Op::And | Op::Nand => fb.ins().band(a, b),
                    Op::Or | Op::Nor => fb.ins().bor(a, b),
                    _ => fb.ins().bxor(a, b),
                };
                match op {
                    Op::Nand | Op::Nor | Op::Xnor => invert(&mut fb, raw),
                    _ => raw,
                }
            }
            Op::Not => invert(&mut fb, a),
            Op::Buf => a,
            Op::Mux => {
                let t = operand(&mut fb, cache, tape.b[slot]);
                let e = operand(&mut fb, cache, tape.c[slot]);
                if swar { fb.ins().bitselect(a, t, e) } else { fb.ins().select(a, t, e) }
            }
        };
        let escapes =
            pinned[slot] || tape.successors(slot as u32).iter().any(|&s| s as usize >= range.end);
        if escapes {
            fb.ins().store(MemFlags::trusted(), value, base, offset(slot));
        }
        cache[slot] = Some(value);
    }

    fb.ins().return_(&[]);
    fb.finalize();
}

impl Engine for JitEngine {
    fn name(&self) -> &'static str {
        "jit"
    }

    fn input_count(&self) -> usize {
        self.tape.input_count()
    }

    fn output_count(&self) -> usize {
        self.tape.output_count()
    }

    fn set_input(&mut self, index: usize, value: bool) {
        self.values[self.tape.input_slots[index] as usize] = u8::from(value);
    }

    fn output(&self, index: usize) -> bool {
        self.values[self.tape.output_slots[index] as usize] != 0
    }

    fn tick(&mut self) {
        apply_edge(&self.tape, &mut self.values, &self.reg_scratch);
        let base = self.values.as_mut_ptr();
        for f in &self.fns {
            // SAFETY: `base` points at a buffer of `slot_count()` bytes and
            // every offset the jitted code touches was validated against it
            // in `Compiled::new`.
            unsafe { f(base) };
        }
        capture_next(&self.tape, &self.values, &mut self.reg_scratch);
    }
}

impl Drop for JitEngine {
    fn drop(&mut self) {
        self.fns.clear();
        if let Some(module) = self.module.take() {
            // SAFETY: all pointers into the module's executable memory were
            // cleared above; nothing can call into it after this point.
            unsafe { module.free_memory() };
        }
    }
}
