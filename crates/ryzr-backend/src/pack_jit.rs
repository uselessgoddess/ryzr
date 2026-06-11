//! Packed JIT engine: the bit-packed word program compiled to native code.
//!
//! [`PackedEngine`](crate::PackedEngine) already does the hard analytical
//! work — it lowers the circuit to a short word-level program of gather
//! segments (immediates, funnel shifts, splats) over a dense bit arena.
//! But it then *interprets* that program: every tick streams tens of
//! kilobytes of segment metadata through the cache and pays a dispatch
//! branch per task. This engine takes the exact same plan and emits it as
//! straight-line Cranelift code instead:
//!
//! - segment metadata (source positions, shifts, masks) become instruction
//!   immediates — zero memory traffic at runtime;
//! - constant shift amounts let funnels drop the branchless `o == 0`
//!   contortion and splats use a two-shift sign-broadcast;
//! - a word-level SSA cache forwards each task's result directly to its
//!   consumers within a chunk, so most arena words are loaded at most once
//!   and hot intermediates never round-trip through memory.
//!
//! The task list is split into chunks, one function per chunk, to keep
//! Cranelift's register allocation time linear in circuit size. Semantics
//! are identical to the packed interpreter (same plan, same task order);
//! the differential suite checks both against the naive interpreter.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::Engine;
use crate::compile::{Compiled, Op};
use crate::pack::{OutSrc, Plan, Seg, TaskOp};

/// Tasks per jitted function; bounds compile time on huge circuits.
const CHUNK: usize = 2048;

/// `fn(bits, staging)`: settle a chunk of tasks; the last chunk also
/// captures register next-state into `staging`.
type ChunkFn = unsafe extern "C" fn(*mut u64, *mut u64);

pub struct PackedJitEngine {
    bits: Vec<u64>,
    staging: Vec<u64>,
    reg_word: usize,
    reg_init: Vec<u64>,
    outputs: Vec<OutSrc>,
    input_count: usize,
    fns: Vec<ChunkFn>,
    /// Owns the executable memory behind `fns`. `Some` until drop.
    module: Option<JITModule>,
}

/// Per-chunk emission state: one SSA value per arena word.
///
/// Soundness: every arena word is either a source region word (never
/// written during settle) or the destination of exactly one task, and task
/// order is topological — no gather reads a task's word before that task
/// ran. So caching the loaded/stored value per word can never observe a
/// stale state.
struct Emit<'a> {
    fb: FunctionBuilder<'a>,
    cache: HashMap<u32, Value>,
    base: Value,
}

impl Emit<'_> {
    fn word(&mut self, w: u32) -> Value {
        match self.cache.get(&w) {
            Some(&v) => v,
            None => {
                let v =
                    self.fb.ins().load(types::I64, MemFlags::trusted(), self.base, (w as i32) * 8);
                self.cache.insert(w, v);
                v
            }
        }
    }

    /// Emit one gather program as constant-folded straight-line code.
    fn gather(&mut self, segs: &[Seg], funnels: usize, imm: u64) -> Value {
        let mut acc =
            if imm != 0 { Some(self.fb.ins().iconst(types::I64, imm as i64)) } else { None };
        for seg in &segs[..funnels] {
            let w = seg.src >> 6;
            let o = seg.src & 63;
            let lo = self.word(w);
            // Source bit `src + k` lands at bit `shift + k`. With `o`
            // known at compile time the `o == 0` case needs no high word.
            let mut v = if o == 0 {
                lo
            } else {
                let hi = self.word(w + 1);
                let l = self.fb.ins().ushr_imm(lo, i64::from(o));
                let h = self.fb.ins().ishl_imm(hi, i64::from(64 - o));
                self.fb.ins().bor(l, h)
            };
            if seg.shift != 0 {
                v = self.fb.ins().ishl_imm(v, i64::from(seg.shift));
            }
            v = self.fb.ins().band_imm(v, seg.mask as i64);
            acc = Some(match acc {
                Some(a) => self.fb.ins().bor(a, v),
                None => v,
            });
        }
        for seg in &segs[funnels..] {
            let w = seg.src >> 6;
            let o = seg.src & 63;
            let lo = self.word(w);
            // Broadcast bit `o` to all 64 lanes with two shifts (arithmetic
            // right shift of the bit parked at the sign position).
            let parked = if o == 63 { lo } else { self.fb.ins().ishl_imm(lo, i64::from(63 - o)) };
            let bcast = self.fb.ins().sshr_imm(parked, 63);
            let v = self.fb.ins().band_imm(bcast, seg.mask as i64);
            acc = Some(match acc {
                Some(a) => self.fb.ins().bor(a, v),
                None => v,
            });
        }
        acc.unwrap_or_else(|| self.fb.ins().iconst(types::I64, 0))
    }

    /// Load arena bit `p` as a 0/1 valued `i64`.
    fn bit_at(&mut self, p: u32) -> Value {
        let w = self.word(p >> 6);
        let s = self.fb.ins().ushr_imm(w, i64::from(p & 63));
        self.fb.ins().band_imm(s, 1)
    }

    /// Assemble an address word from arena bit positions, least significant
    /// first: `sum_j bit(positions[j]) << j`.
    fn assemble_addr(&mut self, positions: &[u32]) -> Value {
        let mut addr = self.fb.ins().iconst(types::I64, 0);
        for (j, &p) in positions.iter().enumerate() {
            let mut bit = self.bit_at(p);
            if j != 0 {
                bit = self.fb.ins().ishl_imm(bit, j as i64);
            }
            addr = self.fb.ins().bor(addr, bit);
        }
        addr
    }
}

/// Advance `cursor` past the next `n` segments and return them.
fn take<'s>(cursor: &mut &'s [Seg], n: usize) -> &'s [Seg] {
    let (head, rest) = cursor.split_at(n);
    *cursor = rest;
    head
}

/// Low `len` bits set (`len <= 64`).
fn ones(len: usize) -> u64 {
    if len >= 64 { !0 } else { (1u64 << len) - 1 }
}

impl PackedJitEngine {
    pub fn new(circuit: &ryzr_core::Circuit) -> Self {
        Self::with_tape(&Compiled::new(circuit))
    }

    pub fn with_tape(tape: &Compiled) -> Self {
        let plan = Plan::new(tape);

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
        signature.params.push(AbiParam::new(pointer));

        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut ids = Vec::new();

        // At least one chunk even with zero tasks: the register capture
        // must still be emitted somewhere.
        let chunks: Vec<&[crate::pack::Task]> =
            if plan.tasks.is_empty() { vec![&[]] } else { plan.tasks.chunks(CHUNK).collect() };
        let last = chunks.len() - 1;
        let mut seg_cursor = plan.segs.as_slice();
        for (k, &tasks) in chunks.iter().enumerate() {
            ctx.func.signature = signature.clone();
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
            let block = fb.create_block();
            fb.append_block_params_for_function_params(block);
            fb.switch_to_block(block);
            fb.seal_block(block);
            let base = fb.block_params(block)[0];
            let staging = fb.block_params(block)[1];
            let mut emit = Emit { fb, cache: HashMap::new(), base };

            for task in tasks {
                let sr = task.streams;
                let sa = take(&mut seg_cursor, sr[0].funnels as usize + sr[0].splats as usize);
                let sb = take(&mut seg_cursor, sr[1].funnels as usize + sr[1].splats as usize);
                let sc = take(&mut seg_cursor, sr[2].funnels as usize + sr[2].splats as usize);

                let a = emit.gather(sa, sr[0].funnels as usize, task.imm[0]);
                let word = match task.op {
                    TaskOp::Gate(Op::Not) => emit.fb.ins().bnot(a),
                    TaskOp::Gate(Op::Buf) => a,
                    TaskOp::Gate(
                        op @ (Op::And | Op::Or | Op::Xor | Op::Nand | Op::Nor | Op::Xnor),
                    ) => {
                        let b = emit.gather(sb, sr[1].funnels as usize, task.imm[1]);
                        let raw = match op {
                            Op::And | Op::Nand => emit.fb.ins().band(a, b),
                            Op::Or | Op::Nor => emit.fb.ins().bor(a, b),
                            _ => emit.fb.ins().bxor(a, b),
                        };
                        match op {
                            Op::Nand | Op::Nor | Op::Xnor => emit.fb.ins().bnot(raw),
                            _ => raw,
                        }
                    }
                    TaskOp::Gate(Op::Mux) => {
                        let b = emit.gather(sb, sr[1].funnels as usize, task.imm[1]);
                        let c = emit.gather(sc, sr[2].funnels as usize, task.imm[2]);
                        emit.fb.ins().bitselect(a, b, c)
                    }
                    // Fused ripple chain: one native add propagates the
                    // carry through all lanes (carry-out at bit `len`).
                    TaskOp::Add => {
                        let b = emit.gather(sb, sr[1].funnels as usize, task.imm[1]);
                        let c = emit.gather(sc, sr[2].funnels as usize, task.imm[2]);
                        let ab = emit.fb.ins().iadd(a, b);
                        emit.fb.ins().iadd(ab, c)
                    }
                    // Fused RAM read: assemble the address, then a single
                    // dynamic-index funnel load pulls `bank[addr]` out of the
                    // register-output region.
                    TaskOp::MemRead(m) => {
                        let mr = &plan.mem_reads[m as usize];
                        let addr = emit.assemble_addr(&mr.addr_pos);
                        // Register file: read index `addr - (addr != 0)` so the
                        // hardwired-zero word 0 reads stored word 1 (never out
                        // of bounds) and is masked to zero below.
                        let (idx, zmask) = if mr.zero_word0 {
                            let nzb = emit.fb.ins().icmp_imm(IntCC::NotEqual, addr, 0);
                            let nz = emit.fb.ins().uextend(types::I64, nzb);
                            let idx = emit.fb.ins().isub(addr, nz);
                            // ineg(1) = !0 (keep), ineg(0) = 0 (zero the read).
                            (idx, Some(emit.fb.ins().ineg(nz)))
                        } else {
                            (addr, None)
                        };
                        let prod = emit.fb.ins().imul_imm(idx, i64::from(mr.width));
                        let srcb = emit.fb.ins().iadd_imm(prod, i64::from(mr.base_bit));
                        let word = emit.fb.ins().ushr_imm(srcb, 6); // / 64
                        let o = emit.fb.ins().band_imm(srcb, 63); // % 64
                        let woff = emit.fb.ins().ishl_imm(word, 3); // word * 8 bytes
                        let ptr = emit.fb.ins().iadd(emit.base, woff);
                        let lo = emit.fb.ins().load(types::I64, MemFlags::trusted(), ptr, 0);
                        let hi = emit.fb.ins().load(types::I64, MemFlags::trusted(), ptr, 8);
                        // Branchless funnel for a runtime offset: at o == 0
                        // the high half `(hi << 63) << 1` shifts out entirely.
                        let l = emit.fb.ins().ushr(lo, o);
                        let s63 = emit.fb.ins().irsub_imm(o, 63); // 63 - o
                        let h0 = emit.fb.ins().ishl(hi, s63);
                        let h = emit.fb.ins().ishl_imm(h0, 1);
                        let merged = emit.fb.ins().bor(l, h);
                        let masked = emit.fb.ins().band_imm(merged, ones(mr.width as usize) as i64);
                        match zmask {
                            Some(z) => emit.fb.ins().band(masked, z),
                            None => masked,
                        }
                    }
                };
                emit.fb.ins().store(MemFlags::trusted(), word, emit.base, (task.dst as i32) * 8);
                emit.cache.insert(task.dst, word);
            }

            // Register capture lives in the last chunk: by then every word
            // a capture gather reads has settled.
            if k == last {
                let mut cap_cursor = plan.cap_segs.as_slice();
                for (i, &(imm, sr)) in plan.capture.iter().enumerate() {
                    let n = sr.funnels as usize + sr.splats as usize;
                    let (head, rest) = cap_cursor.split_at(n);
                    cap_cursor = rest;
                    let v = emit.gather(head, sr.funnels as usize, imm);
                    emit.fb.ins().store(MemFlags::trusted(), v, staging, (i as i32) * 8);
                }

                // Fused RAM writes: the capture already held each bank's
                // contents forward, so a single guarded 128-bit patch (the
                // field may straddle two staging words) overwrites just the
                // addressed word when the store-enable is high — exactly
                // `mux(we_addr, data, self)`, none else.
                let mut mw_cursor = plan.mem_segs.as_slice();
                for mw in &plan.mem_writes {
                    let n = mw.data.funnels as usize + mw.data.splats as usize;
                    let (head, rest) = mw_cursor.split_at(n);
                    mw_cursor = rest;
                    let data = emit.gather(head, mw.data.funnels as usize, mw.data_imm);
                    let addr = emit.assemble_addr(&mw.addr_pos);

                    // Store-enable broadcast to an all-ones / all-zero mask.
                    let ew = emit.word(mw.enable_pos >> 6);
                    let eo = mw.enable_pos & 63;
                    let parked =
                        if eo == 63 { ew } else { emit.fb.ins().ishl_imm(ew, i64::from(63 - eo)) };
                    let cond = emit.fb.ins().sshr_imm(parked, 63);

                    // Destination word + intra-word offset in the staging.
                    let prod = emit.fb.ins().imul_imm(addr, i64::from(mw.width));
                    let off = emit.fb.ins().iadd_imm(prod, i64::from(mw.base_bit));
                    let word = emit.fb.ins().ushr_imm(off, 6);
                    let sh = emit.fb.ins().band_imm(off, 63);
                    let woff = emit.fb.ins().ishl_imm(word, 3);
                    let sptr = emit.fb.ins().iadd(staging, woff);
                    let lo = emit.fb.ins().load(types::I64, MemFlags::trusted(), sptr, 0);
                    let hi = emit.fb.ins().load(types::I64, MemFlags::trusted(), sptr, 8);

                    // 128-bit field placement (mirrors `pack::patch`).
                    let vmask = emit.fb.ins().iconst(types::I64, ones(mw.width as usize) as i64);
                    let vmask128 = emit.fb.ins().uextend(types::I128, vmask);
                    let masked_data = emit.fb.ins().band(data, vmask);
                    let data128 = emit.fb.ins().uextend(types::I128, masked_data);
                    let placed = emit.fb.ins().ishl(data128, sh);
                    let pmask = emit.fb.ins().ishl(vmask128, sh);
                    let cond128 = emit.fb.ins().sextend(types::I128, cond);
                    let effective = emit.fb.ins().band(pmask, cond128);
                    let neff = emit.fb.ins().bnot(effective);
                    let lo128 = emit.fb.ins().uextend(types::I128, lo);
                    let hi128 = emit.fb.ins().uextend(types::I128, hi);
                    let hi_shifted = emit.fb.ins().ishl_imm(hi128, 64);
                    let cur = emit.fb.ins().bor(lo128, hi_shifted);
                    let kept = emit.fb.ins().band(cur, neff);
                    let put = emit.fb.ins().band(placed, effective);
                    let new = emit.fb.ins().bor(kept, put);
                    let new_lo = emit.fb.ins().ireduce(types::I64, new);
                    let new_hi_full = emit.fb.ins().ushr_imm(new, 64);
                    let new_hi = emit.fb.ins().ireduce(types::I64, new_hi_full);
                    emit.fb.ins().store(MemFlags::trusted(), new_lo, sptr, 0);
                    emit.fb.ins().store(MemFlags::trusted(), new_hi, sptr, 8);
                }
            }

            emit.fb.ins().return_(&[]);
            emit.fb.finalize();

            let name = format!("settle{k}");
            let id = module
                .declare_function(&name, Linkage::Export, &ctx.func.signature)
                .expect("declare packed jit chunk");
            module.define_function(id, &mut ctx).expect("compile packed jit chunk");
            module.clear_context(&mut ctx);
            ids.push(id);
        }

        module.finalize_definitions().expect("finalize packed jit module");
        let fns = ids
            .into_iter()
            .map(|id| {
                let ptr = module.get_finalized_function(id);
                // SAFETY: the function was compiled with exactly the
                // `fn(*mut u64, *mut u64)` signature built above.
                unsafe { core::mem::transmute::<*const u8, ChunkFn>(ptr) }
            })
            .collect();

        let mut engine = Self {
            bits: vec![0; plan.words],
            staging: vec![0; plan.staging_words],
            reg_word: plan.reg_word,
            reg_init: plan.reg_init,
            outputs: plan.outputs,
            input_count: plan.input_count,
            fns,
            module: Some(module),
        };
        engine.reset();
        engine
    }

    /// Restore power-on state: register initials latched, inputs low.
    pub(crate) fn reset(&mut self) {
        self.bits.fill(0);
        self.staging.fill(0);
        let regs = self.reg_init.len();
        self.staging[..regs].copy_from_slice(&self.reg_init);
        self.bits[self.reg_word..self.reg_word + regs].copy_from_slice(&self.reg_init);
    }
}

impl Engine for PackedJitEngine {
    fn name(&self) -> &'static str {
        "packed-jit"
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
        // Clock edge: the register region is contiguous and word-aligned, so
        // applying the captured next-state is a straight word copy. Only the
        // live-register words copy back; a staging pad word a write patch may
        // have touched is scratch and stays behind.
        let regs = self.reg_init.len();
        self.bits[self.reg_word..self.reg_word + regs].copy_from_slice(&self.staging[..regs]);

        let bits = self.bits.as_mut_ptr();
        let staging = self.staging.as_mut_ptr();
        for f in &self.fns {
            // SAFETY: `bits` has `plan.words` words (validated against every
            // gather source in the planner) and `staging` has one word per
            // capture entry; the jitted code touches nothing else.
            unsafe { f(bits, staging) };
        }
    }
}

impl Drop for PackedJitEngine {
    fn drop(&mut self) {
        self.fns.clear();
        if let Some(module) = self.module.take() {
            // SAFETY: all pointers into the module's executable memory were
            // cleared above; nothing can call into it after this point.
            unsafe { module.free_memory() };
        }
    }
}
