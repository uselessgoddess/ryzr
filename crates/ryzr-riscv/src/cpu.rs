//! Single-cycle gate-level RV32I core: one tick = one retired instruction.
//!
//! Everything is honest hardware built from the nine ryzr gate kinds: the
//! register file and RAM are D flip-flops behind mux trees, the ALU is a
//! ripple-carry adder plus barrel shifters, instruction decode is wired bit
//! matching. Nothing is pattern-matched into native arithmetic — an `add`
//! really ripples 32 carry bits through gates every tick.
//!
//! Memory follows a simple word-RAM model: the byte address is reduced
//! modulo the RAM size, loads shift the addressed word right by the byte
//! offset (zero fill) before width/sign selection, stores byte-merge into
//! the addressed word. Unimplemented opcodes retire as no-ops (pc advances).
//! The bundled [`Emulator`](crate::Emulator) mirrors this model exactly,
//! including out-of-range wrap-around and misaligned-access behavior, so
//! the two can run in lockstep on arbitrary programs.
//!
//! Architectural state is exported as circuit outputs: `pc` at bit indices
//! `0..32`, register `xN` at `32 * (N + 1) .. 32 * (N + 2)`.

use ryzr_core::{Circuit, CircuitBuilder, Signal};

use crate::rtl::{self, Word, XLEN};

/// `jal x0, 0`: spins in place forever; used to pad the ROM.
pub(crate) const HALT: u32 = 0x0000_006F;

/// ROM image padded with [`HALT`] to a power of two, so the fetch mux tree
/// is complete and the program counter wraps instead of fetching garbage.
pub(crate) fn pad_rom(program: &[u32]) -> Vec<u32> {
    let len = program.len().next_power_of_two().max(2);
    program.iter().copied().chain(core::iter::repeat(HALT)).take(len).collect()
}

/// Build the CPU circuit around a program ROM and `ram_words` words of RAM
/// (must be a power of two).
pub fn build_cpu(program: &[u32], ram_words: usize) -> Circuit {
    assert!(ram_words.is_power_of_two(), "ram_words must be a power of two");
    let rom = pad_rom(program);
    let rom_bits = rom.len().trailing_zeros() as usize;
    let ram_bits = ram_words.trailing_zeros() as usize;

    let mut b = CircuitBuilder::new();
    let zero = b.const_val(false);
    let one = b.const_val(true);

    // ---- State: pc, x1..x31, RAM (x0 is hardwired zero, no storage) ----
    let pc_regs: Vec<_> = (0..XLEN).map(|i| b.reg(format!("PC{i}"), false)).collect();
    let pc: Word = pc_regs.iter().map(|&(_, q)| q).collect();

    let reg_cells: Vec<Vec<_>> =
        (1..32).map(|r| (0..XLEN).map(|i| b.reg(format!("X{r}_{i}"), false)).collect()).collect();
    let mut reg_words: Vec<Word> = vec![vec![zero; XLEN]];
    reg_words.extend(reg_cells.iter().map(|cells| cells.iter().map(|&(_, q)| q).collect()));

    let mem_cells: Vec<Vec<_>> = (0..ram_words)
        .map(|w| (0..XLEN).map(|i| b.reg(format!("M{w}_{i}"), false)).collect())
        .collect();
    let mem_words: Vec<Word> =
        mem_cells.iter().map(|cells| cells.iter().map(|&(_, q)| q).collect()).collect();

    // ---- Fetch: instruction ROM as a mux tree of constants ----
    let rom_words: Vec<Word> = rom.iter().map(|&w| rtl::constant(&mut b, w)).collect();
    let instr = rtl::mux_tree(&mut b, &pc[2..2 + rom_bits], &rom_words);

    // ---- Decode ----
    let rd_sel = &instr[7..12];
    let funct3 = &instr[12..15];
    let rs1_sel = &instr[15..20];
    let rs2_sel = &instr[20..25];
    let instr30 = instr[30];
    let sign = instr[31];

    let opcode = &instr[0..7];
    let is_lui = rtl::match_bits(&mut b, opcode, 0b0110111);
    let is_auipc = rtl::match_bits(&mut b, opcode, 0b0010111);
    let is_jal = rtl::match_bits(&mut b, opcode, 0b1101111);
    let is_jalr = rtl::match_bits(&mut b, opcode, 0b1100111);
    let is_branch = rtl::match_bits(&mut b, opcode, 0b1100011);
    let is_load = rtl::match_bits(&mut b, opcode, 0b0000011);
    let is_store = rtl::match_bits(&mut b, opcode, 0b0100011);
    let is_opimm = rtl::match_bits(&mut b, opcode, 0b0010011);
    let is_op = rtl::match_bits(&mut b, opcode, 0b0110011);

    let f3d = rtl::decoder(&mut b, funct3);

    // Immediates are pure wiring: each bit is an existing instruction wire.
    let imm_i: Word = (0..XLEN).map(|i| if i < 11 { instr[20 + i] } else { sign }).collect();
    let imm_s: Word = (0..XLEN)
        .map(|i| {
            if i < 5 {
                instr[7 + i]
            } else if i < 11 {
                instr[20 + i]
            } else {
                sign
            }
        })
        .collect();
    let imm_b: Word = (0..XLEN)
        .map(|i| {
            if i == 0 {
                zero
            } else if i < 5 {
                instr[7 + i]
            } else if i < 11 {
                instr[20 + i]
            } else if i == 11 {
                instr[7]
            } else {
                sign
            }
        })
        .collect();
    let imm_u: Word = (0..XLEN).map(|i| if i < 12 { zero } else { instr[i] }).collect();
    let imm_j: Word = (0..XLEN)
        .map(|i| {
            if i == 0 {
                zero
            } else if i < 11 {
                instr[20 + i]
            } else if i == 11 {
                instr[20]
            } else if i < 20 {
                instr[i]
            } else {
                sign
            }
        })
        .collect();

    // ---- Register file read ports ----
    let rs1_data = rtl::mux_tree(&mut b, rs1_sel, &reg_words);
    let rs2_data = rtl::mux_tree(&mut b, rs2_sel, &reg_words);

    // ---- ALU ----
    // Second operand: rs2 for OP, the relevant immediate otherwise.
    let st_or_i = rtl::mux_word(&mut b, is_store, &imm_s, &imm_i);
    let ld_or_st = b.or(is_load, is_store);
    let imm_user = b.or(ld_or_st, is_jalr);
    let use_imm = b.or(imm_user, is_opimm);
    let alu_b = rtl::mux_word(&mut b, use_imm, &st_or_i, &rs2_data);

    // One shared adder does ADD, SUB (operand inverted, carry-in 1), the
    // load/store address and the JALR target.
    let op_f30 = b.and(is_op, instr30);
    let sub_flag = b.and(op_f30, f3d[0]);
    let b_maybe_inverted: Word = alu_b.iter().map(|&s| b.xor(s, sub_flag)).collect();
    let (add_sub, _) = rtl::adder(&mut b, &rs1_data, &b_maybe_inverted, sub_flag);

    // Dedicated subtract for comparisons: rs1 - alu_b.
    let not_b = rtl::word_not(&mut b, &alu_b);
    let (diff, borrow_free) = rtl::adder(&mut b, &rs1_data, &not_b, one);
    let ltu = b.not(borrow_free);
    let sign_mix = b.xor(rs1_data[XLEN - 1], alu_b[XLEN - 1]);
    let lt = b.mux(sign_mix, rs1_data[XLEN - 1], diff[XLEN - 1]);
    let eq = rtl::equal(&mut b, &rs1_data, &alu_b);

    let mut slt_w = vec![zero; XLEN];
    slt_w[0] = lt;
    let mut sltu_w = vec![zero; XLEN];
    sltu_w[0] = ltu;

    let shamt = &alu_b[0..5];
    let sll = rtl::shift_left(&mut b, &rs1_data, shamt);
    let sra_fill = b.and(instr30, rs1_data[XLEN - 1]);
    let srx = rtl::shift_right(&mut b, &rs1_data, shamt, sra_fill);

    let xor_w = rtl::word_xor(&mut b, &rs1_data, &alu_b);
    let or_w = rtl::word_or(&mut b, &rs1_data, &alu_b);
    let and_w = rtl::word_and(&mut b, &rs1_data, &alu_b);

    // funct3-indexed result; only consumed by OP/OP-IMM writeback (loads
    // and JALR take `add_sub` directly, dodging the funct3 clash).
    let alu_out = rtl::mux_tree(
        &mut b,
        funct3,
        &[add_sub.clone(), sll, slt_w, sltu_w, xor_w, srx, or_w, and_w],
    );

    // ---- Data memory ----
    let mem_addr = &add_sub;
    let addr0 = mem_addr[0];
    let addr1 = mem_addr[1];
    let mem_word = rtl::mux_tree(&mut b, &mem_addr[2..2 + ram_bits], &mem_words);

    // Loads: shift the word right by the byte offset (zero fill), then
    // select width and sign extension by funct3.
    let by8: Word = (0..XLEN).map(|i| if i + 8 < XLEN { mem_word[i + 8] } else { zero }).collect();
    let half_shift = rtl::mux_word(&mut b, addr0, &by8, &mem_word);
    let by16: Word =
        (0..XLEN).map(|i| if i + 16 < XLEN { half_shift[i + 16] } else { zero }).collect();
    let lane = rtl::mux_word(&mut b, addr1, &by16, &half_shift);

    let lb: Word = (0..XLEN).map(|i| if i < 8 { lane[i] } else { lane[7] }).collect();
    let lh: Word = (0..XLEN).map(|i| if i < 16 { lane[i] } else { lane[15] }).collect();
    let lbu: Word = (0..XLEN).map(|i| if i < 8 { lane[i] } else { zero }).collect();
    let lhu: Word = (0..XLEN).map(|i| if i < 16 { lane[i] } else { zero }).collect();
    let load_data = rtl::mux_tree(
        &mut b,
        funct3,
        &[lb, lh, lane.clone(), lane.clone(), lbu, lhu, lane.clone(), lane],
    );

    // Stores: shift the source left into its byte lane, then merge the
    // enabled bytes into the addressed word.
    let s_by8: Word = (0..XLEN).map(|i| if i >= 8 { rs2_data[i - 8] } else { zero }).collect();
    let s_half = rtl::mux_word(&mut b, addr0, &s_by8, &rs2_data);
    let s_by16: Word = (0..XLEN).map(|i| if i >= 16 { s_half[i - 16] } else { zero }).collect();
    let sdata = rtl::mux_word(&mut b, addr1, &s_by16, &s_half);

    let off_dec = rtl::decoder(&mut b, &[addr0, addr1]);
    let not_a1 = b.not(addr1);
    let byte_en: Vec<Signal> = (0..4)
        .map(|j| {
            let half = if j < 2 { not_a1 } else { addr1 };
            let h_en = b.and(f3d[1], half);
            let b_en = b.and(f3d[0], off_dec[j]);
            let wide = b.or(f3d[2], h_en);
            b.or(wide, b_en)
        })
        .collect();
    let merged: Word = (0..XLEN).map(|i| b.mux(byte_en[i / 8], sdata[i], mem_word[i])).collect();

    let mem_sel_dec = rtl::decoder(&mut b, &mem_addr[2..2 + ram_bits]);
    for (w, cells) in mem_cells.iter().enumerate() {
        let we = b.and(is_store, mem_sel_dec[w]);
        for (i, &(reg, q)) in cells.iter().enumerate() {
            let next = b.mux(we, merged[i], q);
            b.drive(reg, next);
        }
    }

    // ---- Next pc ----
    let four = rtl::constant(&mut b, 4);
    let pc4 = rtl::add(&mut b, &pc, &four);
    let jal_or_b = rtl::mux_word(&mut b, is_jal, &imm_j, &imm_b);
    let pc_off = rtl::mux_word(&mut b, is_auipc, &imm_u, &jal_or_b);
    let pc_target = rtl::add(&mut b, &pc, &pc_off);
    let mut jalr_target = add_sub.clone();
    jalr_target[0] = zero;

    // beq/bne use eq, blt/bge signed less-than, bltu/bgeu unsigned;
    // funct3 bit 0 inverts the condition.
    let lt_kind = b.mux(funct3[1], ltu, lt);
    let base_cond = b.mux(funct3[2], lt_kind, eq);
    let cond = b.xor(base_cond, funct3[0]);
    let taken = b.and(is_branch, cond);
    let any_jal = b.or(is_jal, is_jalr);
    let jump = b.or(taken, any_jal);

    let jump_target = rtl::mux_word(&mut b, is_jalr, &jalr_target, &pc_target);
    let next_pc = rtl::mux_word(&mut b, jump, &jump_target, &pc4);
    for (i, &(reg, _)) in pc_regs.iter().enumerate() {
        b.drive(reg, next_pc[i]);
    }

    // ---- Writeback ----
    let wb_alu = rtl::mux_word(&mut b, is_lui, &imm_u, &alu_out);
    let wb_pc = rtl::mux_word(&mut b, is_auipc, &pc_target, &wb_alu);
    let wb_link = rtl::mux_word(&mut b, any_jal, &pc4, &wb_pc);
    let wb = rtl::mux_word(&mut b, is_load, &load_data, &wb_link);

    let wb_int = b.or(is_opimm, is_op);
    let wb_upper = b.or(is_lui, is_auipc);
    let wb_jump_load = b.or(any_jal, is_load);
    let wb_some = b.or(wb_int, wb_upper);
    let reg_write = b.or(wb_some, wb_jump_load);

    let rd_dec = rtl::decoder(&mut b, rd_sel);
    for (r, cells) in reg_cells.iter().enumerate() {
        let we = b.and(reg_write, rd_dec[r + 1]);
        for (i, &(reg, q)) in cells.iter().enumerate() {
            let next = b.mux(we, wb[i], q);
            b.drive(reg, next);
        }
    }

    // ---- Observable architectural state ----
    for (i, &q) in pc.iter().enumerate() {
        b.output(format!("PC{i}"), q);
    }
    for (r, word) in reg_words.iter().enumerate() {
        for (i, &q) in word.iter().enumerate() {
            b.output(format!("X{r}_{i}"), q);
        }
    }

    b.finish().expect("cpu circuit is acyclic by construction")
}
