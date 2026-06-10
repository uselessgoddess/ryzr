//! Minimal RV32I assembler: instruction-word encoders for programs, tests
//! and benchmarks. Registers are plain numbers (`x0..x31`), branch and jump
//! offsets are byte offsets relative to the instruction itself.

fn rtype(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    funct7 << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode
}

fn itype(imm: i32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (imm as u32 & 0xfff) << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode
}

fn stype(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm = imm as u32;
    (imm >> 5 & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | (imm & 0x1f) << 7 | 0b0100011
}

fn btype(offset: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm = offset as u32;
    (imm >> 12 & 1) << 31
        | (imm >> 5 & 0x3f) << 25
        | rs2 << 20
        | rs1 << 15
        | funct3 << 12
        | (imm >> 1 & 0xf) << 8
        | (imm >> 11 & 1) << 7
        | 0b1100011
}

// OP-IMM ------------------------------------------------------------------

pub fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b000, rd, 0b0010011)
}
pub fn slti(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b010, rd, 0b0010011)
}
pub fn sltiu(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b011, rd, 0b0010011)
}
pub fn xori(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b100, rd, 0b0010011)
}
pub fn ori(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b110, rd, 0b0010011)
}
pub fn andi(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b111, rd, 0b0010011)
}
pub fn slli(rd: u32, rs1: u32, shamt: u32) -> u32 {
    itype(shamt as i32, rs1, 0b001, rd, 0b0010011)
}
pub fn srli(rd: u32, rs1: u32, shamt: u32) -> u32 {
    itype(shamt as i32, rs1, 0b101, rd, 0b0010011)
}
pub fn srai(rd: u32, rs1: u32, shamt: u32) -> u32 {
    itype((0x400 | shamt) as i32, rs1, 0b101, rd, 0b0010011)
}

// OP ----------------------------------------------------------------------

pub fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b000, rd, 0b0110011)
}
pub fn sub(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0b0100000, rs2, rs1, 0b000, rd, 0b0110011)
}
pub fn sll(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b001, rd, 0b0110011)
}
pub fn slt(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b010, rd, 0b0110011)
}
pub fn sltu(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b011, rd, 0b0110011)
}
pub fn xor(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b100, rd, 0b0110011)
}
pub fn srl(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b101, rd, 0b0110011)
}
pub fn sra(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0b0100000, rs2, rs1, 0b101, rd, 0b0110011)
}
pub fn or(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b110, rd, 0b0110011)
}
pub fn and(rd: u32, rs1: u32, rs2: u32) -> u32 {
    rtype(0, rs2, rs1, 0b111, rd, 0b0110011)
}

// Upper immediates and jumps ----------------------------------------------

/// `value`'s upper 20 bits are kept, low 12 dropped (as the ISA does).
pub fn lui(rd: u32, value: u32) -> u32 {
    (value & 0xffff_f000) | rd << 7 | 0b0110111
}
pub fn auipc(rd: u32, value: u32) -> u32 {
    (value & 0xffff_f000) | rd << 7 | 0b0010111
}
pub fn jal(rd: u32, offset: i32) -> u32 {
    let imm = offset as u32;
    (imm >> 20 & 1) << 31
        | (imm >> 1 & 0x3ff) << 21
        | (imm >> 11 & 1) << 20
        | (imm >> 12 & 0xff) << 12
        | rd << 7
        | 0b1101111
}
pub fn jalr(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b000, rd, 0b1100111)
}

// Branches ------------------------------------------------------------------

pub fn beq(rs1: u32, rs2: u32, offset: i32) -> u32 {
    btype(offset, rs2, rs1, 0b000)
}
pub fn bne(rs1: u32, rs2: u32, offset: i32) -> u32 {
    btype(offset, rs2, rs1, 0b001)
}
pub fn blt(rs1: u32, rs2: u32, offset: i32) -> u32 {
    btype(offset, rs2, rs1, 0b100)
}
pub fn bge(rs1: u32, rs2: u32, offset: i32) -> u32 {
    btype(offset, rs2, rs1, 0b101)
}
pub fn bltu(rs1: u32, rs2: u32, offset: i32) -> u32 {
    btype(offset, rs2, rs1, 0b110)
}
pub fn bgeu(rs1: u32, rs2: u32, offset: i32) -> u32 {
    btype(offset, rs2, rs1, 0b111)
}

// Loads and stores -----------------------------------------------------------

pub fn lb(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b000, rd, 0b0000011)
}
pub fn lh(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b001, rd, 0b0000011)
}
pub fn lw(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b010, rd, 0b0000011)
}
pub fn lbu(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b100, rd, 0b0000011)
}
pub fn lhu(rd: u32, rs1: u32, imm: i32) -> u32 {
    itype(imm, rs1, 0b101, rd, 0b0000011)
}
pub fn sb(rs1: u32, rs2: u32, imm: i32) -> u32 {
    stype(imm, rs2, rs1, 0b000)
}
pub fn sh(rs1: u32, rs2: u32, imm: i32) -> u32 {
    stype(imm, rs2, rs1, 0b001)
}
pub fn sw(rs1: u32, rs2: u32, imm: i32) -> u32 {
    stype(imm, rs2, rs1, 0b010)
}
