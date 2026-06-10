//! Instruction-level RV32I reference emulator.
//!
//! Mirrors the gate-level core *exactly*, including its quirks: the fetch
//! and data addresses wrap modulo the ROM/RAM size, loads shift the
//! addressed word right by the byte offset with zero fill before width
//! selection, stores byte-merge with the same enables the hardware
//! computes, undefined branch funct3 values fall back to eq/ne, and
//! unknown opcodes retire as no-ops. Running it in lockstep against the
//! circuit therefore checks every retired instruction bit for bit.

use crate::cpu::pad_rom;

pub struct Emulator {
    pub pc: u32,
    pub regs: [u32; 32],
    pub ram: Vec<u32>,
    rom: Vec<u32>,
}

impl Emulator {
    pub fn new(program: &[u32], ram_words: usize) -> Self {
        assert!(ram_words.is_power_of_two());
        Self { pc: 0, regs: [0; 32], ram: vec![0; ram_words], rom: pad_rom(program) }
    }

    /// Retire one instruction.
    pub fn step(&mut self) {
        let instr = self.rom[(self.pc as usize >> 2) & (self.rom.len() - 1)];
        let opcode = instr & 0x7f;
        let rd = (instr >> 7 & 0x1f) as usize;
        let f3 = instr >> 12 & 7;
        let rs1 = self.regs[(instr >> 15 & 0x1f) as usize];
        let rs2 = self.regs[(instr >> 20 & 0x1f) as usize];

        let imm_i = (instr as i32) >> 20;
        let imm_s = ((instr & 0xfe00_0000) as i32 >> 20) | (instr >> 7 & 0x1f) as i32;
        let imm_b = ((instr & 0x8000_0000) as i32 >> 19)
            | ((instr >> 7 & 1) << 11) as i32
            | ((instr >> 25 & 0x3f) << 5) as i32
            | ((instr >> 8 & 0xf) << 1) as i32;
        let imm_u = instr & 0xffff_f000;
        let imm_j = ((instr & 0x8000_0000) as i32 >> 11)
            | (instr & 0x000f_f000) as i32
            | ((instr >> 20 & 1) << 11) as i32
            | ((instr >> 21 & 0x3ff) << 1) as i32;

        let mut next_pc = self.pc.wrapping_add(4);
        let write = |regs: &mut [u32; 32], value: u32| {
            if rd != 0 {
                regs[rd] = value;
            }
        };

        match opcode {
            0b0110111 => write(&mut self.regs, imm_u),
            0b0010111 => write(&mut self.regs, self.pc.wrapping_add(imm_u)),
            0b1101111 => {
                write(&mut self.regs, next_pc);
                next_pc = self.pc.wrapping_add(imm_j as u32);
            }
            0b1100111 => {
                write(&mut self.regs, next_pc);
                next_pc = rs1.wrapping_add(imm_i as u32) & !1;
            }
            0b1100011 => {
                let taken = match f3 {
                    0 | 2 => rs1 == rs2,
                    1 | 3 => rs1 != rs2,
                    4 => (rs1 as i32) < (rs2 as i32),
                    5 => (rs1 as i32) >= (rs2 as i32),
                    6 => rs1 < rs2,
                    _ => rs1 >= rs2,
                };
                if taken {
                    next_pc = self.pc.wrapping_add(imm_b as u32);
                }
            }
            0b0000011 => {
                let addr = rs1.wrapping_add(imm_i as u32);
                let word = self.ram[(addr as usize >> 2) & (self.ram.len() - 1)];
                let lane = word >> ((addr & 3) * 8);
                let value = match f3 {
                    0 => lane as u8 as i8 as i32 as u32,
                    1 => lane as u16 as i16 as i32 as u32,
                    4 => lane as u8 as u32,
                    5 => lane as u16 as u32,
                    _ => lane,
                };
                write(&mut self.regs, value);
            }
            0b0100011 => {
                let addr = rs1.wrapping_add(imm_s as u32);
                let index = (addr as usize >> 2) & (self.ram.len() - 1);
                let shift = (addr & 3) * 8;
                let mask = match f3 {
                    0 => 0xffu32 << shift,
                    1 => {
                        if addr & 2 != 0 {
                            0xffff_0000
                        } else {
                            0x0000_ffff
                        }
                    }
                    2 => 0xffff_ffff,
                    _ => 0,
                };
                self.ram[index] = (self.ram[index] & !mask) | ((rs2 << shift) & mask);
            }
            0b0010011 | 0b0110011 => {
                let operand = if opcode == 0b0010011 { imm_i as u32 } else { rs2 };
                let bit30 = instr >> 30 & 1 != 0;
                let value = match f3 {
                    0 => {
                        if opcode == 0b0110011 && bit30 {
                            rs1.wrapping_sub(operand)
                        } else {
                            rs1.wrapping_add(operand)
                        }
                    }
                    1 => rs1 << (operand & 31),
                    2 => u32::from((rs1 as i32) < (operand as i32)),
                    3 => u32::from(rs1 < operand),
                    4 => rs1 ^ operand,
                    5 => {
                        if bit30 {
                            ((rs1 as i32) >> (operand & 31)) as u32
                        } else {
                            rs1 >> (operand & 31)
                        }
                    }
                    6 => rs1 | operand,
                    _ => rs1 & operand,
                };
                write(&mut self.regs, value);
            }
            _ => {}
        }

        self.pc = next_pc;
    }

    /// Retire `n` instructions.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Emulator;
    use crate::programs;

    #[test]
    fn fib_20_is_6765() {
        let mut emu = Emulator::new(&programs::fib_terminating(20), 4);
        emu.run(200);
        assert_eq!(emu.regs[10], 6765);
    }

    #[test]
    fn memory_widths_round_trip() {
        use crate::asm::*;
        let program =
            vec![lui(5, 0x1234_5000), addi(5, 5, 0x678), sw(0, 5, 8), lhu(6, 0, 10), lb(7, 0, 9)];
        let mut emu = Emulator::new(&program, 16);
        emu.run(program.len());
        assert_eq!(emu.ram[2], 0x1234_5678);
        assert_eq!(emu.regs[6], 0x1234);
        assert_eq!(emu.regs[7], 0x56);
    }
}
