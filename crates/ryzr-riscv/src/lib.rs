//! Gate-level RV32I core built from `ryzr` primitives, plus the tooling to
//! prove it correct: a tiny assembler, an instruction-level reference
//! emulator that mirrors the hardware exactly, and benchmark programs.
//!
//! This is the project's honesty benchmark. One engine tick retires one
//! instruction, and every bit of that instruction — decode, ripple-carry
//! arithmetic, barrel shifts, register-file and RAM mux trees — is computed
//! through real gates. Tests run the circuit in lockstep with the emulator
//! and compare the full architectural state (pc and all 32 registers) after
//! every instruction.

pub mod asm;
mod cpu;
mod emu;
pub mod programs;
pub mod rtl;

pub use cpu::build_cpu;
pub use emu::Emulator;
