//! Small RV32I programs used by tests and benchmarks.

use crate::asm::*;

/// Endless Fibonacci: every tick retires a useful instruction forever, so
/// throughput benchmarks never run out of work.
///
/// ```text
/// addi a0, x0, 0
/// addi a1, x0, 1
/// loop: add a2, a0, a1
///       addi a0, a1, 0
///       addi a1, a2, 0
///       jal x0, loop
/// ```
pub fn fib_forever() -> Vec<u32> {
    vec![
        addi(10, 0, 0),
        addi(11, 0, 1),
        add(12, 10, 11),
        addi(10, 11, 0),
        addi(11, 12, 0),
        jal(0, -12),
    ]
}

/// Compute `fib(n)` into `a0`, then spin in place. `fib(20) == 6765`.
pub fn fib_terminating(n: i32) -> Vec<u32> {
    vec![
        addi(10, 0, 0),  //  0: a0 = 0
        addi(11, 0, 1),  //  4: a1 = 1
        addi(5, 0, n),   //  8: t0 = n
        beq(5, 0, 24),   // 12: while t0 != 0 {
        add(12, 10, 11), // 16:   a2 = a0 + a1
        addi(10, 11, 0), // 20:   a0 = a1
        addi(11, 12, 0), // 24:   a1 = a2
        addi(5, 5, -1),  // 28:   t0 -= 1
        jal(0, -20),     // 32: }
        jal(0, 0),       // 36: halt
    ]
}

/// Every load/store width and sign mode, including a negative byte and a
/// misaligned-by-design halfword lane.
pub fn memory_exercise() -> Vec<u32> {
    vec![
        lui(5, 0x1234_5000), // t0 = 0x12345000
        addi(5, 5, 0x678),   // t0 = 0x12345678
        sw(0, 5, 8),
        lw(6, 0, 8),    // t1 = 0x12345678
        lb(7, 0, 9),    // t2 = 0x56
        lbu(28, 0, 11), // t3 = 0x12
        lh(29, 0, 8),   // t4 = 0x5678
        lhu(30, 0, 10), // t5 = 0x1234
        add(10, 6, 7),
        add(10, 10, 28),
        add(10, 10, 29),
        add(10, 10, 30),
        lui(31, 0xfedc_b000), // t6: negative bytes
        sw(0, 31, 20),
        lb(8, 0, 21), // 0xffffffb0 (sign extension)
        lh(9, 0, 22), // 0xfffffedc
        add(10, 10, 8),
        add(10, 10, 9),
        sb(0, 10, 13),
        sh(0, 10, 18),
        lw(11, 0, 12),
        lw(12, 0, 16),
        add(10, 11, 12),
        jal(0, 0), // halt
    ]
}

/// Every OP and OP-IMM form, both funct7 variants, plus LUI/AUIPC.
pub fn alu_exercise() -> Vec<u32> {
    vec![
        addi(5, 0, -5),
        addi(6, 0, 3),
        add(10, 5, 6),
        sub(11, 5, 6),
        sll(12, 6, 6),
        slt(13, 5, 6),
        sltu(14, 5, 6),
        xor(15, 5, 6),
        srl(16, 5, 6),
        sra(17, 5, 6),
        or(28, 5, 6),
        and(29, 5, 6),
        slli(30, 5, 4),
        srli(31, 5, 4),
        srai(7, 5, 4),
        slti(8, 5, -4),
        sltiu(9, 5, 100),
        xori(18, 5, 0x55),
        ori(19, 5, 0x0f),
        andi(20, 5, 0xff),
        lui(21, 0xabcd_e000),
        auipc(22, 0x0000_1000),
        jal(0, 0), // halt
    ]
}

/// Every branch kind, taken and not taken, plus JAL/JALR linking.
pub fn branch_exercise() -> Vec<u32> {
    vec![
        addi(5, 0, -1),   //  0: t0 = -1 (0xffffffff unsigned)
        addi(6, 0, 1),    //  4: t1 = 1
        blt(5, 6, 8),     //  8: taken (signed)
        addi(10, 0, 111), // 12: skipped
        bltu(5, 6, 8),    // 16: not taken (unsigned)
        addi(11, 0, 1),   // 20: executed
        bge(6, 5, 8),     // 24: taken
        addi(12, 0, 222), // 28: skipped
        bgeu(6, 5, 8),    // 32: not taken
        addi(13, 0, 2),   // 36: executed
        bne(5, 6, 8),     // 40: taken
        addi(14, 0, 333), // 44: skipped
        beq(5, 5, 8),     // 48: taken
        addi(15, 0, 444), // 52: skipped
        jal(1, 8),        // 56: ra = 60, jump to 64
        addi(16, 0, 555), // 60: skipped
        auipc(7, 0),      // 64: t2 = 64
        jalr(1, 7, 16),   // 68: ra = 72, jump to 80 (halt padding)
    ]
}
