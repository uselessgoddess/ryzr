#![no_std]

#[cfg_attr(not(feature = "std"), macro_use)]
extern crate alloc;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

#[allow(unused_imports)]
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet, hash_map};

#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet, hash_map};

mod backend;
mod circuit;

pub use backend::{Backend, Interpreter};
pub use circuit::{Circuit, CircuitBuilder, GateOp, InstData, Instruction, Reg, Register, Signal};
