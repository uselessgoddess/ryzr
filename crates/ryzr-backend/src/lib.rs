//! High-performance simulation engines for `ryzr` circuits.
//!
//! Every engine consumes the same [`Compiled`] tape (levelized,
//! op-run-scheduled struct-of-arrays form of a [`ryzr_core::Circuit`]) and
//! implements identical synchronous semantics:
//!
//! 1. combinational settle (every gate sees this tick's source values),
//! 2. registers latch their data inputs,
//! 3. outputs reflect the settled combinational values.
//!
//! One tick = one full clock cycle. There are no shortcuts that change
//! observable results — engines differ only in *how* they arrive at the
//! exact same values:
//!
//! | engine | strategy |
//! |---|---|
//! | [`ScalarEngine`] | dense forward pass, per-run dispatch |
//! | [`EventEngine`] | recomputes only the cone affected by actual changes |
//! | [`BatchEngine`] | 64 independent instances bit-packed per word (SWAR) |
//! | [`PackedEngine`] | one instance bit-packed: up to 64 same-op gates per word op |
//! | [`JitEngine`] | tick compiled to native code via Cranelift |
//! | [`ThreadedEngine`] | level-parallel work distribution via rayon |
//! | [`HybridEngine`] | every technique above behind one type; fastest plan picked by racing them on the live circuit |

mod batch;
pub mod compile;
mod event;
#[cfg(all(feature = "jit", feature = "rayon"))]
mod hybrid;
#[cfg(feature = "jit")]
mod jit;
mod pack;
#[cfg(feature = "jit")]
mod pack_jit;
mod scalar;
#[cfg(feature = "rayon")]
mod threaded;

pub use batch::BatchEngine;
pub use compile::Compiled;
pub use event::EventEngine;
#[cfg(all(feature = "jit", feature = "rayon"))]
pub use hybrid::{HybridEngine, Strategy};
#[cfg(feature = "jit")]
pub use jit::JitEngine;
pub use pack::PackedEngine;
#[cfg(feature = "jit")]
pub use pack_jit::PackedJitEngine;
pub use scalar::ScalarEngine;
#[cfg(feature = "rayon")]
pub use threaded::ThreadedEngine;

/// A compiled circuit instance that can be ticked.
///
/// Engines own their state; inputs are sticky (latched until changed).
pub trait Engine: Send {
    fn name(&self) -> &'static str;

    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;

    fn set_input(&mut self, index: usize, value: bool);
    fn output(&self, index: usize) -> bool;

    /// Advance one clock cycle.
    fn tick(&mut self);

    /// Advance `ticks` cycles. Engines may override with a tighter loop.
    fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.tick();
        }
    }
}
