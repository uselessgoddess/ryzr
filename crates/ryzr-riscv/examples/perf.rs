//! Quick single-instance throughput check on the gate-level RV32I core:
//! instructions retired per second by one simulated CPU, per engine.
//! Criterion's `riscv` bench is the rigorous version; this is the fast
//! inner loop for development.

use std::time::Instant;

use ryzr_backend::{Engine, EventEngine, HybridEngine, JitEngine, PackedEngine, ScalarEngine};
use ryzr_riscv::{build_cpu, programs};

fn main() {
    let circuit = build_cpu(&programs::fib_forever(), 256);
    let engines: Vec<Box<dyn Engine>> = vec![
        Box::new(ScalarEngine::new(&circuit)),
        Box::new(EventEngine::new(&circuit)),
        Box::new(PackedEngine::new(&circuit)),
        Box::new(JitEngine::new(&circuit)),
        Box::new(HybridEngine::new(&circuit)),
        // The wide mode ticks 64 CPUs at once; shown here as its
        // single-CPU (lane 0) rate for an apples-to-apples column.
        Box::new(HybridEngine::wide(&circuit)),
    ];

    for mut engine in engines {
        // Warm up, then time enough ticks for a stable read.
        engine.run(200);
        let mut ticks = 1000u64;
        loop {
            let start = Instant::now();
            engine.run(ticks);
            let dt = start.elapsed();
            if dt.as_millis() >= 500 {
                let ips = ticks as f64 / dt.as_secs_f64();
                println!("{:>8}: {:>10.0} instructions/s (one CPU)", engine.name(), ips);
                break;
            }
            ticks *= 4;
        }
    }
}
