//! Deterministic Doom-like RV32I workload for comparing `ryzr` and VCB.
//!
//! The benchmark program is intentionally plain RV32I: no host display
//! shortcuts, just frame/ray/step loops with a reproducible checksum in
//! registers. `--emit-vcbmem` writes the exact instruction stream in the byte
//! order used by the ViPeR/VCB toolchain.

use std::{
    env,
    error::Error,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use ryzr_backend::{
    Engine, EventEngine, HybridEngine, PackedEngine, PackedJitEngine, ScalarEngine, ThreadedEngine,
};
use ryzr_core::Circuit;
use ryzr_riscv::{Emulator, build_cpu, programs};

const DEFAULT_FRAMES: i32 = 180;
const DEFAULT_RAM_WORDS: usize = 256;
const DEFAULT_MIN_MS: u64 = 500;
const VIPER_CLOCK_TICKS_PER_INSTR: u64 = 7;

struct Config {
    frames: i32,
    ram_words: usize,
    min_duration: Duration,
    emit_vcbmem: Option<PathBuf>,
}

struct Completion {
    retired: u64,
    checksum: u32,
    frames: u32,
    rays: u32,
    hits: u32,
}

struct Measurement {
    engine: &'static str,
    instructions: u64,
    elapsed: Duration,
    frames: u32,
    checksum: u32,
    rays: u32,
    hits: u32,
    done: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(config) = parse_args()? else {
        print_help();
        return Ok(());
    };

    let program = programs::doom_like_benchmark(config.frames);
    if let Some(path) = &config.emit_vcbmem {
        write_vcbmem(path, &program)?;
        println!("wrote VCB VMEM: {}", path.display());
    }

    let completion = run_to_completion(&program, config.ram_words)?;
    println!("program words: {}", program.len());
    println!("target frames: {}", config.frames);
    println!("retired to checksum: {}", completion.retired);
    println!("checksum: 0x{:08x}", completion.checksum);
    println!(
        "counters: frames={} rays={} hits={}",
        completion.frames, completion.rays, completion.hits
    );
    println!(
        "ViPeR/VCB nominal clock cost: {} VCB ticks/instruction, {} VCB ticks to same checksum",
        VIPER_CLOCK_TICKS_PER_INSTR,
        completion.retired * VIPER_CLOCK_TICKS_PER_INSTR
    );
    println!();

    let circuit = build_cpu(&program, config.ram_words);
    for measurement in measure_engines(&circuit, config.min_duration) {
        let secs = measurement.elapsed.as_secs_f64();
        let ips = measurement.instructions as f64 / secs;
        let fps = measurement.frames as f64 / secs;
        println!(
            "{:>10}: {:>10.0} instr/s, {:>8.2} frames/s, checksum=0x{:08x}, rays={}, hits={}, done={}",
            measurement.engine,
            ips,
            fps,
            measurement.checksum,
            measurement.rays,
            measurement.hits,
            measurement.done
        );
    }

    Ok(())
}

fn parse_args() -> Result<Option<Config>, Box<dyn Error>> {
    let mut frames = DEFAULT_FRAMES;
    let mut ram_words = DEFAULT_RAM_WORDS;
    let mut min_ms = DEFAULT_MIN_MS;
    let mut emit_vcbmem = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--frames" => frames = parse_value("--frames", args.next())?,
            "--ram-words" => ram_words = parse_value("--ram-words", args.next())?,
            "--min-ms" => min_ms = parse_value("--min-ms", args.next())?,
            "--emit-vcbmem" => {
                emit_vcbmem = Some(PathBuf::from(required_value("--emit-vcbmem", args.next())?))
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Some(Config { frames, ram_words, min_duration: Duration::from_millis(min_ms), emit_vcbmem }))
}

fn parse_value<T: core::str::FromStr>(
    name: &'static str,
    value: Option<String>,
) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    Ok(required_value(name, value)?.parse()?)
}

fn required_value(name: &'static str, value: Option<String>) -> Result<String, Box<dyn Error>> {
    value.ok_or_else(|| format!("{name} needs a value").into())
}

fn print_help() {
    println!(
        "\
Usage: cargo run -p ryzr-riscv --release --example doom_bench -- [options]

Options:
  --frames N          benchmark frames to render before the done flag (default {DEFAULT_FRAMES})
  --ram-words N       gate CPU RAM words, power of two (default {DEFAULT_RAM_WORDS})
  --min-ms N          minimum timing window per engine (default {DEFAULT_MIN_MS})
  --emit-vcbmem PATH  write a ViPeR/VCB-compatible VMEM image
"
    );
}

fn write_vcbmem(path: &PathBuf, program: &[u32]) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(program.len() * 4);
    for word in program {
        bytes.extend_from_slice(&word.to_be_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn run_to_completion(program: &[u32], ram_words: usize) -> Result<Completion, Box<dyn Error>> {
    let mut emu = Emulator::new(program, ram_words);
    for retired in 0..20_000_000u64 {
        if emu.regs[17] != 0 {
            return Ok(Completion {
                retired,
                checksum: emu.regs[10],
                frames: emu.regs[11],
                rays: emu.regs[12],
                hits: emu.regs[13],
            });
        }
        emu.step();
    }
    Err("program did not reach the done flag within 20M instructions".into())
}

fn measure_engines(circuit: &Circuit, min_duration: Duration) -> Vec<Measurement> {
    engines(circuit)
        .into_iter()
        .map(|mut engine| {
            engine.run(512);
            let start_frames = read_reg(engine.as_ref(), 11);
            let start_rays = read_reg(engine.as_ref(), 12);
            let start_hits = read_reg(engine.as_ref(), 13);
            let start = Instant::now();
            let mut instructions = 0;
            while start.elapsed() < min_duration {
                engine.run(2048);
                instructions += 2048;
            }
            let elapsed = start.elapsed();
            Measurement {
                engine: engine.name(),
                instructions,
                elapsed,
                frames: read_reg(engine.as_ref(), 11).saturating_sub(start_frames),
                checksum: read_reg(engine.as_ref(), 10),
                rays: read_reg(engine.as_ref(), 12).saturating_sub(start_rays),
                hits: read_reg(engine.as_ref(), 13).saturating_sub(start_hits),
                done: read_reg(engine.as_ref(), 17) != 0,
            }
        })
        .collect()
}

fn engines(circuit: &Circuit) -> Vec<Box<dyn Engine>> {
    vec![
        Box::new(ScalarEngine::new(circuit)),
        Box::new(EventEngine::new(circuit)),
        Box::new(PackedEngine::new(circuit)),
        Box::new(PackedJitEngine::new(circuit)),
        Box::new(ThreadedEngine::new(circuit)),
        Box::new(HybridEngine::new(circuit)),
    ]
}

fn read_reg(engine: &dyn Engine, register: usize) -> u32 {
    read_word(engine, 32 * (register + 1))
}

fn read_word(engine: &dyn Engine, base: usize) -> u32 {
    (0..32).map(|i| u32::from(engine.output(base + i)) << i).sum()
}
