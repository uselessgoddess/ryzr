set dotenv-load
set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

test_runner := "cargo nextest run"

default:
    @just --list

check:
    cargo check --workspace --all-targets

lint *args:
    cargo clippy --workspace --all-targets {{ args }} -- -D warnings

fmt:
    cargo +nightly fmt --all

pre: fmt lint test

test *args:
    {{ test_runner }} --workspace --all-targets {{ args }}

bench *args:
    cargo bench --workspace {{ args }}

build *args: fmt
    cargo build {{ args }}
