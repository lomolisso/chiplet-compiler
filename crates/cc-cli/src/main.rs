use std::fs::File;
use std::process::ExitCode;
use std::sync::Mutex;

use clap::Parser;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

fn init_tracing(cli: &cc_cli::Cli) {
    let level = match cli.verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter = || EnvFilter::try_from_env("CHIPC_LOG").unwrap_or_else(|_| EnvFilter::new(level));
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_filter(filter());
    // `run` also logs structured JSON into build/.pipeline/logs/.
    let json = cc_cli::run_log_path(cli).and_then(|p| {
        std::fs::create_dir_all(p.parent()?).ok()?;
        let f = File::create(&p).ok()?;
        Some(
            tracing_subscriber::fmt::layer()
                .json()
                .with_writer(Mutex::new(f))
                .with_filter(filter()),
        )
    });
    tracing_subscriber::registry().with(stderr).with(json).init();
}

fn main() -> ExitCode {
    let cli = cc_cli::Cli::parse();
    init_tracing(&cli);
    let mut stdout = std::io::stdout();
    match cc_cli::execute(&cli, cc_passes::registry(), &mut stdout) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
