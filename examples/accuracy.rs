//! Estimate the number of distinct words in *The King in Yellow* many times
//! for each of several values of epsilon, and write each estimate to stdout as
//! CSV.
//!
//! ```shell
//! cargo run --release --example accuracy > accuracy.csv
//! uv run analysis/plot_accuracy.py accuracy.csv cvmcount.png
//! ```

use cvmcount::CVM;
use regex::Regex;
use std::collections::HashSet;
use std::io::{self, BufWriter, Write};
use std::num::NonZeroUsize;
use std::thread;

const INPUT: &str = "benches/kiy.txt";
const RUNS: usize = 1000;
const DELTA: f64 = 0.1;
/// From the CLI's suggested starting point down to the first value whose
/// buffer can hold every distinct word in the text.
const EPSILONS: [f64; 6] = [0.8, 0.6, 0.4, 0.3, 0.2, 0.15];

/// Split and clean the text in the same way as the `cvmcount` CLI.
fn tokens(text: &str) -> Vec<String> {
    let re = Regex::new(r"[^\w\s]").expect("pattern is valid");
    text.lines()
        .flat_map(|line| line.split(' '))
        .map(|word| re.replace_all(word, "").to_lowercase())
        .collect()
}

fn estimate(words: &[String], epsilon: f64) -> f64 {
    let mut cvm: CVM<String> = CVM::<String>::builder()
        .epsilon(epsilon)
        .delta(DELTA)
        .estimated_size(words.len())
        .build()
        .expect("parameters are valid");
    cvm.process_stream(words.iter().cloned())
}

fn main() -> io::Result<()> {
    let text = std::fs::read_to_string(INPUT)?;
    let words = tokens(&text);
    let actual = words.iter().collect::<HashSet<_>>().len();

    let jobs: Vec<(f64, usize)> = EPSILONS
        .iter()
        .flat_map(|&epsilon| (0..RUNS).map(move |run| (epsilon, run)))
        .collect();
    let threads = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let estimates: Vec<f64> = thread::scope(|scope| {
        let words = &words;
        let workers: Vec<_> = jobs
            .chunks(jobs.len().div_ceil(threads))
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&(epsilon, _)| estimate(words, epsilon))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().expect("worker thread panicked"))
            .collect()
    });

    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(out, "epsilon,delta,stream_size,actual,run,estimate")?;
    for (&(epsilon, run), estimate) in jobs.iter().zip(estimates) {
        writeln!(
            out,
            "{epsilon},{DELTA},{},{actual},{run},{estimate}",
            words.len()
        )?;
    }
    out.flush()
}
