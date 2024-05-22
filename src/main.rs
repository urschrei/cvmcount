use clap::{arg, crate_version, value_parser, Command};
use std::io::BufRead;
use std::path::PathBuf;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use cvmcount::CVM;

pub fn open_file<P>(filename: P) -> BufReader<File>
where
    P: AsRef<Path>,
{
    // no need to use a BufReader since we want the entire file
    let f = File::open(filename).expect("Couldn't read from file");
    BufReader::new(f)
}

fn main() {
    // Generate a CLI, and get input filename to process
    let params = Command::new("CVM")
        .version(crate_version!())
        .author("Stephan Hügel <urschrei@gmail.com>")
        .about("Use the CVM algorithm to estimate the number of unique tokens in a stream")
        .arg(arg!(-t --tokens <FILE> "A text file containing words").index(1).required(true).value_parser(value_parser!(PathBuf)))
        .arg(arg!(-e --epsilon <EPSILON> "How close you want your estimate to be to the true number of distinct tokens. A smaller ε means you require a more precise estimate. For example, ε = 0.05 means you want your estimate to be within 5 % of the actual value. An epsilon of 0.8 is a good starting point for most applications").index(2).required(true).value_parser(value_parser!(f64)))
        .arg(arg!(-d --delta <DELTA> "The level of certainty that the algorithm's estimate will fall within the desired accuracy range. A higher confidence (e.g., 99.9 %) means you're very sure the estimate will be accurate, while a lower confidence (e.g., 90 %) means there's a higher chance the estimate might be outside the desired range. A delta of 0.1 is a good starting point for most applications").index(3).required(true).value_parser(value_parser!(f64)))
        .arg(arg!(-s --streamsize <STREAM_SIZE> "This is used to determine buffer size and can be a loose approximation. The closer it is to the stream size, the more accurate the results").index(4).required(true).value_parser(value_parser!(usize)))
        .get_matches();
    let input_file = params.get_one::<PathBuf>("tokens").unwrap();
    let epsilon = params.get_one::<f64>("epsilon").unwrap();
    let delta = params.get_one::<f64>("delta").unwrap();
    let stream_size = params.get_one::<usize>("streamsize").unwrap();

    let mut counter = CVM::new(*epsilon, *delta, *stream_size);
    let br = open_file(input_file);
    for line in br.lines() {
        counter.process_line_tokens(line.unwrap())
    }
    println!(
        "Unique tokens: {:?}",
        counter.calculate_final_result() as i32
    );
}
