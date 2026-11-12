use clap::{Command, arg, crate_version, value_parser};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::path::PathBuf;

use cvmcount::CVM;

mod input;

/// Open `path` for reading. The path `-` is standard input.
fn open_input(path: &Path) -> io::Result<Box<dyn BufRead>> {
    if path == Path::new("-") {
        Ok(Box::new(io::stdin().lock()))
    } else {
        Ok(Box::new(BufReader::new(File::open(path)?)))
    }
}

fn main() {
    // Generate a CLI, and get input filename to process
    let params = Command::new("CVM")
        .version(crate_version!())
        .author("Stephan Hügel <urschrei@gmail.com>")
        .about("Use the CVM algorithm to estimate the number of unique tokens in a stream")
        .arg(arg!(-t --tokens <FILE> "A text file containing words, or - to read from standard input")
            .required(true)
            .value_parser(value_parser!(PathBuf)))
        .arg(arg!(-e --epsilon <EPSILON> "How close you want your estimate to be to the true number of distinct tokens. A smaller ε means you require a more precise estimate. For example, ε = 0.05 means you want your estimate to be within 5 % of the actual value. An epsilon of 0.8 is a good starting point for most applications")
            .required(true)
            .value_parser(clap::value_parser!(f64))
            )
        .arg(arg!(-d --delta <DELTA> "The probability that the algorithm's estimate will fall outside the desired accuracy range; your confidence is 1 - delta. A delta of 0.001 means you're 99.9 % sure the estimate will be accurate, while a delta of 0.1 (90 % confidence) means there's a higher chance the estimate might be outside the desired range. A delta of 0.1 is a good starting point for most applications")
            .required(true)
            .value_parser(value_parser!(f64)))
        .arg(arg!(-s --streamsize <STREAM_SIZE> "This is used to determine buffer size and can be a loose approximation. The closer it is to the stream size, the more accurate the results")
            .required(true)
            .value_parser(value_parser!(usize)))
        .get_matches();

    let input_file = params.get_one::<PathBuf>("tokens").unwrap();
    let epsilon = params.get_one::<f64>("epsilon").unwrap();
    let delta = params.get_one::<f64>("delta").unwrap();
    let stream_size = params.get_one::<usize>("streamsize").unwrap();
    let mut counter: CVM<String> = match CVM::<String>::builder()
        .epsilon(*epsilon)
        .delta(*delta)
        .estimated_size(*stream_size)
        .build()
    {
        Ok(counter) => counter,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = open_input(input_file)
        .and_then(|br| input::for_each_word(br, |word| counter.process_element(word)))
    {
        eprintln!("error: {}: {e}", input_file.display());
        std::process::exit(1);
    }
    println!(
        "Unique tokens: {:?}",
        counter.calculate_final_result() as i32
    );
}
