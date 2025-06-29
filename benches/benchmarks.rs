#[macro_use]
extern crate criterion;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use criterion::Criterion;
use cvmcount::CVM;
use rand::{Rng, thread_rng};
use regex::Regex;

use std::collections::HashSet;

// generate 1 million 7-digit random positive integers
fn generate_random_numbers() -> Vec<i32> {
    let mut rng = thread_rng();

    (0..1_000_000)
        .map(|_| rng.gen_range(1_000_000..10_000_000))
        .collect()
}

fn open_file<P>(filename: P) -> BufReader<File>
where
    P: AsRef<Path>,
{
    let f = File::open(filename).expect("Couldn't read from file");
    BufReader::new(f)
}

fn line_to_word(re: &Regex, cvm: &mut CVM<String>, line: &str) {
    let words = line.split(' ');
    words.for_each(|word| {
        let clean_word = re.replace_all(word, "").to_lowercase();
        cvm.process_element(clean_word)
    })
}

#[allow(unused_must_use)]
fn bench_count_strings_integers(c: &mut Criterion) {
    c.bench_function(
        "Count unique strings in The King in Yellow with regex regularization: e = 0.8, d = 0.1, s = 1000",
        |b| {
            let input_file = "benches/kiy.txt";
            let epsilon = 0.8;
            let delta = 0.1;
            let stream_size = 1000;
            let re = Regex::new(r"[^\w\s]").unwrap();
            b.iter(|| {
                let mut string_counter: CVM<String> = CVM::new(epsilon, delta, stream_size);
                let br = open_file(input_file);
                br.lines()
                    .for_each(|line| line_to_word(&re, &mut string_counter, &line.unwrap()));
                string_counter.calculate_final_result()
            })
        },
    );
    c.bench_function(
        "Count uniques in ten million 7-digit random positive integers: e = 0.8, d = 0.1, s = 1000",
        |b| {
            let epsilon = 0.8;
            let delta = 0.1;
            let stream_size = 1000;
            let digits = generate_random_numbers();
            b.iter(|| {
                let mut int_counter: CVM<i32> = CVM::new(epsilon, delta, stream_size);
                digits
                    .iter()
                    .for_each(|integer| int_counter.process_element(*integer));
                int_counter.calculate_final_result()
            })
        },
    );
    c.bench_function(
        "Count uniques in ten million 7-digit random positive integers: HashSet",
        |b| {
            let digits = generate_random_numbers();
            b.iter(|| {
                let mut hs = HashSet::new();
                digits.iter().for_each(|digit| {
                    hs.insert(digit);
                });
                digits.len()
            })
        },
    );
}

criterion_group!(benches, bench_count_strings_integers,);
criterion_main!(benches);
