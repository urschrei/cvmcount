//! An implementation of the CVM fast element counting algorithm presented in
//! Chakraborty, S., Vinodchandran, N. V., & Meel, K. S. (2022). *Distinct Elements in Streams: An Algorithm for the (Text) Book*. 6 pages, 727571 bytes. <https://doi.org/10.4230/LIPIcs.ESA.2022.34>

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rustc_hash::FxHashSet;
use std::hash::Hash;

/// A counter implementing the CVM algorithm
///
/// Note that the CVM struct's buffer takes ownership of its elements.
pub struct CVM<T: PartialEq + Eq + Hash> {
    buf_size: usize,
    buf: FxHashSet<T>,
    probability: f64,
    rng: StdRng,
}

impl<T: PartialEq + Eq + Hash> CVM<T> {
    /// Initialise the algorithm
    ///
    /// epsilon: how close you want your estimate to be to the true number of distinct elements.
    /// A smaller ε means you require a more precise estimate.
    /// For example, ε = 0.05 means you want your estimate to be within 5% of the actual value.
    /// An epsilon of 0.8 is a good starting point for most applications.
    ///
    /// delta: The level of certainty that the algorithm's estimate will fall within the desired accuracy range. A higher confidence
    /// (e.g. 99.9 %) means you're very sure the estimate will be accurate, while a lower confidence (e.g. 90 %) means there's a
    /// higher chance the estimate might be outside the desired range.
    /// A delta of 0.1 is a good starting point for most applications.
    ///
    /// stream_size: this is used to determine buffer size and can be a loose approximation. The closer it is to the stream size,
    /// the more accurate the result will be.
    pub fn new(epsilon: f64, delta: f64, stream_size: usize) -> Self {
        let bufsize = buffer_size(epsilon, delta, stream_size);
        Self {
            buf_size: bufsize,
            buf: FxHashSet::with_capacity_and_hasher(bufsize, Default::default()),
            probability: 1.0,
            rng: StdRng::from_entropy(),
        }
    }
    /// Add an element, potentially updating the unique element count
    pub fn process_element(&mut self, elem: T) {
        // We should switch to a treap (as per Knuth) to avoid the hash overhead, but FxHash
        // is still a lot faster than linear searching a Vec, even at small (1000) buffer sizes
        // Round 0: if an element exists, remove it. Element is added back due to probability 1
        // When buffer is full, remove half the elements
        // Round 1: if an element exists, remove it. Element MAY be added back due to probability 0.5
        if self.buf.contains(&elem) {
            self.buf.remove(&elem);
        }
        if self.rng.gen_bool(self.probability) {
            self.buf.insert(elem);
        }
        while self.buf.len() == self.buf_size {
            self.clear_about_half();
            self.probability /= 2.0;
        }
    }
    // remove around half of the elements at random
    fn clear_about_half(&mut self) {
        self.buf.retain(|_| self.rng.gen_bool(0.5));
    }
    /// Calculate the current unique element count. You can continue to add elements after calling this method.
    pub fn calculate_final_result(&self) -> f64 {
        self.buf.len() as f64 / self.probability
    }
}

// Calculate threshold (buf_size) value for the F0-Estimator algorithm
fn buffer_size(epsilon: f64, delta: f64, stream_size: usize) -> usize {
    ((12.0 / epsilon.powf(2.0)) * ((8.0 * stream_size as f64) / delta).log2()).ceil() as usize
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
        path::Path,
    };

    use super::*;
    use regex::Regex;

    fn open_file<P>(filename: P) -> BufReader<File>
    where
        P: AsRef<Path>,
    {
        let f = File::open(filename).expect("Couldn't read from file");
        BufReader::new(f)
    }

    fn line_to_word(re: &Regex, hs: &mut FxHashSet<String>, line: &str) {
        let words = line.split(' ');
        words.for_each(|word| {
            let clean_word = re.replace_all(word, "").to_lowercase();
            hs.insert(clean_word);
        })
    }
    #[test]
    fn actual() {
        let input_file = "benches/kiy.txt";
        let re = Regex::new(r"[^\w\s]").unwrap();
        let br = open_file(input_file);
        let mut hs = FxHashSet::with_hasher(Default::default());
        br.lines()
            .for_each(|line| line_to_word(&re, &mut hs, &line.unwrap()));
        assert_eq!(hs.len(), 9016)
    }
}
