//! An implementation of the CVM fast element counting algorithm presented in
//! Chakraborty, S., Vinodchandran, N. V., & Meel, K. S. (2022). *Distinct Elements in Streams: An Algorithm for the (Text) Book*. 6 pages, 727571 bytes. <https://doi.org/10.4230/LIPIcs.ESA.2022.34>
//!
//! This implementation uses a treap data structure as the buffer, following Knuth's original design.

mod treap;

use crate::treap::Treap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A counter implementing the CVM algorithm
///
/// This implementation uses a treap (randomized binary search tree) as the buffer,
/// which provides `O(log n)` operations while maintaining the probabilistic properties
/// needed for the algorithm.
///
/// Note that the CVM struct's buffer takes ownership of its elements.
pub struct CVM<T: Ord> {
    buf_size: usize,
    buf: Treap<T>,
    probability: f64,
    rng: StdRng,
}

impl<T: Ord> CVM<T> {
    /// Initialise the algorithm
    ///
    /// `epsilon`: how close you want your estimate to be to the true number of distinct elements.
    /// A smaller `ε` means you require a more precise estimate.
    /// For example, `ε = 0.05` means you want your estimate to be within 5 % of the actual value.
    /// An epsilon of `0.8` is a good starting point for most applications.
    ///
    /// `delta`: The level of certainty that the algorithm's estimate will fall within the desired accuracy range. A higher confidence
    /// (e.g. 99.9 %) means you're very sure the estimate will be accurate, while a lower confidence (e.g. 90 %) means there's a
    /// higher chance the estimate might be outside the desired range.
    /// A `delta` of `0.1` is a good starting point for most applications.
    ///
    /// `stream_size`: this is used to determine buffer size and can be a loose approximation. The closer it is to the stream size,
    /// the more accurate the result will be.
    pub fn new(epsilon: f64, delta: f64, stream_size: usize) -> Self {
        let bufsize = buffer_size(epsilon, delta, stream_size);
        Self {
            buf_size: bufsize,
            buf: Treap::new(),
            probability: 1.0,
            rng: StdRng::from_entropy(),
        }
    }
    /// Add an element, potentially updating the unique element count
    pub fn process_element(&mut self, elem: T) {
        // The algorithm works as follows:
        // 1. If element exists in buffer, remove it (this ensures proper sampling)
        // 2. Add element back with current probability
        // 3. If buffer is full, remove ~half the elements and halve the probability
        // This creates a geometric sampling scheme that provides an unbiased estimate
        if self.buf.contains(&elem) {
            self.buf.remove(&elem);
        }
        if self.rng.gen_bool(self.probability) {
            self.buf.insert(elem, &mut self.rng);
        }
        while self.buf.len() == self.buf_size {
            self.clear_about_half();
            self.probability /= 2.0;
        }
    }
    // remove around half of the elements at random
    fn clear_about_half(&mut self) {
        // Need to capture rng reference to use in closure
        let rng = &mut self.rng;
        self.buf.retain(|_| rng.gen_bool(0.5));
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

    use regex::Regex;
    use std::collections::HashSet;

    fn open_file<P>(filename: P) -> BufReader<File>
    where
        P: AsRef<Path>,
    {
        let f = File::open(filename).expect("Couldn't read from file");
        BufReader::new(f)
    }

    fn line_to_word(re: &Regex, hs: &mut HashSet<String>, line: &str) {
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
        let mut hs = HashSet::new();
        br.lines()
            .for_each(|line| line_to_word(&re, &mut hs, &line.unwrap()));
        assert_eq!(hs.len(), 9016)
    }
}
