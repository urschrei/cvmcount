//! An implementation of the CVM fast token counting algorithm presented in
//! Chakraborty, S., Vinodchandran, N. V., & Meel, K. S. (2022). *Distinct Elements in Streams: An Algorithm for the (Text) Book*. 6 pages, 727571 bytes. https://doi.org/10.4230/LIPIcs.ESA.2022.34

use rand::rngs::ThreadRng;
use rand::Rng;
use regex::Regex;

pub struct CVM {
    buf_size: usize,
    buf: Vec<String>,
    probability: f64,
    rng: ThreadRng,
    re: Regex,
}
impl CVM {
    /// Initialise the algorithm
    ///
    /// epsilon: how close you want your estimate to be to the true number of distinct elements.
    /// A smaller ε means you require a more precise estimate.
    /// For example, ε = 0.05 means you want your estimate to be within 5% of the actual value.
    /// An epsilon of 0.8 is a good starting point for most applications.
    ///
    /// delta: The level of certainty that the algorithm's estimate will fall within the desired accuracy range. A higher confidence
    /// (e.g., 99.9 %) means you're very sure the estimate will be accurate, while a lower confidence (e.g., 90 %) means there's a
    /// higher chance the estimate might be outside the desired range.
    /// A delta of 0.1 is a good starting point for most applications.
    ///
    /// stream_size: this is used to determine buffer size and can be a loose approximation. The closer it is to the stream size,
    /// the more accurate the results
    pub fn new(epsilon: f64, delta: f64, stream_size: usize) -> Self {
        let bufsize = buffer_size(epsilon, delta, stream_size);
        Self {
            buf_size: bufsize,
            buf: Vec::with_capacity(bufsize),
            probability: 1.0,
            rng: rand::thread_rng(),
            re: Regex::new(r"[^\w\s]").unwrap(),
        }
    }
    /// Count tokens, given a string containing words, e.g. a line of a book
    pub fn process_line_tokens(&mut self, line: String) {
        let words = line.split(' ');
        for word in words {
            let clean_word = self.re.replace_all(word, "").to_lowercase();
            // binary search should be pretty fast
            // I think this will be faster than a hashset for practical sizes
            // but I need some empirical data for this
            if let Some(pos) = self.buf.iter().position(|x| *x == clean_word) {
                self.buf.remove(pos);
            }
            if self.rng.gen_bool(self.probability) {
                self.buf.push(clean_word);
            }
            if self.buf.len() == self.buf_size {
                self.clear_about_half();
                self.probability /= 2.0;
                if self.buf.len() == self.buf_size {
                    panic!("Something has gone proper wrong")
                }
            }
        }
    }
    // remove around half of the elements at random
    fn clear_about_half(&mut self) {
        self.buf.retain(|_| self.rng.gen_bool(0.5));
    }
    /// Calculate the final token count
    pub fn calculate_final_result(&self) -> f64 {
        self.buf.len() as f64 / self.probability
    }
}

// Calculate threshold (buf_size) value for the F0-Estimator algorithm
fn buffer_size(epsilon: f64, delta: f64, stream_size: usize) -> usize {
    ((12.0 / epsilon.powf(2.0)) * ((8.0 * stream_size as f64) / delta).log2()).ceil() as usize
}
