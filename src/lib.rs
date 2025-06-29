//! An implementation of the CVM fast element counting algorithm presented in
//! Chakraborty, S., Vinodchandran, N. V., & Meel, K. S. (2022). *Distinct Elements in Streams: An Algorithm for the (Text) Book*. 6 pages, 727571 bytes. <https://doi.org/10.4230/LIPIcs.ESA.2022.34>
//!
//! This implementation uses a treap data structure as the buffer, following Knuth's original design.

mod treap;

use crate::treap::Treap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Specification for confidence level in the CVM algorithm
#[derive(Debug, Clone, Copy)]
pub enum ConfidenceSpec {
    /// Specify delta directly (probability of failure)
    Delta(f64),
    /// Specify confidence level (probability of success)
    Confidence(f64),
}

impl ConfidenceSpec {
    /// Convert to delta value for internal use
    fn to_delta(self) -> f64 {
        match self {
            ConfidenceSpec::Delta(delta) => delta,
            ConfidenceSpec::Confidence(confidence) => 1.0 - confidence,
        }
    }

    /// Validate the confidence specification
    fn validate(self) -> Result<Self, String> {
        match self {
            ConfidenceSpec::Delta(delta) => {
                if delta <= 0.0 || delta >= 1.0 {
                    Err("Delta must be between 0.0 and 1.0 (exclusive)".to_string())
                } else {
                    Ok(self)
                }
            }
            ConfidenceSpec::Confidence(confidence) => {
                if confidence <= 0.0 || confidence >= 1.0 {
                    Err("Confidence must be between 0.0 and 1.0 (exclusive)".to_string())
                } else {
                    Ok(self)
                }
            }
        }
    }
}

/// Builder for constructing CVM instances with validation and defaults
///
/// # Examples
///
/// ```
/// use cvmcount::CVM;
///
/// // Using defaults (`epsilon=0.8`, `confidence=0.9`, `size=1000`)
/// let cvm: CVM<String> = CVM::<String>::builder().build().unwrap();
///
/// // Custom parameters
/// let cvm: CVM<i32> = CVM::<i32>::builder()
///     .epsilon(0.05)  // 5 % accuracy
///     .confidence(0.99)  // 99 % confidence
///     .estimated_size(10_000)
///     .build()
///     .unwrap();
///
/// // Using delta instead of confidence
/// let cvm: CVM<String> = CVM::<String>::builder()
///     .epsilon(0.1)
///     .delta(0.01)  // 1 % failure probability
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Default)]
pub struct CVMBuilder {
    epsilon: Option<f64>,
    confidence_spec: Option<ConfidenceSpec>,
    stream_size: Option<usize>,
}

impl CVMBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the epsilon parameter (accuracy requirement)
    ///
    /// `Epsilon` determines how close you want your estimate to be to the true number
    /// of distinct elements. A smaller `ε` means you require a more precise estimate.
    /// For example, `ε = 0.05` means you want your estimate to be within 5 % of the
    /// actual value.
    ///
    /// Must be between 0.0 and 1.0 (exclusive).
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    /// Set the confidence level (probability that the estimate will be accurate)
    ///
    /// Confidence represents how certain you want to be that the algorithm's
    /// estimate will fall within the desired accuracy range. For example,
    /// `confidence = 0.99` means you're 99 % sure the estimate will be accurate.
    ///
    /// Must be between 0.0 and 1.0 (exclusive).
    /// Cannot be used together with [`Self::delta`] – the last one called will be used.
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence_spec = Some(ConfidenceSpec::Confidence(confidence));
        self
    }

    /// Set the delta parameter (probability of failure)
    ///
    /// Delta represents the probability that the algorithm's estimate will fall
    /// outside the desired accuracy range. For example, `delta = 0.01` means there's
    /// a 1 % chance the estimate will be inaccurate.
    ///
    /// Must be between 0.0 and 1.0 (exclusive).
    /// Cannot be used together with [`Self::confidence()`] – the last one called will be used.
    pub fn delta(mut self, delta: f64) -> Self {
        self.confidence_spec = Some(ConfidenceSpec::Delta(delta));
        self
    }

    /// Set the estimated stream size
    ///
    /// This is used to determine buffer size and can be a loose approximation.
    /// The closer it is to the actual stream size, the more accurate the results
    /// will be.
    pub fn estimated_size(mut self, size: usize) -> Self {
        self.stream_size = Some(size);
        self
    }

    /// Build the CVM instance with validation
    ///
    /// Uses the following defaults if not specified:
    /// - `epsilon: 0.8` (good starting point for most applications)
    /// - `confidence: 0.9` (90 % confidence, equivalent to delta = 0.1)
    /// - `estimated_size: 1000`
    ///
    /// Returns an error if any parameters are invalid.
    pub fn build<T: Ord>(self) -> Result<CVM<T>, String> {
        // Validate and get epsilon
        let epsilon = self.epsilon.unwrap_or(0.8);
        if epsilon <= 0.0 || epsilon >= 1.0 {
            return Err("Epsilon must be between 0.0 and 1.0 (exclusive)".to_string());
        }

        // Validate and get delta
        let confidence_spec = self
            .confidence_spec
            .unwrap_or(ConfidenceSpec::Confidence(0.9));
        let validated_spec = confidence_spec.validate()?;
        let delta = validated_spec.to_delta();

        // Validate and get stream size
        let stream_size = self.stream_size.unwrap_or(1000);
        if stream_size == 0 {
            return Err("Stream size must be greater than 0".to_string());
        }

        Ok(CVM::new(epsilon, delta, stream_size))
    }
}

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
    /// Create a new builder for constructing CVM instances
    ///
    /// The builder provides a more ergonomic way to construct CVM instances with
    /// validation and sensible defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvmcount::CVM;
    ///
    /// // Using defaults
    /// let cvm: CVM<String> = CVM::<String>::builder().build().unwrap();
    ///
    /// // Custom configuration
    /// let cvm: CVM<i32> = CVM::<i32>::builder()
    ///     .epsilon(0.05)
    ///     .confidence(0.99)
    ///     .estimated_size(10_000)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> CVMBuilder {
        CVMBuilder::new()
    }

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
    /// Process an entire iterator of owned values and return the final estimate
    ///
    /// This is a convenience method that processes all elements from an iterator
    /// and returns the final count estimate. The iterator must yield owned values
    /// that the CVM can take ownership of.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvmcount::CVM;
    ///
    /// // Process owned strings
    /// let words = vec!["hello".to_string(), "world".to_string(), "hello".to_string()];
    /// let mut cvm: CVM<String> = CVM::<String>::builder().build().unwrap();
    /// let estimate = cvm.process_stream(words);
    /// assert!(estimate > 0.0);
    ///
    /// // Process numeric data
    /// let numbers = vec![1, 2, 3, 2, 1, 4];
    /// let mut cvm: CVM<i32> = CVM::<i32>::builder().build().unwrap();
    /// let estimate = cvm.process_stream(numbers);
    /// assert!(estimate > 0.0);
    ///
    /// // When you have borrowed data, clone explicitly
    /// let borrowed_words = vec!["hello", "world", "hello"];
    /// let mut cvm: CVM<String> = CVM::<String>::builder().build().unwrap();
    /// let estimate = cvm.process_stream(borrowed_words.iter().map(|s| s.to_string()));
    /// assert!(estimate > 0.0);
    /// ```
    pub fn process_stream<I>(&mut self, iter: I) -> f64
    where
        I: IntoIterator<Item = T>,
    {
        for item in iter {
            self.process_element(item);
        }
        self.calculate_final_result()
    }

    /// Calculate the current unique element count. You can continue to add elements after calling this method.
    pub fn calculate_final_result(&self) -> f64 {
        self.buf.len() as f64 / self.probability
    }
}

/// Extension trait for iterators to estimate distinct count directly
///
/// This trait provides convenient methods to estimate distinct counts from iterators
/// without manually creating and managing a CVM instance.
///
/// # Examples
///
/// ```
/// use cvmcount::{CVM, EstimateDistinct};
///
/// // Simple usage with default parameters
/// let numbers = vec![1, 2, 3, 2, 1, 4, 5];
/// let estimate = numbers.into_iter().estimate_distinct_count(0.1, 0.1, 1000);
/// assert!(estimate > 0.0);
///
/// // Using builder pattern for more control
/// let words = vec!["hello".to_string(), "world".to_string(), "hello".to_string()];
/// let builder = CVM::<String>::builder().epsilon(0.05).confidence(0.99);
/// let estimate = words.into_iter().estimate_distinct_with_builder(builder).unwrap();
/// assert!(estimate > 0.0);
/// ```
pub trait EstimateDistinct<T: Ord>: Iterator<Item = T> + Sized {
    /// Estimate distinct count using the CVM algorithm with specified parameters
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Accuracy requirement (smaller = more accurate)
    /// * `delta` - Failure probability (smaller = more confident)
    /// * `estimated_size` - Rough estimate of total stream size
    ///
    /// # Returns
    ///
    /// The estimated number of distinct elements
    fn estimate_distinct_count(self, epsilon: f64, delta: f64, estimated_size: usize) -> f64 {
        let mut cvm = CVM::new(epsilon, delta, estimated_size);
        cvm.process_stream(self)
    }

    /// Estimate distinct count using a builder for more ergonomic configuration
    ///
    /// # Arguments
    ///
    /// * `builder` - A configured CVMBuilder instance
    ///
    /// # Returns
    ///
    /// Result containing the estimated number of distinct elements or an error message
    ///
    /// # Examples
    ///
    /// ```
    /// use cvmcount::{CVM, EstimateDistinct};
    ///
    /// let data = vec![1, 2, 3, 2, 1];
    /// let builder = CVM::<i32>::builder().epsilon(0.05).confidence(0.99);
    /// let estimate = data.into_iter().estimate_distinct_with_builder(builder).unwrap();
    /// assert!(estimate > 0.0);
    /// ```
    fn estimate_distinct_with_builder(self, builder: CVMBuilder) -> Result<f64, String> {
        let mut cvm: CVM<T> = builder.build()?;
        Ok(cvm.process_stream(self))
    }
}

/// Implement EstimateDistinct for all iterators that yield Ord types
impl<T: Ord, I: Iterator<Item = T>> EstimateDistinct<T> for I {}

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

    use super::{CVM, ConfidenceSpec, EstimateDistinct};
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

    #[test]
    fn test_builder_defaults() {
        let cvm: CVM<String> = CVM::<String>::builder().build().unwrap();
        // Verify that it's properly constructed with defaults
        assert_eq!(cvm.calculate_final_result(), 0.0); // Empty buffer
    }

    #[test]
    fn test_builder_custom_params() {
        let cvm: CVM<i32> = CVM::<i32>::builder()
            .epsilon(0.05)
            .confidence(0.99)
            .estimated_size(5000)
            .build()
            .unwrap();

        // Test that it works by processing some elements
        let mut cvm = cvm;
        for i in 0..100 {
            cvm.process_element(i);
        }
        let result = cvm.calculate_final_result();
        assert!(result > 0.0);
    }

    #[test]
    fn test_builder_delta_vs_confidence() {
        // Test that confidence and delta give equivalent results
        let cvm1: CVM<i32> = CVM::<i32>::builder().confidence(0.9).build().unwrap();

        let cvm2: CVM<i32> = CVM::<i32>::builder().delta(0.1).build().unwrap();

        // They should have the same internal configuration
        // (we can't directly test this without exposing internals,
        // but we can test they both work)
        assert_eq!(cvm1.calculate_final_result(), 0.0);
        assert_eq!(cvm2.calculate_final_result(), 0.0);
    }

    #[test]
    fn test_builder_last_wins() {
        // Test that the last confidence/delta setting wins
        let cvm: CVM<i32> = CVM::<i32>::builder()
            .confidence(0.9)
            .delta(0.05) // This should override confidence
            .build()
            .unwrap();

        assert_eq!(cvm.calculate_final_result(), 0.0);
    }

    #[test]
    fn test_builder_validation() {
        // Test epsilon validation
        let result = CVM::<i32>::builder().epsilon(0.0).build::<i32>();
        assert!(result.is_err());

        let result = CVM::<i32>::builder().epsilon(1.0).build::<i32>();
        assert!(result.is_err());

        let result = CVM::<i32>::builder().epsilon(-0.5).build::<i32>();
        assert!(result.is_err());

        // Test confidence validation
        let result = CVM::<i32>::builder().confidence(0.0).build::<i32>();
        assert!(result.is_err());

        let result = CVM::<i32>::builder().confidence(1.0).build::<i32>();
        assert!(result.is_err());

        // Test delta validation
        let result = CVM::<i32>::builder().delta(0.0).build::<i32>();
        assert!(result.is_err());

        let result = CVM::<i32>::builder().delta(1.0).build::<i32>();
        assert!(result.is_err());

        // Test stream size validation
        let result = CVM::<i32>::builder().estimated_size(0).build::<i32>();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_method_chaining() {
        let result = CVM::<String>::builder()
            .epsilon(0.1)
            .confidence(0.95)
            .estimated_size(2000)
            .build::<String>();

        assert!(result.is_ok());
    }

    #[test]
    fn test_confidence_spec_conversion() {
        // Test ConfidenceSpec::to_delta conversion
        let confidence_spec = ConfidenceSpec::Confidence(0.9);
        assert!((confidence_spec.to_delta() - 0.1).abs() < f64::EPSILON);

        let delta_spec = ConfidenceSpec::Delta(0.05);
        assert!((delta_spec.to_delta() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_process_stream() {
        let mut cvm: CVM<i32> = CVM::<i32>::builder().build().unwrap();

        // Test with vector
        let numbers = vec![1, 2, 3, 2, 1, 4, 5, 3];
        let estimate = cvm.process_stream(numbers);
        assert!(estimate > 0.0);

        // Test with range
        let mut cvm2: CVM<i32> = CVM::<i32>::builder().build().unwrap();
        let estimate2 = cvm2.process_stream(1..=100);
        assert!(estimate2 > 0.0);
    }

    #[test]
    fn test_process_stream_strings() {
        let mut cvm: CVM<String> = CVM::<String>::builder().build().unwrap();

        // Test with owned strings
        let words = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(),
            "rust".to_string(),
        ];
        let estimate = cvm.process_stream(words);
        assert!(estimate > 0.0);
    }

    #[test]
    fn test_process_stream_with_map() {
        let mut cvm: CVM<String> = CVM::<String>::builder().build().unwrap();

        // Test with borrowed data mapped to owned
        let borrowed_words = ["hello", "world", "hello", "rust"];
        let estimate = cvm.process_stream(borrowed_words.iter().map(|s| s.to_string()));
        assert!(estimate > 0.0);
    }

    #[test]
    fn test_estimate_distinct_trait() {
        // Test simple usage
        let numbers = vec![1, 2, 3, 2, 1, 4, 5];
        let estimate = numbers.into_iter().estimate_distinct_count(0.1, 0.1, 1000);
        assert!(estimate > 0.0);

        // Test with builder
        let words = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(),
        ];
        let builder = CVM::<String>::builder().epsilon(0.05).confidence(0.99);
        let estimate = words
            .into_iter()
            .estimate_distinct_with_builder(builder)
            .unwrap();
        assert!(estimate > 0.0);
    }

    #[test]
    fn test_estimate_distinct_with_cloning() {
        // Test that explicit cloning works as expected
        let borrowed_numbers = [1, 2, 3, 2, 1, 4];
        let estimate = borrowed_numbers
            .iter()
            .cloned()
            .estimate_distinct_count(0.1, 0.1, 100);
        assert!(estimate > 0.0);
    }

    #[test]
    fn test_streaming_integration_with_file_processing() {
        // Simulate file processing pattern
        let lines = vec![
            "hello world".to_string(),
            "world peace".to_string(),
            "hello rust".to_string(),
        ];

        let mut cvm: CVM<String> = CVM::<String>::builder()
            .epsilon(0.1)
            .confidence(0.9)
            .build()
            .unwrap();

        // Process words from all lines
        let words: Vec<String> = lines
            .into_iter()
            .flat_map(|line| {
                line.split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        let estimate = cvm.process_stream(words);

        assert!(estimate > 0.0);
    }

    #[test]
    fn test_streaming_large_dataset() {
        // Test with a larger dataset to verify the algorithm works
        let mut cvm: CVM<i32> = CVM::<i32>::builder()
            .epsilon(0.1)
            .confidence(0.9)
            .estimated_size(10_000)
            .build()
            .unwrap();

        // Create data with known distinct count (1000 unique values, repeated)
        let data: Vec<i32> = (0..1000).cycle().take(10_000).collect();
        let estimate = cvm.process_stream(data);

        // The estimate should be reasonably close to 1000
        // With epsilon=0.1, we expect within 10 % accuracy most of the time
        assert!(estimate > 500.0 && estimate < 2000.0);
    }
}
