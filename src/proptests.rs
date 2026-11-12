//! Hegel property-based tests for the CVM estimator.
//!
//! These live inside the crate so that they can check the estimator's
//! internal state (buffer size, sampling probability) as well as its public
//! behaviour, and so that they can seed its RNG for exact comparisons
//! between equivalent code paths.

use crate::{CVM, CVMBuilder, EstimateDistinct, buffer_size};
use hegel::TestCase;
use hegel::generators::{self as gs, Generator, PrintableGenerator};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashSet;

/// Smallest buffer any valid builder configuration can produce:
/// buffer_size is ceil((12 / ε²) · log2(8n / δ)), and with 0 < ε, δ < 1 and
/// n ≥ 1 that is strictly greater than 12 · log2(8) = 36.
const MIN_VALID_BUFFER: usize = 37;

/// Values in the open interval (0, 1), the documented domain of ε, δ and
/// confidence, including subnormals and the values next to each endpoint.
fn unit_open() -> impl PrintableGenerator<f64> {
    hegel::one_of!(
        gs::floats::<f64>()
            .min_value_exclusive(0.0)
            .max_value_exclusive(1.0),
        gs::sampled_from(vec![
            f64::from_bits(1),        // smallest subnormal
            f64::MIN_POSITIVE,        // smallest normal
            1.0 - f64::EPSILON / 2.0, // largest value below 1
        ])
    )
}

/// Values outside (0, 1): zero (of either sign), negatives, values ≥ 1,
/// infinities, and NaN.
fn not_unit_open() -> impl PrintableGenerator<f64> {
    hegel::one_of!(
        gs::sampled_from(vec![
            0.0,
            -0.0,
            1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
        ]),
        gs::floats::<f64>().max_value(0.0),
        gs::floats::<f64>().min_value(1.0),
        gs::floats::<f64>().filter(|x| !(*x > 0.0 && *x < 1.0))
    )
}

/// A CVM whose RNG is seeded, for reproducible runs.
fn seeded<T: Ord>(epsilon: f64, delta: f64, stream_size: usize, seed: u64) -> CVM<T> {
    let mut cvm = CVM::new(epsilon, delta, stream_size);
    cvm.rng = StdRng::seed_from_u64(seed);
    cvm
}

fn distinct<T: std::hash::Hash + Eq>(xs: &[T]) -> usize {
    xs.iter().collect::<HashSet<_>>().len()
}

// ----- Builder validation -------------------------------------------------

/// Any ε, confidence and estimated size inside their documented domains is
/// accepted, and yields the buffer the formula gives for δ = 1 - confidence.
#[hegel::test(test_cases = 2000)]
fn builder_accepts_valid_confidence(tc: TestCase) {
    let epsilon = tc.draw(unit_open());
    let confidence = tc.draw(unit_open());
    let size = tc.draw(gs::integers::<usize>().min_value(1));
    let cvm: CVM<i32> = CVM::<i32>::builder()
        .epsilon(epsilon)
        .confidence(confidence)
        .estimated_size(size)
        .build()
        .expect("valid parameters must be accepted");
    assert_eq!(cvm.buf_size, buffer_size(epsilon, 1.0 - confidence, size));
    assert!(cvm.buf_size >= MIN_VALID_BUFFER);
}

/// Any ε, δ and estimated size inside their documented domains is accepted,
/// and yields the same buffer as `CVM::new` with those parameters.
#[hegel::test(test_cases = 2000)]
fn builder_accepts_valid_delta(tc: TestCase) {
    let epsilon = tc.draw(unit_open());
    let delta = tc.draw(unit_open());
    let size = tc.draw(gs::integers::<usize>().min_value(1));
    let cvm: CVM<i32> = CVM::<i32>::builder()
        .epsilon(epsilon)
        .delta(delta)
        .estimated_size(size)
        .build()
        .expect("valid parameters must be accepted");
    assert_eq!(cvm.buf_size, CVM::<i32>::new(epsilon, delta, size).buf_size);
    assert!(cvm.buf_size >= MIN_VALID_BUFFER);
}

/// "Must be between 0.0 and 1.0 (exclusive)": an ε outside (0, 1) is rejected.
#[hegel::test(test_cases = 2000)]
fn builder_rejects_invalid_epsilon(tc: TestCase) {
    let epsilon = tc.draw(not_unit_open());
    let result = CVM::<i32>::builder().epsilon(epsilon).build::<i32>();
    assert!(result.is_err(), "epsilon {epsilon:?} was accepted");
}

/// A confidence outside (0, 1) is rejected.
#[hegel::test(test_cases = 2000)]
fn builder_rejects_invalid_confidence(tc: TestCase) {
    let confidence = tc.draw(not_unit_open());
    let result = CVM::<i32>::builder().confidence(confidence).build::<i32>();
    assert!(result.is_err(), "confidence {confidence:?} was accepted");
}

/// A δ outside (0, 1) is rejected.
#[hegel::test(test_cases = 2000)]
fn builder_rejects_invalid_delta(tc: TestCase) {
    let delta = tc.draw(not_unit_open());
    let result = CVM::<i32>::builder().delta(delta).build::<i32>();
    assert!(result.is_err(), "delta {delta:?} was accepted");
}

/// An estimated size of zero is rejected, whatever the other parameters.
#[hegel::test(test_cases = 500)]
fn builder_rejects_zero_size(tc: TestCase) {
    let epsilon = tc.draw(unit_open());
    let delta = tc.draw(unit_open());
    let result = CVM::<i32>::builder()
        .epsilon(epsilon)
        .delta(delta)
        .estimated_size(0)
        .build::<i32>();
    assert!(result.is_err());
}

/// Parameters left unset take the documented defaults (ε = 0.8,
/// confidence = 0.9, estimated size = 1000), and of `confidence` and
/// `delta` the last one set wins.
#[hegel::test(test_cases = 2000)]
fn builder_defaults_and_last_wins(tc: TestCase) {
    let epsilon = tc.draw(gs::optional(unit_open()));
    let size = tc.draw(gs::optional(gs::integers::<usize>().min_value(1)));
    // A sequence of confidence (true) / delta (false) calls.
    let specs = tc.draw(gs::vecs(gs::tuples!(gs::booleans(), unit_open())).max_size(4));
    let mut builder = CVMBuilder::new();
    if let Some(e) = epsilon {
        builder = builder.epsilon(e);
    }
    if let Some(s) = size {
        builder = builder.estimated_size(s);
    }
    for &(is_confidence, v) in &specs {
        builder = if is_confidence {
            builder.confidence(v)
        } else {
            builder.delta(v)
        };
    }
    let delta = match specs.last() {
        None => 1.0 - 0.9,
        Some(&(true, c)) => 1.0 - c,
        Some(&(false, d)) => d,
    };
    let cvm: CVM<i32> = builder.build().unwrap();
    let expected = buffer_size(epsilon.unwrap_or(0.8), delta, size.unwrap_or(1000));
    assert_eq!(cvm.buf_size, expected);
}

// ----- buffer_size --------------------------------------------------------

/// The buffer grows as the requirements tighten: a smaller ε, a smaller δ,
/// or a larger stream never gives a smaller buffer.
#[hegel::test(test_cases = 2000)]
fn buffer_size_is_monotone(tc: TestCase) {
    let (e1, e2) = (tc.draw(unit_open()), tc.draw(unit_open()));
    let (d1, d2) = (tc.draw(unit_open()), tc.draw(unit_open()));
    let (n1, n2) = (
        tc.draw(gs::integers::<usize>().min_value(1)),
        tc.draw(gs::integers::<usize>().min_value(1)),
    );
    let (e_lo, e_hi) = (e1.min(e2), e1.max(e2));
    let (d_lo, d_hi) = (d1.min(d2), d1.max(d2));
    let (n_lo, n_hi) = (n1.min(n2), n1.max(n2));
    assert!(buffer_size(e_lo, d1, n1) >= buffer_size(e_hi, d1, n1));
    assert!(buffer_size(e1, d_lo, n1) >= buffer_size(e1, d_hi, n1));
    assert!(buffer_size(e1, d1, n_hi) >= buffer_size(e1, d1, n_lo));
}

// ----- The exact regime ---------------------------------------------------

/// Elements from a domain of at most `MIN_VALID_BUFFER - 1` values, so that
/// any stream has fewer distinct elements than any valid buffer holds.
fn small_domain_stream() -> impl PrintableGenerator<Vec<i32>> {
    gs::vecs(gs::integers::<i32>().min_value(-18).max_value(17))
}

/// Until the buffer first fills, the sampling probability stays at 1 and
/// every distinct element is held, so the estimate is the exact distinct
/// count. Holds for `process_element`, whatever the order and repetition.
#[hegel::test(test_cases = 2000)]
fn exact_below_buffer_process_element(tc: TestCase) {
    let epsilon = tc.draw(unit_open());
    let delta = tc.draw(unit_open());
    let size = tc.draw(gs::integers::<usize>().min_value(1));
    let stream = tc.draw(small_domain_stream());
    let mut cvm: CVM<i32> = CVM::<i32>::builder()
        .epsilon(epsilon)
        .delta(delta)
        .estimated_size(size)
        .build()
        .unwrap();
    for &x in &stream {
        cvm.process_element(x);
    }
    assert_eq!(cvm.calculate_final_result(), distinct(&stream) as f64);
}

/// As above, through `process_stream`.
#[hegel::test(test_cases = 2000)]
fn exact_below_buffer_process_stream(tc: TestCase) {
    let epsilon = tc.draw(unit_open());
    let confidence = tc.draw(unit_open());
    let stream = tc.draw(small_domain_stream());
    let mut cvm: CVM<i32> = CVM::<i32>::builder()
        .epsilon(epsilon)
        .confidence(confidence)
        .build()
        .unwrap();
    let estimate = cvm.process_stream(stream.clone());
    assert_eq!(estimate, distinct(&stream) as f64);
}

/// As above, through `EstimateDistinct::estimate_distinct_count`.
#[hegel::test(test_cases = 2000)]
fn exact_below_buffer_estimate_distinct_count(tc: TestCase) {
    let epsilon = tc.draw(unit_open());
    let delta = tc.draw(unit_open());
    let size = tc.draw(gs::integers::<usize>().min_value(1));
    let stream = tc.draw(small_domain_stream());
    let estimate = stream
        .clone()
        .into_iter()
        .estimate_distinct_count(epsilon, delta, size);
    assert_eq!(estimate, distinct(&stream) as f64);
}

/// As above, through `EstimateDistinct::estimate_distinct_with_builder`,
/// with `String` elements (the instantiation the CLI uses).
#[hegel::test(test_cases = 2000)]
fn exact_below_buffer_with_builder_strings(tc: TestCase) {
    let epsilon = tc.draw(unit_open());
    let delta = tc.draw(unit_open());
    let words = tc.draw(gs::vecs(gs::text()).max_size(MIN_VALID_BUFFER - 1));
    // Repeat the words in a shuffled, duplicated order.
    let picks = tc.draw(gs::vecs(gs::integers::<usize>()));
    let stream: Vec<String> = if words.is_empty() {
        Vec::new()
    } else {
        picks
            .iter()
            .map(|&i| words[i % words.len()].clone())
            .collect()
    };
    let builder = CVM::<String>::builder().epsilon(epsilon).delta(delta);
    let estimate = stream
        .clone()
        .into_iter()
        .estimate_distinct_with_builder(builder)
        .unwrap();
    assert_eq!(estimate, distinct(&stream) as f64);
}

/// The exact regime extends right up to the buffer size: any stream with
/// fewer than `buf_size` distinct elements is counted exactly.
#[hegel::test(test_cases = 500)]
fn exact_up_to_buffer_size(tc: TestCase) {
    // ε ≥ 0.25 keeps buffers small enough to fill in a test case.
    let epsilon = tc.draw(gs::floats::<f64>().min_value(0.25).max_value_exclusive(1.0));
    let delta = tc.draw(unit_open());
    let size = tc.draw(gs::integers::<usize>().min_value(1).max_value(1 << 20));
    let mut cvm: CVM<u32> = CVM::new(epsilon, delta, size);
    // Cap the materialized stream; a tiny δ can make the buffer enormous.
    let d = tc.draw(gs::integers::<usize>().max_value((cvm.buf_size - 1).min(20_000)));
    let repeats = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
    for _ in 0..repeats {
        for x in 0..d as u32 {
            cvm.process_element(x);
        }
    }
    assert_eq!(cvm.calculate_final_result(), d as f64);
}

// ----- The shape of the estimate ------------------------------------------

/// Whether `p` is 2^-k for some integer k ≥ 0.
fn is_power_of_half(p: f64) -> bool {
    p > 0.0 && p <= 1.0 && p.to_bits() & ((1u64 << 52) - 1) == 0
}

/// After every element: the buffer is below its threshold, the sampling
/// probability is a power of 1/2, and the estimate is exactly
/// |buffer| / p — a non-negative whole number. Runs with small buffers and
/// streams long enough to force repeated halving.
#[hegel::test(test_cases = 1000)]
fn estimate_shape_invariants(tc: TestCase) {
    let epsilon = tc.draw(gs::floats::<f64>().min_value(0.5).max_value_exclusive(1.0));
    let delta = tc.draw(unit_open());
    let seed = tc.draw(gs::integers::<u64>());
    let mut cvm: CVM<u16> = seeded(epsilon, delta, 1, seed);
    let stream = tc.draw(gs::vecs(gs::integers::<u16>()).max_size(5000));
    for x in stream {
        cvm.process_element(x);
        assert!(cvm.buf.len() < cvm.buf_size, "buffer reached its threshold");
        assert!(is_power_of_half(cvm.probability), "p = {}", cvm.probability);
        let estimate = cvm.calculate_final_result();
        assert_eq!(estimate, cvm.buf.len() as f64 / cvm.probability);
        assert!(
            estimate >= 0.0 && estimate.fract() == 0.0,
            "estimate {estimate}"
        );
    }
}

// ----- Agreement between equivalent paths ---------------------------------

/// `process_stream` is `process_element` on each item followed by
/// `calculate_final_result`: with the same seed, the results are identical.
#[hegel::test(test_cases = 1000)]
fn process_stream_matches_element_loop(tc: TestCase) {
    let epsilon = tc.draw(gs::floats::<f64>().min_value(0.5).max_value_exclusive(1.0));
    let seed = tc.draw(gs::integers::<u64>());
    let stream = tc.draw(gs::vecs(gs::integers::<u16>()).max_size(3000));
    let mut a: CVM<u16> = seeded(epsilon, 0.1, 1, seed);
    let mut b: CVM<u16> = seeded(epsilon, 0.1, 1, seed);
    let via_stream = a.process_stream(stream.clone());
    for x in stream {
        b.process_element(x);
    }
    assert_eq!(via_stream.to_bits(), b.calculate_final_result().to_bits());
}

/// `process_borrowed` is `process_element` with an owned copy: with the same
/// seed, the results are identical. The streams are long enough to fill the
/// buffer, so the probability falls below 1 and repeats can be removed.
#[hegel::test(test_cases = 1000)]
fn process_borrowed_matches_element(tc: TestCase) {
    let epsilon = tc.draw(gs::floats::<f64>().min_value(0.5).max_value_exclusive(1.0));
    let seed = tc.draw(gs::integers::<u64>());
    let stream = tc.draw(
        gs::vecs(gs::integers::<u16>().max_value(2000))
            .min_size(1000)
            .max_size(3000),
    );
    let words: Vec<String> = stream.iter().map(u16::to_string).collect();
    let mut owned: CVM<String> = seeded(epsilon, 0.1, 1, seed);
    let mut borrowed: CVM<String> = seeded(epsilon, 0.1, 1, seed);
    for w in &words {
        owned.process_element(w.clone());
        borrowed.process_borrowed(w.as_str());
    }
    assert_eq!(
        owned.calculate_final_result().to_bits(),
        borrowed.calculate_final_result().to_bits()
    );
}

/// Processing a stream in two `process_stream` calls is the same as one
/// call on the concatenation: the counter can keep going after a result.
#[hegel::test(test_cases = 1000)]
fn process_stream_is_resumable(tc: TestCase) {
    let epsilon = tc.draw(gs::floats::<f64>().min_value(0.5).max_value_exclusive(1.0));
    let seed = tc.draw(gs::integers::<u64>());
    let first = tc.draw(gs::vecs(gs::integers::<u16>()).max_size(2000));
    let second = tc.draw(gs::vecs(gs::integers::<u16>()).max_size(2000));
    let mut split: CVM<u16> = seeded(epsilon, 0.1, 1, seed);
    let mut whole: CVM<u16> = seeded(epsilon, 0.1, 1, seed);
    split.process_stream(first.clone());
    let a = split.process_stream(second.clone());
    let b = whole.process_stream(first.into_iter().chain(second));
    assert_eq!(a.to_bits(), b.to_bits());
}

/// `calculate_final_result` does not change the state: calling it again, or
/// processing an empty stream, gives the same value.
#[hegel::test(test_cases = 1000)]
fn calculate_final_result_is_pure(tc: TestCase) {
    let epsilon = tc.draw(gs::floats::<f64>().min_value(0.5).max_value_exclusive(1.0));
    let seed = tc.draw(gs::integers::<u64>());
    let stream = tc.draw(gs::vecs(gs::integers::<u16>()).max_size(3000));
    let mut cvm: CVM<u16> = seeded(epsilon, 0.1, 1, seed);
    cvm.process_stream(stream);
    let first = cvm.calculate_final_result();
    assert_eq!(first.to_bits(), cvm.calculate_final_result().to_bits());
    assert_eq!(first.to_bits(), cvm.process_stream(Vec::new()).to_bits());
}

// ----- Accuracy -----------------------------------------------------------

/// The (ε, δ) guarantee, the main theorem of Chakraborty, Vinodchandran &
/// Meel (ESA 2022; see arXiv:2301.10191 v2 for a corrected proof): the
/// estimate is within a factor (1 ± ε) of the true count with probability
/// at least 1 - δ. Over 200 fixed seeds the number of misses
/// must stay within the δ budget (expected count 10 at δ = 0.05; the bound
/// is loose, so in practice there are far fewer).
#[test]
fn accuracy_within_guarantee() {
    let (epsilon, delta, n) = (0.2, 0.05, 50_000usize);
    let trials = 200;
    let mut misses = 0;
    for seed in 0..trials {
        let mut cvm: CVM<u32> = seeded(epsilon, delta, n, seed);
        // Each value twice, so that removal of present elements is exercised.
        let estimate = cvm.process_stream((0..n as u32).chain(0..n as u32));
        if (estimate - n as f64).abs() > epsilon * n as f64 {
            misses += 1;
        }
    }
    assert!(
        misses as f64 <= delta * trials as f64,
        "{misses} of {trials} estimates missed by more than ε"
    );
}

/// Scale probe: a large stream with a large buffer, through every entry
/// point, finishing with a sane estimate.
#[test]
fn scale_probe() {
    let n = 2_000_000u32;
    let mut cvm: CVM<u32> = seeded(0.05, 0.01, n as usize, 7);
    let estimate = cvm.process_stream(0..n);
    assert!((estimate - n as f64).abs() <= 0.05 * n as f64, "{estimate}");
    let estimate = (0..n).estimate_distinct_count(0.05, 0.01, n as usize);
    assert!((estimate - n as f64).abs() <= 0.05 * n as f64, "{estimate}");
}
