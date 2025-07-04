# A Rust implementation of the CVM Algorithm for Counting Distinct Elements

This library implements the algorithm described in

> Chakraborty, S., Vinodchandran, N. V., & Meel, K. S. (2022). *Distinct Elements in Streams: An Algorithm for the (Text) Book*. 6 pages, 727571 bytes. https://doi.org/10.4230/LIPIcs.ESA.2022.34

The accompanying article in Quanta is here: https://www.quantamagazine.org/computer-scientists-invent-an-efficient-new-way-to-count-20240516/

## What does that mean
The count-distinct problem, or cardinality-estimation problem refers to counting the number of distinct elements in a data stream with repeated elements. As a concrete example, imagine that you want to count the unique words in a book. If you have enough memory, you can keep track of every unique element you encounter. However, you may not have enough working memory due to resource constraints, or the number of potential elements may be enormous. This constraint is referred to as the bounded-storage constraint in the literature.

In order to overcome this constraint, streaming algorithms have been developed: [Flajolet-Martin](https://en.wikipedia.org/wiki/Flajolet–Martin_algorithm), LogLog, [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog). The algorithm implemented by this library is an improvement on these in one particular sense: it is extremely simple. Instead of hashing, it uses a sampling method to compute an [unbiased estimate](https://www.statlect.com/glossary/unbiased-estimator#:~:text=An%20estimator%20of%20a%20given,Examples) of the cardinality.

# What is an Element
In this implementation, an element is anything implementing the `Ord` trait: various integer flavours, strings, any Struct on which you have implemented the trait. Not `f32` / `f64`, however (unless wrapped in an ordered wrapper type).

## Ownership
The buffer has to keep ownership of its elements. In practice, this is not a problem: relative to its input stream size, the buffer is very small. This is also the point of the algorithm: your data set is very large and your working memory is small; you **don't** want to keep the original data around in order to store references to it! Thus, if you have `&str` elements you will need to create new `String`s to store them. If you're processing text data you'll probably want to strip punctuation and regularise the case, so you'll need new `String`s anyway. If you're processing strings containing numeric values, parsing them to the appropriate integer type (which implements `Copy`) first seems like a reasonable approach.

## Further Details
Don Knuth has written about the algorithm (he refers to it as **Algorithm D**) at https://cs.stanford.edu/~knuth/papers/cvm-note.pdf, and does a far better job than I do at explaining it. You will note that on p1 he describes the buffer he uses as a data structure – called a [treap](https://en.wikipedia.org/wiki/Treap#:~:text=7%20External%20links-,Description,(randomly%20chosen)%20numeric%20priority.) – as a binary tree
> "that’s capable of holding up to _s_ ordered pairs (_a_, _u_), where _a_ is an element of the stream and _u_ is a real number, 0 ≤ _u_ < 1."

where _s_ >= 1. This implementation uses a treap as a buffer, following Knuth's original design. While this results in O(log n) operations instead of O(1) for hash-based approaches, it provides better cache locality for small buffers and eliminates hash collision overhead.

# What does this library provide
Two things: the crate / library, and a command-line utility (`cvmcount`) which will count the unique strings in an input text file.

# Installation
Binaries and installation instructions are available for x64 Linux, Apple Silicon and Intel, and x64 Windows in [releases](https://github.com/urschrei/cvmcount/releases)

You can also build your own if you have Rust installed: `cargo install cvmcount`.

# CLI Example

```shell
cvmcount -t file.txt -e 0.8 -d 0.1 -s 5000
```
`-t --tokens`: a valid path to a text file

`-e --epsilon`: how close you want your estimate to be to the true number of distinct tokens. A smaller ε means you require a more precise estimate. For example, ε = 0.05 means you want your estimate to be within 5 % of the actual value. An epsilon of 0.8 is a good starting point for most applications.

`-d --delta`: the level of certainty that the algorithm's estimate will fall within your desired accuracy range. A higher confidence (e.g. 99.9 %) means you're very sure the estimate will be accurate, while a lower confidence (e.g. 90 %) means there's a higher chance the estimate may be outside your desired range. A delta of 0.1 is a good starting point for most applications.

`-s --streamsize`: this is used to determine buffer size and can be a loose approximation. The closer it is to the stream size, the more accurate the results.

The `--help` option is available.

# Library Usage

The library provides both a simple constructor and a builder pattern for more ergonomic usage:

## Simple Constructor

```rust
use cvmcount::CVM;

let mut cvm = CVM::new(0.05, 0.01, 10_000);
for item in data_stream {
    cvm.process_element(item);
}
let estimate = cvm.calculate_final_result();
```

## Builder Pattern (Recommended)

The builder pattern provides better readability and validation:

```rust
use cvmcount::CVM;

// Using defaults (epsilon=0.8, confidence=0.9, size=1000)
let mut cvm: CVM<String> = CVM::builder().build().unwrap();

// Custom configuration with confidence level
let mut cvm: CVM<i32> = CVM::builder()
    .epsilon(0.05)        // 5 % accuracy
    .confidence(0.99)     // 99 % confidence
    .estimated_size(50_000)
    .build()
    .unwrap();

// Using delta (failure probability) instead of confidence
let mut cvm: CVM<String> = CVM::builder()
    .epsilon(0.1)         // 10 % accuracy
    .delta(0.01)          // 1 % chance of failure
    .estimated_size(1_000)
    .build()
    .unwrap();

// Process your data
for word in text.split_whitespace() {
    cvm.process_element(word.to_string());
}

let estimate = cvm.calculate_final_result();
println!("Estimated unique words: {}", estimate as usize);
```

The builder validates parameters and provides clear error messages for invalid inputs.

## Streaming Interface

For processing iterators directly, you can use the streaming methods:

```rust
use cvmcount::{CVM, EstimateDistinct};

// Process an entire iterator with CVM instance
let mut cvm: CVM<i32> = CVM::builder().epsilon(0.05).build().unwrap();
let numbers = vec![1, 2, 3, 2, 1, 4, 5];
let estimate = cvm.process_stream(numbers);

// Or use the iterator extension trait for one-liners
let estimate = (1..=1000)
    .cycle()
    .take(10_000)
    .estimate_distinct_count(0.1, 0.1, 10_000);

// With builder pattern
let words = vec!["hello".to_string(), "world".to_string(), "hello".to_string()];
let builder = CVM::<String>::builder().epsilon(0.05).confidence(0.99);
let estimate = words.into_iter().estimate_distinct_with_builder(builder).unwrap();

// When working with borrowed data, map to owned explicitly
let borrowed_words = vec!["hello", "world", "hello"];
let estimate = borrowed_words
    .iter()
    .map(|s| s.to_string())
    .estimate_distinct_count(0.1, 0.1, 1000);
```

The streaming interface accepts owned values to avoid cloning within the algorithm, making the ownership requirements explicit.

## Analysis

![](cvmcount.png)
```text
Mean: 9015.744000
Std: 534.076058
Min 7552.000000
25% 8672.000000
50% 9024.000000
75% 9344.000000
Max 11072.00000
```

## Note
If you're thinking about using this library, you presumably know that it only provides an estimate (within the specified bounds), similar to something like HyperLogLog. You are trading accuracy for speed and memory usage!

## Perf
Calculating the unique tokens in a [418K UTF-8 text file](https://www.gutenberg.org/ebooks/8492) using the CLI takes 7.2 ms ± 0.3 ms on an M2 Pro. Counting 10e6 7-digit integers takes around 13.5 ms. An exact count using the same regex and HashSet runs in around 18 ms. Run `cargo bench` for more.

## Implementation Details
The CLI app strips punctuation from input tokens using a regex. I assume there is a small performance penalty, but it seems like a small price to pay for increased practicality.

 
