# A Rust implementation of the CVM Algorithm for Counting Distinct Elements

This library implements the algorithm described in

> Chakraborty, S., Vinodchandran, N. V., & Meel, K. S. (2022). *Distinct Elements in Streams: An Algorithm for the (Text) Book*. 6 pages, 727571 bytes. https://doi.org/10.4230/LIPIcs.ESA.2022.34

The accompanying article in Quanta is here: https://www.quantamagazine.org/computer-scientists-invent-an-efficient-new-way-to-count-20240516/

## What does that mean
The count-distinct problem, or cardinality-estimation problem refers to counting the number of distinct elements in a data stream with repeated elements. As a concrete example, imagine that you want to count the unique words in a book. If you have enough memory, you can keep track of every unique element you encounter. However, you may not have enough working memory due to resource constraints, or the number of potential elements may be enormous. This constraint is referred to as the bounded-storage constraint in the literature.

In order to overcome this constraint, streaming algorithms have been developed: [Flajolet-Martin](https://en.wikipedia.org/wiki/Flajolet–Martin_algorithm), LogLog, [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog). The algorithm implemented by this library is an improvement on these in one particular sense: it is extremely simple. Instead of hashing, it uses a sampling method to compute an [unbiased estimate](https://www.statlect.com/glossary/unbiased-estimator#:~:text=An%20estimator%20of%20a%20given,Examples) of the cardinality.

# What is an Element
In this implementation, an element is anything implementing the [`PartialOrd`](https://doc.rust-lang.org/std/cmp/trait.PartialOrd.html) and [`PartialEQ`](https://doc.rust-lang.org/std/cmp/trait.PartialEq.html) traits: various integer flavours, strings, any Struct on which you have implemented the traits. Not `f32` / `f64`, however.

You will also note that I didn't mention `&str`: that's because the buffer has to keep ownership of its elements. In practice, this is not a problem: relative to its input stream size, the buffer is very small.

## Further Details
Don Knuth has written about the algorithm (he refers to it as **Algorithm D**) at https://cs.stanford.edu/~knuth/papers/cvm-note.pdf, and does a far better job than I do at explaining it. You will note that on p1 he describes the buffer he uses as a data structure called a [treap](https://en.wikipedia.org/wiki/Treap#:~:text=7%20External%20links-,Description,(randomly%20chosen)%20numeric%20priority.)
> "that’s capable of holding up to _s_ ordered pairs (_a_, _u_), where _a_ is an element of the stream and _u_ is a real number, 0 ≤ _u_ < 1.", where _s_ >= 1.

This implementation doesn't use a treap as a buffer; it uses a Vec and performs a binary search during step **D4**. Note in particular his modification of step **D6** on p5: **D6'**: halving the buffer.

I may switch to a treap implementation eventually; for many practical applications a binary search is considerably faster than the hashing algorithms under consideration. If your application assumes a buffer containing 100k+ elements, you may wish to consider using a treap.

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

## Note
If you're thinking about using this library, you presumably know that it only provides an estimate (within the specified bounds), similar to something like HyperLogLog. You are trading accuracy for speed!

## Perf
Calculating the unique tokens in a [418K UTF-8 text file](https://www.gutenberg.org/ebooks/8492) takes 18.6 ms ± 0.3 ms on an M2 Pro

## Implementation Details
This library strips punctuation from input tokens using a regex. I assume there is a small performance penalty, but it seems like a small price to pay for increased practicality.
