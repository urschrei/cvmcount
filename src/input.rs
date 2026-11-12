//! Read the input and split it into words for the counter.

use std::io::{self, BufRead};

use regex::Regex;

/// Split the bytes of `reader` into words at ASCII whitespace, clean each word, and give it to `f`.
///
/// To clean a word, remove all characters that are not word characters, then make it lowercase.
/// A word that is empty after cleaning is skipped.
///
/// Byte sequences that are not valid UTF-8 are removed in the same way as other non-word characters.
///
/// This function reads the input one buffer at a time. It keeps only the part of a word that
/// crosses the end of a buffer, so it uses memory proportional to the longest word, not to the
/// longest line.
///
/// # Errors
///
/// Returns the first error from `reader`, other than [`io::ErrorKind::Interrupted`].
pub fn for_each_word<R: BufRead>(mut reader: R, mut f: impl FnMut(String)) -> io::Result<()> {
    let re = Regex::new(r"\W").unwrap();
    let mut emit = |bytes: &[u8]| {
        let word = String::from_utf8_lossy(bytes);
        let clean = re.replace_all(&word, "");
        if !clean.is_empty() {
            f(clean.to_lowercase());
        }
    };
    // The start of a word that continues in the next buffer.
    let mut partial = Vec::new();
    loop {
        let buf = match reader.fill_buf() {
            Ok([]) => break,
            Ok(buf) => buf,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        let mut pieces = buf.split(u8::is_ascii_whitespace).peekable();
        while let Some(piece) = pieces.next() {
            if pieces.peek().is_none() {
                // No whitespace follows the last piece in this buffer, so the word can continue.
                partial.extend_from_slice(piece);
            } else if partial.is_empty() {
                emit(piece);
            } else {
                partial.extend_from_slice(piece);
                emit(&partial);
                partial.clear();
            }
        }
        let len = buf.len();
        reader.consume(len);
    }
    emit(&partial);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hegel::TestCase;
    use hegel::generators as gs;

    fn words(input: &str) -> Vec<String> {
        let mut out = Vec::new();
        for_each_word(input.as_bytes(), |w| out.push(w)).unwrap();
        out
    }

    #[test]
    fn cleans_and_lowercases() {
        assert_eq!(words("The cat's hat."), ["the", "cats", "hat"]);
    }

    #[test]
    fn reads_every_line() {
        assert_eq!(
            words("one two\nthree\r\nfour"),
            ["one", "two", "three", "four"]
        );
    }

    #[test]
    fn splits_on_ascii_whitespace() {
        assert_eq!(
            words("one\ttwo  three\u{a0}four"),
            ["one", "two", "threefour"]
        );
    }

    #[test]
    fn skips_empty_words() {
        assert_eq!(
            words("  leading -- and trailing  \n\n"),
            ["leading", "and", "trailing"]
        );
    }

    #[test]
    fn keeps_non_ascii_word_characters() {
        assert_eq!(words("Grüße, Ölçü"), ["grüße", "ölçü"]);
    }

    #[test]
    fn replaces_invalid_utf8() {
        let mut out = Vec::new();
        for_each_word(&b"ok bad\xffbyte\n"[..], |w| out.push(w)).unwrap();
        assert_eq!(out, ["ok", "badbyte"]);
    }

    #[test]
    fn returns_read_errors() {
        struct Failing;
        impl io::Read for Failing {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("read failed"))
            }
        }
        let err = for_each_word(io::BufReader::new(Failing), |_| {}).unwrap_err();
        assert_eq!(err.to_string(), "read failed");
    }

    /// The words, found by reading all of `input` at once.
    fn reference_words(input: &[u8]) -> Vec<String> {
        let re = Regex::new(r"\W").unwrap();
        String::from_utf8_lossy(input)
            .split(|c: char| c.is_ascii_whitespace())
            .map(|w| re.replace_all(w, "").to_lowercase())
            .filter(|w| !w.is_empty())
            .collect()
    }

    fn words_with_capacity(input: &[u8], capacity: usize) -> Vec<String> {
        let mut out = Vec::new();
        for_each_word(io::BufReader::with_capacity(capacity, input), |w| {
            out.push(w)
        })
        .unwrap();
        out
    }

    #[test]
    fn joins_words_across_buffers() {
        assert_eq!(
            words_with_capacity("Grüße aus Köln".as_bytes(), 3),
            ["grüße", "aus", "köln"]
        );
    }

    #[test]
    fn reads_input_without_newlines() {
        let input = "word ".repeat(10_000);
        assert_eq!(words_with_capacity(input.as_bytes(), 64).len(), 10_000);
    }

    #[hegel::test(test_cases = 1000)]
    fn buffer_size_does_not_change_text_words(tc: TestCase) {
        let input = tc.draw(gs::text());
        let capacity = tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
        assert_eq!(
            words_with_capacity(input.as_bytes(), capacity),
            reference_words(input.as_bytes())
        );
    }

    #[hegel::test(test_cases = 1000)]
    fn buffer_size_does_not_change_binary_words(tc: TestCase) {
        let input = tc.draw(gs::binary());
        let capacity = tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
        assert_eq!(
            words_with_capacity(&input, capacity),
            reference_words(&input)
        );
    }
}
