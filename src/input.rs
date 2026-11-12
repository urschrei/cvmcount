//! Read the input and split it into words for the counter.

use std::io::{self, BufRead};

use regex::Regex;

/// Split each line of `reader` into words at whitespace, clean each word, and give it to `f`.
///
/// To clean a word, remove all characters that are not word characters, then make it lowercase.
/// A word that is empty after cleaning is skipped.
///
/// Byte sequences that are not valid UTF-8 are removed in the same way as other non-word characters.
///
/// # Errors
///
/// Returns the first error from `reader`.
pub fn for_each_word<R: BufRead>(mut reader: R, mut f: impl FnMut(String)) -> io::Result<()> {
    let re = Regex::new(r"\W").unwrap();
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(());
        }
        String::from_utf8_lossy(&line)
            .split_whitespace()
            .for_each(|word| {
                let clean = re.replace_all(word, "");
                if !clean.is_empty() {
                    f(clean.to_lowercase());
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn splits_on_any_whitespace() {
        assert_eq!(words("one\ttwo  three"), ["one", "two", "three"]);
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
}
