//! Read the input and split it into words for the counter.

use std::io::BufRead;

use regex::Regex;

/// Split each line of `reader` into words at whitespace, clean each word, and give it to `f`.
///
/// To clean a word, remove all characters that are not word characters, then make it lowercase.
/// A word that is empty after cleaning is skipped.
pub fn for_each_word<R: BufRead>(reader: R, mut f: impl FnMut(String)) {
    let re = Regex::new(r"\W").unwrap();
    reader.lines().for_each(|line| {
        line.unwrap().split_whitespace().for_each(|word| {
            let clean = re.replace_all(word, "");
            if !clean.is_empty() {
                f(clean.to_lowercase());
            }
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(input: &str) -> Vec<String> {
        let mut out = Vec::new();
        for_each_word(input.as_bytes(), |w| out.push(w));
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
}
