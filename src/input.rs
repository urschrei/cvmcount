//! Read the input and split it into words for the counter.

use std::io::BufRead;

use regex::Regex;

/// Split each line of `reader` into words, clean each word, and give it to `f`.
///
/// A clean word has no characters other than word characters and whitespace, and is lowercase.
pub fn for_each_word<R: BufRead>(reader: R, mut f: impl FnMut(String)) {
    let re = Regex::new(r"[^\w\s]").unwrap();
    reader.lines().for_each(|line| {
        line.unwrap()
            .split(' ')
            .for_each(|word| f(re.replace_all(word, "").to_lowercase()))
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
    fn keeps_non_ascii_word_characters() {
        assert_eq!(words("Grüße, Ölçü"), ["grüße", "ölçü"]);
    }
}
