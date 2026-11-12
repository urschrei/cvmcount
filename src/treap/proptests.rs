//! Hegel property-based tests for the treap.
//!
//! The treap is checked against a `BTreeSet` model. Priorities come from
//! Hegel-controlled randomness (`randoms()`), so Hegel can steer the tree
//! into degenerate shapes (constant or monotone priorities) and shrink
//! failing priority sequences, as well as from a seeded `StdRng` for
//! realistic shapes.

use super::*;
use hegel::TestCase;
use hegel::extras::rand::{HegelRandom, randoms};
use hegel::generators::{self as gs, Generator, PrintableGenerator};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// In-order keys, cloned.
fn inorder<T: Clone>(treap: &Treap<T>) -> Vec<T> {
    treap.keys().into_iter().cloned().collect()
}

/// The treap's full observable state agrees with the model: BST order and
/// contents, heap order, the cached size, and the arena's bookkeeping.
fn assert_matches_model<T: Ord + Clone + Debug>(treap: &Treap<T>, model: &BTreeSet<T>) {
    let keys = inorder(treap);
    let expected: Vec<T> = model.iter().cloned().collect();
    assert_eq!(keys, expected, "in-order keys differ from model");
    assert!(treap.heap_ok(), "heap property violated");
    assert!(treap.arena_ok(), "arena slots are inconsistent");
    assert_eq!(treap.len(), model.len(), "len() differs from model");
    assert_eq!(treap.is_empty(), model.is_empty());
}

/// Keys from a narrow range (so that duplicates, hits and misses are common)
/// mixed with the full `i32` domain (including the extremes).
fn keys() -> impl PrintableGenerator<i32> {
    hegel::one_of!(
        gs::integers::<i32>().min_value(-16).max_value(16),
        gs::integers::<i32>()
    )
}

struct TreapMachine {
    treap: Treap<i32>,
    model: BTreeSet<i32>,
    rng: HegelRandom,
}

#[hegel::state_machine]
impl TreapMachine {
    #[rule(weight = 8)]
    fn insert(&mut self, tc: TestCase) {
        let k = tc.draw(keys());
        assert_eq!(self.treap.insert(k, &mut self.rng), self.model.insert(k));
        tc.event_value("len after insert", self.model.len() as f64);
    }

    #[rule(weight = 4)]
    fn insert_borrowed(&mut self, tc: TestCase) {
        let k = tc.draw(keys());
        assert_eq!(
            self.treap.insert_borrowed(&k, &mut self.rng),
            self.model.insert(k)
        );
    }

    #[rule(weight = 3)]
    fn remove(&mut self, tc: TestCase) {
        let k = tc.draw(keys());
        assert_eq!(self.treap.remove(&k), self.model.remove(&k));
    }

    #[rule(weight = 3)]
    fn contains(&mut self, tc: TestCase) {
        let k = tc.draw(keys());
        assert_eq!(self.treap.contains(&k), self.model.contains(&k));
    }

    #[rule]
    fn retain(&mut self, tc: TestCase) {
        let mut decisions: Vec<(i32, bool)> = Vec::new();
        self.treap.retain(|&k| {
            let keep = tc.draw(gs::booleans());
            decisions.push((k, keep));
            keep
        });
        // The predicate is applied exactly once to every element.
        let mut seen: Vec<i32> = decisions.iter().map(|&(k, _)| k).collect();
        seen.sort_unstable();
        let expected: Vec<i32> = self.model.iter().copied().collect();
        assert_eq!(
            seen, expected,
            "retain must visit each element exactly once"
        );
        let decisions: HashMap<i32, bool> = decisions.into_iter().collect();
        self.model.retain(|k| decisions[k]);
    }

    #[rule]
    fn clear(&mut self, _: TestCase) {
        self.treap.clear();
        self.model.clear();
    }

    #[invariant]
    fn matches_model(&self, _: TestCase) {
        assert_matches_model(&self.treap, &self.model);
    }
}

/// Arbitrary interleavings of every operation agree with a `BTreeSet`, and
/// keep the BST, heap, and size invariants, under Hegel-chosen priorities.
#[hegel::test(test_cases = 1000)]
fn treap_matches_btreeset_model(tc: TestCase) {
    let machine = TreapMachine {
        treap: Treap::new(),
        model: BTreeSet::new(),
        rng: tc.draw(randoms()),
    };
    hegel::stateful::machine(machine).steps(200).run(tc);
}

/// As above, with priorities from a seeded `StdRng`: realistic tree shapes.
#[hegel::test(test_cases = 1000)]
fn treap_matches_btreeset_model_true_random(tc: TestCase) {
    let machine = TreapMachine {
        treap: Treap::new(),
        model: BTreeSet::new(),
        rng: tc.draw(randoms().use_true_random(true)),
    };
    hegel::stateful::machine(machine).steps(200).run(tc);
}

#[derive(Debug, Clone)]
enum Op<T> {
    Insert(T),
    Remove(T),
    Contains(T),
    RetainHalf(u64),
}

/// Run a sequence of operations against the treap and a model, checking
/// return values after each step and full structure at the end.
fn run_ops<T: Ord + Clone + Debug + std::hash::Hash>(ops: Vec<Op<T>>, rng: &mut impl Rng) {
    let mut treap = Treap::new();
    let mut model = BTreeSet::new();
    for op in ops {
        match op {
            Op::Insert(k) => assert_eq!(treap.insert(k.clone(), rng), model.insert(k)),
            Op::Remove(k) => assert_eq!(treap.remove(&k), model.remove(&k)),
            Op::Contains(k) => assert_eq!(treap.contains(&k), model.contains(&k)),
            Op::RetainHalf(salt) => {
                // A deterministic predicate of the key, so the model can replay it.
                let keep = |k: &T| {
                    use std::hash::{BuildHasher, BuildHasherDefault};
                    let h = BuildHasherDefault::<rustc_hash::FxHasher>::default().hash_one(k);
                    (h ^ salt) & 1 == 0
                };
                treap.retain(keep);
                model.retain(keep);
            }
        }
        assert_eq!(treap.len(), model.len());
    }
    assert_matches_model(&treap, &model);
}

fn ops<
    T: Clone + Debug + Send + Sync + 'static,
    G: PrintableGenerator<T> + Send + Sync + 'static,
>(
    key: impl Fn() -> G,
) -> impl PrintableGenerator<Op<T>> {
    hegel::one_of!(
        key().map(Op::Insert),
        key().map(Op::Insert),
        key().map(Op::Remove),
        key().map(Op::Contains),
        gs::integers::<u64>().map(Op::RetainHalf)
    )
}

/// The `String` instantiation (what the `cvmcount` binary uses) agrees with
/// the model, including empty strings, NUL, and arbitrary Unicode.
#[hegel::test(test_cases = 1000)]
fn string_treap_matches_btreeset_model(tc: TestCase) {
    let key = || {
        hegel::one_of!(
            gs::sampled_from(vec![
                String::new(),
                "\0".to_string(),
                "a".to_string(),
                "A".to_string(),
                "é".to_string()
            ]),
            gs::text()
        )
    };
    let ops = tc.draw(gs::vecs(ops(key)));
    let mut rng = tc.draw(randoms());
    run_ops(ops, &mut rng);
}

/// The `u8` instantiation: a tiny key space, so the treap is repeatedly
/// filled to every possible key and emptied again.
#[hegel::test(test_cases = 1000)]
fn u8_treap_matches_btreeset_model(tc: TestCase) {
    let ops = tc.draw(gs::vecs(ops(gs::integers::<u8>)));
    let mut rng = tc.draw(randoms());
    run_ops(ops, &mut rng);
}

/// An RNG yielding an arithmetic progression of priorities: constant
/// (`step == 0`, every comparison a tie), ascending, or descending. These are
/// the shapes most likely to break rotation and merge logic.
struct PatternRng {
    next: u32,
    step: u32,
}
impl rand::TryRng for PatternRng {
    type Error = std::convert::Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let v = self.next;
        self.next = self.next.wrapping_add(self.step);
        Ok(v)
    }
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let hi = u64::from(self.try_next_u32()?);
        Ok((hi << 32) | u64::from(self.try_next_u32()?))
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in dest.chunks_mut(4) {
            let v = self.try_next_u32()?.to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        Ok(())
    }
}

/// Degenerate priority sequences (all ties, strictly ascending, strictly
/// descending, or any fixed stride) still give a correct set with valid
/// BST and heap order.
#[hegel::test(test_cases = 1000)]
fn degenerate_priorities_match_model(tc: TestCase) {
    let next = tc.draw(gs::integers::<u32>());
    let step = tc.draw(hegel::one_of!(
        gs::sampled_from(vec![0u32, 1, u32::MAX]),
        gs::integers::<u32>()
    ));
    let ops = tc.draw(gs::vecs(ops(keys)));
    run_ops(ops, &mut PatternRng { next, step });
}

/// A key whose `Ord` ignores its tag, to observe which of two equal keys is stored.
#[derive(Debug, Clone)]
struct Tagged {
    key: i8,
    tag: usize,
}
impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Tagged {}
impl PartialOrd for Tagged {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Tagged {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// "Insert doesn't update existing keys": inserting a key equal to one
/// already present leaves the original element in place.
#[hegel::test(test_cases = 1000)]
fn insert_keeps_first_of_equal_keys(tc: TestCase) {
    let keys: Vec<i8> = tc.draw(gs::vecs(gs::integers::<i8>()));
    let mut rng = tc.draw(randoms());
    let mut treap = Treap::new();
    let mut first_tag: HashMap<i8, usize> = HashMap::new();
    for (tag, key) in keys.into_iter().enumerate() {
        let inserted = treap.insert(Tagged { key, tag }, &mut rng);
        assert_eq!(inserted, !first_tag.contains_key(&key));
        first_tag.entry(key).or_insert(tag);
    }
    for t in inorder(&treap) {
        assert_eq!(t.tag, first_tag[&t.key], "key {} was replaced", t.key);
    }
}

/// Build a treap from `keys` with the given RNG.
fn build(keys: &[i32], rng: &mut impl Rng) -> Treap<i32> {
    let mut treap = Treap::new();
    for &k in keys {
        treap.insert(k, rng);
    }
    treap
}

/// `retain` with an always-true predicate leaves the tree exactly as it was:
/// same keys, same priorities, same shape.
#[hegel::test(test_cases = 1000)]
fn retain_all_is_identity(tc: TestCase) {
    let keys: Vec<i32> = tc.draw(gs::vecs(keys()));
    let mut rng = tc.draw(randoms());
    let mut treap = build(&keys, &mut rng);
    let before: Vec<Option<(i32, u32)>> = treap
        .shape()
        .into_iter()
        .map(|o| o.map(|(k, p)| (*k, p)))
        .collect();
    let len = treap.len();
    treap.retain(|_| true);
    let after: Vec<Option<(i32, u32)>> = treap
        .shape()
        .into_iter()
        .map(|o| o.map(|(k, p)| (*k, p)))
        .collect();
    assert_eq!(after, before);
    assert_eq!(treap.len(), len);
}

/// `retain` with an always-false predicate empties the treap.
#[hegel::test(test_cases = 1000)]
fn retain_none_empties(tc: TestCase) {
    let keys: Vec<i32> = tc.draw(gs::vecs(keys()));
    let mut rng = tc.draw(randoms());
    let mut treap = build(&keys, &mut rng);
    treap.retain(|_| false);
    assert!(treap.is_empty());
    assert!(treap.keys().is_empty());
    assert!(treap.arena_ok());
}

/// Removing a present key and re-inserting it restores membership and size,
/// and leaves every other key untouched.
#[hegel::test(test_cases = 1000)]
fn remove_insert_roundtrip(tc: TestCase) {
    let keys: Vec<i32> = tc.draw(gs::vecs(keys()).min_size(1));
    let mut rng = tc.draw(randoms());
    let mut treap = build(&keys, &mut rng);
    let before = inorder(&treap);
    let k = tc.draw(gs::sampled_from(keys));
    assert!(treap.remove(&k));
    assert!(!treap.contains(&k));
    assert_eq!(treap.len(), before.len() - 1);
    assert!(treap.insert(k, &mut rng));
    assert_eq!(inorder(&treap), before);
    assert_eq!(treap.len(), before.len());
}

thread_local! {
    /// Number of `Fused` comparisons left before one panics.
    static FUSE: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// An `i32` whose comparisons panic once the thread-local fuse runs out.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fused(i32);
impl PartialOrd for Fused {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Fused {
    fn cmp(&self, other: &Self) -> Ordering {
        FUSE.with(|f| {
            let left = f.get();
            if left == 0 {
                panic!("Ord::cmp panicked");
            }
            f.set(left - 1);
        });
        self.0.cmp(&other.0)
    }
}

#[derive(Debug, Clone, Copy)]
enum PanickingOp {
    Insert(i32),
    Remove(i32),
    Contains(i32),
    Retain,
}

/// If a user callback panics part-way through an operation (`Ord::cmp` in
/// `insert`/`remove`/`contains`, or the predicate in `retain`), the treap is
/// left a valid treap whose `len()` counts the elements it actually holds.
/// Any of these operations that panics leaves the contents unchanged. (If the
/// fuse outlasts the operation, it must simply have its normal effect.)
#[hegel::test(test_cases = 3000)]
fn consistent_after_panicking_callback(tc: TestCase) {
    let initial: Vec<i32> = tc.draw(gs::vecs(keys()));
    let mut rng = tc.draw(randoms());
    FUSE.with(|f| f.set(usize::MAX));
    let mut treap = Treap::new();
    for &k in &initial {
        treap.insert(Fused(k), &mut rng);
    }
    let before: BTreeSet<i32> = initial.iter().copied().collect();
    let op = tc.draw(hegel::one_of!(
        keys().map(PanickingOp::Insert),
        keys().map(PanickingOp::Remove),
        keys().map(PanickingOp::Contains),
        gs::just(PanickingOp::Retain)
    ));
    let fuse = tc.draw(gs::integers::<usize>().max_value(16));
    let mut rejected = BTreeSet::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        FUSE.with(|f| f.set(fuse));
        match op {
            PanickingOp::Insert(k) => {
                treap.insert(Fused(k), &mut rng);
            }
            PanickingOp::Remove(k) => {
                treap.remove(&Fused(k));
            }
            PanickingOp::Contains(k) => {
                treap.contains(&Fused(k));
            }
            PanickingOp::Retain => {
                // Reject every other element; the fuse counts predicate calls here.
                treap.retain(|k| {
                    let left = FUSE.with(|f| f.get());
                    if left == 0 {
                        panic!("predicate panicked");
                    }
                    FUSE.with(|f| f.set(left - 1));
                    let keep = left % 2 == 0;
                    if !keep {
                        rejected.insert(k.0);
                    }
                    keep
                });
            }
        }
    }));
    FUSE.with(|f| f.set(usize::MAX));
    let mut expected = before.clone();
    if result.is_ok() {
        match op {
            PanickingOp::Insert(k) => {
                expected.insert(k);
            }
            PanickingOp::Remove(k) => {
                expected.remove(&k);
            }
            PanickingOp::Retain => expected.retain(|k| !rejected.contains(k)),
            PanickingOp::Contains(_) => {}
        }
    } else {
        tc.event("panicked");
    }
    let actual: Vec<i32> = inorder(&treap).into_iter().map(|k| k.0).collect();
    assert_eq!(actual, expected.iter().copied().collect::<Vec<_>>());
    assert!(treap.heap_ok(), "heap property violated");
    assert!(treap.arena_ok(), "arena slots are inconsistent");
    assert_eq!(treap.len(), expected.len(), "len() is stale after a panic");
}

/// Scale probe: a large treap built with real randomness, exercising every
/// operation, and finally dropped. Catches per-element recursion depth and
/// accidentally quadratic behaviour that generator-sized inputs never reach.
#[test]
fn scale_probe() {
    const N: i32 = 500_000;
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let mut treap = Treap::new();
    // Sorted insertion order: the worst case for an unbalanced BST.
    for k in 0..N {
        assert!(treap.insert(k, &mut rng));
    }
    assert_eq!(treap.len(), N as usize);
    assert!(treap.heap_ok());
    // A random treap's expected height is ~2.99 log2(n) (Devroye 1986, for
    // random BSTs); 4 log2(n) catches a rotation or priority mistake that
    // leaves the tree correct but unbalanced.
    let h = treap.height();
    assert!(
        h <= 4 * (N as f64).log2().ceil() as usize,
        "height {h} is unbalanced"
    );
    for k in (0..N).step_by(7) {
        assert!(treap.contains(&k));
    }
    let mut kept = 0usize;
    treap.retain(|&k| {
        let keep = k % 2 == 0;
        kept += usize::from(keep);
        keep
    });
    assert_eq!(treap.len(), kept);
    assert_eq!(inorder(&treap), (0..N).step_by(2).collect::<Vec<_>>());
    for k in (0..N).step_by(2) {
        assert!(treap.remove(&k));
    }
    assert!(treap.is_empty());
    for k in (0..N).rev() {
        treap.insert(k, &mut rng);
    }
    drop(treap);
}

/// Scale probe for degenerate priorities: ascending keys with ascending
/// priorities rotate every new node to the root, building a left path of
/// height N. No operation (or drop) may recurse per level, so none of them
/// may overflow the stack.
#[test]
fn scale_probe_degenerate_path() {
    const N: i32 = 200_000;
    let mut rng = PatternRng { next: 0, step: 1 };
    let mut treap = Treap::new();
    for k in 0..N {
        assert!(treap.insert(k, &mut rng));
    }
    assert_eq!(treap.height(), N as usize);
    assert!(treap.contains(&0));
    assert!(!treap.contains(&-1));
    assert!(treap.remove(&0));
    assert!(treap.insert(0, &mut PatternRng { next: 0, step: 0 }));
    treap.retain(|&k| k % 2 == 0);
    assert_eq!(treap.len(), (N / 2) as usize);
    assert!(treap.heap_ok());
    assert!(treap.arena_ok());
    drop(treap);
}
