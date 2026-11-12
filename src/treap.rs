//! A randomized binary search tree (treap) implementation
//!
//! A treap maintains both BST property (for keys) and heap property (for priorities).
//!
//! This implementation was inspired by the treap exploration in <https://github.com/apanda/cvm>
//! (BSD-2-Clause license), but is an independent implementation tailored specifically
//! for the CVM algorithm's requirements.
//!
//! ## Key Differences from apanda/cvm treap:
//!
//! 1. **Simpler structure**: We don't use a separate Element type; keys and priorities are
//!    stored directly in nodes
//! 2. **Random priorities**: apanda's implementation expects explicit priorities, while ours
//!    generates random priorities at insertion time
//! 3. **No allocation tracking**: apanda uses `alloc_counter` for performance analysis
//! 4. **Simplified delete**: Our delete returns a bool, apanda's has more complex handling
//! 5. **Retain operation**: We added a specialized `retain` method for CVM's "clear half"
//! 6. **No Display trait**: We focus on the minimal API needed for CVM
//! 7. **Insert behavior**: apanda's `insert_or_replace` updates existing elements; ours
//!    keeps the original (no update) which is what CVM needs
//!
//! ## Design Decisions
//!
//! Unlike general-purpose treap implementations, this one is optimized for CVM:
//! - No key-value mapping: CVM only needs to track unique elements
//! - Simplified API: Only operations needed for CVM are implemented
//! - Efficient `retain`: Optimized for the "clear about half" operation
//! - RNG integration: Accepts an external RNG for consistent randomness
//!
//! ## Layout
//!
//! Nodes live in a single `Vec` arena and refer to their children by `u32`
//! index rather than through `Box` pointers. For small keys this halves the
//! per-node footprint (an `i32` node is 16 bytes, against a 32-byte heap
//! allocation per boxed node), so more of the tree stays in cache, and
//! inserting no longer allocates once the arena has grown. Slots freed by
//! `remove` are reused by later inserts; `retain` compacts the arena.
//!
//! Every operation is iterative, so tree height (which is O(log n) only in
//! expectation) never translates into recursion depth.

use rand::{Rng, RngExt};
use std::borrow::Borrow;
use std::cmp::Ordering;

/// Index of a node in the arena; `NIL` marks an empty subtree.
type Idx = u32;
const NIL: Idx = Idx::MAX;
const LEFT: usize = 0;
const RIGHT: usize = 1;

/// A node in the treap
struct Node<T> {
    key: T,
    priority: u32,
    /// Children, indexed by `LEFT` and `RIGHT`. An array (rather than two
    /// fields) lets a search pick the next node without a branch.
    child: [Idx; 2],
}

/// A position that holds a subtree: the root, or a child field of a node.
#[derive(Clone, Copy)]
enum Link {
    Root,
    Left(Idx),
    Right(Idx),
}

/// A treap data structure
///
/// Key differences from typical treap implementations:
/// 1. Priorities are generated at insertion time using the provided RNG
/// 2. The `retain` operation is optimized for the CVM algorithm's "clear half" operation
/// 3. No support for key-value pairs - only keys are stored (values are implicit)
/// 4. No split operation as it's not needed for CVM
/// 5. Insert doesn't update existing keys - matching CVM's requirement
pub struct Treap<T> {
    /// Node arena. Slots listed in `free` are dead: unreachable from `root`,
    /// awaiting reuse. Their keys are dropped when the slot is reused, or when
    /// `retain` or `clear` compacts the arena.
    nodes: Vec<Node<T>>,
    free: Vec<Idx>,
    root: Idx,
    size: usize,
    /// Scratch space for `insert`'s search path, kept to avoid reallocating.
    path: Vec<(Idx, bool)>,
}

impl<T> Treap<T> {
    fn node(&self, i: Idx) -> &Node<T> {
        &self.nodes[i as usize]
    }

    fn node_mut(&mut self, i: Idx) -> &mut Node<T> {
        &mut self.nodes[i as usize]
    }

    fn set(&mut self, link: Link, child: Idx) {
        match link {
            Link::Root => self.root = child,
            Link::Left(p) => self.node_mut(p).child[LEFT] = child,
            Link::Right(p) => self.node_mut(p).child[RIGHT] = child,
        }
    }

    /// Store a new, unlinked node, reusing a dead slot if there is one.
    fn alloc(&mut self, key: T, priority: u32) -> Idx {
        let node = Node {
            key,
            priority,
            child: [NIL, NIL],
        };
        if let Some(i) = self.free.pop() {
            *self.node_mut(i) = node;
            i
        } else {
            let i = Idx::try_from(self.nodes.len())
                .ok()
                .filter(|&i| i != NIL)
                .expect("treap capacity exceeded");
            self.nodes.push(node);
            i
        }
    }

    /// Merge two treaps, where every key in `left` is less than every key in
    /// `right`, returning the root of the result. Compares priorities only.
    fn merge(&mut self, mut left: Idx, mut right: Idx) -> Idx {
        let mut root = NIL;
        // Where the next piece of the result goes; `None` means `root`.
        let mut hole: Option<Link> = None;
        loop {
            let piece = if left == NIL || right == NIL {
                if left == NIL { right } else { left }
            } else if self.node(left).priority > self.node(right).priority {
                left
            } else {
                right
            };
            match hole {
                None => root = piece,
                Some(link) => self.set(link, piece),
            }
            if left == NIL || right == NIL {
                return root;
            }
            // The higher-priority root stays on top; merge the rest beneath it.
            if piece == left {
                hole = Some(Link::Right(left));
                left = self.node(left).child[RIGHT];
            } else {
                hole = Some(Link::Left(right));
                right = self.node(right).child[LEFT];
            }
        }
    }
}

impl<T: Ord> Treap<T> {
    /// Create a new empty treap
    pub fn new() -> Self {
        Treap {
            nodes: Vec::new(),
            free: Vec::new(),
            root: NIL,
            size: 0,
            path: Vec::new(),
        }
    }

    /// Get the number of elements in the treap
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the treap is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Insert a key with a random priority
    ///
    /// Returns `true` if the key was inserted, `false` if it already existed
    /// (in which case the existing element is kept unchanged). A priority is
    /// drawn from `rng` in either case.
    pub fn insert<R: Rng>(&mut self, key: T, rng: &mut R) -> bool {
        let priority = rng.random();
        // Search, recording the path. All comparisons happen here, before any
        // mutation, so a panicking `Ord` leaves the treap untouched.
        self.path.clear();
        let mut i = self.root;
        while i != NIL {
            let n = &self.nodes[i as usize];
            let went_left = match key.cmp(&n.key) {
                Ordering::Less => true,
                Ordering::Greater => false,
                Ordering::Equal => return false,
            };
            self.path.push((i, went_left));
            i = if went_left {
                n.child[LEFT]
            } else {
                n.child[RIGHT]
            };
        }
        // Attach as a leaf, then rotate up while it outranks its parent.
        let x = self.alloc(key, priority);
        let mut link = match self.path.last() {
            None => Link::Root,
            Some(&(p, true)) => Link::Left(p),
            Some(&(p, false)) => Link::Right(p),
        };
        self.set(link, x);
        while let Some((p, went_left)) = self.path.pop() {
            if self.node(x).priority <= self.node(p).priority {
                break;
            }
            if went_left {
                let moved = self.node(x).child[RIGHT];
                self.node_mut(p).child[LEFT] = moved;
                self.node_mut(x).child[RIGHT] = p;
            } else {
                let moved = self.node(x).child[LEFT];
                self.node_mut(p).child[RIGHT] = moved;
                self.node_mut(x).child[LEFT] = p;
            }
            link = match self.path.last() {
                None => Link::Root,
                Some(&(g, true)) => Link::Left(g),
                Some(&(g, false)) => Link::Right(g),
            };
            self.set(link, x);
        }
        self.size += 1;
        true
    }

    /// Check if the treap contains a key
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut i = self.root;
        while i != NIL {
            let n = self.node(i);
            let ord = key.cmp(n.key.borrow());
            if ord == Ordering::Equal {
                return true;
            }
            i = n.child[usize::from(ord == Ordering::Greater)];
        }
        false
    }

    /// Remove a key from the treap
    ///
    /// Returns `true` if the key was present.
    pub fn remove<Q>(&mut self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // All comparisons happen before any mutation, so a panicking `Ord`
        // leaves the treap untouched.
        let mut link = Link::Root;
        let mut i = self.root;
        loop {
            if i == NIL {
                return false;
            }
            let n = self.node(i);
            match key.cmp(n.key.borrow()) {
                Ordering::Less => {
                    link = Link::Left(i);
                    i = n.child[LEFT];
                }
                Ordering::Greater => {
                    link = Link::Right(i);
                    i = n.child[RIGHT];
                }
                Ordering::Equal => break,
            }
        }
        let (left, right) = (self.node(i).child[LEFT], self.node(i).child[RIGHT]);
        let merged = self.merge(left, right);
        self.set(link, merged);
        self.free.push(i);
        self.size -= 1;
        true
    }

    /// Clear the treap
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.root = NIL;
        self.size = 0;
    }

    /// Apply a function to each element, removing those for which it returns false
    ///
    /// The predicate is called exactly once per element, in post-order (left
    /// subtree, right subtree, node), not in key order. If it panics, the
    /// treap is left unchanged.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        // Phase 1: ask the predicate about every element, mutating nothing.
        let mut keep = vec![false; self.nodes.len()];
        let mut kept = 0;
        let mut stack: Vec<(Idx, bool)> = Vec::new();
        if self.root != NIL {
            stack.push((self.root, false));
        }
        while let Some((i, children_done)) = stack.pop() {
            let n = self.node(i);
            if children_done {
                let k = f(&n.key);
                keep[i as usize] = k;
                kept += usize::from(k);
            } else {
                stack.push((i, true));
                for child in [n.child[RIGHT], n.child[LEFT]] {
                    if child != NIL {
                        stack.push((child, false));
                    }
                }
            }
        }
        if kept == self.size {
            return;
        }

        // Phase 2: compact the arena to the kept nodes (dropping the rest,
        // including dead slots), remembering where each one moved to.
        let mut new_index = vec![NIL; self.nodes.len()];
        let in_order = self.in_order_slots(&keep);
        let old = std::mem::take(&mut self.nodes);
        self.nodes.reserve_exact(kept);
        for (i, node) in old.into_iter().enumerate() {
            if keep[i] {
                new_index[i] = self.nodes.len() as Idx;
                self.nodes.push(Node {
                    child: [NIL, NIL],
                    ..node
                });
            }
        }
        self.free.clear();
        self.size = kept;

        // Phase 3: rebuild the tree from the kept nodes in key order. A treap
        // is the Cartesian tree of its keys and priorities, which a stack
        // builds in linear time; the stack holds the current right spine.
        let mut spine: Vec<Idx> = Vec::new();
        for old_i in in_order {
            let x = new_index[old_i as usize];
            let p = self.node(x).priority;
            let mut last = NIL;
            while let Some(&top) = spine.last() {
                if self.node(top).priority >= p {
                    break;
                }
                last = top;
                spine.pop();
            }
            self.node_mut(x).child[LEFT] = last;
            if let Some(&top) = spine.last() {
                self.node_mut(top).child[RIGHT] = x;
            }
            spine.push(x);
        }
        self.root = spine.first().copied().unwrap_or(NIL);
    }

    /// Slots of the nodes marked in `keep`, in key (in-order) order.
    fn in_order_slots(&self, keep: &[bool]) -> Vec<Idx> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        let mut i = self.root;
        loop {
            while i != NIL {
                stack.push(i);
                i = self.node(i).child[LEFT];
            }
            match stack.pop() {
                None => return out,
                Some(n) => {
                    if keep[n as usize] {
                        out.push(n);
                    }
                    i = self.node(n).child[RIGHT];
                }
            }
        }
    }
}

impl<T: Ord> Default for Treap<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Read-only views of the tree, for the tests' structural checks.
#[cfg(test)]
impl<T> Treap<T> {
    /// Keys in order (an in-order walk).
    fn keys(&self) -> Vec<&T> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        let mut i = self.root;
        loop {
            while i != NIL {
                stack.push(i);
                i = self.node(i).child[LEFT];
            }
            match stack.pop() {
                None => return out,
                Some(n) => {
                    out.push(&self.node(n).key);
                    assert!(out.len() <= self.nodes.len(), "cycle in tree");
                    i = self.node(n).child[RIGHT];
                }
            }
        }
    }

    /// (slot, depth) for every reachable node, in pre-order.
    fn reachable(&self) -> Vec<(Idx, usize)> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        if self.root != NIL {
            stack.push((self.root, 1));
        }
        while let Some((i, d)) = stack.pop() {
            out.push((i, d));
            assert!(out.len() <= self.nodes.len(), "cycle in tree");
            let n = self.node(i);
            for child in [n.child[RIGHT], n.child[LEFT]] {
                if child != NIL {
                    stack.push((child, d + 1));
                }
            }
        }
        out
    }

    /// Every parent's priority is >= its children's.
    fn heap_ok(&self) -> bool {
        self.reachable().iter().all(|&(i, _)| {
            let n = self.node(i);
            [n.child[LEFT], n.child[RIGHT]]
                .into_iter()
                .all(|c| c == NIL || self.node(c).priority <= n.priority)
        })
    }

    /// Height: the number of nodes on the longest root-to-leaf path.
    fn height(&self) -> usize {
        self.reachable().iter().map(|&(_, d)| d).max().unwrap_or(0)
    }

    /// The tree's shape: (key, priority) in pre-order, with `None` for
    /// empty subtrees, independent of where nodes sit in the arena.
    fn shape(&self) -> Vec<Option<(&T, u32)>> {
        let mut out = Vec::new();
        let mut stack = vec![self.root];
        while let Some(i) = stack.pop() {
            assert!(out.len() <= 2 * self.nodes.len() + 1, "cycle in tree");
            if i == NIL {
                out.push(None);
            } else {
                let n = self.node(i);
                out.push(Some((&n.key, n.priority)));
                stack.push(n.child[RIGHT]);
                stack.push(n.child[LEFT]);
            }
        }
        out
    }

    /// Arena bookkeeping holds: every slot is either reachable (exactly
    /// once) or free (exactly once), never both, and `size` counts the
    /// reachable ones.
    fn arena_ok(&self) -> bool {
        let mut seen = vec![0u8; self.nodes.len()];
        let reachable = self.reachable();
        for &(i, _) in &reachable {
            seen[i as usize] += 1;
        }
        for &i in &self.free {
            seen[i as usize] += 1;
        }
        reachable.len() == self.size && seen.iter().all(|&c| c == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // Helper: collect keys via in-order traversal (should yield sorted order for valid BST)
    fn collect_inorder<T: Clone>(treap: &Treap<T>) -> Vec<T> {
        treap.keys().into_iter().cloned().collect()
    }

    // Helper: verify max-heap property (parent priority >= child priorities)
    fn verify_heap_property<T>(treap: &Treap<T>) -> bool {
        treap.heap_ok()
    }

    #[test]
    fn test_insert_and_contains() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(42);

        treap.insert(5, &mut rng);
        treap.insert(3, &mut rng);
        treap.insert(7, &mut rng);

        assert!(treap.contains(&5));
        assert!(treap.contains(&3));
        assert!(treap.contains(&7));
        assert!(!treap.contains(&1));
        assert_eq!(treap.len(), 3);
    }

    #[test]
    fn test_remove() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(42);

        treap.insert(5, &mut rng);
        treap.insert(3, &mut rng);
        treap.insert(7, &mut rng);

        assert!(treap.remove(&3));
        assert!(!treap.contains(&3));
        assert_eq!(treap.len(), 2);

        assert!(!treap.remove(&3)); // Already removed
    }

    #[test]
    fn test_borrowed_keys() {
        let mut treap: Treap<String> = Treap::new();
        let mut rng = StdRng::seed_from_u64(42);

        treap.insert("b".to_owned(), &mut rng);
        treap.insert("a".to_owned(), &mut rng);

        assert!(treap.contains("a"));
        assert!(!treap.contains("c"));
        assert!(treap.remove("b"));
        assert!(!treap.contains("b"));
        assert_eq!(treap.len(), 1);
    }

    #[test]
    fn test_retain() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..10 {
            treap.insert(i, &mut rng);
        }

        treap.retain(|&x| x % 2 == 0);
        assert_eq!(treap.len(), 5);

        for i in 0..10 {
            if i % 2 == 0 {
                assert!(treap.contains(&i));
            } else {
                assert!(!treap.contains(&i));
            }
        }
    }

    #[test]
    fn test_duplicate_insertion() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Insert elements
        assert!(treap.insert(5, &mut rng)); // First insertion returns true
        assert!(treap.insert(3, &mut rng));
        assert!(treap.insert(7, &mut rng));
        assert_eq!(treap.len(), 3);

        // Try to insert duplicates - should return false and not change size
        assert!(!treap.insert(5, &mut rng));
        assert!(!treap.insert(3, &mut rng));
        assert!(!treap.insert(7, &mut rng));
        assert_eq!(treap.len(), 3); // Size unchanged

        // Verify elements still exist
        assert!(treap.contains(&5));
        assert!(treap.contains(&3));
        assert!(treap.contains(&7));

        // Remove and re-insert should work
        assert!(treap.remove(&5));
        assert_eq!(treap.len(), 2);
        assert!(treap.insert(5, &mut rng)); // Re-insertion returns true
        assert_eq!(treap.len(), 3);
    }

    #[test]
    fn test_bst_property() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(123);

        // Insert elements in random order
        let elements = vec![50, 25, 75, 10, 30, 60, 90, 5, 15, 27, 35];
        for elem in &elements {
            treap.insert(*elem, &mut rng);
        }

        // In-order traversal should yield sorted keys
        let inorder = collect_inorder(&treap);
        let mut sorted = elements.clone();
        sorted.sort();
        assert_eq!(inorder, sorted);

        // Test after some removals
        treap.remove(&25);
        treap.remove(&75);
        let inorder_after = collect_inorder(&treap);
        let expected: Vec<i32> = sorted.into_iter().filter(|&x| x != 25 && x != 75).collect();
        assert_eq!(inorder_after, expected);
    }

    #[test]
    fn test_heap_property() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(456);

        // Insert many elements
        for i in 0..100 {
            treap.insert(i, &mut rng);
            assert!(
                verify_heap_property(&treap),
                "Heap property violated after inserting {}",
                i
            );
        }

        // Test after removals
        for i in (0..100).step_by(3) {
            treap.remove(&i);
            assert!(
                verify_heap_property(&treap),
                "Heap property violated after removing {}",
                i
            );
        }

        // Test after retain
        treap.retain(|&x| x % 2 == 0);
        assert!(
            verify_heap_property(&treap),
            "Heap property violated after retain"
        );
    }

    #[test]
    fn test_stress_insert_remove() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(789);

        // Insert 1000 elements
        for i in 0..1000 {
            treap.insert(i, &mut rng);
        }
        assert_eq!(treap.len(), 1000);

        // Verify all elements present
        for i in 0..1000 {
            assert!(treap.contains(&i), "Element {} should be present", i);
        }

        // Remove every other element
        for i in (0..1000).step_by(2) {
            assert!(treap.remove(&i), "Should remove {}", i);
        }
        assert_eq!(treap.len(), 500);

        // Verify correct elements remain
        for i in 0..1000 {
            if i % 2 == 0 {
                assert!(!treap.contains(&i), "Element {} should be removed", i);
            } else {
                assert!(treap.contains(&i), "Element {} should remain", i);
            }
        }

        // Re-insert removed elements
        for i in (0..1000).step_by(2) {
            assert!(treap.insert(i, &mut rng), "Should insert {}", i);
        }
        assert_eq!(treap.len(), 1000);

        // Verify invariants
        let inorder = collect_inorder(&treap);
        let expected: Vec<i32> = (0..1000).collect();
        assert_eq!(inorder, expected);
        assert!(verify_heap_property(&treap));
    }

    #[test]
    fn test_empty_tree_operations() {
        let mut treap: Treap<i32> = Treap::new();
        let mut rng = StdRng::seed_from_u64(999);

        // Operations on empty treap
        assert!(treap.is_empty());
        assert_eq!(treap.len(), 0);
        assert!(!treap.contains(&42));
        assert!(!treap.remove(&42));

        // Retain on empty treap (should be no-op)
        treap.retain(|_| true);
        assert!(treap.is_empty());

        // Clear on empty treap
        treap.clear();
        assert!(treap.is_empty());

        // Insert then clear
        treap.insert(1, &mut rng);
        treap.insert(2, &mut rng);
        assert_eq!(treap.len(), 2);
        treap.clear();
        assert!(treap.is_empty());
        assert!(!treap.contains(&1));
        assert!(!treap.contains(&2));
    }

    #[test]
    fn test_single_element() {
        let mut treap = Treap::new();
        let mut rng = StdRng::seed_from_u64(111);

        // Single element operations
        treap.insert(42, &mut rng);
        assert_eq!(treap.len(), 1);
        assert!(treap.contains(&42));
        assert!(!treap.contains(&0));

        // Duplicate of single element
        assert!(!treap.insert(42, &mut rng));
        assert_eq!(treap.len(), 1);

        // Remove single element
        assert!(treap.remove(&42));
        assert!(treap.is_empty());
        assert!(!treap.contains(&42));

        // Re-insert
        treap.insert(42, &mut rng);
        assert_eq!(treap.len(), 1);

        // Retain that keeps the element
        treap.retain(|&x| x == 42);
        assert_eq!(treap.len(), 1);

        // Retain that removes the element
        treap.retain(|&x| x != 42);
        assert!(treap.is_empty());
    }
}

#[cfg(test)]
mod proptests;
