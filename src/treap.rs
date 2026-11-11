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

use rand::{Rng, RngExt};
use std::cmp::Ordering;

/// A node in the treap
struct Node<T> {
    key: T,
    priority: u32,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

impl<T> Node<T> {
    fn new(key: T, priority: u32) -> Self {
        Node {
            key,
            priority,
            left: None,
            right: None,
        }
    }
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
    root: Option<Box<Node<T>>>,
    size: usize,
}

impl<T: Ord> Treap<T> {
    /// Create a new empty treap
    pub fn new() -> Self {
        Treap {
            root: None,
            size: 0,
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
        let inserted = Self::insert_at(&mut self.root, key, priority);
        if inserted {
            self.size += 1;
        }
        inserted
    }

    /// Check if the treap contains a key
    pub fn contains(&self, key: &T) -> bool {
        Self::contains_node(&self.root, key)
    }

    /// Remove a key from the treap
    ///
    /// Returns `true` if the key was present.
    pub fn remove(&mut self, key: &T) -> bool {
        // Find the link holding the key. All comparisons happen before any
        // mutation, so a panicking `Ord` leaves the treap untouched.
        let mut link = &mut self.root;
        loop {
            let ord = match link.as_deref() {
                None => return false,
                Some(n) => key.cmp(&n.key),
            };
            link = match ord {
                Ordering::Equal => break,
                Ordering::Less => &mut link.as_mut().expect("link is non-empty").left,
                Ordering::Greater => &mut link.as_mut().expect("link is non-empty").right,
            };
        }
        let node = link.take().expect("link holds the key");
        *link = Self::merge(node.left, node.right);
        self.size -= 1;
        true
    }

    /// Clear the treap
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.root = None;
        self.size = 0;
    }

    /// Apply a function to each element, removing those for which it returns false
    ///
    /// The predicate is called exactly once per element, in post-order (left
    /// subtree, right subtree, node), not in key order. If it panics, the
    /// treap keeps every element it had not yet rejected and remains valid.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        /// Keeps `size` in step with the tree even if the predicate panics.
        struct SizeGuard<'a> {
            size: &'a mut usize,
            removed: usize,
        }
        impl Drop for SizeGuard<'_> {
            fn drop(&mut self) {
                *self.size -= self.removed;
            }
        }
        let mut guard = SizeGuard {
            size: &mut self.size,
            removed: 0,
        };
        Self::retain_at(&mut self.root, &mut f, &mut guard.removed);
    }

    // Insert below `link`, rotating the new node up to restore heap order.
    // Comparisons happen on the way down, before any mutation, so a panicking
    // `Ord` leaves the treap untouched. Returns whether the key was inserted.
    fn insert_at(link: &mut Option<Box<Node<T>>>, key: T, priority: u32) -> bool {
        let Some(node) = link else {
            *link = Some(Box::new(Node::new(key, priority)));
            return true;
        };
        match key.cmp(&node.key) {
            Ordering::Less => {
                if !Self::insert_at(&mut node.left, key, priority) {
                    return false;
                }
                if node
                    .left
                    .as_ref()
                    .is_some_and(|l| l.priority > node.priority)
                {
                    Self::rotate_right(link);
                }
                true
            }
            Ordering::Greater => {
                if !Self::insert_at(&mut node.right, key, priority) {
                    return false;
                }
                if node
                    .right
                    .as_ref()
                    .is_some_and(|r| r.priority > node.priority)
                {
                    Self::rotate_left(link);
                }
                true
            }
            Ordering::Equal => false, // Key already exists, do nothing
        }
    }

    // Helper function to check if a node contains a key
    fn contains_node(node: &Option<Box<Node<T>>>, key: &T) -> bool {
        match node {
            None => false,
            Some(n) => match key.cmp(&n.key) {
                Ordering::Less => Self::contains_node(&n.left, key),
                Ordering::Greater => Self::contains_node(&n.right, key),
                Ordering::Equal => true,
            },
        }
    }

    // Merge two subtrees, where every key in `left` is less than every key in `right`
    fn merge(left: Option<Box<Node<T>>>, right: Option<Box<Node<T>>>) -> Option<Box<Node<T>>> {
        match (left, right) {
            (None, right) => right,
            (left, None) => left,
            (Some(l), Some(r)) => {
                if l.priority > r.priority {
                    let mut l = l;
                    l.right = Self::merge(l.right, Some(r));
                    Some(l)
                } else {
                    let mut r = r;
                    r.left = Self::merge(Some(l), r.left);
                    Some(r)
                }
            }
        }
    }

    // Rotate the subtree at `link` right: its left child becomes its root
    fn rotate_right(link: &mut Option<Box<Node<T>>>) {
        let mut node = link.take().expect("rotate_right on empty link");
        let mut new_root = node.left.take().expect("rotate_right without left child");
        node.left = new_root.right.take();
        new_root.right = Some(node);
        *link = Some(new_root);
    }

    // Rotate the subtree at `link` left: its right child becomes its root
    fn rotate_left(link: &mut Option<Box<Node<T>>>) {
        let mut node = link.take().expect("rotate_left on empty link");
        let mut new_root = node.right.take().expect("rotate_left without right child");
        node.right = new_root.left.take();
        new_root.left = Some(node);
        *link = Some(new_root);
    }

    // Retain nodes that satisfy the predicate, in place and in post-order.
    // Each node stays attached until the predicate has accepted or rejected
    // it, so a panic leaves a valid treap; `removed` counts rejected nodes.
    fn retain_at<F>(link: &mut Option<Box<Node<T>>>, f: &mut F, removed: &mut usize)
    where
        F: FnMut(&T) -> bool,
    {
        let Some(node) = link else { return };
        Self::retain_at(&mut node.left, f, removed);
        Self::retain_at(&mut node.right, f, removed);
        if !f(&node.key) {
            let node = link.take().expect("link is non-empty");
            *link = Self::merge(node.left, node.right);
            *removed += 1;
        }
    }
}

impl<T: Ord> Default for Treap<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // Helper: collect keys via in-order traversal (should yield sorted order for valid BST)
    fn collect_inorder<T: Ord + Clone>(node: &Option<Box<Node<T>>>) -> Vec<T> {
        match node {
            None => vec![],
            Some(n) => {
                let mut result = collect_inorder(&n.left);
                result.push(n.key.clone());
                result.extend(collect_inorder(&n.right));
                result
            }
        }
    }

    // Helper: verify max-heap property (parent priority >= child priorities)
    fn verify_heap_property<T: Ord>(node: &Option<Box<Node<T>>>) -> bool {
        match node {
            None => true,
            Some(n) => {
                let left_ok = n.left.as_ref().is_none_or(|l| n.priority >= l.priority);
                let right_ok = n.right.as_ref().is_none_or(|r| n.priority >= r.priority);
                left_ok
                    && right_ok
                    && verify_heap_property(&n.left)
                    && verify_heap_property(&n.right)
            }
        }
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
        let inorder = collect_inorder(&treap.root);
        let mut sorted = elements.clone();
        sorted.sort();
        assert_eq!(inorder, sorted);

        // Test after some removals
        treap.remove(&25);
        treap.remove(&75);
        let inorder_after = collect_inorder(&treap.root);
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
                verify_heap_property(&treap.root),
                "Heap property violated after inserting {}",
                i
            );
        }

        // Test after removals
        for i in (0..100).step_by(3) {
            treap.remove(&i);
            assert!(
                verify_heap_property(&treap.root),
                "Heap property violated after removing {}",
                i
            );
        }

        // Test after retain
        treap.retain(|&x| x % 2 == 0);
        assert!(
            verify_heap_property(&treap.root),
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
        let inorder = collect_inorder(&treap.root);
        let expected: Vec<i32> = (0..1000).collect();
        assert_eq!(inorder, expected);
        assert!(verify_heap_property(&treap.root));
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

