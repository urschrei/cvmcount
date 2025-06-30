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

use rand::Rng;
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
    pub fn insert<R: Rng>(&mut self, key: T, rng: &mut R) {
        let priority = rng.random();
        self.root = Self::insert_node(self.root.take(), key, priority);
        self.size += 1;
    }

    /// Check if the treap contains a key
    pub fn contains(&self, key: &T) -> bool {
        Self::contains_node(&self.root, key)
    }

    /// Remove a key from the treap
    pub fn remove(&mut self, key: &T) -> bool {
        let (new_root, removed) = Self::remove_node(self.root.take(), key);
        self.root = new_root;
        if removed {
            self.size -= 1;
        }
        removed
    }

    /// Clear the treap
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.root = None;
        self.size = 0;
    }

    /// Apply a function to each element, removing those for which it returns false
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let (new_root, new_size) = Self::retain_node(self.root.take(), &mut f);
        self.root = new_root;
        self.size = new_size;
    }

    // Helper function to insert a node
    fn insert_node(node: Option<Box<Node<T>>>, key: T, priority: u32) -> Option<Box<Node<T>>> {
        match node {
            None => Some(Box::new(Node::new(key, priority))),
            Some(mut n) => {
                match key.cmp(&n.key) {
                    Ordering::Less => {
                        n.left = Self::insert_node(n.left, key, priority);
                        // Maintain heap property
                        if n.left.as_ref().unwrap().priority > n.priority {
                            Self::rotate_right(n)
                        } else {
                            Some(n)
                        }
                    }
                    Ordering::Greater => {
                        n.right = Self::insert_node(n.right, key, priority);
                        // Maintain heap property
                        if n.right.as_ref().unwrap().priority > n.priority {
                            Self::rotate_left(n)
                        } else {
                            Some(n)
                        }
                    }
                    Ordering::Equal => Some(n), // Key already exists, do nothing
                }
            }
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

    // Helper function to remove a node
    fn remove_node(node: Option<Box<Node<T>>>, key: &T) -> (Option<Box<Node<T>>>, bool) {
        match node {
            None => (None, false),
            Some(mut n) => match key.cmp(&n.key) {
                Ordering::Less => {
                    let (new_left, removed) = Self::remove_node(n.left, key);
                    n.left = new_left;
                    (Some(n), removed)
                }
                Ordering::Greater => {
                    let (new_right, removed) = Self::remove_node(n.right, key);
                    n.right = new_right;
                    (Some(n), removed)
                }
                Ordering::Equal => {
                    // Found the node to remove
                    (Self::merge(n.left, n.right), true)
                }
            },
        }
    }

    // Merge two subtrees
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

    // Rotate right
    fn rotate_right(mut node: Box<Node<T>>) -> Option<Box<Node<T>>> {
        let mut new_root = node.left.take().unwrap();
        node.left = new_root.right.take();
        new_root.right = Some(node);
        Some(new_root)
    }

    // Rotate left
    fn rotate_left(mut node: Box<Node<T>>) -> Option<Box<Node<T>>> {
        let mut new_root = node.right.take().unwrap();
        node.right = new_root.left.take();
        new_root.left = Some(node);
        Some(new_root)
    }

    // Retain nodes that satisfy the predicate
    fn retain_node<F>(node: Option<Box<Node<T>>>, f: &mut F) -> (Option<Box<Node<T>>>, usize)
    where
        F: FnMut(&T) -> bool,
    {
        match node {
            None => (None, 0),
            Some(mut n) => {
                let (new_left, left_size) = Self::retain_node(n.left, f);
                let (new_right, right_size) = Self::retain_node(n.right, f);

                if f(&n.key) {
                    n.left = new_left;
                    n.right = new_right;
                    (Some(n), left_size + right_size + 1)
                } else {
                    // Remove this node by merging its subtrees
                    let merged = Self::merge(new_left, new_right);
                    (merged, left_size + right_size)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
}
