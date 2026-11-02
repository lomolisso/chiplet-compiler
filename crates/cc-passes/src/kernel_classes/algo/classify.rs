//! Kernel classes: identity graphs bucketed by 1-WL hash, each bucket split into isomorphism classes
//! with VF2 (docs/passes/kernel_classes.md §4.3). Equal hashes prove nothing (Cai, Fürer, Immerman,
//! Combinatorica 12(4):389-410, 1992), so every member is confirmed by an isomorphism, which is its
//! certificate.

use std::collections::BTreeMap;

use cc_ir::absgraph::{Hash128, hash_bytes, isomorphism, wl};
use rayon::prelude::*;

use super::encode::Encoded;
use crate::kernel_classes::json::ZNode;

/// Members of a class: `(item, map)`.
pub type Members = Vec<(usize, Vec<usize>)>;

/// One class: its id, the WL hash, and its members as `(item, map)`: `map[i]` is the item's node that
/// node `i` of the representative (the first member) maps to.
#[derive(Debug, Clone)]
pub struct Found {
    pub id: String,
    pub hash: Hash128,
    pub members: Members,
}

pub fn key(z: &ZNode) -> Vec<u8> {
    postcard::to_stdvec(z).unwrap_or_default()
}

/// The WL hash of an identity graph.
pub fn hash(g: &Encoded, rounds: u32) -> (Hash128, Vec<Hash128>) {
    let w = wl(&g.graph, |z| hash_bytes(&key(z)), rounds);
    (w.hash, w.colors)
}

/// `z<first 16 hex digits of the hash>`, with `-k` for the `k`-th class of a bucket.
pub fn class_id(h: Hash128, k: usize) -> String {
    let base = format!("z{:016x}", (h >> 64) as u64);
    if k == 0 { base } else { format!("{base}-{k}") }
}

/// Classify the items (in their order: the first item of a class is its representative). Returns the
/// classes sorted by id and the number of buckets with more than one class.
pub fn classify(items: &[&Encoded], rounds: u32, budget: u64) -> Result<(Vec<Found>, u64), String> {
    let hashes: Vec<(Hash128, Vec<Hash128>)> = items.par_iter().map(|g| hash(g, rounds)).collect();
    let keys: Vec<Vec<Vec<u8>>> = items
        .par_iter()
        .map(|g| g.graph.nodes.iter().map(key).collect())
        .collect();
    let mut buckets: BTreeMap<Hash128, Vec<usize>> = BTreeMap::new();
    for (i, (h, _)) in hashes.iter().enumerate() {
        buckets.entry(*h).or_default().push(i);
    }
    let per: Vec<Result<Vec<Found>, String>> = buckets
        .par_iter()
        .map(|(h, idx)| {
            let mut reps: Vec<(usize, Members)> = Vec::new();
            'item: for &i in idx {
                for (r, members) in reps.iter_mut() {
                    let (cr, ci) = (&hashes[*r].1, &hashes[i].1);
                    let (kr, ki) = (&keys[*r], &keys[i]);
                    let m = isomorphism(
                        &items[*r].graph,
                        &items[i].graph,
                        |a, b| cr[a] == ci[b] && kr[a] == ki[b],
                        budget,
                    )
                    .map_err(|e| format!("VF2 over budget ({} pairs) in bucket {h:032x}", e.0))?;
                    if let Some(m) = m {
                        members.push((i, m));
                        continue 'item;
                    }
                }
                let id: Vec<usize> = (0..items[i].graph.node_count()).collect();
                reps.push((i, vec![(i, id)]));
            }
            Ok(reps
                .into_iter()
                .enumerate()
                .map(|(k, (_, members))| Found {
                    id: class_id(*h, k),
                    hash: *h,
                    members,
                })
                .collect())
        })
        .collect();
    let mut out = Vec::new();
    let mut split = 0;
    for p in per {
        let p = p?;
        if p.len() > 1 {
            split += 1;
        }
        out.extend(p);
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((out, split))
}
