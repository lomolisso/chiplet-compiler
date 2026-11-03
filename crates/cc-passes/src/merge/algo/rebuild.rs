//! The merged unit of a clique (Moreano et al. 2005, Sec. V): each vertex mapping becomes one block
//! (members of a group block by block), every other block is copied; the configurations of both sides
//! are carried over through the block correspondence, and a chosen arc mapping that exchanges a
//! commutative pair flips B's orientation on that block. Wires follow from the configurations
//! (`Fu::wiring`): a shared wire is one source, the others add multiplexer inputs.

use std::collections::BTreeMap;

use super::compat::Model;
use super::fu::{FBlock, Fu, UCfg};

pub fn rebuild(a: &Fu, b: &Fu, m: &Model, chosen: &[usize], host: &str) -> Fu {
    let nv = m.vm.len();
    // Per block of A and of B: the vertex mapping (and member) that merges it.
    let mut a_of: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    let mut b_of: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    for &i in chosen.iter().filter(|&&i| i < nv) {
        let v = &m.vm[i];
        for (k, (&x, &y)) in m.sa.members[v.sa as usize]
            .iter()
            .zip(&m.sb.members[v.sb as usize])
            .enumerate()
        {
            a_of.insert(x, (i, k));
            b_of.insert(y, (i, k));
        }
    }
    let mut flip = vec![false; b.blocks.len()];
    for &i in chosen.iter().filter(|&&i| i >= nv) {
        let am = &m.am[i - nv];
        if am.perm {
            flip[am.b.2 as usize] = true;
        }
    }

    let mut blocks: Vec<FBlock> = Vec::new();
    let mut groups: Vec<String> = Vec::new();
    let mut ma = vec![u32::MAX; a.blocks.len()];
    let mut mb = vec![u32::MAX; b.blocks.len()];
    let mut merged: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    let mut group_a: BTreeMap<u32, u32> = BTreeMap::new();
    let mut group_b: BTreeMap<u32, u32> = BTreeMap::new();
    let mut group_vm: BTreeMap<usize, u32> = BTreeMap::new();

    let new_group = |groups: &mut Vec<String>, pattern: &str| -> u32 {
        groups.push(pattern.to_string());
        groups.len() as u32 - 1
    };

    for (x, bx) in a.blocks.iter().enumerate() {
        match a_of.get(&(x as u32)) {
            Some(&(i, k)) => {
                let v = &m.vm[i];
                let y = m.sb.members[v.sb as usize][k];
                let by = &b.blocks[y as usize];
                let group = bx.group.map(|g| {
                    *group_vm
                        .entry(i)
                        .or_insert_with(|| new_group(&mut groups, &a.groups[g as usize]))
                });
                let literal = bx.literal && by.literal && bx.attrs == by.attrs;
                let mut attrs = bx.attrs.clone();
                if !literal {
                    attrs.remove("value");
                }
                let id = blocks.len() as u32;
                blocks.push(FBlock {
                    kind: bx.kind,
                    op: bx.op.clone(),
                    attrs,
                    label: bx.label.clone(),
                    types: v.types[k].clone(),
                    literal,
                    group,
                    member: bx.member,
                });
                merged.insert((i, k), id);
                ma[x] = id;
                mb[y as usize] = id;
            }
            None => {
                let group = bx.group.map(|g| {
                    *group_a
                        .entry(g)
                        .or_insert_with(|| new_group(&mut groups, &a.groups[g as usize]))
                });
                ma[x] = blocks.len() as u32;
                blocks.push(FBlock { group, ..bx.clone() });
            }
        }
    }
    for (y, by) in b.blocks.iter().enumerate() {
        if mb[y] != u32::MAX {
            continue;
        }
        let group = by.group.map(|g| {
            *group_b
                .entry(g)
                .or_insert_with(|| new_group(&mut groups, &b.groups[g as usize]))
        });
        mb[y] = blocks.len() as u32;
        blocks.push(FBlock { group, ..by.clone() });
    }

    let mut cfgs: Vec<UCfg> = a
        .cfgs
        .iter()
        .map(|c| UCfg {
            unit: c.unit,
            map: c.map.iter().map(|&x| ma[x as usize]).collect(),
            swap: c.swap.clone(),
        })
        .collect();
    for c in &b.cfgs {
        cfgs.push(UCfg {
            unit: c.unit,
            map: c.map.iter().map(|&y| mb[y as usize]).collect(),
            swap: c
                .swap
                .iter()
                .zip(&c.map)
                .map(|(&s, &y)| s ^ flip[y as usize])
                .collect(),
        });
    }
    Fu {
        blocks,
        groups,
        cfgs,
        host: host.to_string(),
    }
}

/// The vertex mappings whose merged blocks lie on `cycle` (blocks of the rebuilt unit): a no-good.
pub fn nogood(a: &Fu, m: &Model, chosen: &[usize], cycle: &[u32]) -> Vec<usize> {
    // Merged blocks are created in A's block order: recompute which vertex mapping produced each.
    let nv = m.vm.len();
    let mut of_a: BTreeMap<u32, usize> = BTreeMap::new();
    for &i in chosen.iter().filter(|&&i| i < nv) {
        for &x in &m.sa.members[m.vm[i].sa as usize] {
            of_a.insert(x, i);
        }
    }
    // Block ids of the rebuilt unit: A's blocks keep their order (merged or copied), so block x of A is
    // block x of the result.
    let _ = a;
    let mut g: Vec<usize> = cycle.iter().filter_map(|&blk| of_a.get(&blk).copied()).collect();
    g.sort_unstable();
    g.dedup();
    g
}
