//! Smoke suite (CI): framework behaviour on toy and fixture passes. No external tools beyond
//! `/bin/echo`, no Docker, seconds in total.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod cache;
mod cli;
mod components;
mod config;
mod dag;
mod datapaths;
mod eda;
mod env;
mod export;
mod import;
mod iteration_space;
mod kernel_classes;
mod kernels;
mod normalize;
mod pipeline;
mod runners;
mod schemas;
mod smt;
mod verify;
