//! Process supervision shared by the runners: poll the child, sample memory, enforce caps, timeout and
//! cancellation. The libc calls here are the crate's only `unsafe`.
#![allow(unsafe_code)]

use std::process::{Child, Command, ExitStatus};
use std::time::{Duration, Instant};

use crate::cancel::CancelToken;
use crate::runner::KillReason;

pub(crate) struct Supervisor<'a> {
    pub timeout: Duration,
    pub cancel: &'a CancelToken,
    pub sample_interval: Duration,
    /// Kill when a sample exceeds this many bytes (host watchdog). `None`: sample only (Docker, where
    /// the kernel enforces the cap).
    pub cap_bytes: Option<u64>,
    pub sample: &'a dyn Fn() -> Option<u64>,
    pub kill: &'a dyn Fn(),
}

pub(crate) struct Supervised {
    pub status: ExitStatus,
    pub killed: Option<KillReason>,
    pub peak_bytes: Option<u64>,
}

pub(crate) fn supervise(child: &mut Child, s: &Supervisor<'_>) -> std::io::Result<Supervised> {
    let start = Instant::now();
    let mut peak: Option<u64> = None;
    let mut killed: Option<KillReason> = None;
    // First sample after a short delay: sub-50 ms tools never pay for a /proc scan.
    let mut next_sample = start + s.sample_interval.min(Duration::from_millis(50));
    let mut last_kill: Option<Instant> = None;
    let mut backoff = Duration::from_millis(1);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Supervised {
                status,
                killed,
                peak_bytes: peak,
            });
        }
        let now = Instant::now();
        if now >= next_sample {
            if let Some(rss) = (s.sample)() {
                peak = Some(peak.map_or(rss, |p| p.max(rss)));
                if killed.is_none() && s.cap_bytes.is_some_and(|cap| rss > cap) {
                    killed = Some(KillReason::MemoryCap);
                }
            }
            next_sample = now + s.sample_interval;
        }
        if killed.is_none() && now.duration_since(start) > s.timeout {
            killed = Some(KillReason::Timeout);
        }
        if killed.is_none() && s.cancel.is_cancelled() {
            killed = Some(KillReason::Cancelled);
        }
        // Re-issue the kill every second until the child is gone (docker kill can race container start).
        if killed.is_some() && last_kill.is_none_or(|t| now.duration_since(t) > Duration::from_secs(1)) {
            (s.kill)();
            last_kill = Some(now);
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(Duration::from_millis(20));
    }
}

/// Put the child in its own process group (so the whole tree can be measured and killed) and disable
/// core dumps.
pub(crate) fn isolate(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);
    // SAFETY: setrlimit is async-signal-safe and touches no memory shared with the parent.
    unsafe {
        cmd.pre_exec(|| {
            let lim = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_CORE, &lim) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// SIGKILL every process in group `pgid`.
pub(crate) fn kill_group(pgid: i32) {
    if pgid > 1 {
        // SAFETY: plain syscall; failure (ESRCH: group already gone) is harmless.
        unsafe {
            libc::killpg(pgid, libc::SIGKILL);
        }
    }
}

fn page_size() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if n > 0 { n as u64 } else { 4096 }
}

/// Sum of RSS over all processes whose process group is `pgid`, from `/proc/<pid>/stat`
/// (fields 5 `pgrp` and 24 `rss` in pages; proc(5)). `None` if no member was found.
pub(crate) fn group_rss_bytes(pgid: i32) -> Option<u64> {
    let page = page_size();
    let mut total = 0u64;
    let mut found = false;
    for entry in std::fs::read_dir("/proc").ok()?.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        // `comm` (field 2) is parenthesised and may contain spaces: parse after the last ')'.
        let Some(rest) = stat.rfind(')').map(|i| &stat[i + 1..]) else {
            continue;
        };
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let (Some(pgrp), Some(rss)) = (fields.get(2), fields.get(21)) else {
            continue;
        };
        if pgrp.parse::<i32>().ok() == Some(pgid) {
            found = true;
            total += rss.parse::<u64>().unwrap_or(0) * page;
        }
    }
    found.then_some(total)
}

/// `MemAvailable` from `/proc/meminfo` in bytes (proc(5)); `None` if unreadable.
pub fn mem_available_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

pub(crate) fn current_uid_gid() -> (u32, u32) {
    // SAFETY: getuid/getgid always succeed.
    unsafe { (libc::getuid(), libc::getgid()) }
}

pub(crate) fn exit_parts(status: &ExitStatus) -> (Option<i32>, Option<i32>) {
    use std::os::unix::process::ExitStatusExt;
    (status.code(), status.signal())
}
