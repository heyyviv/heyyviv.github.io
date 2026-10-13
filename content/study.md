+++
title = 'Study Plan'
date = '2026-10-14T00:00:00+05:30'
lastmod = '2026-10-14T00:00:00+05:30'
draft = false
description = "Daily plan for working through MIT 6.5840 Distributed Systems and CMU 15-721 Advanced Database Systems."
+++

# Study Plan: MIT 6.5840 + CMU 15-721

*Updated daily. Last update: **Wed, 14 Oct 2026** · Day 1 · Week 1 of ~24 · Target finish: end of March 2027*

## Today: Day 1, Wed 14 Oct

**Theme:** MapReduce and lab setup.

### 🏢 Office (30–45 min, split however you like)

1. Read the [MapReduce paper](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf), sections 1–3 (about 7 pages). Skip §4 onward for now.
2. Skim the [Lecture 1 notes](https://pdos.csail.mit.edu/6.824/notes/l01.txt).
3. *Optional, if time:* first 30 min of [2020 Lecture 1 video](https://www.youtube.com/playlist?list=PLrw6a1wE39_tb2fErI4-WkMbsvGQk9_UB).

While reading, keep one question in mind: *how does the master know a worker died, and what does it redo?*

### 🌙 Evening (2 h hard stop)

| Time | Task |
| --- | --- |
| 0:00–0:20 | **Fix Go.** `/usr/local/go/bin/go` fails with "bad CPU type" (wrong architecture for this Mac). Run `brew install go` (or install the arm64 pkg from go.dev), make sure `which go` points to the new one, and check `go version`. |
| 0:20–0:35 | `git clone git://g.csail.mit.edu/6.5840-golabs-2026 6.5840`, `cd 6.5840/src`, run `make mr`. Failing tests are expected. |
| 0:35–1:15 | Read the whole [Lab 1 spec](https://pdos.csail.mit.edu/6.824/labs/lab-mr.html), then `main/mrsequential.go` and the stubs in `mr/`. |
| 1:15–1:50 | **Predict (no code yet).** In `notes/lab1-design.md`, write: the RPCs between worker and coordinator (names and fields), the coordinator's task states, how a crashed or slow worker is detected (the lab suggests 10 s), and why output is written to a temp file and then renamed. |
| 1:50–2:00 | Send me the report below. Stop. |

**Done when:** `go version` works, `make mr` runs, and the design note exists.

### 📝 Tonight's report (copy, fill in, send)

```
Day 1 report
Office: read ___ (paper §1–3 / notes / video)
Evening: Go fixed? y/n · make mr runs? y/n · design note? y/n
Stuck on: ___
Energy (1–5): _  · Actual time: office __ min, evening __ min
```

---

## This week (Week 1: 14–17 Oct)

| Day | Office | Evening |
| --- | --- | --- |
| Wed 14 | MapReduce §1–3, L1 notes | Go fix, clone, Lab 1 spec, design note |
| Thu 15 | MapReduce §4–end, MIT [lecture question](https://pdos.csail.mit.edu/6.824/schedule.html) | Coordinator: task table + `RequestTask` RPC |
| Fri 16 | L1 video (rest) | Worker: map phase writes `mr-X-Y` intermediate files |
| Sat 17 | Catch-up only, or rest | Catch-up only, or rest |
| Sun 18 | Off | Off |

## The whole course

| Track | When | Where |
| --- | --- | --- |
| **MIT 6.5840** labs: MapReduce → KV server → Raft (3A–3D) → KV Raft → Sharded KV | Evenings, weeks 1–19 | [schedule](https://pdos.csail.mit.edu/6.824/schedule.html) |
| **MIT 6.5840** papers and lectures (21) | Office, weeks 1–20 | [2020 videos](https://www.youtube.com/playlist?list=PLrw6a1wE39_tb2fErI4-WkMbsvGQk9_UB) |
| **CMU 15-721** lectures and papers (23) | Office, weeks 2–20 | [schedule](https://15721.courses.cs.cmu.edu/spring2024/schedule.html) · [videos](https://www.youtube.com/playlist?list=PLSE8ODhjZXjYa_zX-KeMJui7pcN1rIaIJ) |
| **CMU-style Go mini-engine**: columnar scan → vectorized ops → hash join → morsel scheduler → join ordering | Evenings, weeks 20–23 | stands in for the CMU group project |

### Milestones

| Target week | Milestone | Status |
| --- | --- | --- |
| 2 (by 24 Oct) | Lab 1 MapReduce passes | ⬜ |
| 3 (by 31 Oct) | Lab 2 KV server passes | ⬜ |
| 6 (by 21 Nov) | Raft 3A + 3B pass 100 runs | ⬜ |
| 10 (by 19 Dec) | All of Raft (3A–3D) passes 200 runs | ⬜ |
| 11 | 🌴 Deload week | ⬜ |
| 14 (by 16 Jan) | Lab 4 KV Raft passes | ⬜ |
| 19 (by 20 Feb) | Lab 5 Sharded KV passes · **MIT done** | ⬜ |
| 20 (by 27 Feb) | All 23 CMU lectures watched with notes · **CMU reading done** | ⬜ |
| 24 (by 27 Mar) | Mini-engine and final write-up | ⬜ |

**Progress:** MIT lectures 0/21 · MIT labs 0/5 · CMU lectures 0/23 · Mini-engine 0/4

## Rules that keep this sustainable

- Two hours in the evening is a cap. Stop mid-function if you have to, and write one line saying where to resume.
- Falling behind moves the calendar. It never means doing two days at once.
- Sunday is off. Saturday is catch-up, and a rest day if nothing is behind.
- Two days in a row at energy 2/5 or lower → the next day is light (reading or rerunning tests only).
- Raft is where most people get stuck. A day of debugging with no new code still counts.

## Log

| Day | Date | Planned | Done |
| --- | --- | --- | --- |
| 1 | Wed 14 Oct | MapReduce §1–3 · Go fix · Lab 1 design note | *(pending report)* |
