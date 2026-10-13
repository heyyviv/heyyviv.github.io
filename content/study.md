+++
title = 'Study Plan'
date = '2026-10-14T00:00:00+05:30'
lastmod = '2026-10-14T00:45:00+05:30'
draft = false
description = "Daily plan for working through MIT 6.5840 Distributed Systems and CMU 15-721 Advanced Database Systems, and daily DSA practice."
+++

# Study Plan: MIT 6.5840 + CMU 15-721 + daily DSA

*Updated daily. Last update: **Wed, 14 Oct 2026** · Day 1 · Week 1 of ~31 · Target finish: mid-May 2027*

## Today: Day 1, Wed 14 Oct

**Theme:** RPC and threads in Go, and Lab 1 setup. *(MapReduce paper already done: see [my notes](/blog/mapper-reducer/).)*

### 🏢 Office (30–45 min, split however you like)

1. Read the [Lecture 2 notes: RPC and threads](https://pdos.csail.mit.edu/6.824/notes/l-rpc.txt). This is the Go `net/rpc` + goroutines + mutex toolkit Lab 1 is built on.
2. Skim [crawler.go](https://pdos.csail.mit.edu/6.824/notes/crawler.go) and [kv.go](https://pdos.csail.mit.edu/6.824/notes/kv.go) from the same lecture (you'll copy these patterns tonight).
3. *Optional:* the 2020 Lecture 2 video, "RPC and Threads" ([playlist](https://www.youtube.com/playlist?list=PLrw6a1wE39_tb2fErI4-WkMbsvGQk9_UB)).

While reading, keep one question in mind: *in kv.go, what does the client see if the reply is lost, and is a retried Put safe?* (This becomes Lab 2.)

### 🌙 Evening (2 h hard stop)

| Time | Task |
| --- | --- |
| 0:00–0:30 | **DSA.** In `DSA_prep`, run `python3 scripts/dsa.py review`. Re-solve **Two Sum** and **Course Schedule** from scratch (no notes, C++), then rate them. You have 26 cards overdue since July; clearing about 2–3 a day takes 2 weeks. |
| 0:30–0:50 | **Fix Go.** `/usr/local/go/bin/go` fails with "bad CPU type" (wrong architecture for this Mac). Run `brew install go` (or install the arm64 pkg from go.dev), make sure `which go` points to the new one, and check `go version`. |
| 0:50–1:00 | `git clone git://g.csail.mit.edu/6.5840-golabs-2026 6.5840`, `cd 6.5840/src`, run `make mr`. Failing tests are expected. |
| 1:00–1:30 | Read the whole [Lab 1 spec](https://pdos.csail.mit.edu/6.824/labs/lab-mr.html), then `main/mrsequential.go` and the stubs in `mr/`. |
| 1:30–1:55 | **Design note (no code yet).** You know the paper, so keep it short. In `notes/lab1-design.md`: RPC names and fields between worker and coordinator, task states (idle / in-progress / done + start time), the 10 s timeout rule, and how the lab differs from the paper (shared filesystem, no master checkpoint, the coordinator must tell workers when to exit). |
| 1:55–2:00 | Send me the report below. Stop. |

**Done when:** 2 SRS cards re-solved, `go version` works, `make mr` runs, and the design note exists.

### 📝 Tonight's report (copy, fill in, send)

```
Day 1 report
Office: read ___ (L2 notes / crawler.go / kv.go / video)
DSA: problems ___ (solved clean / needed hint / failed)
Evening: Go fixed? y/n · make mr runs? y/n · design note? y/n
Stuck on: ___
Energy (1–5): _  · Actual time: office __ min, evening __ min
```

---

## This week (Week 1: 14–17 Oct)

| Day | Office | DSA (30 min) | Lab (90 min) |
| --- | --- | --- | --- |
| Wed 14 | L2 RPC & threads notes, crawler.go / kv.go | 2 overdue SRS cards | Go fix, clone, Lab 1 spec, design note |
| Thu 15 | L2 [Go lecture question](https://pdos.csail.mit.edu/6.824/questions.html?q=q-gointro&lec=2) (write answer first) + [Go tour FAQ](https://pdos.csail.mit.edu/6.824/papers/tour-faq.txt) | 2–3 overdue cards | `mr/rpc.go` structs + coordinator task table + `RequestTask` RPC |
| Fri 16 | Start [GFS paper](https://pdos.csail.mit.edu/6.824/papers/gfs.pdf) §1–3 | 2–3 overdue cards | Worker: map phase writes `mr-X-Y` intermediate files |
| Sat 17 | — | 60 min timed set: 1 medium + 1 hard, unseen | Catch-up only, or rest |
| Sun 18 | Off | Off | Off |

## The whole course

| Track | When | Where |
| --- | --- | --- |
| **MIT 6.5840** labs: MapReduce → KV server → Raft (3A–3D) → KV Raft → Sharded KV | Evenings (90 min), weeks 1–~25 | [schedule](https://pdos.csail.mit.edu/6.824/schedule.html) |
| **MIT 6.5840** papers and lectures (21) | Office, weeks 1–20 | [2020 videos](https://www.youtube.com/playlist?list=PLrw6a1wE39_tb2fErI4-WkMbsvGQk9_UB) |
| **CMU 15-721** lectures and papers (23) | Office, weeks 2–20 | [schedule](https://15721.courses.cs.cmu.edu/spring2024/schedule.html) · [videos](https://www.youtube.com/playlist?list=PLSE8ODhjZXjYa_zX-KeMJui7pcN1rIaIJ) |
| **DSA**: clear the overdue SRS backlog, then 1 new problem (`/start`, weakest pattern) + due reviews | Evenings 30 min, Mon–Fri; Saturday 60 min timed set | `~/Desktop/projects/JOB/DSA_prep` |
| **CMU-style Go mini-engine**: columnar scan → vectorized ops → hash join → morsel scheduler → join ordering | Evenings, after Lab 5 | stands in for the CMU group project |

### Milestones

| Target week | Milestone | Status |
| --- | --- | --- |
| 3 (by 31 Oct) | Lab 1 MapReduce passes | ⬜ |
| 4 (by 7 Nov) | Lab 2 KV server passes | ⬜ |
| 8 (by 5 Dec) | Raft 3A + 3B pass 100 runs | ⬜ |
| 14 (by 16 Jan) | All of Raft (3A–3D) passes 200 runs | ⬜ |
| 11 (21–26 Dec) | 🌴 Deload week | ⬜ |
| 19 (by 20 Feb) | Lab 4 KV Raft passes | ⬜ |
| 25 (by 3 Apr) | Lab 5 Sharded KV passes · **MIT done** | ⬜ |
| 20 (by 27 Feb) | All 23 CMU lectures watched with notes · **CMU reading done** | ⬜ |
| 31 (by 15 May) | Mini-engine and final write-up | ⬜ |

**Progress:** MIT lectures 1/21 (L1 MapReduce, read Jul 2025) · MIT labs 0/5 · CMU lectures 0/23 · Mini-engine 0/4 · DSA overdue cards 26 → 0

## Rules that keep this sustainable

- The evening is 2 hours **including** DSA: 30 min DSA first, then 90 min lab. Two hours is a cap. Stop mid-function if you have to, and write one line saying where to resume.
- Falling behind moves the calendar. It never means doing two days at once.
- Sunday is off. Saturday is catch-up, and a rest day if nothing is behind.
- Two days in a row at energy 2/5 or lower → the next day is light (reading or rerunning tests only).
- Raft is where most people get stuck. A day of debugging with no new code still counts.

## Log

| Day | Date | Planned | Done |
| --- | --- | --- | --- |
| 0 | Jul 2025 | — | MapReduce paper read; [notes](/blog/mapper-reducer/) cover re-execution, atomic rename, stragglers ✅ |
| 1 | Wed 14 Oct | L2 RPC notes · 2 SRS cards · Go fix · Lab 1 design note | *(pending report)* |
