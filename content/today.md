+++
title = 'Today'
date = '2026-10-26T00:00:00+05:30'
lastmod = '2026-10-27T00:50:00+05:30'
draft = false
description = "What to read and build today: inference engineering, MIT 6.5840, CMU 15-721 and LeetCode."
+++

# What to read today

*Updated daily. Last update: **Tue, 27 Oct 2026** · Day 2 · Week 1 · **Phase 1 of 4: Resume + proof** (Nov–Dec) · Next checkpoint: **AI-infra resume done by 30 Nov***

## Today: Day 2, Tue 27 Oct

**Theme:** the read / write / compute arithmetic, made to click. Carried over from Day 1, and that's fine.

### 🏢 Office

Read kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/), only up to the KV cache and flops-vs-bytes part. Slowly, with paper next to you.

### 🌙 Evening (~1.5 h)

| Time | Task |
| --- | --- |
| 0:00–0:30 | **By hand, one matmul only.** `(1 × 4096) @ (4096 × 4096)`: count FLOPs, bytes read, bytes written, then FLOPs ÷ bytes. Then redo it for batch 64: `(64 × 4096) @ (4096 × 4096)`. See the intensity jump from ~1 to ~64. |
| 0:30–1:00 | **Code it:** a `matmul_cost(M, K, N)` function that prints exactly those numbers. Then the whole Llama-3-8B decode step = 16 GB weights read, 16 GFLOPs per token. |
| 1:00–1:20 | 1 LeetCode |
| 1:20–1:25 | **Notes:** what clicked, what's still fuzzy. |

**Done when:** you can explain to an empty room why batch 1 decode wastes the GPU, using the matmul numbers.

6.5840 check moves to Thursday.

---

## This week (Week 1: 26 Oct – 1 Nov)

| Day | 🏢 Office (read / watch) | 🌙 Evening (build) |
| --- | --- | --- |
| Mon 26 | — | Book Ch 2 (first half) + arithmetic-intensity script |
| Tue 27 | kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (first half) | Day 1 carry-over: matmul cost by hand + `matmul_cost` script · 1 LeetCode |
| Wed 28 | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the rooflines chapter | 1 LeetCode · Book Ch 2 (rest) · start GPT-2 inference from scratch: load the real weights into your own PyTorch model, check outputs match Hugging Face |
| Thu 29 | Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | 1 LeetCode · 6.5840: check where MapReduce / Raft 3A stand, continue |
| Fri 30 | CMU 15-721 [#01 Modern analytical DBs](https://15721.courses.cs.cmu.edu/spring2024/schedule.html) at 1.5x + skim the Lakehouse paper | GPT-2: add a KV cache |
| Sat 31 | — | 1 LeetCode · 5–6 h: GPT-2 benchmark: tokens/sec and memory with vs without KV cache, across batch sizes · or 6.5840 catch-up |
| Sun 1 | Light | Weekly contest · **Resume pass 1:** list every bullet missing a number (p99, throughput, users, hours saved) and collect the numbers from work · first weekly notes post |

## Roadmap (Oct 2026 – Sep 2027)

| Phase | When | Focus |
| --- | --- | --- |
| **1. Resume + proof** | Nov – Dec 2026 | AI-infra resume with numbers · GPT-2 + KV cache benchmark · Narad benchmarks · Raft 3A–3B · 15-721 #01–#08 · LeetCode 5–6/week · DDIA Ch 5, 8, 9 |
| 2. Interview prep | Jan – Mar 2027 | LeetCode daily · system design (classic + ML system design) · project deep-dive stories · behavioral · mock interviews · mini inference engine |
| 3. Applications | Apr – Jun 2027 | Referrals, applications in waves, interviews |
| 4. Offer + move | Jul – Sep 2027 | Negotiate, paperwork, move |

### Checkpoints

| By | Checkpoint | Status |
| --- | --- | --- |
| 30 Nov 2026 | AI-infra resume variant done, every bullet has a number | ⬜ |
| 31 Dec 2026 | GPT-2 + KV cache benchmark published · Raft 3B passes | ⬜ |
| 31 Jan 2027 | 30 LeetCode in Jan · ML system design notes for 3 designs | ⬜ |
| 28 Feb 2027 | 6 mock interviews · mini inference engine runs | ⬜ |
| 31 Mar 2027 | Deep-dive stories ready · contest rating 2000+ | ⬜ |
| 30 Jun 2027 | Offer signed | ⬜ |

## Reading queue (Phase 1)

- [ ] kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [ ] Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [ ] Google, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (rooflines, parallelism)
- [ ] Llama 2 / Llama 3 papers: GQA and what it does to KV cache size
- [ ] Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [ ] [vLLM / PagedAttention blog post](https://blog.vllm.ai/2023/06/20/vllm.html) + Anyscale, [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ ] Martin Kleppmann, [distributed systems lectures](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB) (pairs with 6.5840)
- [ ] [Raft paper](https://raft.github.io/raft.pdf) (before Raft 3A)
- [ ] GPU MODE lectures on YouTube (closer to Phase 2)
- [ ] *Inference Engineering*: Ch 2 ⬜ · Ch 3 ⬜ · Ch 5 ⬜
- [ ] Kleppmann, *Designing Data-Intensive Applications* (1st ed.), one chapter a week from Week 2, office reading: Ch 5 Replication ⬜ · Ch 8 Trouble with distributed systems ⬜ · Ch 9 Consistency and consensus ⬜ (these three go with Raft) · then Ch 3 Storage ⬜ · Ch 7 Transactions ⬜ · Ch 6 Partitioning ⬜ (Ch 3 goes with 15-721 #02–03)

### CMU 15-721: Advanced Database Systems (Spring 2024)

[Schedule, slides, papers](https://15721.courses.cs.cmu.edu/spring2024/schedule.html) · [YouTube playlist](https://www.youtube.com/playlist?list=PLSE8ODhjZXjYa_zX-KeMJui7pcN1rIaIJ). One lecture a week at the office + its main paper, and the build on Friday evening.

| Week | Lecture (office, ~1 h at 1.5x) | Main paper | Build (Friday evening) |
|---|---|---|---|
| 1 (30 Oct) | #01 Modern analytical DBs | Lakehouse (CIDR '21) | — (KV cache week) |
| 2 | #19 Snowflake (early: I use it daily) | Snowflake Elastic DW (SIGMOD '16) | Read real Snowflake query profiles: pruning, spilling, warehouse size |
| 3 | #02 Data formats I | Columnar formats evaluation (VLDB '23) | Same table as CSV vs Parquet (plain / dictionary / RLE / zstd) with pyarrow + DuckDB: file size and scan time |
| 4 | #03 Data formats II | FastLanes (VLDB '23) | Frame-of-reference + bit-packing in Go: decode GB/s |
| 5 | #04 Query execution I | MonetDB/X100 (CIDR '05) | **Tuple-at-a-time (Volcano) vs vectorized (1024-row batches)** filter + sum over 100M ints in Go: same idea as batching in inference |
| 6 | #05 Query execution II | Velox (VLDB '22) | Read Velox / DuckDB vector code (see expert code) |
| 7 | #06 Vectorization (SIMD) | Lang et al. (VLDBJ '20) | SIMD filter in Rust (`std::simd`) vs scalar |
| 8 | #07 Compilation | Neumann, HyPer (VLDB '11) | Optional: hand-fuse the Week 5 pipeline into one loop and compare (operator fusion = what FlashAttention does) |
| 9 | #08 Scheduling | Morsel-driven parallelism (SIGMOD '14) | Morsel-driven parallel aggregation with a goroutine pool: speedup vs cores |
| 10 | #09 Hash joins | 13 equi-joins compared (SIGMOD '16) | Naive vs radix-partitioned hash join in Go: time vs table size crossing the cache size |
| 11 | #13 Optimizer I | Chaudhuri overview (PODS '98) | — |
| 12 | #16 Cost models (slides only) | How good are query optimizers? (VLDB '15) | Postgres: `EXPLAIN ANALYZE` a 3-way join on correlated columns; estimated vs actual rows |
| 13 | #17 Dremel + #18 Photon | Dremel decade (VLDB '20), Photon (SIGMOD '22) | — |
| 14 | #20 DuckDB | DuckDB (SIGMOD '19) | Run TPC-H SF10 in DuckDB, read the profile output |

Optional, only if curious: #10 multi-way joins, #11 UDFs, #12 networking, #14–15 optimizer, #21 Yellowbrick, #22 Redshift.

### DDIA: learn by doing

Each chapter read at the office gets one build that week, in the normal evening slots.

| Ch | Read | Build (prove it) | Size |
|---|---|---|---|
| 5 | Replication | Postgres primary + async replica in Docker Compose: measure replication lag under load, kill the primary mid-write and count lost writes, show a read-your-writes violation | 1 evening |
| 8 | Trouble with distributed systems | Last-write-wins with skewed clocks: two Go nodes, one clock +200 ms, show silently lost writes. Then add network partitions with toxiproxy | 1 evening |
| 9 | Consistency and consensus | Raft 3B/3C is the build. Extra: run 6.5840's porcupine linearizability checker against a deliberately broken KV store and read the failing history | 1 evening |
| 3 | Storage and retrieval | **Mini LSM-tree KV store in Go**: WAL → memtable → SSTables → compaction → bloom filter. Benchmark against bbolt (B+tree): write amp, read latency | 2–3 Saturdays, after Raft 3B (portfolio piece) |
| 7 | Transactions | Two psql sessions: reproduce lost update, write skew and phantoms under READ COMMITTED / REPEATABLE READ / SERIALIZABLE, and record which level stops which anomaly | 1 evening |
| 6 | Partitioning | Consistent-hashing router in Go with virtual nodes: measure keys moved when a node joins/leaves vs mod-N hashing. Connects to llm-d cache-aware routing | 1 evening |

### Reading expert code

After each build, read how experts wrote the same thing.

| After you build… | Read the expert version | Start with |
|---|---|---|
| Arithmetic-intensity script, GPT-2 + KV cache | **gpt-fast** (PyTorch team, ~1k lines: KV cache, int8/int4, speculative decoding) | `model.py`, `generate.py` |
| Mini inference engine (Phase 2) | **nano-vllm** (~1.2k lines of vLLM ideas), then **vLLM** itself | nano-vllm `engine/`; vLLM `v1/core/sched/scheduler.py`, `v1/core/kv_cache_manager.py` |
| Raft 3A–3C | **etcd-io/raft** (the Raft used by etcd, K8s, CockroachDB) | `raft.go`, `log.go`, `node.go` |
| Mini LSM KV store | **LevelDB** (Jeff Dean & Sanjay Ghemawat, small and clean C++), then **Pebble** (CockroachDB's LSM in Go) | LevelDB `db/db_impl.cc`, `table/`; Pebble `compaction.go` |
| Paged KV cache in the mini engine | **Postgres** buffer manager (clock-sweep eviction, maps to PagedAttention) | `src/backend/storage/buffer/bufmgr.c`, `freelist.c` |
| LSM vs B+tree benchmark | **bbolt** (small Go B+tree), **SQLite** (heavily commented) | bbolt `node.go`, `cursor.go`; SQLite `btree.c` |
| 15-721 vectorized executor builds | **DuckDB** (C++, vectorized), **Velox** (Meta) | DuckDB `src/execution/`, `src/common/vector_operations/` |
| Go services at work | **Go standard library** | `net/http/server.go`, `sync/mutex.go`, `context/` |

**How to read it:** build your own version first, then read theirs and list 3 things they did differently and why. Trace one request or operation end to end instead of reading files top to bottom. Read the tests, which show what the authors were worried about. Use `git blame` and the linked PRs to find out *why* the code is shaped that way. Code reading counts as office reading time.

## Rules

- Office is for reading and watching. Evenings are for building.
- After every read: 2–3 lines of notes, rolled up into the Sunday post.
- LeetCode: 5–6 a week in Phase 1, daily from January. Timed, weak topics, explained out loud.
- Falling behind moves the calendar. It never means doing two days at once.
- Eight months of showing up beats two weeks of 14-hour days. Sunday stays light.

## Log

| Day | Date | Planned | Done |
| --- | --- | --- | --- |
| 1 | Mon 26 Oct | Book Ch 2 (first half) · intensity script | Read Ch 2 · went through ops:byte, S/P matrices, attention read/write/compute. Calculation didn't click yet → redo from one matmul on Day 2 |
| 2 | Tue 27 Oct | kipply (first half) · matmul cost by hand + script | *(pending)* |

<script>
(function () {
  var d = new Date();
  var key = d.toLocaleDateString('en-GB', { weekday: 'short' }) + ' ' + d.getDate();
  document.querySelectorAll('table tr').forEach(function (row) {
    var cell = row.querySelector('td');
    if (cell && cell.textContent.trim() === key) {
      row.style.background = 'rgba(49,130,206,0.15)';
      row.style.fontWeight = '600';
    }
  });
})();
</script>
