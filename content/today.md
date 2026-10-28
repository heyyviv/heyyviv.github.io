+++
title = 'Today'
date = '2026-10-26T00:00:00+05:30'
lastmod = '2026-10-28T21:00:00+05:30'
draft = false
description = "What to read and build today: inference, ML data + eval, MIT 6.5840 and LeetCode."
+++

# What to read today

*Updated daily. Last update: **Wed, 28 Oct 2026** · Day 3 · Week 1 · **Phase 1 of 4: Resume + proof** (Nov–Dec) · Next checkpoint: **AI-infra resume done by 30 Nov***

## Today: Day 3, Wed 28 Oct

**Theme:** rooflines. kipply is done (whole post); the matmul evening work from Day 2 carries over to tonight.

### 🏢 Office

[How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the rooflines chapter. Same idea as kipply's flops-vs-bytes, drawn as a graph: find where the ridge point sits for an A100/H100 and why batch-1 decode lands far left of it.

### 🌙 Evening (~1.5 h, Day 2 carry-over)

| Time | Task |
| --- | --- |
| 0:00–0:30 | **By hand, one matmul only.** `(1 × 4096) @ (4096 × 4096)`: count FLOPs, bytes read, bytes written, then FLOPs ÷ bytes. Then redo it for batch 64: `(64 × 4096) @ (4096 × 4096)`. See the intensity jump from ~1 to ~64. |
| 0:30–1:00 | **Code it:** a `matmul_cost(M, K, N)` function that prints exactly those numbers. Then the whole Llama-3-8B decode step = 16 GB weights read, 16 GFLOPs per token. Check it against kipply's latency numbers. |
| 1:00–1:20 | 1 LeetCode |
| 1:20–1:25 | **Notes:** 2–3 lines on kipply + the rooflines chapter: what clicked, what's still fuzzy. |

**Done when:** you can explain to an empty room why batch 1 decode wastes the GPU, using the matmul numbers, and at what batch size it stops being memory-bound.

GPT-2 from scratch moves to Thursday, the 6.5840 check to Friday.

---

## This week (Week 1: 26 Oct – 1 Nov)

| Day | 🏢 Office (read / watch) | 🌙 Evening (build) |
| --- | --- | --- |
| Mon 26 | — | Book Ch 2 (first half) + arithmetic-intensity script |
| Tue 27 | kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) ✅ (whole post) | Matmul cost by hand + `matmul_cost` script · 1 LeetCode → moved to Wed |
| Wed 28 | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the rooflines chapter | Day 2 carry-over: matmul cost by hand + `matmul_cost` script · 1 LeetCode |
| Thu 29 | Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | 1 LeetCode · Book Ch 2 (rest) · start GPT-2 inference from scratch: load the real weights into your own PyTorch model, check outputs match Hugging Face |
| Fri 30 | Data topic 1: CMU 15-721 #02 *Data Formats & Encoding* (video, 1.5x) + Arrow blog, *Querying Parquet with Millisecond Latency* | 6.5840: check where MapReduce / Raft 3A stand, continue |
| Sat 31 | — | 1 LeetCode · 5–6 h: GPT-2 add a KV cache, then benchmark tokens/sec and memory with vs without it across batch sizes |
| Sun 1 | Light | Weekly contest · **Resume pass 1:** list every bullet missing a number (p99, throughput, users, hours saved) and collect the numbers from work · first weekly notes post |

## Roadmap (Oct 2026 – Sep 2027)

| Phase | When | Focus |
| --- | --- | --- |
| **1. Resume + proof** | Nov – Dec 2026 | AI-infra resume with numbers · GPT-2 + KV cache benchmark · Narad benchmarks · Raft 3A–3B · data topics 1–4 · eval set at work · LeetCode 5–6/week · DDIA Ch 5, 8, 9 |
| 2. Interview prep | Jan – Mar 2027 | LeetCode daily · system design (classic + ML system design) · project deep-dive stories · behavioral · mock interviews · LLM data → eval → device pipeline |
| 3. Applications | Apr – Jun 2027 | Referrals, applications in waves, interviews |
| 4. Offer + move | Jul – Sep 2027 | Negotiate, paperwork, move |

### Checkpoints

| By | Checkpoint | Status |
| --- | --- | --- |
| 30 Nov 2026 | AI-infra resume variant done, every bullet has a number | ⬜ |
| 31 Dec 2026 | GPT-2 + KV cache benchmark published · Raft 3B passes | ⬜ |
| 31 Jan 2027 | 30 LeetCode in Jan · ML system design notes for 3 designs | ⬜ |
| 28 Feb 2027 | 6 mock interviews · data → eval → device pipeline runs end to end | ⬜ |
| 31 Mar 2027 | Deep-dive stories ready · contest rating 2000+ | ⬜ |
| 30 Jun 2027 | Offer signed | ⬜ |

## Reading queue (Phase 1)

- [x] kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [ ] Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [ ] Google, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (rooflines, parallelism)
- [ ] Llama 2 / Llama 3 papers: GQA and what it does to KV cache size
- [ ] Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [ ] [vLLM / PagedAttention blog post](https://blog.vllm.ai/2023/06/20/vllm.html) + Anyscale, [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ ] Martin Kleppmann, [distributed systems lectures](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB) (pairs with 6.5840)
- [ ] [Raft paper](https://raft.github.io/raft.pdf) (before Raft 3A)
- [ ] Hamel Husain, *Your AI Product Needs Evals*
- [ ] Chip Huyen, *AI Engineering* Ch 3–4 (evaluation), Ch 8 (dataset engineering)
- [ ] Stanford CS336 (Spring 2025): data lectures + evaluation lecture
- [ ] *Inference Engineering*: Ch 2 ⬜ · Ch 3 ⬜ · Ch 5 ⬜
- [ ] Kleppmann, *Designing Data-Intensive Applications* (1st ed.), one chapter a week from Week 2, office reading: Ch 5 Replication ⬜ · Ch 8 Trouble with distributed systems ⬜ · Ch 9 Consistency and consensus ⬜ (these three go with Raft) · then Ch 3 Storage ⬜ · Ch 7 Transactions ⬜ · Ch 6 Partitioning ⬜ · Ch 10 Batch ⬜ · Ch 11 Streams ⬜ (Ch 3 goes with data topic 1)

### Data engineering for ML (topic-wise)

One topic a week: watch the talk at the office (1.5x), read the blog, skim the paper only if needed, then build on Friday evening. Priority order; stop wherever time runs out.

| # | Topic | Watch | Blog / read | Build (Friday evening) |
|---|---|---|---|---|
| 1 | Columnar formats (Parquet, encodings) | CMU 15-721 (Spring 2024) #02 *Data Formats & Encoding* | Arrow blog, *Querying Parquet with Millisecond Latency* · Parquet docs | CSV vs Parquet (plain / dictionary / RLE / zstd) with pyarrow + DuckDB: size and scan time |
| 2 | Partitioning, pruning, skew | CMU 15-445 (Fall 2023) *Parallel Query Execution* | Databricks, *Adaptive Query Execution* · Snowflake docs, *Micro-partitions & Data Clustering* · DDIA Ch 6 | Read real Snowflake query profiles; skewed group-by in DuckDB/Ray, fixed with salting |
| 3 | Distributed execution (Spark, Ray) | Matei Zaharia, *Resilient Distributed Datasets* (NSDI '12) · a Ray Summit talk on Ray Data | Anyscale blog on Ray Data streaming execution · Ray Data docs | Same dedup job in pandas vs Ray Data: throughput vs workers, memory |
| 4 | Dedup at scale (shuffle, MinHash) | Stanford CS336 (Spring 2025) data lectures | HuggingFace, *Large-scale Near-deduplication Behind BigCode* · *FineWeb* blog · DDIA Ch 10 | Exact vs near-dedup (MinHash LSH) with Ray: recall, time, shuffle size |
| 5 | Table formats + data versioning | CMU 15-721 #01 *Modern Analytical DBs* | Dremio, *Apache Iceberg: An Architectural Look Under the Covers* · lakeFS / DVC docs | Iceberg/Delta with DuckDB: write, time-travel, compare with DVC |
| 6 | Vectorized execution | CMU 15-721 #04 *Query Execution* · CMU DB seminar talk on DuckDB | DuckDB execution engine docs · MonetDB/X100 (skim) | Row-at-a-time vs vectorized (1024-row batches) over 100M ints in Go |
| 7 | Streaming (Kafka, logs, backpressure) | Kleppmann, *Turning the Database Inside Out* | Jay Kreps, *The Log* · Tyler Akidau, *Streaming 101* · DDIA Ch 11 | Extend Narad: lag + throughput under backpressure |
| 8 | Data quality + lineage | A Data Council / Airflow Summit talk on data quality or OpenLineage | Chip Huyen, *Data Distribution Shifts and Monitoring* · OpenLineage docs | Quality checks + lineage events in an Airflow DAG |
| 9 | Real systems | CMU 15-721 #19 *Snowflake*, #20 *DuckDB* | Snowflake (SIGMOD '16), Dremel decade (VLDB '20), skim | TPC-H SF10 in DuckDB, read the profile |

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
| Data topic 6 vectorized build | **DuckDB** (C++, vectorized), **Velox** (Meta) | DuckDB `src/execution/`, `src/common/vector_operations/` |
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
| 2 | Tue 27 Oct | kipply (first half) · matmul cost by hand + script | Read the **whole** kipply post (ahead of plan). Evening tasks not done → Wed |
| 3 | Wed 28 Oct | Rooflines chapter · matmul by hand + `matmul_cost` script · 1 LeetCode | *(pending)* |

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
