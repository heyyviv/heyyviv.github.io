+++
title = 'Today'
date = '2026-10-26T00:00:00+05:30'
lastmod = '2026-10-30T09:00:00+05:30'
draft = false
description = "What to read and build today: LLM inference and serving, distributed systems, MIT 6.5840 and LeetCode."
+++

# What to read today

*Updated daily. Last update: **Fri, 30 Oct 2026** · Day 5 · Week 1 · **Stage 1: Foundations** (Nov–Dec)*

## Today: Day 5, Fri 30 Oct

✅ Horace He, *Making Deep Learning Go Brrrr* (done ahead of Monday).

**Morning:** the matmul exercise (see Fri row below). It's the same compute-vs-memory question as Horace He, now with real numbers.

**Evening:** 2–3 lines of notes on kipply + Horace He + matmul · 1 LeetCode (optional)

## Daily rhythm

- **Morning block (1 h):** read + one hands-on exercise that ends in code or a number
- **Evening (short):** one small task; LeetCode optional
- **Saturday:** long build block
- **Sunday:** light, notes post
- A task that slips twice moves to the next morning block or Saturday

---

## This week (Week 1: 26 Oct – 1 Nov)

| Day | ☀️ Morning block | 🌙 Evening (short) |
| --- | --- | --- |
| Mon 26 | — | Book Ch 2 (first half) ✅ (calculation didn't click yet) |
| Tue 27 | kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) ✅ (whole post) | Matmul exercise → slipped |
| Wed 28 | — | Matmul exercise → **moved to Fri morning** |
| Thu 29 | — | 1 LeetCode ✅ |
| Fri 30 | Horace He ✅ (ahead) · **Matmul by hand (25 min):** `(1 × 4096) @ (4096 × 4096)`: FLOPs, bytes read, bytes written, FLOPs ÷ bytes. Redo for batch 64 and see intensity jump from ~1 to ~64. **Then code it (35 min):** `matmul_cost(M, K, N)` printing those numbers; check one Llama-3-8B decode step (16 GB read, 16 GFLOPs/token) against kipply | 2–3 lines of notes on kipply + Horace He + matmul: what clicked, what's fuzzy · 1 LeetCode (optional) |
| Sat 31 | — | **5–6 h build:** 6.5840 check (30 min: where MapReduce / Raft 3A stand) · GPT-2 from scratch: load the real weights into my own PyTorch model, outputs match Hugging Face (~3 h) · start the KV cache if time (benchmark moves to next Saturday) · 1 LeetCode (optional) |
| Sun 1 | — | Light: first weekly notes post · optional contest |

**Next week starts with:** Mon morning: classify 3 ops (matmul, softmax, layernorm) as compute / memory / overhead bound with numbers, using Horace He's framing. Then the rooflines chapter of *How to Scale Your Model*.

## Roadmap

| Stage | When | Focus |
| --- | --- | --- |
| **1. Foundations** | Nov – Dec 2026 | GPT-2 + KV cache benchmark · Narad benchmarks · Raft 3A–3B · distributed systems topics 1–6 · LeetCode 5–6/week |
| 2. Serving engine | Jan – Mar 2027 | Mini inference engine (paged attention, continuous batching) · multi-GPU router · system design · flagship topic picked 31 Jan |
| 3. Flagship + open source | Apr 2027 → | Flagship build and write-up · first merged PR in vLLM / SGLang / llm-d |

### Checkpoints

| By | Checkpoint | Status |
| --- | --- | --- |
| 31 Dec 2026 | GPT-2 + KV cache benchmark published · Raft 3B passes | ⬜ |
| 31 Jan 2027 | Flagship topic picked · ML system design notes for 3 designs | ⬜ |
| 28 Feb 2027 | Mini inference engine + router runs with benchmarks | ⬜ |
| 31 Mar 2027 | Flagship write-up drafted | ⬜ |

## Reading queue (Phase 1)

- [x] kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [x] Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [ ] Google, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (rooflines, parallelism)
- [ ] Llama 2 / Llama 3 papers: GQA and what it does to KV cache size
- [ ] Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [ ] [vLLM / PagedAttention blog post](https://blog.vllm.ai/2023/06/20/vllm.html) + Anyscale, [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ ] Martin Kleppmann, [distributed systems lectures](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB) (used per topic in the table below)
- [ ] [Raft paper](https://raft.github.io/raft.pdf) (before Raft 3A)
- [ ] Hamel Husain, *Your AI Product Needs Evals*
- [ ] Chip Huyen, *AI Engineering* Ch 3–4 (evaluation), Ch 8 (dataset engineering)
- [ ] Stanford CS336 (Spring 2025): data lectures + evaluation lecture
- [ ] *Inference Engineering*: Ch 2 ⬜ · Ch 3 ⬜ · Ch 5 ⬜
- [ ] Distributed systems topics 1–6 (see table below: video + blog + build, DDIA as reference)

### Data engineering for ML (topic-wise) — supporting, lower priority

One topic a week: watch the talk (1.5x), read the blog, skim the paper only if needed, then build on Friday evening. Priority order; stop wherever time runs out.

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

### Distributed systems (topic-wise)

Same pattern: watch the lecture (1.5x), read the blog, DDIA or the paper only where it helps, then build. Videos: MIT 6.824 Spring 2020 (Robert Morris) and Kleppmann's Cambridge *Distributed Systems* playlists on YouTube. The 6.5840 labs are the builds for topics 1, 5, 7 and 11.

| # | Topic | Watch | Blog / read | Build |
|---|---|---|---|---|
| 1 | MapReduce + GFS (the lab warm-up) | MIT 6.824 (2020, Robert Morris) Lec 1 *Introduction / MapReduce* and Lec 3 *GFS* | MapReduce paper (OSDI '04), GFS paper (SOSP '03): skim | 6.5840 Lab 1 MapReduce |
| 2 | Replication | Kleppmann, *Distributed Systems* (Cambridge) replication lectures · MIT 6.824 Lec 4 *Primary-Backup (VMware FT)* | DDIA Ch 5 | Postgres primary + async replica in Docker Compose: replication lag under load, kill the primary mid-write and count lost writes, show a read-your-writes violation (1 evening) |
| 3 | Failures, timeouts, retries | Kleppmann Cambridge lectures on system models + failure detectors | Marc Brooker, *Timeouts, retries and backoff with jitter* (Amazon Builders' Library) · Jeff Hodges, *Notes on Distributed Systems for Young Bloods* · DDIA Ch 8 | Retry storm demo in Go: N clients, naive retries vs exponential backoff + jitter against a flaky server; plot load (1 evening) |
| 4 | Time and clocks | Kleppmann Cambridge lectures on physical / logical clocks | Justin Sheehy, *There Is No Now* (ACM Queue) · Lamport, *Time, Clocks, and the Ordering of Events* (skim) | Last-write-wins with skewed clocks: two Go nodes, one clock +200 ms, show silently lost writes; then partitions with toxiproxy (1 evening) |
| 5 | Consensus: Raft | MIT 6.824 Lec 6–7 *Raft* · Diego Ongaro's Raft lecture (YouTube) | Raft paper (extended) · *The Secret Lives of Data* Raft visualization · Eli Bendersky, *Implementing Raft* series | 6.5840 Raft 3A–3D |
| 6 | Linearizability + consistency models | MIT 6.824 lecture on linearizability (Zookeeper / chain replication weeks) | Kyle Kingsbury (aphyr), *Strong Consistency Models* · Jepsen *Consistency Models* map · DDIA Ch 9 | Run porcupine against a deliberately broken KV store and read the failing history (1 evening) |
| 7 | Fault-tolerant KV on Raft | MIT 6.824 Lec 8–9 *Zookeeper*, *CRAQ* | Jepsen analysis of etcd (pick one recent report) | 6.5840 Lab 4 fault-tolerant KV |
| 8 | Partitioning + consistent hashing | Kleppmann Cambridge lecture on partitioning / quorums | Tom White, *Consistent Hashing* (blog) · Werner Vogels, *Amazon's Dynamo* (All Things Distributed) · DDIA Ch 6 | Consistent-hashing router in Go with virtual nodes: keys moved on join/leave vs mod-N. Connects to llm-d cache-aware routing (1 evening) |
| 9 | Storage engines (LSM vs B-tree) | CMU 15-445 (Fall 2023) storage lectures (pick the LSM / indexes one) | Ben Stopford, *Log Structured Merge Trees* · DDIA Ch 3 | **Mini LSM-tree KV store in Go**: WAL → memtable → SSTables → compaction → bloom filter, vs bbolt: write amp, read latency (2–3 Saturdays, after Raft 3B, portfolio piece) |
| 10 | Transactions + isolation | Kleppmann, *Transactions: Myths, Surprises and Opportunities* (Strange Loop 2015) | Jepsen consistency models (isolation side) · DDIA Ch 7 | Two psql sessions: lost update, write skew, phantoms under READ COMMITTED / REPEATABLE READ / SERIALIZABLE (1 evening) |
| 11 | Sharded KV + global systems | MIT 6.824 *Spanner* lecture | Spanner paper (OSDI '12), skim | 6.5840 Lab 5 sharded KV |

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
| Go services | **Go standard library** | `net/http/server.go`, `sync/mutex.go`, `context/` |

**How to read it:** build your own version first, then read theirs and list 3 things they did differently and why. Trace one request or operation end to end instead of reading files top to bottom. Read the tests, which show what the authors were worried about. Use `git blame` and the linked PRs to find out *why* the code is shaped that way. Code reading counts as morning reading time.

## Rules

- Mornings: read + one small exercise. Evenings stay short. Saturday is for building.
- After every read: 2–3 lines of notes, rolled up into the Sunday post.
- LeetCode: 5–6 a week in Phase 1, daily from January. Timed, weak topics, explained out loud.
- Falling behind moves the calendar. It never means doing two days at once.
- Showing up daily beats long sprints. Sunday stays light.

## Log

| Day | Date | Planned | Done |
| --- | --- | --- | --- |
| 1 | Mon 26 Oct | Book Ch 2 (first half) · intensity script | Read Ch 2 · went through ops:byte, S/P matrices, attention read/write/compute. Calculation didn't click yet → redo from one matmul on Day 2 |
| 2 | Tue 27 Oct | kipply (first half) · matmul cost by hand + script | Read the **whole** kipply post (ahead of plan). Matmul → Wed |
| 3 | Wed 28 Oct | Rooflines chapter · matmul by hand + `matmul_cost` script · 1 LeetCode | Study tracks reorganised topic-wise. Matmul → Fri |
| 4 | Thu 29 Oct | Horace He · GPT-2 start | Schedule reshaped around a morning block · 1 LeetCode |
| 5 | Fri 30 Oct | Matmul exercise + `matmul_cost` script · notes | Horace He ✅ (ahead of Mon) |

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
