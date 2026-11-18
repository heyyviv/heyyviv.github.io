+++
title = "Interview Study Guide"
slug = "interview-study-guide"
date = "2026-11-18T21:00:00+05:30"
description = "Everything to study so no backend/infra interview question is a surprise: every resume topic, the concepts to know, and a one-line answer for each."
tags = ["blog","interview","backend","notes"]
+++

**How to use this.** Each line is one concept with its short answer. A line is *done* only when you can say it out loud without looking, then answer "why?" and "what breaks?" about it. Tick as you go. Order = most likely to be asked first.

Deeper notes: [Go Runtime & Concurrency](/blog/go-runtime-concurrency/) · [DBMS](/blog/dbms_content/) · [Distributed Systems](/blog/dis_sys_content/)

---

## 1. Your story and numbers (asked in every interview)

- [ ] **60-second intro** — who you are → what you build at Samsung (Go services, data pipelines, AI tooling) → one big number → why Singapore / this role.
- [ ] **For every number** (1M assets/day, 500K/day at 99%, −54.6% / $610K, 1–2h → 10 min, 405 ms, 14x) know: where it came from, the baseline, the time window, how it was measured, and one caveat.
- [ ] **For every bullet** know: the problem, your part vs the team's, the design, one trade-off, the hardest bug, what breaks at 10x, what you'd change now.
- [ ] **"3-stage", "2-service", "4-component"** on the SG resume — name each stage/service/component.

## 2. Go

- [ ] **Goroutine vs thread** — managed by the Go runtime; starts with ~2 KB growable stack (threads ~1–8 MB fixed); switch is in user space, so it's cheap.
- [ ] **G/M/P scheduler** — G = goroutine, M = OS thread, P = processor (run queue, count = `GOMAXPROCS`). An M needs a P to run Gs. Idle Ps steal work. Blocking syscalls hand the P to another M.
- [ ] **Channels** — unbuffered = sender and receiver meet (sync); buffered = queue of N. Send on closed channel panics; receive from closed returns zero value. Nil channel blocks forever.
- [ ] **`select`** — waits on many channel ops; picks a random ready one; `default` makes it non-blocking.
- [ ] **`context`** — carries cancellation, deadlines, request values down the call tree. Goroutines `select` on `ctx.Done()` to stop. Always `defer cancel()`.
- [ ] **`sync.Mutex` / `RWMutex`** — protect shared data; RW allows many readers or one writer.
- [ ] **`sync.WaitGroup`** — wait for N goroutines; `Add` before `go`, `Done` in a defer.
- [ ] **`sync/atomic`** — lock-free counters/flags.
- [ ] **Race detector** — `go test -race`; only finds races that actually execute in the test.
- [ ] **Goroutine leak** — goroutine blocked forever on a channel no one reads; fix with context/close/buffer.
- [ ] **Worker pool** — N goroutines read jobs from one channel → bounded concurrency + backpressure.
- [ ] **Errors** — values, not exceptions; wrap with `fmt.Errorf("...: %w", err)`, check with `errors.Is/As`. `panic` only for bugs.
- [ ] **Interfaces** — implicit; small interfaces (`io.Reader`). Nil interface vs interface holding nil pointer gotcha.
- [ ] **Slices** — pointer + len + cap over an array; `append` may reallocate; two slices can share memory.
- [ ] **Maps** — not safe for concurrent writes (runtime panic); use mutex or `sync.Map`.
- [ ] **GC** — concurrent mark-and-sweep, low pause; tune with `GOGC`; reduce allocations to reduce GC.
- [ ] **Escape analysis** — compiler decides stack vs heap; returning a pointer to a local moves it to the heap (`go build -gcflags=-m`).
- [ ] **pprof** — CPU, heap, goroutine profiles; how you'd find a slow path.

## 3. Python

- [ ] **GIL** — only one thread runs Python bytecode at a time. Threads help **I/O-bound** work (network, disk — GIL released while waiting); **CPU-bound** work needs `multiprocessing`.
- [ ] **threads vs processes vs asyncio** — threads: I/O, simple; processes: CPU, separate memory; asyncio: one thread, thousands of I/O tasks, cooperative (`await`).
- [ ] **Bounded concurrency** — `asyncio.Semaphore(n)` or `ThreadPoolExecutor(max_workers=n)`.
- [ ] **Timeouts / retries** — `asyncio.wait_for`; retry with exponential backoff + jitter; only retry transient errors.
- [ ] **Generators** — lazy iteration with `yield`; constant memory over big data.
- [ ] **Decorators, context managers** (`with` → `__enter__/__exit__`), **mutable default args** gotcha, **list vs tuple vs dict vs set** complexity.

## 4. Concurrency (general)

- [ ] **Race condition** — result depends on timing of unsynchronised access to shared data.
- [ ] **Deadlock** — circular wait. Needs: mutual exclusion, hold-and-wait, no preemption, circular wait. Avoid: always lock in the same order; timeouts.
- [ ] **Livelock / starvation** — threads keep reacting but no progress / one thread never gets the resource.
- [ ] **Mutex vs semaphore** — mutex = 1 owner; semaphore = N permits.
- [ ] **Backpressure** — slow consumer pushes back on producer (bounded queue blocks / rejects) instead of memory blowing up.
- [ ] **Producer–consumer** — bounded queue between them.

## 5. SQL and databases

- [ ] **Joins** — inner, left, right, full; NULLs don't equal each other (`IS NULL`).
- [ ] **GROUP BY / HAVING**, **window functions** (`ROW_NUMBER`, `RANK`, `LAG`, running totals), **CTEs**.
- [ ] **Classic queries** — top-N per group, dedupe (keep latest), second highest, running total, gaps.
- [ ] **Index = B+ tree** — sorted keys → row location; lookup O(log n) vs full scan; leaves linked for range scans. Writes slower: every index updated.
- [ ] **Composite index** — leftmost-prefix rule: index (a, b) helps `WHERE a=` and `a= AND b=`, not `b=` alone.
- [ ] **Covering index**, **when an index isn't used** (function on column, low selectivity, leading `%LIKE`).
- [ ] **`EXPLAIN ANALYZE`** — read seq scan vs index scan, rows estimated vs actual.
- [ ] **ACID** — Atomic (all or nothing), Consistent (constraints hold), Isolated (concurrent tx don't interfere), Durable (committed survives crash, via WAL).
- [ ] **Isolation anomalies** — dirty read, non-repeatable read, phantom, lost update, write skew.
- [ ] **Isolation levels** — Read Committed (Postgres default), Repeatable Read, Serializable.
- [ ] **Lost update fix** — atomic `UPDATE ... SET x = x - 1 WHERE x >= 1`; `SELECT ... FOR UPDATE`; optimistic lock with version column.
- [ ] **MVCC** — readers see a snapshot, don't block writers.
- [ ] **WAL** — write log first, then data; crash recovery replays it.
- [ ] **Normalisation vs denormalisation** — fewer anomalies vs faster reads.
- [ ] **Replication** — primary takes writes, replicas lag → stale reads; read-your-writes by reading the primary after a write.
- [ ] **Sharding** — split data by key; hot shards; cross-shard queries are hard.
- [ ] **SQL vs NoSQL** — choose by access pattern, consistency need, scale.
- [ ] **TimescaleDB** — Postgres extension; **hypertable** auto-split into time **chunks**; retention = drop old chunks; compression; continuous aggregates = auto-refreshed rollups.

## 6. Kafka

- [ ] **Topic → partitions → offsets** — a partition is an append-only log; offset = position.
- [ ] **Ordering** — only within a partition. Same key → same partition. Adding partitions remaps keys.
- [ ] **Producer acks** — `acks=0` fire-and-forget, `1` leader wrote, `all` all in-sync replicas wrote.
- [ ] **Replication** — leader + followers; ISR = in-sync replicas; leader dies → an ISR member takes over.
- [ ] **Consumer group** — each partition read by one consumer in the group; more consumers than partitions = idle consumers.
- [ ] **Rebalance** — consumer joins/leaves → partitions reassigned; processing pauses.
- [ ] **Offset commit** — commit after processing = at-least-once (duplicates possible); before = at-most-once (loss possible).
- [ ] **Exactly-once** — idempotent producer + transactions; in practice: at-least-once + **idempotent consumer** (dedupe by event ID).
- [ ] **Consumer lag** — latest offset − committed offset; the main health metric.
- [ ] **Retention** — by time/size; log compaction keeps the latest value per key.
- [ ] **Dead-letter topic** — park poison messages so the partition keeps moving.
- [ ] **Kafka vs RabbitMQ/SQS** — Kafka: replayable log, high throughput, ordering per partition; queues: per-message ack, delete on consume.

## 7. Distributed systems

- [ ] **CAP** — during a network partition, pick consistency (refuse) or availability (serve possibly stale).
- [ ] **Consistency models** — strong/linearizable, read-your-writes, eventual.
- [ ] **Replication** — leader-follower, multi-leader, leaderless (quorum: W + R > N).
- [ ] **Partitioning** — range vs hash; **consistent hashing** moves few keys when nodes change.
- [ ] **Consensus (Raft)** — elect a leader, replicate a log to a majority; survives minority failures. Used by etcd, Kafka KRaft.
- [ ] **Leader election, split brain** — two leaders; fixed by majority quorum + fencing tokens.
- [ ] **Clocks** — wall clocks drift; use logical clocks / versions for ordering.
- [ ] **Distributed transactions** — 2PC (blocking); **saga** = local steps + compensations.
- [ ] **Outbox pattern** — write event to DB table in same tx, relay to Kafka → no lost/dual writes.
- [ ] **Idempotency** — same request twice = same effect; idempotency key.
- [ ] **Retries** — exponential backoff + jitter; retry budget to avoid retry storms.
- [ ] **Circuit breaker** — stop calling a failing dependency, fail fast, probe later.
- [ ] **Timeouts everywhere**, **bulkheads** (isolate resources), **graceful degradation**.

## 8. Backend / API design

- [ ] **HTTP methods** — GET/PUT/DELETE idempotent; POST not. **Status codes** — 200, 201, 204, 400, 401, 403, 404, 409, 429, 500, 503.
- [ ] **Pagination** — offset (simple, slow + shifting) vs **cursor** (stable, fast).
- [ ] **Auth** — authN = who you are (JWT, OAuth, API keys); authZ = what you may do.
- [ ] **Rate limiting** — token bucket / sliding window; return 429.
- [ ] **Caching** — cache-aside (read cache → miss → DB → fill); on write update DB then **delete** key; TTL as safety net. **Stampede** fix: lock or early refresh. Write-through vs write-back.
- [ ] **Versioning**, consistent error format, input validation at the boundary.
- [ ] **Microservices** — clear boundaries, sync (HTTP/gRPC) vs async (queue), service discovery, each hop needs timeout + retry + metric.
- [ ] **Observability** — logs (what), metrics (how much — RED: rate, errors, duration), traces (where time went).

## 9. Networking

- [ ] **URL → response** — DNS (name → IP) → TCP handshake (SYN, SYN-ACK, ACK) → TLS handshake (certs, keys) → HTTP request → load balancer → app → DB → response.
- [ ] **TCP vs UDP** — TCP: connection, ordered, reliable, flow + congestion control. UDP: no guarantees, low latency (video calls, DNS, games).
- [ ] **HTTP/1.1 vs 2 vs 3** — keep-alive; multiplexing over one connection; QUIC over UDP.
- [ ] **TLS** — encrypts + authenticates the server with a certificate.
- [ ] **Load balancer** — L4 (TCP) vs L7 (HTTP, path routing); health checks; round robin / least connections.
- [ ] **DNS** — records (A, CNAME), TTL caching. **CDN**, **reverse proxy**.
- [ ] **WebSocket vs polling**, **gRPC** (HTTP/2 + protobuf).

## 10. OS and Linux

- [ ] **Process vs thread** — threads share heap, code, globals, file descriptors; each has own stack + registers. Thread switch cheaper (no address-space change).
- [ ] **Context switch** — save/restore registers; process switch also changes page tables (TLB flush).
- [ ] **Virtual memory** — each process sees its own address space; page tables map to physical; page fault loads a page.
- [ ] **Stack vs heap** — stack: function frames, automatic; heap: dynamic, GC/free.
- [ ] **File descriptors**, **signals** (SIGTERM graceful, SIGKILL not catchable), **pipes**.
- [ ] **I/O models** — blocking, non-blocking, epoll (how Go's netpoller handles thousands of connections).
- [ ] **Page cache**, `fsync`.
- [ ] **Debug commands** — `top/htop`, `ps`, `lsof`, `netstat/ss`, `strace`, `df`, `free`, `journalctl`, `curl -v`, `dig`.

## 11. Data stack

**Airflow**
- [ ] DAG of tasks; scheduler; executor (Local/Celery/K8s); retries + timeouts per task; **pools** cap concurrency; tasks must be **idempotent** (re-runs, backfills); XCom only for small data.

**Spark / EMR**
- [ ] Driver plans, executors run tasks; lazy transformations, actions trigger a job.
- [ ] Narrow (map, filter) vs wide (groupBy, join → **shuffle**). Shuffle = expensive network + disk.
- [ ] **Skew** — one key dominates → one slow task; fix: salting, broadcast join.
- [ ] Broadcast join for small tables; partitions sizing; caching; fault tolerance via lineage.
- [ ] EMR — managed Spark cluster on EC2; data in S3; autoscaling; spot instances for cost.

**Snowflake + dbt**
- [ ] Storage separate from compute; **virtual warehouses** (XS…4XL) billed in credits per second while running; auto-suspend/resume; queuing when busy.
- [ ] Bigger warehouse = faster but more credits/hour; right-sizing = route each workload to the smallest warehouse that meets its SLA (**your $610K story**).
- [ ] Result cache, micro-partitions, clustering keys, query profile.
- [ ] dbt — SQL models + dependencies (DAG), materialisations (view, table, incremental), tests, docs.

## 12. AI stack

- [ ] **Embeddings** — text → vector; similar meaning = close vectors (cosine similarity).
- [ ] **RAG** — chunk docs → embed → store in vector DB → at query time retrieve top-k → put in prompt → generate.
- [ ] **Chunking**, **top-k**, **reranking** (cross-encoder rescoring), **metadata filters**.
- [ ] **Vector index** — HNSW (graph-based approximate nearest neighbour); Qdrant uses it.
- [ ] **Evaluate separately** — retrieval (did we fetch the right schema?) vs generation (is the SQL correct?).
- [ ] **Text-to-SQL safety** — read-only role, parse/validate SQL, row limits, timeouts, allow-listed tables, audit log.
- [ ] **LangGraph** — state graph: nodes (steps/agents), edges (conditional routing), shared state, retries, stop condition. Why multi-agent: separate retrieve / generate / validate / fix steps.
- [ ] **MCP** — protocol letting AI apps call tools: host ↔ client ↔ server; server exposes tools with JSON schemas. Risks: prompt injection, over-broad permissions → least privilege, validate inputs.
- [ ] **Bedrock** — AWS managed LLM API; quotas, throttling → retries; data stays in AWS.
- [ ] **FastAPI** — ASGI, async endpoints, Pydantic validation, dependency injection; never block the event loop with CPU work.
- [ ] **LLM serving basics** (for AI-infra roles) — tokens, prefill vs decode, **KV cache**, batching, latency (TTFT) vs throughput.

## 13. Infra and DevOps

- [ ] **Docker** — image = layers; container = running image; layer caching (copy deps first); multi-stage builds; volumes; non-root user.
- [ ] **Kubernetes** — Pod, Deployment (replicas + rolling update), Service (stable IP/DNS), Ingress; ConfigMap/Secret.
- [ ] **Probes** — liveness (restart if dead), readiness (send traffic only when ready), startup.
- [ ] **Requests/limits** — scheduling vs cap; over memory limit → OOMKilled.
- [ ] **HPA** — scale pods on CPU/custom metrics. **Rollback** — `kubectl rollout undo`.
- [ ] **Helm** — templated K8s manifests + values; releases with rollback.
- [ ] **AWS** — EC2 (VMs), S3 (object store, strongly consistent now), IAM (roles + least privilege), VPC basics.
- [ ] **CI/CD** — build → test → deploy; blue-green vs canary vs rolling; secrets never in repo.
- [ ] **Git** — merge vs rebase, resolving conflicts, revert vs reset.

## 14. System design

**Method (use every time):** requirements (functional + scale numbers) → back-of-envelope (QPS, storage) → API → data model → high-level diagram → deep dive on the bottleneck → failures + scaling → trade-offs.

- [ ] Numbers to know: 1 day ≈ 86,400 s ≈ 10^5 s; 1M/day ≈ 12/s avg; memory ~100 ns, SSD ~100 µs, same-region network ~0.5 ms, cross-region ~100 ms.
- [ ] **Your three designs** — (1) 500K/day creative validation pipeline; (2) Narad at 10x logs; (3) safe text-to-SQL.
- [ ] **Classics** — URL shortener, rate limiter, notification system, chat, news feed, distributed cache, job scheduler, web crawler (you built one!).

## 15. DSA (live coding)

- [ ] **Protocol** — clarify → examples → brute force → complexity → optimise → code → dry run → edge cases. Talk the whole time.
- [ ] **Patterns** — two pointers, sliding window, binary search (incl. on answer), prefix sums, hashing, stack/monotonic stack, heap/top-k, BFS/DFS, topological sort, union-find, Dijkstra, DP (1D, 2D, knapsack, intervals), backtracking, tries, intervals, LRU cache.

## 16. Other languages on the resume (be honest)

- [ ] **Rust** — ownership, borrowing, lifetimes, `Result/Option`, `Arc<Mutex>`, `Send/Sync`, async (tokio). Explain one web-shooter request path.
- [ ] **C++** — RAII, smart pointers, move semantics, STL complexity, virtual functions.
- [ ] **Java** — JVM/GC, collections, `equals/hashCode`, executors, `volatile/synchronized`.
- [ ] If you can't defend one, say "used it for X, not production depth" — or remove it.

## 17. Behavioural

- [ ] **STAR stories** (situation, task, action, result with a number) for: hardest bug, a conflict, a failure, ownership beyond your role, a tight deadline, disagreeing with a senior, learning something fast.
- [ ] **Logistics** — why Singapore, notice period, EP sponsorship needed, expected salary, start date.
- [ ] **Questions to ask them** — team's biggest technical problem, on-call, how success is measured in 6 months.
