+++
title = 'Topics'
date = '2026-11-20T00:00:00+05:30'
lastmod = '2026-11-20T00:00:00+05:30'
draft = false
description = "Everything I'm learning in AI engineering, LLM serving and distributed systems, one topic at a time, each ending in a measured result."
+++

# Topics

Everything I'm working through in AI engineering, LLM serving, backend and distributed systems. **One topic at a time, in this order.** A topic counts as done only when its proof is done: a number, a demo, or something I can explain without notes.

### 1. LLM serving basics
*Read:* Chip Huyen, *AI Engineering* Ch 9 · Raschka, *Build an LLM from Scratch* Ch 2–4 · *How to Scale Your Model*, inference chapter

- [ ] Transformer basics: embeddings → attention (Q/K/V) → MLP → logits, only as deep as serving needs (Raschka Ch 3–4)
- [ ] Decoding + sampling: greedy, temperature, top-p/top-k, stop sequences, why output varies run to run
- [ ] Tokenization: BPE, why token count drives latency + cost, tokens vs words across languages
- [ ] Tokens, prefill vs decode, why decode is memory-bound
- [ ] Context windows: limits, why long context is slow and costly (KV grows linearly, attention quadratically in prefill), lost-in-the-middle
- [ ] KV cache: what it stores, size formula `2 × layers × kv_heads × head_dim × bytes × tokens`, sized for Llama-3-8B at 4K
- [ ] GQA and what it does to KV size
- [ ] Batching basics, TTFT / TPOT / throughput, streaming (SSE)
- [ ] LLM gateway: one API over many models/providers, fallback, per-user quotas, cost tracking, semantic caching (and when it returns wrong answers)
- [ ] Quantization at a high level (INT8/FP8: memory vs quality)
- [ ] Latency + cost optimization: smaller model routing, prompt caching, shorter prompts, batching, rate limits per token, fallback + timeouts, cost per request
- [ ] Streaming responses: SSE end to end, time to first token, cancelling mid-stream, partial-output errors
- [ ] **Proof:** GPT-2 decode time per token with vs without KV cache; cache MB vs sequence length

### 2. Continuous batching + paged KV
*Read:* Huyen Ch 9 (batching) · Orca (OSDI '22) · vLLM / PagedAttention (SOSP '23)

- [ ] Static batching and padding waste
- [ ] Continuous (iteration-level) batching — Orca
- [ ] Paged KV: block table, free list, copy-on-write, fragmentation — vLLM
- [ ] Preemption / eviction when memory runs out
- [ ] Prefix caching
- [ ] **Proof:** static vs continuous on a toy scheduler: throughput + TTFT; allocated vs used blocks

### 3. Go concurrency
*Read:* Cox-Buday, *Concurrency in Go* Ch 3–4

- [ ] Goroutines + scheduler (G/M/P), channels (buffered vs not), `select`
- [ ] `sync`: Mutex, RWMutex, WaitGroup, Once; `atomic`
- [ ] `context`: cancellation, deadlines
- [ ] Patterns: worker pool, pipeline, fan-out/fan-in, bounded concurrency, backpressure
- [ ] Goroutine leaks, data races, `-race`, pprof
- [ ] **Proof:** worker pool vs goroutine-per-task: throughput cliff, `-race` clean, pprof profile

### 4. Kafka + streaming
*Read:* Kleppmann, *DDIA* Ch 11

- [ ] Topics, partitions, offsets, retention
- [ ] Producers: acks, idempotent producer, ordering
- [ ] Consumer groups, rebalancing, lag
- [ ] At-most / at-least / exactly-once; transactions
- [ ] Idempotent consumers, dedup, dead-letter topics, backpressure, schema evolution
- [ ] **Proof:** Narad: show a duplicate delivery, then prove the idempotent consumer; lag under backpressure

### 5. Memory, storage, indexes, transactions
*Read:* *DDIA* Ch 3, Ch 7 · *Use The Index, Luke*

- [ ] Latency numbers (cache vs RAM vs SSD vs network), batching vs row-at-a-time
- [ ] B-tree vs LSM-tree, WAL, pages
- [ ] Indexes, query plans, `EXPLAIN`
- [ ] SQL: joins + NULLs, aggregation, window functions, CTEs, dedup, top-N
- [ ] ACID, isolation levels, lost update, write skew, phantoms, MVCC, optimistic vs pessimistic locking
- [ ] TimescaleDB: hypertables, chunks, compression, retention, continuous aggregates
- [ ] **Proof:** lost-update + write-skew demo in 2 psql sessions; one `EXPLAIN` before/after an index; row vs batched latency

### 6. Embeddings, RAG, vector search
*Read:* Huyen Ch 6 (RAG), Ch 7 (finetuning, first part) · HNSW paper

- [ ] Embeddings + similarity metrics, dimensions
- [ ] Chunking strategies
- [ ] Vector indexes: brute force vs HNSW; metadata filters
- [ ] Top-k, reranking (cross-encoder), hybrid search
- [ ] RAG failure modes: retrieval vs generation errors, stale index
- [ ] Hallucinations: why models make things up, grounding in retrieved text, citations, detecting unsupported claims
- [ ] Fine-tuning vs RAG: knowledge (RAG) vs behaviour/format (fine-tune), cost, freshness; LoRA at a high level (Huyen Ch 7, first part)
- [ ] **Proof:** recall@k + latency, brute force vs HNSW on 100K vectors; with vs without reranking

### 7. Agents, tools, context (bigger topic, ~1.5 weeks)
*Read:* Huyen Ch 2, 5, 6 (agents) · Anthropic, *Building Effective Agents* · MCP spec

- [ ] Function / tool calling: JSON-schema tool definitions, model picks the call, your code runs it, parallel calls, tool errors fed back
- [ ] Structured outputs: JSON schema, constrained decoding, validate + retry on bad output
- [ ] Agentic workflows vs agents: fixed chains, routing, orchestrator-worker, evaluator loop vs an open-ended agent loop (Anthropic, *Building Effective Agents*)
- [ ] Agent loop, ReAct, planning, termination, step limits
- [ ] Prompt engineering basics: system vs user prompt, few-shot examples, instructions that hold, prompt versioning
- [ ] Context engineering: what goes into the window and in what order, trimming, summarising, tool-result size
- [ ] Memory systems: short-term (conversation), long-term (stored facts), retrieval-based memory, when to forget
- [ ] LangGraph: state graph, nodes/edges, conditional routing, checkpoints, retries
- [ ] MCP: host/client/server, tools + schemas, transports, auth
- [ ] Guardrails: input checks (prompt injection, PII), output checks (policy, schema, groundedness), least privilege, read-only SQL, audit logs
- [ ] Bedrock: model access, quotas, retries, data boundaries, cost
- [ ] **Proof:** tiny text-to-SQL agent: % correct SQL on 20 labelled questions; unsafe queries blocked

### 8. Python concurrency + FastAPI
*Read:* *Fluent Python* (2nd ed) Ch 19–21

- [ ] GIL; threads vs processes vs asyncio — when each wins
- [ ] `asyncio`: event loop, tasks, semaphores, timeouts
- [ ] FastAPI: ASGI, sync vs async endpoints, dependencies, Pydantic, what never runs in a web worker
- [ ] **Proof:** threads vs processes vs asyncio on 10K HTTP checks: throughput + memory

### 9. Networking + API resilience
*Read:* Google SRE book Ch 21–22 · *High Performance Browser Networking* Ch 1–4

- [ ] TCP (handshake, flow/congestion control), UDP, TLS, HTTP/1.1 vs 2, DNS
- [ ] REST design: idempotency, pagination, versioning, errors, auth, rate limits
- [ ] Timeouts, retries, backoff + jitter, circuit breakers, load shedding, cascading failures
- [ ] Tail latency (p99) and why fan-out makes it worse
- [ ] Microservices: boundaries, sync vs async, discovery, load balancing, tracing
- [ ] **Proof:** retry storm: naive vs backoff+jitter, plotted load

### 10. Containers + Kubernetes
*Read:* *Kubernetes: Up and Running* (Pods, Services, Deployments)

- [ ] Docker: images, layers, caching, multi-stage, volumes, networking, limits, health checks
- [ ] K8s: Pods, Deployments, Services, Ingress, ConfigMaps/Secrets
- [ ] Probes, requests/limits, HPA, rollouts + rollbacks
- [ ] Helm: values, templates, releases
- [ ] **Proof:** forced failed rollout + rollback, detection and recovery time measured

### 11. Replication + partitioning
*Read:* *DDIA* Ch 5–6

- [ ] Leader-follower, multi-leader, leaderless (quorums)
- [ ] Replication lag, read-your-writes, monotonic reads
- [ ] Partitioning by key vs hash, consistent hashing, hot spots, rebalancing
- [ ] **Proof:** toy leader-follower: lag + read-your-writes violation shown

### 12. Failures, clocks, consensus (2–3 week block)
*Read:* *DDIA* Ch 8–9 · Raft paper · MIT 6.5840 Raft labs

- [ ] Partial failures, timeouts, failure detection
- [ ] Clocks: wall vs monotonic, Lamport / vector clocks
- [ ] CAP in practice, linearizability
- [ ] Leader election, Raft (log replication, terms, safety), split brain, fencing tokens
- [ ] **Proof:** 6.5840 Raft 3A–3B pass 3× on different days; leader re-election time measured

### 13. Evaluation (build lands with engine step 6)
*Read:* Huyen Ch 3–4, Ch 10 · Hamel Husain, *Your AI Product Needs Evals*

- [ ] Benchmarks vs task-specific eval sets
- [ ] LLM-as-judge: biases, multiple samples, aggregation
- [ ] Confidence intervals, sample size
- [ ] Eval as a gate for changes (quality vs speed)
- [ ] LLM observability: trace every call (prompt, tools, tokens, latency, cost), log for debugging, feed failures back into the eval set
- [ ] Evaluating RAG + agents: retrieval metrics, answer faithfulness, tool-call success, end-to-end task success
- [ ] **Proof:** harness gating one engine change: quality vs speed table

### 14. OS + memory
*Read:* OSTEP: event-based concurrency + virtual memory chapters

- [ ] Processes vs threads, context switches, scheduling
- [ ] Virtual memory, paging, TLB, page faults
- [ ] File descriptors, signals, pipes
- [ ] epoll/kqueue, event-driven servers, how Go netpoll uses them
- [ ] **Proof:** epoll/kqueue echo server: connections vs memory

### 15. Rust
*Read:* *The Rust Book* Ch 4, 10, 16, 17

- [ ] Ownership, borrowing, lifetimes
- [ ] Traits, generics, `Option`/`Result`, error handling
- [ ] `Arc`/`Mutex`, `Send`/`Sync`, async basics
- [ ] **Proof:** change one web-shooter feature; re-run the 405 ms benchmark

### 16. Distributed serving
*Read:* *How to Scale Your Model* (parallelism, inference) · DistServe (OSDI '24)

- [ ] Tensor vs pipeline parallelism (what each costs)
- [ ] Prefill/decode disaggregation (DistServe)
- [ ] Routing: round-robin vs least-load vs cache-aware
- [ ] **Proof:** cache-aware vs round-robin router: hit rate + p99

## Build: mini inference engine

- [ ] GPT-2 124M + own KV cache → tokens/sec, memory
- [ ] Static batching → padding waste measured
- [ ] Paged KV cache (block table, free list, copy-on-write)
- [ ] Continuous batching + preemption/eviction
- [ ] 2 replicas + cache-aware router → hit rate vs round-robin
- [ ] Quality gate (GSM8K ± INT8) → quality vs speed table

