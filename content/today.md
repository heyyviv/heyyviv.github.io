+++
title = 'Today'
date = '2026-10-26T00:00:00+05:30'
lastmod = '2026-10-26T22:45:00+05:30'
draft = false
description = "What to read and build today: inference engineering, MIT 6.5840, CMU 15-445 and LeetCode."
+++

# What to read today

*Updated daily. Last update: **Mon, 26 Oct 2026** · Day 1 · Week 1 · **Phase 1 of 4: Resume + proof** (Nov–Dec) · Next checkpoint: **AI-infra resume done by 30 Nov***

## Today: Day 1, Mon 26 Oct

**Theme:** why decode is memory-bound and prefill is compute-bound.

### 🏢 Office

Already over today. From tomorrow, this slot is for reading and watching only.

### 🌙 Evening (~1 h, it's a late start)

| Time | Task |
| --- | --- |
| 0:00–0:40 | **Read** *Inference Engineering* Ch 2 (models and bottlenecks), up to and including prefill vs decode. Stop there. |
| 0:40–1:00 | **Code.** A small Python script for Llama-3-8B: FLOPs and bytes moved per token for prefill and for decode, arithmetic intensity (FLOPs ÷ bytes) for each, compared with your machine's compute-to-bandwidth ratio. |
| 1:00–1:05 | **Notes.** 2–3 lines: the key idea, one question, how it maps to DB / distributed systems ideas. |

**Done when:** the script prints both intensities and shows decode is memory-bound.

---

## This week (Week 1: 26 Oct – 1 Nov)

| Day | 🏢 Office (read / watch) | 🌙 Evening (build) |
| --- | --- | --- |
| Mon 26 | — | Book Ch 2 (first half) + arithmetic-intensity script |
| Tue 27 | kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) | 1 LeetCode · 6.5840: check where MapReduce / Raft 3A stand, continue |
| Wed 28 | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the rooflines chapter | 1 LeetCode · Book Ch 2 (rest) · start GPT-2 inference from scratch: load the real weights into your own PyTorch model, check outputs match Hugging Face |
| Thu 29 | Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | 1 LeetCode · 6.5840 lab |
| Fri 30 | CMU 15-445 [Lecture 1](https://15445.courses.cs.cmu.edu/) at 1.5x | GPT-2: add a KV cache (15-445 labs wait until Raft is done) |
| Sat 31 | — | 1 LeetCode · 5–6 h: GPT-2 benchmark: tokens/sec and memory with vs without KV cache, across batch sizes · or 6.5840 catch-up |
| Sun 1 | Light | Weekly contest · **Resume pass 1:** list every bullet missing a number (p99, throughput, users, hours saved) and collect the numbers from work · first weekly notes post |

## Roadmap (Oct 2026 – Sep 2027)

| Phase | When | Focus |
| --- | --- | --- |
| **1. Resume + proof** | Nov – Dec 2026 | AI-infra resume with numbers · GPT-2 + KV cache benchmark · Narad benchmarks · Raft 3A–3B · LeetCode 5–6/week |
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
- [ ] Kleppmann, *Designing Data-Intensive Applications* (1st ed.), one chapter a week from Week 2, office reading: Ch 5 Replication ⬜ · Ch 8 Trouble with distributed systems ⬜ · Ch 9 Consistency and consensus ⬜ (these three go with Raft) · then Ch 3 Storage ⬜ · Ch 7 Transactions ⬜ · Ch 6 Partitioning ⬜ (these go with 15-445)

## Rules

- Office is for reading and watching. Evenings are for building.
- After every read: 2–3 lines of notes, rolled up into the Sunday post.
- LeetCode: 5–6 a week in Phase 1, daily from January. Timed, weak topics, explained out loud.
- Falling behind moves the calendar. It never means doing two days at once.
- Eight months of showing up beats two weeks of 14-hour days. Sunday stays light.

## Log

| Day | Date | Planned | Done |
| --- | --- | --- | --- |
| 1 | Mon 26 Oct | Book Ch 2 (first half) · intensity script | *(pending)* |

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
