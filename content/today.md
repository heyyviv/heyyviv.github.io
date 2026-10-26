+++
title = 'Today'
date = '2026-10-26T00:00:00+05:30'
lastmod = '2026-10-26T22:00:00+05:30'
draft = false
description = "What to read and build today: inference engineering, MIT 6.5840, CMU 15-445 and LeetCode."
+++

# What to read today

*Updated daily. Last update: **Mon, 26 Oct 2026** · Day 1 · Week 1 · Phase 1 (Foundations, Months 1–2)*

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
| Tue 27 | kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) | 1 LeetCode warm-up · 6.5840: check where MapReduce / Raft 3A stand, continue |
| Wed 28 | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) + 3Blue1Brown [GPT and attention videos](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (ch 5–6) | Book Ch 2 (rest) · start Karpathy [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (code along) |
| Thu 29 | Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | 1 LeetCode warm-up · 6.5840 lab |
| Fri 30 | CMU 15-445 [Lecture 1](https://15445.courses.cs.cmu.edu/) at 1.5x | Karpathy GPT continued (15-445 labs wait until Raft is done) |
| Sat 31 | — | 5–6 h: finish Karpathy GPT, train the tiny model, or 6.5840 catch-up |
| Sun 1 | Light | First weekly notes post on this blog · optional LeetCode / contest |

## Reading queue (Phase 1)

- [ ] kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [ ] Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) + 3Blue1Brown transformer videos
- [ ] Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [ ] Google, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (rooflines, parallelism)
- [ ] Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [ ] [vLLM / PagedAttention blog post](https://blog.vllm.ai/2023/06/20/vllm.html) + Anyscale, [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ ] Martin Kleppmann, [distributed systems lectures](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB) (pairs with 6.5840)
- [ ] [Raft paper](https://raft.github.io/raft.pdf) (before Raft 3A)
- [ ] GPU MODE lectures on YouTube (closer to Phase 2)
- [ ] *Inference Engineering*: Ch 2 ⬜ · Ch 3 ⬜ · Ch 5 ⬜

## Rules

- Office is for reading and watching. Evenings are for building.
- After every read: 2–3 lines of notes, rolled up into the Sunday post.
- LeetCode: 3–4 a week, timed, weak topics, explained out loud.
- Falling behind moves the calendar. It never means doing two days at once.

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
