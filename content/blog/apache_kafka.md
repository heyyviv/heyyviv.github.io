+++
title = "Apache Kafka"
slug = "apache-kafka"
date = "2026-11-21T17:22:50+05:30"
description = "Kafka from the ground up: why a log, why it is fast on plain disks, producers and consumers, offsets and lag, replication, exactly-once, backpressure and schema evolution."
tags = ["blog","kafka","distributed-systems","backend","system-design","notes"]
+++

<style>
.kf { --ink:#2D3748; --muted:#718096; --line:#CBD5E0; --card:#FFFFFF; --s1:#2a78d6; --s2:#eb6834; --s3:#2f9e6e; margin:1.6em 0; }
body.dark-mode .kf { --ink:#E2E8F0; --muted:#A0AEC0; --line:#4A5568; --card:#2D3748; --s1:#3987e5; --s2:#d95926; --s3:#3fb07f; }
.kf svg { width:100%; height:auto; display:block; font-family:var(--font-primary, sans-serif); }
.kf .cap { font-size:.85em; color:var(--muted); text-align:center; margin-top:.4em; }
.kf .box { fill:var(--card); stroke:var(--line); stroke-width:1.5; }
.kf .box.user { stroke:var(--s2); stroke-dasharray:5 4; }
.kf .lbl { fill:var(--ink); font-size:13px; font-weight:600; }
.kf .sub { fill:var(--muted); font-size:11px; }
.kf .wire { stroke:var(--line); stroke-width:2; fill:none; }
.kf .c1 { fill:var(--s1); } .kf .c2 { fill:var(--s2); } .kf .c3 { fill:var(--s3); }
.kf .cell { fill:var(--s1); stroke:var(--card); stroke-width:2; }
.kf .cell.f { fill:var(--s3); }
.kf .okline { fill:none; stroke:var(--s3); stroke-width:2.5; }
.kf .done { fill:var(--line); }
.kf .replay { fill:color-mix(in srgb, var(--s2) 35%, var(--card)); stroke:var(--s2); stroke-width:1.5; }
.kf .ahead { fill:var(--card); stroke:var(--line); stroke-width:1.5; }
.kf .unsafe { fill:var(--card); stroke:var(--muted); stroke-width:1.5; stroke-dasharray:4 3; }
.kf .mark { fill:var(--ink); }
.kf .fade { opacity:0; animation:kfIn .6s ease-out forwards; }
.kf .fade.d1 { animation-delay:.4s; } .kf .fade.d2 { animation-delay:.9s; } .kf .fade.d3 { animation-delay:1.4s; } .kf .fade.d4 { animation-delay:1.9s; }
.kf .pulse { animation:kfPulse 1.6s ease-in-out infinite; }
@keyframes kfIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
@keyframes kfPulse { 50% { opacity:.35; } }
@media (prefers-reduced-motion: reduce) {
  .kf * { animation:none !important; opacity:1 !important; }
  .kf .moving { display:none; }
}
</style>

Kafka is often called a "message queue", but at its heart it is something simpler: a **log**. Once you understand the log, almost every Kafka design decision follows from it.

## 1. What is a log?

A log is a data structure with two rules:

1. **Append only.** You only ever add records at the end. Nothing is updated in place.
2. **Ordered.** Records stay in the order they were appended. Each one gets a number, its **offset**.

<div class="kf">
<svg viewBox="0 0 640 150" role="img" aria-label="A log: records are appended one after another at the end, numbered by offset 0 to 7">
<g>
<rect class="cell" x="20" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.05;.08;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
<rect class="cell" x="95" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.15;.18;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
<rect class="cell" x="170" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.25;.28;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
<rect class="cell" x="245" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.35;.38;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
<rect class="cell" x="320" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.45;.48;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
<rect class="cell" x="395" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.55;.58;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
<rect class="cell" x="470" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.65;.68;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
<rect class="cell" x="545" y="50" width="70" height="44" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.75;.78;.88;.95;1" dur="8s" repeatCount="indefinite"/></rect>
</g>
<g class="sub" text-anchor="middle">
<text x="55" y="115">0</text><text x="130" y="115">1</text><text x="205" y="115">2</text><text x="280" y="115">3</text><text x="355" y="115">4</text><text x="430" y="115">5</text><text x="505" y="115">6</text><text x="580" y="115">7</text>
</g>
<text class="sub" x="20" y="135">offset</text>
<text class="sub" x="320" y="30" text-anchor="middle">old records on the left · new records always land at the end →</text>
</svg>
<div class="cap">Writers only append. Readers move left to right. Nobody edits the middle.</div>
</div>

### Logs inside databases

Databases already depend on logs:

- **Write-ahead log (WAL).** Every transaction is first written to a log. Only then is it applied to the B-tree. After a crash, the database replays the log to recover.
- **Replication.** The primary writes everything it does to a log. A replica reads that log and replays it to end up in the same state.

### Logs in distributed systems

When you have a leader and replicas, the hard question is: how do they stay consistent? There are two classic answers, and both are built on an ordered log.

| Model | How it works | Why the log matters |
|---|---|---|
| **Active-active** (state machine replication) | Leader and replicas all apply the same instructions | Everyone must apply them **in the same order**, and the log gives that order |
| **Primary-backup** | The primary does the work, then copies the result to the backups | The log is the stream of results being copied |

Kafka takes this idea and turns it into a product: a log that many machines can write to and read from.

## 2. Why Kafka is fast on "slow" disks

### Disks are slow only at random access

Kafka stores everything on disk, and people say disks are slow. That is only true for **random** access, when the disk head has to jump around. Kafka's log is append only, so:

- **Writes** always go to the end of a file: sequential.
- **Reads** go from some offset forward: also sequential.

Kafka also doesn't keep its own cache. It writes to the **OS page cache** and lets the operating system flush to disk in the background. Data sitting in the page cache can be lost if the machine dies before the flush, and that is why Kafka relies on **replicas** on other machines for durability.

### Why not just use a database?

A traditional design puts messages in a database table, backed by a B-tree.

| | B-tree (database) | Log (Kafka) |
|---|---|---|
| Lookup cost | O(log n), which sounds fast… | O(1) to append, sequential to read |
| On a real disk | Each level can be a random seek, which is slow | Pure sequential I/O |
| Maintenance | Rebalancing the tree on writes is costly | Nothing to rebalance |
| Readers vs writers | Compete for locks and pages | Writers at the end, readers behind: they don't block each other |
| Keeping old data | Expensive | Disks are cheap, so keep messages for days or weeks |

### Zero-copy: skip user space

To send data to a consumer, a normal program reads the file into its own memory (user space) and then writes it back to a socket. The bytes cross the kernel/user boundary twice, and the CPU does the copying.

Kafka uses the `sendfile` system call instead. The bytes go straight from the page cache to the network card, all inside the kernel, mostly by DMA. The CPU stays free.

<div class="kf">
<svg viewBox="0 0 700 250" role="img" aria-label="Normal send path goes disk, page cache, app buffer in user space, socket buffer, network card. Zero-copy path goes disk, page cache, network card.">
<path id="kfNormal" class="wire" d="M65,70 L625,70"/>
<path id="kfZero" class="wire" d="M65,190 L625,190"/>
<circle class="c2 moving" r="6"><animateMotion dur="4s" repeatCount="indefinite"><mpath href="#kfNormal"/></animateMotion></circle>
<circle class="c2 moving" r="6"><animateMotion dur="4s" begin="2s" repeatCount="indefinite"><mpath href="#kfNormal"/></animateMotion></circle>
<circle class="c1 moving" r="6"><animateMotion dur="2s" repeatCount="indefinite"><mpath href="#kfZero"/></animateMotion></circle>
<circle class="c1 moving" r="6"><animateMotion dur="2s" begin="1s" repeatCount="indefinite"><mpath href="#kfZero"/></animateMotion></circle>
<text class="lbl" x="10" y="25">Normal read() + write()</text><text class="sub" x="690" y="25" text-anchor="end">4 copies, 2 of them by the CPU</text>
<rect class="box" x="10" y="45" width="110" height="50" rx="8"/><text class="lbl" x="65" y="75" text-anchor="middle">Disk</text>
<rect class="box" x="150" y="45" width="110" height="50" rx="8"/><text class="lbl" x="205" y="75" text-anchor="middle">Page cache</text>
<rect class="box user" x="290" y="45" width="110" height="50" rx="8"/><text class="lbl" x="345" y="68" text-anchor="middle">App buffer</text><text class="sub" x="345" y="84" text-anchor="middle">user space</text>
<rect class="box" x="430" y="45" width="110" height="50" rx="8"/><text class="lbl" x="485" y="75" text-anchor="middle">Socket buffer</text>
<rect class="box" x="570" y="45" width="110" height="50" rx="8"/><text class="lbl" x="625" y="75" text-anchor="middle">NIC</text>
<text class="lbl" x="10" y="145">Kafka: sendfile()</text><text class="sub" x="690" y="145" text-anchor="end">never leaves the kernel</text>
<rect class="box" x="10" y="165" width="110" height="50" rx="8"/><text class="lbl" x="65" y="195" text-anchor="middle">Disk</text>
<rect class="box" x="150" y="165" width="110" height="50" rx="8"/><text class="lbl" x="205" y="195" text-anchor="middle">Page cache</text>
<text class="sub" x="415" y="182" text-anchor="middle">user space skipped</text>
<rect class="box" x="570" y="165" width="110" height="50" rx="8"/><text class="lbl" x="625" y="195" text-anchor="middle">NIC</text>
</svg>
<div class="cap">Same bytes, same destination. The blue path is shorter and doesn't burn CPU on copies.</div>
</div>

### Compress the batch, not the message

Compressing each message on its own is wasteful: small messages compress badly. Kafka compresses a **whole batch** instead:

1. The producer compresses the batch once.
2. The broker stores it **as is**, still compressed.
3. The consumer decompresses it.

One compression, one decompression, and the broker never pays for either.

## 3. Producers

### Vocabulary

| Term | Meaning |
|---|---|
| **Broker** | One Kafka server |
| **Topic** | A named stream of messages, e.g. `page_views` |
| **Partition** | A topic split into several separate logs |
| **Leader** | The one broker in charge of a partition. All writes go to it; other brokers hold copies (replicas) |

The user picks how many partitions a topic has. Each partition has its own leader, and the leaders are spread across brokers. A topic `pg` with 3 partitions might look like this: `b0` leads P2, `b1` leads P1, `b2` leads P0.

### Batching: when does the producer send?

The producer doesn't send each message right away. It fills a batch and sends when **either** the batch is full **or** a wait time runs out (`batch.size` and `linger.ms`).

| | Big batches | Small batches |
|---|---|---|
| Throughput | High | Low |
| Compression | Better | Worse |
| Latency | Higher | Lower |

### No load balancer in the middle

If producers sent to a load balancer, which then forwarded to the right broker, every message would take two network hops. Kafka avoids that:

1. The producer asks **any** broker: "who leads each partition of this topic?" Every broker knows the answer.
2. It then sends batches **directly** to the leader of the right partition.

### Which partition?

- **Round robin or random**, when messages have no key.
- **`hash(key) % number_of_partitions`**, the default when a message has a key.

The keyed option is the important one. All messages for a user (key = `user_id`) land in the **same partition**, so:

- **Order is preserved** for that user, because one partition is one ordered log.
- **Locality is preserved.** The consumer reading that partition sees all of that user's events, so it never has to ask another consumer "what's the state of this user?"

<div class="kf">
<svg viewBox="0 0 700 240" role="img" aria-label="Producer sends messages to three partitions. Each key color always goes to the same partition.">
<path id="kfP0" class="wire" d="M140,120 C300,120 330,45 480,45"/>
<path id="kfP1" class="wire" d="M140,120 L480,120"/>
<path id="kfP2" class="wire" d="M140,120 C300,120 330,195 480,195"/>
<circle class="c2 moving" r="6"><animateMotion dur="3s" repeatCount="indefinite"><mpath href="#kfP0"/></animateMotion></circle>
<circle class="c2 moving" r="6"><animateMotion dur="3s" begin="1.5s" repeatCount="indefinite"><mpath href="#kfP0"/></animateMotion></circle>
<circle class="c1 moving" r="6"><animateMotion dur="3s" begin=".5s" repeatCount="indefinite"><mpath href="#kfP1"/></animateMotion></circle>
<circle class="c1 moving" r="6"><animateMotion dur="3s" begin="2s" repeatCount="indefinite"><mpath href="#kfP1"/></animateMotion></circle>
<circle class="c3 moving" r="6"><animateMotion dur="3s" begin="1s" repeatCount="indefinite"><mpath href="#kfP2"/></animateMotion></circle>
<circle class="c3 moving" r="6"><animateMotion dur="3s" begin="2.5s" repeatCount="indefinite"><mpath href="#kfP2"/></animateMotion></circle>
<rect class="box" x="10" y="93" width="130" height="54" rx="10"/><text class="lbl" x="75" y="117" text-anchor="middle">Producer</text><text class="sub" x="75" y="133" text-anchor="middle">hash(key) % 3</text>
<rect class="box" x="480" y="20" width="210" height="50" rx="10"/><text class="lbl" x="585" y="42" text-anchor="middle">P0 · leader b2</text><text class="sub" x="585" y="58" text-anchor="middle">key = user B</text>
<rect class="box" x="480" y="95" width="210" height="50" rx="10"/><text class="lbl" x="585" y="117" text-anchor="middle">P1 · leader b1</text><text class="sub" x="585" y="133" text-anchor="middle">key = user A</text>
<rect class="box" x="480" y="170" width="210" height="50" rx="10"/><text class="lbl" x="585" y="192" text-anchor="middle">P2 · leader b0</text><text class="sub" x="585" y="208" text-anchor="middle">key = user C</text>
</svg>
<div class="cap">Each color is one key. The same key always takes the same path, so its messages stay in order.</div>
</div>

## 4. Consumers

A consumer asks the partition leader: "give me data starting at offset X". The broker sends messages from X onward.

### Push vs pull

| | Push | Pull |
|---|---|---|
| Who decides the speed? | The sender | The receiver |
| Kafka uses it for | Producer → broker | Broker → consumer |

Why pull for consumers? With push, a fast producer can **flood a slow consumer** and crush it. With pull, each consumer asks for "data from offset X, up to Y MB" and takes only what it can handle.

### Long polling

Pull has one problem. If there is no new data, the consumer keeps asking and gets nothing back: a **busy wait** that wastes CPU and network.

Kafka fixes this with **long polling**. The consumer says: "if you have nothing, hold my request for up to *T* ms, and reply as soon as at least *N* KB arrives" (`fetch.max.wait.ms`, `fetch.min.bytes`).

### One consumer per partition

Within a **consumer group**, each partition is read by **exactly one** consumer at a time. One consumer can read several partitions, but two consumers never share one.

Why? If two consumers read the same partition, a slow acknowledgement could let both process the same message, and the order would be lost. With one reader per partition, the progress is just **one number**, the offset, instead of per-message acks.

**Static membership:** give each consumer a `group.instance.id`. If it restarts, it gets the **same partitions** back without a full rebalance of the group.

## 5. How many partitions?

- **Producer side:** a producer can write to many partitions in parallel.
- **Consumer side:** each partition has only one reader in a group, so **consumer parallelism = number of partitions**. Extra consumers sit idle.

More partitions means more throughput. For a target throughput *t*, with *p* = what one producer can push into one partition and *c* = what one consumer can read from one partition:

```text
partitions = max(t / p, t / c)
```

Example: t = 100 MB/s, p = 20 MB/s, c = 10 MB/s → max(5, 10) = **10 partitions**.

Don't overdo it. Every partition is a set of open files on the broker, and more partitions means slower leader elections and rebalances.

## 6. Offsets: four positions in one partition

<div class="kf">
<svg viewBox="0 0 680 160" role="img" aria-label="Partition offsets 0 to 10. Last committed offset is 1, current position is 6, high watermark is 9, log end is 10.">
<rect class="done" x="20" y="55" width="52" height="40" rx="5"/>
<rect class="replay" x="78" y="55" width="52" height="40" rx="5"/><rect class="replay" x="136" y="55" width="52" height="40" rx="5"/><rect class="replay" x="194" y="55" width="52" height="40" rx="5"/><rect class="replay" x="252" y="55" width="52" height="40" rx="5"/><rect class="replay" x="310" y="55" width="52" height="40" rx="5"/>
<rect class="ahead" x="368" y="55" width="52" height="40" rx="5"/><rect class="ahead" x="426" y="55" width="52" height="40" rx="5"/><rect class="ahead" x="484" y="55" width="52" height="40" rx="5"/><rect class="ahead" x="542" y="55" width="52" height="40" rx="5"/>
<rect class="unsafe" x="600" y="55" width="52" height="40" rx="5"/>
<g class="sub" text-anchor="middle"><text x="46" y="80">0</text><text x="104" y="80">1</text><text x="162" y="80">2</text><text x="220" y="80">3</text><text x="278" y="80">4</text><text x="336" y="80">5</text><text x="394" y="80">6</text><text x="452" y="80">7</text><text x="510" y="80">8</text><text x="568" y="80">9</text><text x="626" y="80">10</text></g>
<g class="fade d1"><path class="mark" d="M104,100 l-7,10 h14 z"/><text class="lbl" x="104" y="128" text-anchor="middle">last committed</text></g>
<g class="fade d2"><path class="mark pulse" d="M394,100 l-7,10 h14 z"/><text class="lbl" x="394" y="128" text-anchor="middle">current position</text></g>
<g class="fade d3"><path class="mark" d="M568,50 l-7,-10 h14 z"/><text class="lbl" x="575" y="30" text-anchor="end">high watermark</text></g>
<g class="fade d4"><path class="mark" d="M626,50 l-7,-10 h14 z"/><text class="lbl" x="619" y="30" text-anchor="start">log end</text></g>
<text class="sub" x="220" y="150" text-anchor="middle">orange: read but not committed (replayed after a crash)</text>
<text class="sub" x="626" y="150" text-anchor="middle">leader only</text>
</svg>
</div>

| Position | Meaning | Reading a book |
|---|---|---|
| **Last committed** (1) | Last progress saved to Kafka | The bookmark you put in |
| **Current position** (6) | Where the consumer is actually reading | The page your eyes are on |
| **High watermark** (9) | Last message copied to all in-sync replicas | Pages printed in every copy of the book |
| **Log end** (10) | Last message written to the leader | Pages the author has written so far |

Two rules follow:

1. **After a crash, the next consumer resumes from the committed offset**, not the current position. Here the consumer read up to 6 but only saved 1. Whoever takes over starts at 1, so messages 1–5 are processed **again**.
2. **Consumers can only read up to the high watermark.** Messages past it exist only on the leader. If the leader dies they could disappear, so Kafka hides them until they are replicated. You never read something that could later vanish.

## 7. Committing offsets

### Delivery guarantees

The order of "process" and "commit" decides what happens on a crash:

| Strategy | Order | On crash | Result |
|---|---|---|---|
| **At least once** | Process, then commit | Messages after the last commit are processed again | No loss, possible duplicates |
| **At most once** | Commit, then process | Committed but unprocessed messages are skipped | No duplicates, possible loss |

At least once is the common choice. Make your processing **idempotent** so duplicates don't hurt.

### How often to commit

| When | Verdict |
|---|---|
| After every message | ❌ A round trip per message clogs the network and kills throughput |
| After every N messages | ✅ |
| After each batch from a partition | ✅ |
| Auto-commit every X seconds | ✅ Simple, but you may replay up to X seconds of work |

### commitSync vs commitAsync

| | `commitSync` | `commitAsync` |
|---|---|---|
| Waits for the broker? | Yes, blocks | No, keeps processing |
| Retries on failure? | Yes | No (a late retry could overwrite a newer commit) |
| Throughput | Lower | Higher |
| Learn about failure | Immediately | Later, in a callback |

A common pattern: `commitAsync` in the normal loop for speed, and one `commitSync` on shutdown or rebalance so the final position is saved for sure.

## 8. Consumer lag

```text
lag = log end offset − committed offset
```

Lag is how far behind the consumer is.

- **Lag ≈ 0 and stable:** the consumer keeps up. ✅
- **Lag growing steadily:** consumers are slower than producers. 🚨 Add consumers (up to the partition count), make processing faster, or add partitions.

Lag is the first number to watch on any Kafka dashboard.

## 9. Producer reliability: acks, retries, idempotence

Four producer settings decide whether a message can be lost, duplicated or reordered.

### `acks`: when does the leader say "OK"?

| `acks` | "OK" means | Speed | If the leader dies just after |
|---|---|---|---|
| `0` | Nothing. The producer doesn't wait at all | Fastest | The message may never have arrived, and you won't know |
| `1` | The leader wrote it to its own log | Fast | Lost, because the followers hadn't copied it yet |
| `all` (same as `-1`) | Every in-sync replica has it | Slower | Safe, as long as one in-sync replica survives (but see [the trap](#the-acksall-trap--mininsyncreplicas)) |

Default: `all` since Kafka 3.0. Whatever the producer picks, consumers only see a message once it is **committed**.

### `retries`: resend on failure?

Failures like "leader is changing" or "network timeout" are **transient**, so resending usually works.

- `retries` is how many times to resend. Default: `2147483647`, which is effectively forever.
- In practice the limit is **time, not count**. `delivery.timeout.ms` (default 120,000 ms = 2 minutes) is the total time allowed from `send()` to success or failure.
- The docs say: leave `retries` alone and control retrying with `delivery.timeout.ms`.
- `retries=0` passes every error straight to your code.

Retries cause two problems.

**Problem A: duplicates.** The broker wrote the batch, but the producer never heard back, so it sent it again.

```text
Producer sends batch B   ──► Leader writes B ✓
Producer ◄── "OK" lost on the network ✗
Producer times out, retries ──► Leader writes B AGAIN ❌ duplicate
```

**Problem B: reordering.** This only happens with more than one request in flight.

```text
Send batch 1  ──► fails ✗
Send batch 2  ──► succeeds ✓  (offset 100)
Retry batch 1 ──► succeeds ✓  (offset 101)
Log order: 2, 1 ❌   You sent 1, 2.
```

### `max.in.flight.requests.per.connection`

"In flight" means sent but not yet acknowledged.

| Value | Behaviour | Trade-off |
|---|---|---|
| `1` | Send, wait for "OK", send the next | Order is safe, but you wait a full round trip every time |
| `5` (default) | Send up to 5 before waiting | Much faster because the pipe stays full, but Problem B can happen |

Think of letters. With 1 in flight you post a letter and wait for the reply before posting the next. With 5 you post five, and if letter 1 gets lost and is re-posted, it arrives after letter 2.

The old trade-off was **fast (5) or ordered (1): pick one**. Idempotence removes it.

### `enable.idempotence`: the fix for both problems

*Idempotent* means doing something twice has the same effect as doing it once. Here is how it works:

1. The broker gives the producer a **Producer ID (PID)**.
2. The producer tags every batch with a **sequence number per partition**: 0, 1, 2, 3…
3. The broker remembers the last sequence number it accepted for each PID and partition.

| Broker receives | Broker does |
|---|---|
| Sequence = last + 1 | ✅ Writes it |
| Sequence ≤ last (already seen) | Drops it as a duplicate, **but still replies "OK"**. This fixes Problem A |
| Sequence > last + 1 (a gap) | Rejects it as "out of order", and the producer resends in the right order. This fixes Problem B |

<div class="kf">
<svg viewBox="0 0 700 230" role="img" aria-label="Idempotent producer: batch with sequence 3 is written, the OK reply is lost, the producer retries sequence 3, and the broker drops the duplicate but still replies OK.">
<path id="kfSend" class="wire" d="M150,65 L550,65"/>
<path id="kfAck" class="wire" d="M550,120 L150,120"/>
<path id="kfRetry" class="wire" d="M150,175 L550,175"/>
<rect class="box" x="10" y="35" width="140" height="170" rx="10"/><text class="lbl" x="80" y="115" text-anchor="middle">Producer</text><text class="sub" x="80" y="133" text-anchor="middle">PID 42</text>
<rect class="box" x="550" y="35" width="140" height="170" rx="10"/><text class="lbl" x="620" y="115" text-anchor="middle">Broker</text><text class="sub" x="620" y="133" text-anchor="middle">remembers last seq</text>
<g class="sub" text-anchor="middle"><text x="350" y="55">① send batch, seq 3</text><text x="350" y="110">② "OK" on its way back</text><text x="350" y="165">③ retry, seq 3</text></g>
<text class="lbl c3" x="350" y="88" text-anchor="middle" opacity="0">written, last seq = 3 ✓<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.2;.23;.9;.95;1" dur="9s" repeatCount="indefinite"/></text>
<text class="lbl c2" x="350" y="143" text-anchor="middle" opacity="0">✗ lost on the network<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.38;.41;.9;.95;1" dur="9s" repeatCount="indefinite"/></text>
<text class="lbl c3" x="350" y="198" text-anchor="middle" opacity="0">3 ≤ 3 → duplicate dropped, still replies OK<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.7;.73;.9;.95;1" dur="9s" repeatCount="indefinite"/></text>
<circle class="c1 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;.01;.2;.21;1" dur="9s" repeatCount="indefinite"/><animateMotion dur="9s" repeatCount="indefinite" calcMode="linear" keyPoints="0;1;1" keyTimes="0;.2;1"><mpath href="#kfSend"/></animateMotion></circle>
<circle class="c3 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.25;.26;.38;.39;1" dur="9s" repeatCount="indefinite"/><animateMotion dur="9s" repeatCount="indefinite" calcMode="linear" keyPoints="0;0;.5;.5" keyTimes="0;.25;.38;1"><mpath href="#kfAck"/></animateMotion></circle>
<circle class="c1 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.5;.51;.7;.71;1" dur="9s" repeatCount="indefinite"/><animateMotion dur="9s" repeatCount="indefinite" calcMode="linear" keyPoints="0;0;1;1" keyTimes="0;.5;.7;1"><mpath href="#kfRetry"/></animateMotion></circle>
</svg>
<div class="cap">The retry carries the same sequence number, so the broker can tell it is a duplicate.</div>
</div>

Default: `true`. Idempotence requires all three of these:

- `acks=all`
- `retries > 0`
- `max.in.flight.requests.per.connection ≤ 5`

**Why ≤ 5?** From the docs: "the broker only retains at most 5 batches for each producer." The broker keeps the last 5 batches per producer for duplicate checking. With more than 5 in flight, it may already have forgotten an older batch and can't tell whether a retry is a duplicate.

## 10. Replication

Every partition is copied to several brokers, for example 3 copies.

- One copy is the **leader**. Producers write to it, and consumers normally read from it too (reading from followers is optional).
- The others are **followers**. They keep asking the leader "anything new?" (a fetch request) and copy it in order. They pull the same way a normal consumer does, so a follower grabs many messages per fetch and batching comes for free.
- Every copy has the same messages at the same offsets. The leader may just have a few new ones at the end that haven't been copied yet.
- The total number of copies is the **replication factor (RF)**. A topic with "no replication" is just RF = 1. Replication isn't an add-on; it's built in from the start.
- A cluster has thousands of partitions, and leaders are spread evenly across brokers so no single server does all the writing.

<div class="kf">
<svg viewBox="0 0 700 250" role="img" aria-label="A leader appends messages. Follower b2 copies them right behind. Follower b3 stalls after two messages and is removed from the in-sync replica set.">
<text class="lbl" x="10" y="52">Leader · b1</text><text class="sub" x="10" y="68">producers write here</text>
<text class="lbl" x="10" y="117">Follower · b2</text><text class="sub" x="10" y="133">in ISR</text>
<text class="lbl" x="10" y="182">Follower · b3</text><text class="sub" x="10" y="198">lagging</text>
<g>
<rect class="cell" x="150" y="35" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.04;.06;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell" x="230" y="35" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.13;.15;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell" x="310" y="35" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.22;.24;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell" x="390" y="35" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.31;.33;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell" x="470" y="35" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.40;.42;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell" x="550" y="35" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.49;.51;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
</g>
<g>
<rect class="cell f" x="150" y="100" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.09;.11;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell f" x="230" y="100" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.18;.20;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell f" x="310" y="100" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.27;.29;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell f" x="390" y="100" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.36;.38;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell f" x="470" y="100" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.45;.47;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell f" x="550" y="100" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.54;.56;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
</g>
<g>
<rect class="cell f" x="150" y="165" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.12;.14;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
<rect class="cell f" x="230" y="165" width="70" height="40" rx="6" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.24;.26;.9;.95;1" dur="10s" repeatCount="indefinite"/></rect>
</g>
<text class="lbl c2" x="320" y="190" opacity="0">✗ stuck too long → removed from ISR<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.68;.71;.9;.95;1" dur="10s" repeatCount="indefinite"/></text>
<text class="sub" x="385" y="235" text-anchor="middle">ISR = the replicas that are caught up enough to count</text>
</svg>
<div class="cap">b2 keeps up and stays in sync. b3 falls behind and gets dropped, so writes stop waiting for it.</div>
</div>

### ISR: in-sync replicas

The leader keeps a list of followers that are "caught up enough", the **ISR**. It matters for two reasons:

1. A message is **committed** (safe) only when every member of the ISR has it. Only then does an `acks=all` producer get "OK, saved".
2. If the leader dies, the new leader is picked **only from the ISR**. Every ISR member has every committed message, so nothing is lost.

That creates a tension:

- A slow follower in the ISR slows **every** write, because each commit waits for the slowest member.
- So slow followers must be kicked out of the ISR quickly.
- But kicking out a healthy follower by mistake is bad too. You have fewer safe copies, and an "under-replicated partition" alert fires and wakes someone up at 3 a.m.

### Who stays in the ISR?

Today a replica must pass two checks:

| Condition | Checked by | Fails when |
|---|---|---|
| Heartbeating the controller (the cluster's manager) | Controller | No heartbeat within `broker.session.timeout.ms`, so the broker counts as offline |
| Keeping up with the leader | Leader | Not caught up for `replica.lag.time.max.ms` |

### The old rule, and why it was dropped

Kafka 0.8.2 used two settings:

1. `replica.lag.time.max.ms`: "if a follower hasn't asked me for data in X ms, it's stuck. Kick it out." This worked fine, and it is still the rule today.
2. `replica.lag.max.messages`: "if a follower is more than N messages behind, it's slow. Kick it out." **This was the problem.**

Why is "number of messages behind" a bad rule? To set N correctly you'd have to guess your traffic: how big your batches will be, on every topic, forever. One cluster has many topics with different traffic, and traffic changes over time. A single burst can put a perfectly healthy follower N messages behind for a moment, and it gets kicked out. No single number works for all topics, so the count-based setting was removed and only the time-based rule remains.

### The core guarantee

> A committed message is never lost as long as at least one in-sync replica stays alive.

### Why not majority voting?

**Approach A: majority voting** (Raft, Zab, Paxos)

- With 2f+1 copies, a write is committed once f+1 (a majority) have it.
- The new leader is chosen by a majority vote.
- Any two majorities share at least one node, so some voter always has every committed message.

The upside is that latency depends on the **fastest** majority. With 3 copies you wait for the faster follower and can ignore the slow one. The downside is cost: surviving 1 failure needs 3 copies, and surviving 2 needs 5. That is 5× the disk and roughly ⅕ the throughput.

**Approach B: Kafka's ISR**

- A commit needs **every** ISR member, not a majority.
- Only ISR members can become leader, and every one of them has every committed message.
- So with f+1 copies, Kafka survives f failures.

| To survive 1 failure | Majority vote | Kafka ISR |
|---|---|---|
| Copies needed | 3 | **2** |
| Acks waited for | 1 follower | 1 follower |
| Disk cost | 3× | **2×** |
| Slow follower | Ignored, only the faster majority counts | Slows commits until it's removed from the ISR |

Kafka's reasoning: for large volumes of data, storing 3 to 5 copies of everything costs too much. That's why majority voting is usually kept for small metadata (ZooKeeper, KRaft, the HDFS NameNode journal) while bulk data uses cheaper schemes. The slow-follower downside is limited because slow replicas get removed from the ISR.

The closest academic design to Kafka is **PacificA** (Microsoft Research). You don't need to know it in detail, but mentioning it in an interview is a bonus.

### Bonus: no fsync needed

Many consensus algorithms assume a crashed node comes back with all its data intact, which means calling `fsync` on every write. That can be 100–1000× slower. Kafka doesn't assume this. A replica that crashes must fully catch up again before it rejoins the ISR, even if it lost data that was never flushed to disk. Safety comes from **copies on other servers**, not from forcing every write to disk. This is the replica-based durability from [section 2](#2-why-kafka-is-fast-on-slow-disks).

### Unclean leader election: what if every ISR replica dies?

The guarantee needs at least one in-sync replica alive. When none are left, there are two choices:

| Option | Result | Picks |
|---|---|---|
| Wait for an ISR replica to come back | Partition is down until it returns, and down forever if its disk is gone | Consistency (no data loss) |
| Take any replica that comes back, even one not in the ISR | Partition is back up fast, but messages it never received are lost | Availability |

- Config: `unclean.leader.election.enable`
- Default: `false` (wait) since Kafka 0.11.
- This is the CAP trade-off in practice. Majority-vote systems face the same choice when a majority dies for good.

### The `acks=all` trap → `min.insync.replicas`

`acks=all` means every replica **currently in the ISR**, not every replica that exists.

```text
RF = 2. One broker dies        → ISR = {leader}
acks=all now means "the leader alone has it" → write succeeds
Leader dies                    → message lost 💀
```

Two settings fix this:

1. `unclean.leader.election.enable=false`: never promote a replica that isn't in sync.
2. `min.insync.replicas=2`: an `acks=all` write succeeds only if the ISR still has at least 2 members. If it shrinks below that, the producer gets a `NotEnoughReplicas` error instead of a false "OK".

The common production setup is **RF = 3, `min.insync.replicas=2`, `acks=all`**. You can lose one broker and keep accepting writes, and every acknowledged write is on at least two machines.

## 11. Exactly once: transactions

Idempotence stops duplicates from a single producer's retries. Real apps do more: they **read** from one topic and **write** to another. Take a payment processor that reads payments from `payments` and writes debits to `ledger`. Three things can go wrong:

| # | Failure | What goes wrong | Fix |
|---|---|---|---|
| 1 | Producer retries a send after a lost ack | The debit is written twice | Idempotent producer ([section 9](#enableidempotence-the-fix-for-both-problems)) |
| 2 | App crashes after writing the debit but before saving its offset | On restart it reads the payment again and writes the debit again | Atomic transactions |
| 3 | An app is wrongly declared dead and a replacement starts, but the old one is still running (a "zombie") | Two instances write the same debits | Zombie fencing |

Failure 2, step by step:

```text
read payment ✓  →  write debit ✓  →  💥 crash  →  save offset ✗
```

On restart the committed offset still points at the payment, so the app reads it again and writes the debit again.

Failure 3: instance #1 freezes, say on a long GC pause or a network blip. The system decides it's dead and starts instance #2. Then #1 wakes up and keeps working. Now two instances are writing the same debits.

Transactions fix failures 2 and 3.

### The key trick: "consumed" is just another write

Kafka records that a message was consumed by writing its offset to an internal topic, `__consumer_offsets`. Committing an offset is therefore **a write to a partition**, just like writing the debit.

So the problem becomes: two writes have to happen together, the debit and the offset. A transaction gives you exactly that. All writes in it become visible together, or none do.

```text
begin transaction
   write debit    → ledger
   write offset   → __consumer_offsets
commit transaction → both become real at once
```

<div class="kf">
<svg viewBox="0 0 700 230" role="img" aria-label="The app reads a payment, then writes a debit to the ledger and an offset to __consumer_offsets inside one transaction. On commit both become visible at the same moment.">
<path id="kfTxIn" class="wire" d="M140,115 L220,115"/>
<path id="kfTxA" class="wire" d="M340,115 C410,115 410,52 480,52"/>
<path id="kfTxB" class="wire" d="M340,115 C410,115 410,177 480,177"/>
<rect class="box" x="10" y="88" width="130" height="54" rx="10"/><text class="lbl" x="75" y="112" text-anchor="middle">payments</text><text class="sub" x="75" y="128" text-anchor="middle">input · tp0</text>
<rect class="box" x="220" y="88" width="120" height="54" rx="10"/><text class="lbl" x="280" y="112" text-anchor="middle">App</text><text class="sub" x="280" y="128" text-anchor="middle">read → process</text>
<rect class="box" x="480" y="25" width="210" height="55" rx="10"/><text class="lbl" x="585" y="48" text-anchor="middle">ledger · tp1</text><text class="sub" x="585" y="65" text-anchor="middle">the debit</text>
<rect class="box" x="480" y="150" width="210" height="55" rx="10"/><text class="lbl" x="585" y="173" text-anchor="middle">__consumer_offsets</text><text class="sub" x="585" y="190" text-anchor="middle">"payment consumed"</text>
<rect class="okline" x="480" y="25" width="210" height="55" rx="10" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.6;.63;.9;.95;1" dur="7s" repeatCount="indefinite"/></rect>
<rect class="okline" x="480" y="150" width="210" height="55" rx="10" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.6;.63;.9;.95;1" dur="7s" repeatCount="indefinite"/></rect>
<text class="sub" x="585" y="110" text-anchor="middle">one transaction</text>
<text class="lbl c3" x="585" y="130" text-anchor="middle" opacity="0">COMMIT: both visible ✓<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.6;.63;.9;.95;1" dur="7s" repeatCount="indefinite"/></text>
<circle class="c2 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;.01;.18;.19;1" dur="7s" repeatCount="indefinite"/><animateMotion dur="7s" repeatCount="indefinite" calcMode="linear" keyPoints="0;1;1" keyTimes="0;.18;1"><mpath href="#kfTxIn"/></animateMotion></circle>
<circle class="c1 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.22;.23;.45;.46;1" dur="7s" repeatCount="indefinite"/><animateMotion dur="7s" repeatCount="indefinite" calcMode="linear" keyPoints="0;0;1;1" keyTimes="0;.22;.45;1"><mpath href="#kfTxA"/></animateMotion></circle>
<circle class="c1 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.22;.23;.45;.46;1" dur="7s" repeatCount="indefinite"/><animateMotion dur="7s" repeatCount="indefinite" calcMode="linear" keyPoints="0;0;1;1" keyTimes="0;.22;.45;1"><mpath href="#kfTxB"/></animateMotion></circle>
</svg>
<div class="cap">Until the commit, consumers reading with <code>isolation.level=read_committed</code> see neither write.</div>
</div>

If the app crashes before the commit, both writes are thrown away. On restart it reads the payment again, and that is fine, because the first debit never became visible. Exactly one debit ends up in the ledger.

### Zombie fencing

- Each producer has a stable **`transactional.id`**, the same one across restarts, e.g. `payment-processor-0`. The new instance #2 uses the same id as the "dead" instance #1.
- On startup it calls `initTransactions()`. The broker finishes or aborts any half-done transaction for that id and **increments an epoch**, a version number for the id, say from 5 to 6.
- Any write that carries an older epoch is rejected. When zombie #1 wakes up and tries to write with epoch 5, it is cut off.

<div class="kf">
<svg viewBox="0 0 700 210" role="img" aria-label="Instance 2 with epoch 6 writes successfully. The zombie instance 1 with epoch 5 tries to write and is rejected.">
<path id="kfZ1" class="wire" d="M180,50 C340,50 340,95 500,95"/>
<path id="kfZ2" class="wire" d="M180,160 C340,160 340,115 500,115"/>
<rect class="box user" x="10" y="20" width="170" height="60" rx="10"/><text class="lbl" x="95" y="46" text-anchor="middle">Instance #1 (zombie)</text><text class="sub" x="95" y="63" text-anchor="middle">epoch 5</text>
<rect class="box" x="10" y="130" width="170" height="60" rx="10"/><text class="lbl" x="95" y="156" text-anchor="middle">Instance #2</text><text class="sub" x="95" y="173" text-anchor="middle">epoch 6</text>
<rect class="box" x="500" y="70" width="190" height="70" rx="10"/><text class="lbl" x="595" y="98" text-anchor="middle">Broker</text><text class="sub" x="595" y="116" text-anchor="middle">payment-processor-0 → epoch 6</text>
<text class="lbl c3" x="340" y="190" text-anchor="middle" opacity="0">epoch 6 = current → written ✓<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.35;.38;.9;.95;1" dur="7s" repeatCount="indefinite"/></text>
<text class="lbl c2" x="340" y="28" text-anchor="middle" opacity="0">epoch 5 &lt; 6 → rejected ✗<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.72;.75;.9;.95;1" dur="7s" repeatCount="indefinite"/></text>
<circle class="c3 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.05;.06;.35;.36;1" dur="7s" repeatCount="indefinite"/><animateMotion dur="7s" repeatCount="indefinite" calcMode="linear" keyPoints="0;0;1;1" keyTimes="0;.05;.35;1"><mpath href="#kfZ2"/></animateMotion></circle>
<circle class="c2 moving" r="7" opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;.45;.46;.72;.73;1" dur="7s" repeatCount="indefinite"/><animateMotion dur="7s" repeatCount="indefinite" calcMode="linear" keyPoints="0;0;.9;.9" keyTimes="0;.45;.72;1"><mpath href="#kfZ1"/></animateMotion></circle>
</svg>
<div class="cap">Like a hotel that changes the key-card code for each new guest: the previous guest's card stops working.</div>
</div>

It's the same idea as a **fencing token**, or a **term number** in Raft. If you know one of them, you know this.

**The catch:** fencing only works if the same input partitions always map to the same `transactional.id`. If `payments` partition 0 was handled by T0 and later moves to T1, Kafka sees two different ids and nothing stops T0 from writing alongside T1. The mapping has to stay fixed, e.g. id = "processor for partition 0". Kafka Streams does this for you by deriving the id from the input partition.

### How it works inside

```text
Producer ──(1) initTransactions / addPartitions / commit ──► Transaction Coordinator
                                                               │ (runs inside a broker)
                                                               ▼
                                                      __transaction_state topic
                                                      (stores txn state only, not data)
Producer ──(2) normal sends ──► tp1, tp2, __consumer_offsets
Coordinator ──(3) COMMIT / ABORT markers ──► every partition in the txn
```

1. The producer registers with the **transaction coordinator** and tells it which partitions the transaction touches. The coordinator logs this in `__transaction_state`.
2. The actual messages go straight to their partitions as usual, offsets included.
3. On commit, the coordinator writes a **COMMIT marker** into every partition involved. `read_committed` consumers only deliver messages once they see that marker, and skip aborted ones.


## 12. Backpressure: when the consumer is slower than the producer

Producers write 1,000 msg/s, the consumer handles 500 msg/s. A backlog builds up. The good news: the backlog waits safely in Kafka, not in your app. The job is to slow down *without* getting kicked out of the group.

### Symptoms

| Symptom | What's happening |
|---|---|
| Lag increasing | The consumer's position falls further behind the newest message |
| Memory pressure | You fetched too many records at once and they're sitting in RAM |
| Rebalancing | Processing one batch took longer than `max.poll.interval.ms`, so you're kicked, and the next owner re-reads, causing duplicates |
| Timeout errors | Heartbeats stopped (the process is frozen or dead) for longer than `session.timeout.ms` |

### Pause and resume, step by step

```java
while (true) {
    checkAndResumePartitions();                          // 1
    records = consumer.poll(Duration.ofMillis(1000));    // 2
    for (TopicPartition partition : records.partitions()) {
        int recordCount = records.records(partition).size();
        partitionBacklog.put(partition, currentBacklog + recordCount);   // 3
        if (partitionBacklog.get(partition) > backlogThreshold) {
            pausePartition(partition);                   // 4
        }
        for (ConsumerRecord record : records.records(partition)) {
            try { processRecord(record); }
            finally { backlog -= 1; }                    // 5
        }
    }
}
```

1. For each paused partition, if its counter is under 500 (half of 1,000), resume it.
2. Ask Kafka for records and wait up to 1 s. Paused partitions return nothing, but this call still tells Kafka "I'm alive". That's the point of pausing instead of stopping.
3. For each partition that returned records, add the count to its counter. Got 100 from partition 0, so `counter[P0] += 100`.
4. If the counter is over 1,000, pause this partition.
5. Each finished record lowers the counter again.

Pausing at 1,000 and resuming at 500 (not at 999) stops the partition from flipping on and off every poll. That gap is called **hysteresis**.

### The other tools

- **Batch writes:** write 500 rows to the database in one call instead of 500 calls.
- **Bound in-flight work:** a fixed number of workers or a semaphore, never "one goroutine per message".
- **Rate limit** calls to a slow downstream.
- **Don't** add partitions or consumers when the slow part is the downstream. More consumers just push more load onto the thing that is already slow.

What to watch: **consumer lag**. The real danger is lag growing older than the topic's **retention**: Kafka then deletes messages nobody has read yet.

### Long jobs: pause the container (Spring Kafka example)

Settings: `max.poll.records = 1` (one message per poll), `max.poll.interval.ms = 30s`, `concurrency = 20` (20 consumer threads).

When a message arrives:

1. Pause the container: "give me nothing more".
2. Hand the message to a background thread (executor), which starts the long job.
3. The listener function **returns immediately**, so the main thread goes back to polling. It gets nothing, but stays alive.
4. When the job finishes (success or error), resume the container: "OK, give me the next one".

So each consumer handles one message at a time, in the background, and keeps polling in the foreground so the group knows it's alive. A job can take 10 minutes even though the deadline is 30 s, because polling never stops.

**The weak spot (the offset trap from earlier):** Spring commits the offset when the listener function returns, which is step 3. The job is still running at that point. If the app crashes at minute 5 of the job:

- Kafka already has "this message is done" saved.
- After restart, the consumer starts at the next message.
- The crashed message is lost.

**The fix:** turn off auto-commit and commit manually inside the success callback (step 4), after the job really finished.

## 13. Schema evolution

Kafka keeps messages for days and lets you replay them. So one topic always holds data written by **several versions** of the producer code, read by **several versions** of the consumer code. You can't upgrade everything at once. The schema has to change in a way both sides survive.

### Two directions

| Name | Means | Needed when |
|---|---|---|
| **Backward** compatible | New code reads old data | You upgrade **consumers first** (and always, for replay) |
| **Forward** compatible | Old code reads new data | You upgrade **producers first** |
| **Full** | Both | Either side can deploy first |

### Protobuf: fields are numbers

On the wire, each field is `tag number + type + value`. Field **names are never sent**.

- **Add a field:** safe, with a new tag number. Old code skips tags it doesn't know (forward). New code reading old data gets the default value (backward).
- **Remove a field:** safe, as long as the tag number is **never reused**. Mark it `reserved 4;` so nobody can.
- **Rename a field:** safe in binary (the name isn't on the wire). Breaks the JSON mapping, though.
- **Change a type:** mostly unsafe. A few pairs are wire-compatible (`int32`/`int64`/`bool`), but they can truncate.

### Avro: no tags, two schemas

Avro binary is just the values back to back, in schema order. No tags, no names. You **cannot** decode it without the exact schema it was written with (the **writer's schema**). The reader also has its own schema (the **reader's schema**), and Avro matches fields between the two **by name**:

- **Add a field:** only with a **default**. A new reader reading old data fills in the default.
- **Remove a field:** only if it **had a default**. An old reader reading new data fills it in.
- **Rename a field:** rejected by default. Avro sees it as "`msg` deleted, `message` added with no default", so a new consumer reading an old message finds no `message` and nothing to fill in. Only an **alias** in the reader's schema makes it work.
- **Change a type:** only listed promotions: `int → long → float → double`, `string ↔ bytes`.

The rule for both formats: **add optional fields, never rename or retype, never reuse a field id.**

### Schema Registry: how the consumer finds the writer's schema

Avro needs the writer's schema, but sending the whole schema with each message would be wasteful. The Schema Registry solves that:

```text
Producer ──register schema──► Schema Registry ──► schema ID 42
Producer ──► Kafka message = [magic byte 0][4-byte schema ID = 42][payload]
Consumer ──reads ID 42──► Registry (cached) ──► writer's schema ──► decode
```

At registration, the registry **checks compatibility** against the previous versions and **rejects** a breaking schema before any bad message is produced. Modes: `BACKWARD` (default), `FORWARD`, `FULL`, `NONE`, plus `_TRANSITIVE` variants that check against *all* past versions, not just the last one. `BACKWARD` means "upgrade consumers first."

### In my project: Narad

[Narad](https://github.com/heyyviv/narad) sends plain JSON events, with no registry. How it copes (`internal/consumer/kafka.go`):

| Change | What happens | |
|---|---|---|
| Producer adds a new field | Go's `encoding/json` ignores it | Forward ✅ (field not stored) |
| Old event has no `id` | ID built from topic/partition/offset, so replay is idempotent | Backward ✅ |
| No `ts` / `service` / `level` | Kafka time / `"unknown"` / `"INFO"` | Backward ✅ |
| New business ID | Goes into the `dims` map, no schema change | ✅ |
| Rename `msg` → `message` | Decodes fine, `msg` stored **empty**, no alert | ❌ silent |
| `ts` sent as a number | Decode fails → dead-letter topic | ❌ loud, but caught |

The gap: a rename passes silently. The fix is Protobuf or Avro with a Schema Registry in `BACKWARD` mode, so the bad schema is rejected when it's registered, before any message is sent.
