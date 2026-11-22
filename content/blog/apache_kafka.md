+++
title = "Apache Kafka"
slug = "apache-kafka"
date = "2026-11-21T17:22:50+05:30"
description = "Kafka from the ground up: why a log, why it is fast on plain disks, how producers pick partitions, how consumers pull, and what offsets, commits and lag really mean."
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
