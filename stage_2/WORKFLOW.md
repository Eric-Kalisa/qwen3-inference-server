# Execution and ownership workflow

```mermaid
flowchart TD
    A[HTTP completion requests] --> B[Bounded submission queue]
    B --> C[Single GPU owner worker]
    C --> D{Exact cached prompt prefix?}
    D -->|Yes| E[Retain read-only blocks]
    D -->|No| F[Allocate prompt blocks]
    E --> G[Prefill remaining prompt in chunks]
    F --> G
    G --> H[Greedy token, independent samples, or beam candidates]
    H --> I[Fork surviving block tables and retain references]
    I --> J{Append touches a shared partial block?}
    J -->|Yes| K[Copy block before writing; release old reference]
    J -->|No| L[Use private tail or allocate next page]
    K --> M[Pack decode tokens and prompt chunks]
    L --> M
    M --> N[CUDA projections, KV writes, paged attention and MLP]
    N --> O[Wait for stream completion]
    O --> P[Publish cache state and next-token logits]
    P --> Q{Finished or cancelled?}
    Q -->|No| H
    Q -->|Yes| R[Release sequence references]
    R --> S[Return zero-reference blocks to pool]
    S --> C
```

* HTTP threads never mutate the cache or call the model/tokenizer.
* Each live sequence and prefix entry holds one reference to each mapped block.
* Samples share full prefix blocks. Branches writing into a shared partial tail
  first make a private copy of that tail; different branches can then diverge.
* Beam pruning releases discarded candidates only after surviving candidates
  have acquired their references. Ordinary single-child steps reuse the parent.
* A completed GPU call is the boundary at which mappings can safely change.
* When memory fills, evict prefix-registry references first, then reset younger
  request groups for later recomputation. A still-live reference prevents reuse.
* Every scheduling iteration can admit new requests and remove finished ones.
