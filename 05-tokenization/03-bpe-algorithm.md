# Tokenization 3 · The Byte-Pair Encoding (BPE) Algorithm

> **The core of the tokenizer video.** By the end you'll understand exactly how a real tokenizer is trained and used, at the level of the actual code.
> **Previous:** [02-utf8-from-scratch.md](02-utf8-from-scratch.md) · **Next:** [04-regex-tiktoken-special-tokens.md](04-regex-tiktoken-special-tokens.md)
> **Runnable code:** [code/bpe_tokenizer.py](code/bpe_tokenizer.py)

---

## 3.1 The idea, on a tiny example

Byte-Pair Encoding started life as a **data-compression algorithm**. Karpathy uses the example from its Wikipedia page:

```
aaabdaaabac        vocab: {a, b, c, d}   (4 tokens)    length: 11
```

**Step 1.** Most frequent adjacent pair: `aa`. Create a new token `Z = aa` and replace every occurrence (left to right, non-overlapping):

```
aaabdaaabac   →   ZabdZabac        vocab: {a, b, c, d, Z}         length: 9
```

**Step 2.** Count again. Most frequent pair now: `ab`. Create `Y = ab`:

```
ZabdZabac     →   ZYdZYac          vocab: {a, b, c, d, Z, Y}      length: 7
```

**Step 3.** Most frequent pair: `ZY`. Create `X = ZY`. **Merged tokens can be merged again:**

```
ZYdZYac       →   XdXac            vocab: {a, b, c, d, Z, Y, X}   length: 5
```

**Result:** sequence 11 → 5, vocabulary 4 → 7.

> **Each merge adds one token to the vocabulary and shortens the sequence.** That's the vocabulary-size vs sequence-length trade-off, happening in front of you.

**Decoding** reverses it: `X → ZY → aaab`, and so on.

> **Ties:** in step 2, `Za` and `ab` both appear twice — a tie. Wikipedia picks `ab`; the code in [code/bpe_tokenizer.py](code/bpe_tokenizer.py) picks whichever pair it counted first (`Za`). Different tie-break, same algorithm — both end at length 5.

---

## 3.2 Applying it to real text

For an LLM tokenizer, the starting symbols are the **256 possible byte values** (0–255). Text is converted with UTF-8:

```python
ids = list(text.encode("utf-8"))     # a list of integers, each 0–255
```

Starting vocabulary = 256 tokens. **Each new merged token gets the next free ID: 256, 257, 258…** To reach GPT-4's ~100k vocabulary you'd do about 100,000 − 256 merges.

---

## 3.3 Function 1 — counting pairs

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):          # every adjacent pair
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

- `zip(ids, ids[1:])` pairs each element with the next: for `[5, 6, 6, 7]` → `(5,6), (6,6), (6,7)`.
- The dictionary counts each pair.

```python
get_stats([5, 6, 6, 7, 5, 6])
# {(5, 6): 2, (6, 6): 1, (6, 7): 1, (7, 5): 1}
```

`max(stats, key=stats.get)` then returns the most frequent pair → `(5, 6)`.

---

## 3.4 Function 2 — merging a pair

```python
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        # if the pair starts at position i, replace both with the new token
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2                          # skip both elements of the pair
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

Trace `merge([5, 6, 6, 7, 5, 6], pair=(5,6), idx=99)`:

| i | ids[i], ids[i+1] | Is it (5,6)? | Action | newids |
|---|---|---|---|---|
| 0 | 5, 6 | Yes | append 99, skip 2 | [99] |
| 2 | 6, 7 | No | append 6 | [99, 6] |
| 3 | 7, 5 | No | append 7 | [99, 6, 7] |
| 4 | 5, 6 | Yes | append 99, skip 2 | [99, 6, 7, 99] |

Result `[99, 6, 7, 99]` — length 6 → 4. The check `i < len(ids) - 1` stops us reading past the end of the list.

---

## 3.5 Training the tokenizer

```python
vocab_size = 276                  # final vocabulary size we want
num_merges = vocab_size - 256     # = 20 merges
ids = list(text.encode("utf-8"))

merges = {}                       # records every merge: (pair) -> new token id
for i in range(num_merges):
    stats = get_stats(ids)                  # count all pairs
    pair = max(stats, key=stats.get)        # most frequent pair
    idx = 256 + i                           # next free id
    ids = merge(ids, pair, idx)             # replace it everywhere
    merges[pair] = idx                      # remember this merge
```

- **First merges:** in Karpathy's example text, the first merge was `(101, 32)` — that's `e` followed by a space, extremely common since many English words end in "e". Later merges build on earlier ones, just like Z and Y combined into X.
- **Compression:** after only 20 merges the text was about **1.27× shorter**. Real tokenizers do tens of thousands of merges.

> **The `merges` dictionary *is* the trained tokenizer.** A BPE tokenizer is just: the 256 base bytes + an **ordered list of merge rules**. Everything else is derived from it.

---

## 3.6 Decoding: token IDs → text

Build a lookup from each token ID to the bytes it represents:

```python
vocab = {idx: bytes([idx]) for idx in range(256)}   # 0–255: single bytes
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]              # merged token = concatenation
```

This must go **in the order the merges were created** (token 260 may be built from token 258, so 258 must exist first). Python dictionaries keep insertion order, so this works.

```python
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")
```

**Why `errors="replace"`?** Not every byte sequence is valid UTF-8. Decoding token `128` alone: `10000000` starts with `10` → a **continuation byte with no start byte** → invalid. An LLM can output token IDs in any order, including invalid byte sequences, so we replace invalid bytes with `�` instead of crashing. (Full story: [02-utf8-from-scratch.md, Step 10](02-utf8-from-scratch.md#step-10--connecting-it-to-llms).)

---

## 3.7 Encoding: text → token IDs

Start from the text's bytes and **apply the learned merges in the same order they were learned**:

```python
def encode(text):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # among the pairs present, pick the one learned EARLIEST (lowest merge id)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                        # no learned merge applies any more
        idx = merges[pair]
        ids = merge(ids, pair, idx)
    return ids
```

The trickiest line: `pair = min(stats, key=lambda p: merges.get(p, float("inf")))`
- For each pair present, look up its merge ID; unlearned pairs get **infinity**.
- `min` picks the pair **learned first**.
- When **no** pair has a learned merge, every pair maps to infinity, `min` returns an unlearned pair, `pair not in merges` is true → stop.

### Doubt: why earliest-learned first, not "most frequent pair in the new text"?

**The same text must always produce the same tokens — exactly as during the LLM's training.** Example. Suppose training learned, in order:

```
Merge 1 (ID 256):  a + b  → X
Merge 2 (ID 257):  X + c  → Y
Merge 3 (ID 258):  b + c  → Z
```

Encode the new text **"abc"**.

**✅ Correct — earliest-learned first:**
- Pairs present: `(a,b)` [merge 1], `(b,c)` [merge 3]. Earliest = merge 1 → `X c`.
- Now `(X,c)` [merge 2] → `Y`.
- **Result: `[Y]`** — one token, exactly how "abc" was tokenized when the LLM was trained.

**❌ Wrong — most frequent pair in the new text first:**
- Suppose the document is "abc bcbc bcbc", so `(b,c)` is most frequent. Merge it first: `a b c` → `a Z`.
- `(a,Z)` was never learned → stop.
- **Result: `[a, Z]`** — two different tokens for the same "abc".

The LLM always saw "abc" as `Y` during training; now it gets an unfamiliar spelling of a familiar word. Worse, the tokenization of "abc" would **change depending on whatever else is in the document**.

> **The merge order is a fixed recipe.** Repeat the steps in the order originally performed — regardless of what the new text looks like.

### A subtle point (good for interviews)

- `decode(encode(text)) == text` is **always true** — encoding loses nothing.
- `encode(decode(ids)) == ids` is **not always true** — an arbitrary ID sequence may not be the tokenizer's natural encoding. If the model outputs `t` and `he` as separate tokens, decoding gives "the", but re-encoding gives the single token `the`.

---

## 3.8 Doubt: what does "the merges glue bytes back into characters" mean?

It just describes what the **first merges** end up doing for multi-byte characters — there's no separate step.

In byte-level BPE, the Hindi letter `न` starts as **3 separate tokens**:

```
न  →  [224] [164] [168]
```

**The algorithm has no idea these 3 bytes belong together.** The only way `न` becomes one token is if training happens to merge them — which only occurs if those pairs are frequent. On text with plenty of Hindi:

```
Step 1:  (224, 164) is VERY frequent
         (almost every Hindi letter starts with these same two bytes)
         → merge into token A               न → [A] [168]

Step 2:  (A, 168) is frequent, because न is a common letter
         → merge into token B               न → [B]   ← whole letter = one token
```

Those two merges **reassembled** the 3 bytes into one character. Only after that can later merges combine letters into syllables and words.

**If the training text had little Hindi**, those merges may never happen (the merge budget goes to frequent English patterns) → `न` stays as `[224][164][168]` or `[A][168]` — one letter split across several tokens. That's why Hindi costs so many tokens, and why `�` can appear.

**English skips this stage entirely:** `h` is already a complete character in 1 byte, so merges go straight to useful combinations (`t`+`h` → `th`, `th`+`e` → `the`). A big part of why English tokenizes so efficiently.

---

## 3.9 The tokenizer is a separate model, with its own training data

The tokenizer is trained **separately from the LLM**, on its own dataset, then frozen. The LLM only ever sees token IDs.

**What goes into the tokenizer's training data decides which merges it learns:**
- More **code** → more whitespace and code-pattern merges → code takes fewer tokens.
- More **Hindi** → more Hindi merges → Hindi becomes cheaper and fits more in context.

A deliberate design decision — and exactly why GPT-2 was bad at code while GPT-4's tokenizer is much better.

---

## Summary

```
TRAINING:  text → UTF-8 bytes → repeatedly merge most frequent pair → merges table
ENCODING:  text → UTF-8 bytes → apply merges in learned order → token IDs
DECODING:  token IDs → look up bytes → join → UTF-8 decode (replace invalid bytes)
```

---

## Recall tasks (do these on paper)

1. Redo the `aaabdaaabac` example by hand.
2. Trace `merge([1, 2, 1, 2, 3], (1, 2), 256)` step by step. *(Answer: `[256, 256, 3]`)*
3. Explain why encoding applies merges in the order they were learned, using the "abc" example.

## Interview questions for this file

1. How does BPE tokenization work — training and encoding?
2. Why does vocabulary size trade off against sequence length?
3. Why do you sometimes see `�` in LLM output?

Model answers → [interview-question-bank.md](../interview-question-bank.md#5-tokenization)
