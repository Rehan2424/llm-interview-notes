# Tokenization 1 · Why Tokenization Matters, and How Text Becomes Numbers

> **Source:** Karpathy, *"Let's build the GPT Tokenizer"*, first ~25 minutes, plus the doubts we cleared up in our sessions.
> **Next:** [02-utf8-from-scratch.md](02-utf8-from-scratch.md)

---

## 1.1 Where tokenization sits

A transformer can't read text. It only processes numbers. So before anything else:

```
"hello world"  →  [31373, 995]  →  embedding lookup  →  vectors  →  transformer
     text           token IDs       (one row per ID)
```

You know the embedding part already: an **embedding table** has **one row per token** in the vocabulary; each token ID selects its row as the input vector. Those rows are **learned** during training.

Two numbers define the model's input side:

- **Vocabulary size** — how many distinct tokens exist (= how many rows in the embedding table).
- **Sequence length** — how many tokens a piece of text becomes.

**Tokenization decides both.** It is a **completely separate stage** from the neural network: built *before* the LLM is trained, with its own algorithm and its own training data, then **frozen**.

In Karpathy's earlier *"Let's build GPT"* video he used the simplest tokenizer possible — **one token per character**. Shakespeare's text has only 65 distinct characters, so vocabulary = 65. Real LLMs instead use **chunks of characters**, built with **Byte-Pair Encoding** ([03-bpe-algorithm.md](03-bpe-algorithm.md)).

---

## 1.2 "Tokenization is at the heart of much weirdness"

Karpathy opens with a list of strange LLM behaviors — all caused by tokenization:

1. Can't spell words reliably or count letters in them.
2. Struggles with simple string tasks, like reversing a word.
3. Worse at non-English languages.
4. Bad at simple arithmetic.
5. GPT-2 was unusually bad at Python code.
6. Strange behavior when it sees the text `<|endoftext|>`.
7. Warnings about "trailing whitespace" in prompts.
8. The word "SolidGoldMagikarp" made models behave bizarrely.
9. YAML works better than JSON for feeding structured data to LLMs.

**Every item is explained** in [05-sentencepiece-vocab-size-strange-behaviors.md](05-sentencepiece-vocab-size-strange-behaviors.md#54-the-strange-behaviors-explained).

---

## 1.3 What the live tokenizer demo shows

Karpathy pastes text into a website called **Tiktokenizer**, which colors each token. What to notice, and **why it matters**:

### Observation 1 — spaces attach to the *following* word
`"hello world"` → `hello` + `" world"`. So `world` and `" world"` (with a space) are **different tokens with different IDs**.

### Observation 2 — the same word can become many different tokens
`egg`, `" egg"`, `Egg`, `EGG` can all be different tokens depending on capitalization and position. To the model these start as **unrelated IDs**; it must **learn from data** that they mean the same thing.

### Observation 3 — numbers are split inconsistently
In his demo, `127` became one token while `677` became `6` + `77`. How a number is chunked depends on **how often those digit groups appeared** in the tokenizer's training data — not on place value. The model never sees digits neatly arranged as ones, tens, hundreds. **A major reason LLMs are bad at arithmetic.**

### Observation 4 — non-English text uses many more tokens
The same sentence in Korean, Japanese or Hindi produces **far more tokens** than in English (why: [section 1.7](#17-doubt-why-does-non-english-text-cost-more-tokens)). Consequences:
- fills the **context window** faster;
- less meaning per token → **worse performance**;
- **costs more**, since APIs (Application Programming Interfaces) charge per token.

### Observation 5 — code and whitespace
In **GPT-2's** tokenizer, **each space of Python indentation was its own token**. A deeply indented block wasted dozens of tokens on spaces alone → GPT-2 handled code poorly. **GPT-4's** tokenizer merges runs of spaces, so the same code takes far fewer tokens. A deliberate design improvement.

### Observation 6 — vocabulary size grew
| Tokenizer | Vocabulary size |
|---|---|
| GPT-2 | 50,257 |
| GPT-4 (`cl100k_base`) | 100,277 |

Bigger vocabulary → bigger chunks → the same text becomes roughly **half as many tokens** → the model sees about **twice as much text** in the same context window. But bigger vocabularies cost something too ([section 5.2](05-sentencepiece-vocab-size-strange-behaviors.md#52-where-vocabulary-size-shows-up-inside-the-model)).

---

## 1.4 Text in Python: Unicode code points

In Python, a string is a sequence of **Unicode code points**. **Unicode** is a worldwide standard that gives a unique integer to every character in every writing system, plus emoji and symbols. `ord()` returns it:

```python
ord("h")    # 104
ord("न")    # 2344     (Hindi)
ord("안")   # 50504    (Korean)
ord("👋")   # 128075   (emoji)
```

**Why not use code points directly as tokens?**
1. **Vocabulary would be huge** — Unicode defines ~150,000 characters → 150,000 embedding rows, most for characters almost never seen (so they'd barely be trained).
2. **It isn't stable** — Unicode keeps adding characters; you can't keep changing a trained model's vocabulary.

So we need another layer: **encodings** → turning code points into **bytes**. That's the whole of [02-utf8-from-scratch.md](02-utf8-from-scratch.md). The short version: in **UTF-8**, English characters take 1 byte, Hindi/Korean 3 bytes, emoji 4 bytes:

```python
text = "hello 안녕 👋"            # 10 characters
list(text.encode("utf-8"))        # 17 bytes:
# [104, 101, 108, 108, 111, 32,          ← "hello " : 1 byte each
#  236, 149, 136,                        ← "안"     : 3 bytes
#  235, 133, 149,                        ← "녕"     : 3 bytes
#  32,                                   ← " "      : 1 byte
#  240, 159, 145, 139]                   ← "👋"     : 4 bytes
```

**Any text in any language becomes a sequence of numbers from 0 to 255.**

---

## 1.5 Why not just use raw bytes as tokens?

Bytes look perfect at first: vocabulary of only **256**, and they handle any language or emoji with **no unknown characters**. But:

- **Sequences become far too long** — every character is at least one token; an essay becomes tens of thousands of tokens.
- The **context window is limited**, so the model could only see a little text at a time.
- **Attention cost grows quadratically with sequence length** ([section 1.10](#110-doubt-why-does-attention-cost-grow-quadratically)) — long sequences are expensive.

Research into "tokenizer-free" byte-level models exists, but it hasn't been proven at large scale — every major LLM still uses a tokenizer.

---

## 1.6 The key takeaway

Two extremes, neither works:

```
Raw bytes:          vocab = 256        →  sequences way too LONG
Whole code points:  vocab ≈ 150,000    →  vocabulary too BIG and unstable
```

**What we want:** start from **UTF-8 bytes** (so any text, any language, is supported) and then **compress** them by **merging frequently occurring byte sequences into new tokens** → a vocabulary in the tens of thousands and much shorter sequences.

**That compression algorithm is Byte-Pair Encoding** → [03-bpe-algorithm.md](03-bpe-algorithm.md).

---

# Doubts we cleared up

## 1.7 Doubt: why does non-English text cost more tokens?

**How a tokenizer is trained (preview of BPE):** start from bytes, repeatedly count adjacent pairs in a big training text, and **merge the most frequent pair** into a new token. So **only sequences that are frequent in the tokenizer's training text become big tokens.**

- If that text is 90% English, sequences like `" the"`, `" world"`, `"ing"` appear millions of times → they become **single tokens**.
- Hindi or Korean sequences appear far less → **few merges** → they stay broken into **small pieces**.
- **Second effect:** in UTF-8 each Hindi/Korean character **already takes 3 bytes**, so these scripts start out longer before any merging.

```
English: " world"     → 1 token
Hindi:   " दुनिया"    → many tokens (few merges learned + 3 bytes per character)
```

**Why that fills the context window faster:** the context window is measured in **tokens**, not words. If a sentence takes 10 tokens in English and 30 in Hindi, the model fits **3× less Hindi content** in the same window.

**What "compute is per token" means:** every token passes through **every layer**. The same meaning in 3× the tokens costs about 3× the compute — or more, because of attention. That's why APIs price per token: **tokens are the real unit of work.**

**Why the model "gets less meaning per token" and performs worse:** when `" world"` is one token, its embedding directly carries the word's meaning. When a word is split into 4 fragments, each fragment means almost nothing alone — the model must spend early layers **reassembling the word** before it can reason about it. *Honest note:* the bigger reason models do worse in other languages is simply **less training data** in those languages; inefficient tokenization is a second, related reason (both come from English-heavy data).

## 1.8 Doubt: what's the difference between Unicode and embeddings? Both turn text into numbers.

| | Unicode code point | Embedding |
|---|---|---|
| **What it is** | A **single integer** | A **vector of many numbers** (768 in GPT-2 small; thousands in large models) |
| **Who decides it** | A standards committee, fixed forever | **Learned** during model training |
| **Carries meaning?** | **No** — just an ID label | **Yes** — similar meanings get nearby vectors |
| **Purpose** | Storing and exchanging text between computers | Giving the neural network something meaningful to compute with |

**Analogy:** a code point is a **roll number**. `h` = 104 and `i` = 105 being adjacent says nothing about the letters being related — just as roll numbers 101 and 102 say nothing about two students being similar. An embedding is a detailed **profile** of you — skills, interests — where similar profiles really are similar.

**Token IDs are also just labels (roll numbers).** Meaning only appears at the embedding step:

```
text  →  UTF-8 bytes  →  token IDs   →  embedding vectors  →  transformer
          (encoding)     (tokenizer)     (learned lookup)
         no meaning      no meaning      MEANING lives here
```

## 1.9 Doubt: who learns what? ("encoder", "tokenizer", "embedder")

First, **"encoder" means two completely different things** in ML:
- The **tokenizer's `encode()` function** (text → token IDs) — just a procedure; learns nothing. GPT-2's file `encoder.json` is named after this; it's simply the tokenizer's vocabulary file.
- A **transformer encoder** — the neural network block from the original transformer and BERT (Bidirectional Encoder Representations from Transformers). GPT models are **decoder-only**; they don't have one.

| Stage | Learns? | How | What it ends up knowing |
|---|---|---|---|
| UTF-8 encoding | No | Fixed rule | Nothing — converts characters to bytes |
| **Tokenizer training** | **Yes** — by **counting**, not gradient descent | Counts pair frequencies in its own training text | **The merge rules**: which pairs merge, in what order |
| Token IDs | No | Handed out in order: 256, 257, 258... | Nothing — just labels |
| `encode()` / `decode()` | No | Apply the merge rules | Nothing |
| **Embedding table** | **Yes** — by **gradient descent** | Trained with the whole LLM | A meaningful vector per token ID |
| **Transformer layers** | **Yes** — by **gradient descent** | Same training | How to combine tokens using context |

> **The tokenizer decides *what the pieces are*** (trained once, by counting, before the LLM exists, then frozen).
> **The embedding table learns *what each piece means*** (trained by backpropagation as part of the LLM).
> *The tokenizer hands out roll numbers; the LLM builds a profile for each one.*

## 1.10 Doubt: why does attention cost grow quadratically?

Every token's **query** is compared with every token's **key**. For a sequence of *n* tokens, computing QKᵀ produces an **n × n matrix of attention scores**; each score is a dot product of *d* numbers → cost ≈ **n² × d**.

```
n = 1,000 tokens   →  1,000 × 1,000   =   1 million scores
n = 2,000 tokens   →  2,000 × 2,000   =   4 million scores   (2× length = 4× cost)
n = 10,000 tokens  →                     100 million scores  (10× length = 100× cost)
```

- This happens in **every head of every layer**, and the n × n matrix also takes **memory**.
- Causal (masked) attention masks out about half the matrix, but n²/2 is **still quadratic**.
- Compare the **feed-forward layers**: they process each token independently → cost grows **linearly** with n. For short sequences feed-forward dominates; for long sequences attention's n² takes over.

**Why it matters for tokenization:** halving the number of tokens cuts attention cost by about **4×**.

---

## Interview questions for this file

1. Why are LLMs bad at arithmetic?
2. Why do non-English users pay more for the same API usage?
3. Why do LLMs use byte-level tokenization with merges, instead of characters or raw bytes?
4. What's the difference between a token ID and an embedding?
5. Why does attention cost grow quadratically with sequence length?

Model answers → [interview-question-bank.md](../interview-question-bank.md#5-tokenization)
