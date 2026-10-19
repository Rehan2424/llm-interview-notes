# Tokenization 4 · Regex Pre-Tokenization, tiktoken, and Special Tokens

> **What this adds:** the extra pieces that real tokenizers (GPT-2, GPT-4) put on top of the plain BPE you built in [03-bpe-algorithm.md](03-bpe-algorithm.md) — and why each exists.
> **Previous:** [03-bpe-algorithm.md](03-bpe-algorithm.md) · **Next:** [05-sentencepiece-vocab-size-strange-behaviors.md](05-sentencepiece-vocab-size-strange-behaviors.md)

---

## 4.1 The problem with plain BPE

Plain BPE merges whatever pair is most frequent, **with no regard for what kind of characters they are.** The GPT-2 paper pointed out the problem: the word "dog" appears in many forms —

```
dog.    dog!    dog?    dog,    dog's
```

Plain BPE might create **separate tokens for each**: `dog.`, `dog!`, `dog?`… That's bad because:
1. It **wastes vocabulary slots** on near-duplicates of the same word.
2. It **mixes meaning with punctuation** — the model must learn separately that `dog.` and `dog!` are both about dogs.

Same problem for letters glued to numbers (`world123`) and spaces glued to punctuation.

---

## 4.2 The fix: split into chunks first (pre-tokenization)

Before BPE, GPT-2 **splits the text into chunks** with a pattern. Then:
- BPE runs **inside each chunk separately**.
- **Merges never cross chunk boundaries.**

So if letters and punctuation always land in different chunks, they can **never** merge into one token.

```
"Hello world123!!"
        │  split into chunks first
        ▼
["Hello", " world", "123", "!!"]
        │  BPE within each chunk
        ▼
token IDs   (no token can ever contain both "world" and "123")
```

This splitting step is called **pre-tokenization**. The pattern is a **regex**.

### Doubt: so how does a whole sentence with punctuation get tokenized?

Chunks are **not** thrown away or kept apart in the output. Each chunk is tokenized separately, then **all tokens are put back in their original order as one sequence.** "Not merged across chunks" only means **no single token can span two chunks.**

```
Sentence:  "Hello, world! It's 2024."

Step 1 — regex chunks (GPT-2 rules):
   ["Hello", ",", " world", "!", " It", "'s", " 2024", "."]

Step 2 — BPE inside each chunk (common chunks are already single tokens)

Step 3 — join in order:
   [Hello] [,] [ world] [!] [ It] ['s] [ 2024] [.]   → one sequence → the model
```

The model still sees the comma right after "Hello". Nothing is lost. You just never get a token like `world!`.

**Punctuation can still merge with other punctuation inside its own chunk** — `"..."` or `"!!!"` can become a single token if common. That's fine: still one kind of thing.

### Doubt: what's the essence of chunking?

Chunking **builds walls that BPE can't merge across.** It achieves three things:

1. **Different kinds of text stay separate** — letters, numbers, punctuation, whitespace never merge → no `dog.`, `dog!`, `world123` tokens → vocabulary spent on meaningful units.
2. **The same word gets the same token everywhere** — `" dog"` is always its own chunk, so it's the same token whether followed by `.`, `!`, `'s`, or nothing. The model learns "dog" once, not five times.
3. **A chunk is the largest a token can ever be** — merges happen only inside chunks, so the regex sets an upper limit on what a token can contain.

> **In one sentence:** chunking makes tokens follow natural boundaries (words, numbers, punctuation), so the vocabulary is spent on meaningful units instead of accidental combinations.

---

## 4.3 Reading GPT-2's regex

A **regex (regular expression)** is a small pattern language for finding pieces of text. You need only a few symbols:

| Symbol | Meaning |
|---|---|
| `\p{L}` | any **letter**, in any language |
| `\p{N}` | any **number** (digit) |
| `\s` | any **whitespace** (space, tab, newline) |
| `+` | "one or more" of the previous thing |
| `?` | "optional" (zero or one) |
| `[^...]` | "anything **except** these" |
| `\|` | "or" — try each alternative |

GPT-2's pattern:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

| Piece | Meaning | Example match |
|---|---|---|
| `'s\|'t\|'re\|'ve\|'m\|'ll\|'d` | Common English contractions as their own chunks | `'s` in "how's" |
| ` ?\p{L}+` | Optional space, then a run of letters (a word) | `" world"` |
| ` ?\p{N}+` | Optional space, then a run of digits | `"123"` |
| ` ?[^\s\p{L}\p{N}]+` | Optional space, then a run of punctuation/symbols | `"!!!?"` |
| `\s+(?!\S)` | A run of spaces, **except the last one** if a word follows | (see below) |
| `\s+` | Any leftover whitespace | |

**Worked example (Karpathy's) — verified output:**

```python
import regex as re      # the "regex" package; Python's built-in "re" doesn't support \p{L}
pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

re.findall(pattern, "Hello've world123 how's are you!!!?")
# ['Hello', "'ve", ' world', '123', ' how', "'s", ' are', ' you', '!!!?']
```

Notice:
- **Spaces attach to the front of the next word** (`' world'`, `' how'`) — this explains why `" world"` is one token.
- Letters, numbers and punctuation are always in **separate chunks**.
- Contractions like `'s` are split off from their word.

**The tricky piece `\s+(?!\S)`:** `(?!\S)` means "only if the next character is *not* a non-space". For `"   world"` (three spaces) it grabs only the first **two** spaces, leaving the **last** space to attach to the word:

```python
re.findall(pattern, "   world")    # ['  ', ' world']
```

The point: keep `" world"` consistent — a word after a space always gets the same token, no matter how many spaces came before.

**How the regex picks at each position:** it tries the alternatives **left to right and takes the first that matches** (not the longest).

---

## 4.4 Flaws in GPT-2's pattern

### Doubt: how are `how's` and `HOW'S` handled differently?

Regex matching is **case-sensitive** by default — `'s` matches only a lowercase `s`.

**Lowercase `" how's"`:**
- `" how"` matches ` ?\p{L}+` (space + letters), stopping at the apostrophe.
- At `'s`, the first alternative `'s` matches → one chunk.

```
" how's"  →  [" how", "'s"]          ← 2 chunks
```

**Uppercase `" HOW'S"`:**
- `" HOW"` matches as letters.
- At `'S`, the contraction `'s` **doesn't match** (it expects lowercase). The regex moves on: letters? no (`'` isn't a letter). numbers? no. punctuation? **yes** → `'` alone becomes a chunk.
- Then `S` matches as letters → its own chunk.

```
" HOW'S"  →  [" HOW", "'", "S"]      ← 3 chunks, split differently
```

Same meaning, different chunks → different tokens. The model must learn separately that they mean the same. (Verified by running the pattern.)

### Other flaws
- **Contractions are English-only** — other languages' apostrophe patterns aren't handled.
- **Whitespace in code:** in practice GPT-2 kept runs of spaces as individual tokens, so each indentation space cost a token. OpenAI only released GPT-2's tokenizer **usage** code, not the code used to **train** it, so some rules can only be inferred from its behavior.

---

## 4.5 What GPT-4's tokenizer (`cl100k_base`) changed

| Change | Effect | Verified example |
|---|---|---|
| **Case-insensitive contractions** | `HOW'S` handled like `how's` | `" HOW'S"` → `[' HOW', "'S"]` |
| **Numbers split into at most 3 digits** (`\p{N}{1,3}`) | No token ever has more than 3 digits; more consistent numbers; partly addresses arithmetic | `"1234567"` → `'123', '456', '7'` |
| **Space before a number is its own chunk** | Numbers don't absorb a leading space | `"Total 1234567"` → `['Total', ' ', '123', '456', '7']` |
| **Better whitespace handling** | Runs of spaces can be merged → code takes far fewer tokens | |
| **Vocabulary ~doubled** | ~50k → ~100k tokens | |

(GPT-4o's newer tokenizer, `o200k_base`, goes up to ~200k tokens.)

---

## 4.6 tiktoken — OpenAI's tokenizer library

`tiktoken` is OpenAI's official library for **using** their tokenizers. It's fast, and it only encodes/decodes — it **can't train** new tokenizers.

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")          # GPT-2's tokenizer
enc = tiktoken.get_encoding("cl100k_base")   # GPT-4's tokenizer

ids = enc.encode("hello world")
text = enc.decode(ids)
```

**Day-to-day use in AI jobs:** **counting tokens** before calling an API — to estimate cost, or check whether a document fits the context window. Never estimate token counts from word counts.

---

## 4.7 GPT-2's released tokenizer = exactly what you built

OpenAI released GPT-2's tokenizer as two files:

| GPT-2 file | What it is | Your BPE equivalent |
|---|---|---|
| `vocab.bpe` | The list of merges | `merges` |
| `encoder.json` | Token ID ↔ bytes mapping | `vocab` |

(One extra layer in their code maps bytes to printable characters — a cosmetic detail.) Otherwise: **the BPE algorithm + the regex split.** You understand how a real production tokenizer works.

---

## 4.8 Special tokens

**Special tokens** are tokens that **don't come from BPE merges.** They're **added by hand** — meaning the model's designers decide they exist and assign their IDs themselves; no algorithm learns them. They mark **structure** in the data.

### The classic one: `<|endoftext|>`

```
GPT-2 vocabulary:  256 (bytes) + 50,000 (merges) + 1 (special token) = 50,257
```

That one special token is `<|endoftext|>`, **ID 50256** in GPT-2 (**100257** in GPT-4's `cl100k_base`). During training, documents are joined into one long stream with `<|endoftext|>` between them:

```
...end of a Wikipedia article.<|endoftext|>Start of a cooking blog...
```

The model learns that **whatever comes after `<|endoftext|>` is unrelated to what came before** — "reset, new document".

**How they're handled:** the tokenizer **first scans the text for the exact special-token strings**, cuts them out, and inserts their IDs directly. Everything else goes through the normal regex split + BPE.

### Special tokens in chat models

The chat template from [03-post-training-sft](../03-post-training-sft/README.md#32-how-a-conversation-becomes-tokens):

```
<|im_start|>user<|im_sep|>What is 2+2?<|im_end|>
```

`<|im_start|>`, `<|im_sep|>`, `<|im_end|>` are special tokens marking who speaks and where messages end. GPT-4's tokenizer also has **fill-in-the-middle** tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`), used to train code models to fill a gap in the middle of a file, not just continue from the end.

### The security angle: what if user text contains the literal string `<|endoftext|>`?

If the tokenizer turned it into the **real** special token, a user could inject fake structure (fake end-of-document markers, fake chat turns) to manipulate the model. It's the same idea as **SQL injection** (Structured Query Language injection), where user input gets treated as a command instead of data.

So `tiktoken` **refuses by default** (behavior confirmed from tiktoken's source code):

```python
enc = tiktoken.get_encoding("cl100k_base")
text = "hello <|endoftext|>"

enc.encode(text)
# ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'...

# Option A — treat it as the REAL special token (one control-token ID):
enc.encode(text, allowed_special={"<|endoftext|>"})

# Option B — treat it as ORDINARY text (several normal tokens, harmless):
enc.encode(text, disallowed_special=())
```

**Correct practice:**
- **User-provided text:** always encode as ordinary text (Option B) — it can never create control tokens.
- **Special tokens:** only **your own code** inserts them, e.g. when building the chat template.

This also explains a strange behavior from the list: models acting oddly when they see `<|endoftext|>` — if it gets parsed as the real token, the model thinks the document ended.

---

## Summary

```
Real tokenizer = regex split into chunks (pre-tokenization)
               + BPE within each chunk (merges never cross chunks)
               + special tokens inserted directly (bypassing BPE)
```

| | GPT-2 | GPT-4 (`cl100k_base`) |
|---|---|---|
| Vocabulary | ~50k (50,257) | ~100k (100,277) |
| Contractions | Case-sensitive (buggy) | Case-insensitive |
| Numbers | Any length | At most 3 digits per chunk |
| Spaces in code | Mostly 1 token per space | Merged → far fewer tokens |
| `<\|endoftext\|>` ID | 50256 | 100257 |

---

## What to retain about regex for interviews

**Don't memorize the pattern** — no interviewer asks you to write it. Remember these four points:

1. **The name:** splitting text before BPE is called **pre-tokenization**. (Hugging Face's tokenizer library uses exactly this term.)
2. **What it does:** splits text into chunks by category (words, numbers, punctuation, whitespace); merges never cross chunks.
3. **Why:** avoid wasting vocabulary on `dog.`, `dog!`, `dog?`, and keep tokens meaningful and consistent.
4. **GPT-4's improvements:** numbers at most 3 digits, case-insensitive contractions, merged whitespace for code.

---

## Recall tasks

1. Explain why BPE needs the regex split, using the "dog." example.
2. Split `"I'm testing 4567 tokens!!"` by hand with GPT-2's rules. Then: what changes for the number under GPT-4's rules?
   *(GPT-2: `['I', "'m", ' testing', ' 4567', ' tokens', '!!']` · GPT-4: `['I', "'m", ' testing', ' ', '456', '7', ' tokens', '!!']`)*
3. Explain what `<|endoftext|>` does during training.

## Interview questions for this file

1. Why do GPT tokenizers split text with a regex before applying BPE?
2. What are special tokens? Give an example.
3. How would you check whether a document fits in a model's context window?
4. A user's message contains `<|endoftext|>`. What should your system do?

Model answers → [interview-question-bank.md](../interview-question-bank.md#5-tokenization)
