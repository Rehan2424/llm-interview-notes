# Tokenization 5 · SentencePiece, Vocabulary Size, Adding Tokens, and the Strange Behaviors Explained

> **The final tokenization lesson.** By the end, you can explain every strange behavior from [file 01](01-why-tokenization-and-text-as-numbers.md#12-tokenization-is-at-the-heart-of-much-weirdness).
> **Previous:** [04-regex-tiktoken-special-tokens.md](04-regex-tiktoken-special-tokens.md)

---

## 5.1 SentencePiece — the other major tokenizer library

`tiktoken` (OpenAI) only **uses** tokenizers. **SentencePiece** (Google) can **train and use** them. It was used by **Llama 1 and 2, Mistral, T5** and many others — you'll meet it often.

### The key difference: what the starting pieces are

Both use BPE — repeatedly merge the most frequent pair. **The only difference is the starting pieces**, the smallest building blocks before any merging:

| | Byte-level BPE (GPT, tiktoken) | SentencePiece |
|---|---|---|
| Starting pieces | **UTF-8 bytes** (256 base tokens) | **Unicode characters** (code points) |
| What gets merged | Byte pairs | Character pairs |

### Making it crystal clear: the Hindi word `नम`

Two characters, `न` and `म`. In UTF-8 each is 3 bytes:

```
न  →  224, 164, 168
म  →  224, 164, 174
```

**Byte-level BPE** starts from **6 pieces** (one per byte):

```
[224] [164] [168] [224] [164] [174]
```

The merges first have to **reassemble bytes into characters** (224 + 164 → new token, then + 168 → `न`), and only then glue characters into bigger pieces. With little Hindi in training, it may never finish → one character split across tokens → possible `�`. *(Details: [03-bpe-algorithm.md §3.8](03-bpe-algorithm.md#38-doubt-what-does-the-merges-glue-bytes-back-into-characters-mean).)*

**SentencePiece** starts from **2 pieces** (one per character):

```
[न] [म]
```

Characters are already whole. Merges go straight to combining characters (`न` + `म` → `नम`). **A character is never split.**

### The Lego analogy

- **Byte-level** gives you a box of **tiny bricks** — only 256 kinds. You must first assemble them into letters, then letters into words. Small box, but you can build *anything*.
- **SentencePiece** gives you a box of **ready-made letter pieces** — thousands of kinds. You start from letters. But **the box only includes common letters**.

### What happens to rare characters in SentencePiece?

SentencePiece keeps only the characters covering almost all of its training text (about 99.95%). For a character outside that set:

- **Without byte fallback:** it becomes `<unk>` (the "unknown" token) → **information lost**. Bad for an LLM.
- **With byte fallback** (Llama 2 uses it): the character falls back to tiny bricks — its **UTF-8 bytes**, each as a token:

```
"안" (rare in training) → <0xEC> <0x95> <0x88>      (= 236, 149, 136 in hexadecimal)
```

| | Byte-level (GPT, tiktoken) | Character-level (SentencePiece) |
|---|---|---|
| Starting pieces | 256 bytes | Thousands of characters |
| Can one character be split across tokens? | **Yes** (can produce `�`) | **No** (except via byte fallback) |
| Unknown characters? | **Never** — 256 bytes cover everything | Rare ones need **byte fallback**, else `<unk>` |

> **One line:** *both merge frequent pairs. Byte-level starts from bytes, so it must rebuild characters first. SentencePiece starts from whole characters, so it never splits one — but needs byte fallback for characters missing from its vocabulary.*

### Other SentencePiece details

1. **Treats text as one raw stream** and replaces every space with the visible symbol `▁`. So `▁hello` means `" hello"`. It doesn't rely on spaces to find words → works the same for languages without spaces (Chinese, Japanese).
2. **Adds a space at the start** of the text by default, so `hello` at the start of a sentence gets the same token as `▁hello` mid-sentence (same consistency goal as regex chunking).
3. **Decoding:** join the pieces and turn `▁` back into spaces. Nothing is lost.

Karpathy's opinion: SentencePiece works, but has many confusing settings and historical baggage. Notably, **Llama 3 moved to a tiktoken-style byte-level tokenizer** with ~128k tokens.

### Tokenizer algorithms worth knowing ("BPE vs WordPiece" is a common question)

| Algorithm | How it builds the vocabulary | Used by |
|---|---|---|
| **BPE (Byte-Pair Encoding)** | Merges the **most frequent** pair, repeatedly | GPT models, Llama |
| **WordPiece** | Merges the pair with the highest **score** = frequency(ab) ÷ (frequency(a) × frequency(b)) — favors pairs that occur together **more than chance would predict**, not just common ones | BERT (Bidirectional Encoder Representations from Transformers) |
| **Unigram** | Starts with a **large** vocabulary and repeatedly **removes** the tokens that matter least | T5 (via SentencePiece) |

### What to retain about SentencePiece for interviews

- **Google's library for training and using tokenizers** (Llama 1/2, T5, many others).
- **Merges characters, not bytes**; uses **byte fallback** so rare characters don't become `<unk>`.
- **Represents spaces as `▁`** → language-independent.
- Llama 3 switched to a tiktoken-style byte-level tokenizer.

---

## 5.2 Where vocabulary size shows up inside the model

You know the transformer, so place vocabulary size inside it. It appears in **exactly two places**:

```
token IDs → [EMBEDDING TABLE: vocab_size × d_model] → transformer blocks
          → [OUTPUT LAYER: d_model × vocab_size] → logits → softmax → next-token probabilities
```

(`d_model` = size of each token's vector, e.g. 768 in GPT-2 small.)

**What happens as vocabulary size grows:**

| Effect | Why |
|---|---|
| ✅ **Shorter sequences** | More merges → bigger chunks |
| ✅ More text fits in the context window; attention cheaper | Fewer tokens (attention is quadratic in length) |
| ❌ **More parameters** | Both the embedding table and the output layer grow |
| ❌ **More compute per token** at the output | Softmax runs over every vocabulary entry, for every token |
| ❌ **Undertrained tokens** | Each token appears more rarely in training → its embedding gets fewer updates |
| ❌ **Too much meaning packed into one token** | Each token gets one fixed pass of computation ("models need tokens to think") |

So in practice vocabularies land in the **tens of thousands to ~200k**: GPT-2 ~50k, GPT-4 ~100k, Llama 3 ~128k, GPT-4o ~200k.

---

## 5.3 Adding new tokens to an existing model ("model surgery")

Common in real work — e.g. adding chat tokens like `<|im_start|>` when fine-tuning a base model into a chat model, or tokens for tool calls.

1. Add the new token to the tokenizer with the next free ID.
2. **Add a new row to the embedding table** (initialized to small random values, or the average of existing embeddings).
3. **Add a new row to the output layer**, so the model can *predict* the new token.
4. **Train.** Often you **freeze the rest of the model** and train only the new rows — cheap, and doesn't disturb what the model already knows.

It's called model surgery because you're **resizing two layers of an already trained network.**

**Beyond text:** images and audio can also be turned into tokens (e.g. an image split into patches, each patch → a token). The same transformer processes them — that's the basis of multimodal models. Karpathy's point: **you usually don't change the transformer; you change what gets tokenized.**

---

## 5.4 The strange behaviors explained

Back to the list from [file 01](01-why-tokenization-and-text-as-numbers.md#12-tokenization-is-at-the-heart-of-much-weirdness):

**1. Can't spell or count letters.** Some tokens are long. In Karpathy's demo, `.DefaultCellStyle` was a **single token** in GPT-4's tokenizer; asked how many letter "l"s it contains, GPT-4 got it wrong. The model never sees the letters inside a token — only one ID.

**2. Can't reverse a string.** Same cause. Karpathy's workaround: first ask the model to write the word **letter by letter, separated by spaces**, then reverse it. Works, because each letter is now its own token.

**3. Worse at non-English.** Less of those languages in the tokenizer's training data → fewer merges; plus UTF-8 uses 3 bytes per character for many scripts → more tokens per meaning. And the LLM itself trained on less data in those languages.

**4. Bad at arithmetic.** Numbers are chunked inconsistently (`127` one token, `677` → `6` + `77`), so digits don't line up by place value. Fixes: GPT-4's 3-digit maximum, Llama 2 splitting every digit into its own token — and better still, using code or a calculator tool.

**5. GPT-2 bad at Python.** Every indentation space was its own token → huge share of context wasted on whitespace. GPT-4 merges runs of spaces.

**6. Strange behavior around `<|endoftext|>`.** If parsed as the **real** special token, the model thinks the document ended → it may stop or behave oddly. That's why tiktoken refuses it in ordinary text by default.

**7. Warnings about trailing whitespace.** A prompt ending with a space, e.g. `"Here is a tagline for an ice cream shop: "`:
- In training data, a space is almost always **attached to the next word** (`" Oh"`, `" Delicious"`).
- Here the space becomes **its own separate token** at the end of the prompt.
- The model has rarely seen a lone space token followed by a word → **out-of-distribution** → output quality drops.
- **General lesson:** prompts that end **partway through a typical token** confuse the model. (Another example: ending a prompt with `.DefaultCellSty`, cutting the usual single token in half.)

**8. SolidGoldMagikarp.** ⭐ Great interview story.
- "SolidGoldMagikarp" was a **Reddit username** of a user who posted constantly in one subreddit.
- That Reddit data was in the **tokenizer's** training data → the name was frequent enough to get **its own token**.
- That data was **not** in the **LLM's** training data → the LLM never saw this token during training.
- So its embedding row **was never updated** — it stayed at its random starting values.
- Using the token at inference feeds the model a meaningless vector → behavior goes haywire (evasive answers, insults, nonsense).
- **Lesson:** the tokenizer and the LLM are trained on **different data**; a mismatch creates **undertrained ("glitch") tokens**. Fix: train the tokenizer on the same kind of data as the LLM, and check for undertrained tokens before release.

**9. YAML better than JSON.** JSON (JavaScript Object Notation) is full of quotes, braces and commas — each costs tokens. YAML (YAML Ain't Markup Language) expresses the same data with indentation and far less punctuation → **noticeably fewer tokens** → cheaper and fits more in context. **Token efficiency is a real cost lever in production.**

---

## 5.5 Karpathy's practical advice, and what it means for you

- **Reuse an existing tokenizer** if you can (e.g. GPT-4's via tiktoken). Training your own is rarely necessary.
- If you must train one, SentencePiece BPE with byte fallback is the common choice — check its settings carefully.

**As an intern, the practical skills that matter:**
- **Count tokens** (with the model's own tokenizer) to estimate cost and check context limits.
- Know that **non-English text, JSON, and code** can be token-expensive.
- Avoid **trailing whitespace** in prompts.
- Know that **numbers and spelling** are weak spots — tools or code fix them.

---

## The full chain — you now understand all of it

```
text → UTF-8 bytes → regex chunks → BPE merges → token IDs
     → embeddings → transformer → output layer → next-token probabilities
```

---

## Recall tasks

1. How does SentencePiece differ from tiktoken-style tokenizers? What is byte fallback?
2. The two places vocabulary size appears in the model, and 3 effects of increasing it.
3. SolidGoldMagikarp, step by step.
4. Why trailing whitespace hurts output.

## Interview questions for this file

1. What happens if you increase vocabulary size?
2. How would you add a new special token to a pretrained model?
3. What are undertrained / glitch tokens?
4. Why would you send YAML instead of JSON to an LLM?
5. BPE vs WordPiece vs Unigram — what's the difference?

Model answers → [interview-question-bank.md](../interview-question-bank.md#5-tokenization)
