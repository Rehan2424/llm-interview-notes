# 05 · Tokenization

> **Source:** Andrej Karpathy, *"Let's build the GPT Tokenizer"* (~2h13m), taught as 5 detailed lessons, plus every doubt we cleared up along the way.

Read in order — each file builds only on the ones before it.

| # | File | What you'll understand |
|---|---|---|
| 1 | [Why tokenization matters, and how text becomes numbers](01-why-tokenization-and-text-as-numbers.md) | Where tokenization sits, the tokenizer demo, Unicode, why neither bytes nor characters work alone. **Doubts:** non-English cost, Unicode vs embeddings, who learns what, quadratic attention |
| 2 | [UTF-8 from scratch](02-utf8-from-scratch.md) | How any character becomes 1–4 bytes, the "train" picture, and exactly when `�` appears in LLM output |
| 3 | [The BPE algorithm](03-bpe-algorithm.md) | Training, encoding, decoding — at the level of real code. **Doubts:** why merges apply in learned order, "gluing bytes back into characters" |
| 4 | [Regex, tiktoken, special tokens](04-regex-tiktoken-special-tokens.md) | Pre-tokenization, GPT-2 vs GPT-4, `<\|endoftext\|>`, injection safety. **Doubts:** `how's` vs `HOW'S`, essence of chunking |
| 5 | [SentencePiece, vocabulary size, strange behaviors](05-sentencepiece-vocab-size-strange-behaviors.md) | Bytes vs characters, model surgery, and **every strange LLM behavior explained** (SolidGoldMagikarp, trailing whitespace...) |
| 💻 | [code/bpe_tokenizer.py](code/bpe_tokenizer.py) | A complete, runnable BPE tokenizer you can run in Google Colab |

## The whole topic in 5 lines

1. **Text → UTF-8 bytes** (English 1 byte, Hindi 3, emoji 4) so any language works.
2. **Regex splits text into chunks** (words, numbers, punctuation, spaces) — merges never cross chunks.
3. **BPE merges the most frequent pairs** inside chunks → vocabulary of ~50k–200k, much shorter sequences.
4. **Special tokens** (`<|endoftext|>`, chat markers) are inserted directly, bypassing BPE.
5. **Most strange LLM behaviors** — spelling, arithmetic, non-English cost, glitch tokens — **trace back to tokenization.**
