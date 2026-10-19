# LLM Interview Notes

**Detailed, from-scratch notes on how Large Language Models work — built for AI/ML internship interviews.**

Covers how ChatGPT-style models are built (pretraining → fine-tuning → reinforcement learning) and tokenization in full depth, with every concept explained from first principles, verified code, and a bank of interview questions with model answers.

> **Background assumed:** machine learning, deep learning, and the transformer architecture (attention, embeddings, layers). Everything beyond that is explained here.

---

## How to use this repo

1. **First time:** read the topic folders in order (01 → 05). Each file builds on the previous ones.
2. **Revising:** follow [how-to-revise.md](how-to-revise.md) — handwrite only the *questions*, answer them out loud, check against the [question bank](interview-question-bank.md).
3. **Stuck on a term?** → [glossary.md](glossary.md) (every abbreviation with its full form).

---

## Contents

| # | Topic | What you'll be able to explain |
|---|---|---|
| 01 | [The LLM lifecycle](01-llm-lifecycle/README.md) | The full pipeline in one diagram — **start here** |
| 02 | [Pretraining](02-pretraining/README.md) | Data, the training objective, loss, sampling & temperature, base models, few-shot prompting |
| 03 | [Supervised Fine-Tuning](03-post-training-sft/README.md) | Base model → assistant, chat templates, hallucinations, working memory, why chain-of-thought works |
| 04 | [Reinforcement learning](04-reinforcement-learning/README.md) | Thinking models, DeepSeek-R1, AlphaGo, RLHF, reward hacking, DPO / PPO / GRPO |
| 05 | [Tokenization](05-tokenization/README.md) | UTF-8, BPE (with runnable code), regex, special tokens, SentencePiece, every strange LLM behavior |
| ❓ | [Interview question bank](interview-question-bank.md) | 50+ questions with model answers, incl. practical scenarios |
| 📖 | [Glossary](glossary.md) | Every term and abbreviation, in plain words |
| 🔁 | [How to revise](how-to-revise.md) | Cue pages to handwrite + the pre-interview routine |

---

## The whole repo in five sentences

1. **Pretraining** teaches a transformer to predict the next token on trillions of filtered internet tokens, producing a base model with knowledge stored as a lossy compression in its weights.
2. **Supervised Fine-Tuning** — same objective, conversation data — turns it into an assistant that imitates an expert labeler, which is also where its confident hallucination style comes from.
3. **RLHF** polishes open-ended qualities but must stop early because the model games the reward model; **RL with verifiable rewards** can run long and produces thinking models.
4. The **context window** is working memory: facts placed there are far more reliable than facts recalled from the weights.
5. **Tokenization** (UTF-8 bytes → regex chunks → BPE merges) decides cost, context usage, and most of an LLM's strange failures — spelling, arithmetic, non-English cost, glitch tokens.

---

## Progress

- [x] LLM lifecycle, pretraining, post-training, RL — *Karpathy, "Deep Dive into LLMs like ChatGPT"*
- [x] Tokenization — *Karpathy, "Let's build the GPT Tokenizer"*
- [ ] Building GPT from scratch — *Karpathy, "Let's build GPT"*
- [ ] Modern transformer changes — RoPE, RMSNorm, KV cache, grouped-query attention
- [ ] Fine-tuning with LoRA / QLoRA
- [ ] RAG and LLM evaluation
- [ ] Project: LLM-derived sentiment as an alpha signal

---

## Sources

- Andrej Karpathy — [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) (YouTube)
- Andrej Karpathy — [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) (YouTube) and [minbpe](https://github.com/karpathy/minbpe)
- OpenAI — [tiktoken](https://github.com/openai/tiktoken)
- Google — [SentencePiece](https://github.com/google/sentencepiece)

*Notes written in my own study sessions; numbers marked as Karpathy's examples come from his videos. All code and UTF-8 / regex examples were run and verified.*
