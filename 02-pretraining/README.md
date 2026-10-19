# 02 · Pretraining: From Internet Text to a Base Model

> **Source:** Andrej Karpathy, *"Deep Dive into LLMs like ChatGPT"*, first ~hour.
> **You already know:** the transformer architecture (attention, embeddings, layers).
> **This file explains:** what the transformer is *trained on*, *how*, and what comes out.

---

## 2.1 Step 1 — Getting the data

Pretraining needs an enormous amount of text. It comes from the **internet**.

- **Common Crawl** is an organization that has been crawling the web since 2007, saving billions of web pages.
- Raw crawled pages are messy: HTML code, navigation menus, ads, spam, adult content, duplicate pages.
- So the raw crawl goes through a **filtering pipeline** before training. A well-known public example is **FineWeb** (by Hugging Face): roughly **44 TB of text, about 15 trillion tokens**.

### The filtering pipeline

| Step | What it does | Why |
|---|---|---|
| **URL filtering** | Drop pages from blocklisted domains (spam, malware, adult, marketing sites) | Remove harmful / low-value content |
| **Text extraction** | Strip HTML, menus, buttons — keep only the readable text | The model should learn language, not page layouts |
| **Language filtering** | Keep pages mostly in the target language(s) (e.g. >65% English) | Control what languages the model is good at |
| **Deduplication** | Remove repeated pages / passages | Avoid the model over-memorizing repeated text |
| **PII removal** | Remove Personally Identifiable Information (addresses, ID numbers…) | Privacy |

**Why filter at all?** Garbage in, garbage out. Data quality decides model quality.

> ⚠️ **Common mistake:** listing *tokenization* as a filtering step. It isn't. Filtering produces clean **text**; tokenization comes **after**, turning clean text into **token IDs**.

---

## 2.2 Step 2 — Turning text into tokens

A neural network can't read text; it needs numbers. The clean text is converted into a long sequence of **token IDs** using **Byte-Pair Encoding (BPE)**:

```
"hello world"  →  [31373, 995]
```

- GPT-4's tokenizer has about **100,000** possible tokens (exactly 100,277).
- A token is often a word or part of a word, e.g. `" world"` (with its leading space) is one token.

Full detail in [05-tokenization](../05-tokenization/). For now: **text → sequence of integers.**

---

## 2.3 Step 3 — What the network takes in and puts out

Take a window of tokens from the dataset (the **context**, up to the model's context length, e.g. 8,000 tokens). The network's job:

> **Given the tokens so far, predict the next token.**

**Input:** a sequence of token IDs (each looked up in the embedding table to get a vector — the part you already know).

**Output:** at **every position**, a **probability distribution over the entire vocabulary** — about 100,000 numbers that sum to 1, produced by a **softmax** over the raw scores (**logits**).

```
Input:   "The cat sat on the"
Output at last position:
   " mat"    0.40
   " floor"  0.25
   " sofa"   0.15
   ...       (all ~100,000 tokens get some probability)
```

### Why "at every position"?

In one pass, the model predicts the next token after *each* prefix of the window simultaneously — after "The", after "The cat", after "The cat sat"... — because **causal (masked) attention** stops each position from seeing the future. So one window of 8,000 tokens gives 8,000 training examples at once. This is why transformer training is so efficient.

---

## 2.4 Step 4 — The loss: how the model learns

We **know** the correct next token — it's simply the next token in the real text. So:

- **Label** = the actual next token in the data. Labels come *free* from the text itself, so this is called **self-supervised learning**. No human labeling needed.
- **Loss** = **cross-entropy** = −log(probability the model assigned to the correct next token), averaged over all positions.

```
If the true next token is " mat" and the model gave it 0.40:
   loss = −log(0.40) ≈ 0.92
If the model had given it 0.95:
   loss = −log(0.95) ≈ 0.05   ← much better
```

- **Gradients** of this loss nudge all the parameters so the correct token gets a slightly higher probability next time.
- Repeat across trillions of tokens, and the loss slowly falls. As it falls, the model's samples go from random gibberish to fluent text.

### A sense of scale (Karpathy's examples)

| Model | Parameters | Context | Training tokens |
|---|---|---|---|
| **GPT-2** (2019) | ~1.5 billion | 1,024 tokens | ~100 billion |
| **Llama 3.1** (2024) | up to 405 billion | much longer | ~15 trillion |

Karpathy notes GPT-2 originally cost on the order of tens of thousands of dollars to train; today it can be reproduced for a few hundred dollars, thanks to better data, hardware and software. Modern frontier pretraining runs take **months on thousands of GPUs** — by far the most expensive stage.

---

## 2.5 Inference — how the model generates text

Once trained, generating text is a loop:

```
1. Feed in the tokens so far
2. Model outputs a probability distribution for the next token
3. CHOOSE a token from that distribution
4. Append it to the sequence
5. Repeat until done
```

Step 3 is the interesting part.

### Greedy decoding vs sampling

> ❓ *"I thought an LLM always takes the highest-probability token?"* — That's one option, called **greedy decoding**. Most chat models don't use it by default.

| Method | How it picks | Result |
|---|---|---|
| **Greedy decoding** | Always the single most likely token | Deterministic: same prompt → same output. But dull, repetitive, can get stuck in loops. |
| **Sampling** | Randomly, *according to the probabilities* (" mat" 40% of the time, " floor" 25%...) | Varied, more natural text. **This is where randomness comes from.** |

**The model itself is deterministic** — the same input always gives the same probabilities. **The randomness is entirely in the sampling step.** One different token early on sends the rest of the text in a different direction, which is why the same prompt gives different answers.

### Temperature

Before the softmax, the logits `z` are divided by a temperature `T`:

```
p_i = softmax(z_i / T)
```

| Temperature | Effect | Use for |
|---|---|---|
| **T → 0** | Distribution becomes very sharp; top token gets ~all the probability ≈ **greedy** | Extraction, classification, factual Q&A |
| **T = 1** | The model's own probabilities, unchanged | General use |
| **T > 1** | Distribution flattens; unlikely tokens picked more often | Brainstorming, creative writing (more errors too) |

### Top-k and top-p (safety nets)

Both remove the long tail of unlikely tokens before sampling:

- **Top-k:** keep only the *k* most likely tokens (e.g. k = 50).
- **Top-p (nucleus sampling):** keep the smallest set of tokens whose probabilities add up to *p* (e.g. 0.9).

---

## 2.6 The base model — what pretraining produces

After pretraining you have a **base model**. Karpathy calls it an **"internet document simulator"** — a very advanced autocomplete.

- Give it text, and it continues it **the way a web page would continue**.
- Ask it a question and it might answer — or produce more questions, or a forum thread, because that's what web pages look like.
- **It is not an assistant**, because it has **never seen conversations** in a question → helpful-answer format.

### Memorization (regurgitation)

Karpathy gave the base model (Llama 3.1 405B) the first sentence of a Wikipedia article. It **continued the article almost word for word**.

**Why?** High-quality sources like Wikipedia are seen **multiple times** during training (they're often deliberately up-weighted). Seen often enough, text gets partly memorized into the weights. Ordinary pages seen once are not memorized this way.

### Knowledge as lossy compression

Fifteen trillion tokens of text get squeezed into a few hundred billion parameters. So the parameters hold a **lossy compression of the internet**: common facts are stored well; rare facts are stored fuzzily — which is exactly why models can **hallucinate** rare details.

### Getting assistant-like behavior from a base model: few-shot prompting

Even without any fine-tuning, you can coax a base model into answering questions by writing the prompt **as a document that looks like a conversation**:

```
The following is a conversation between a helpful AI assistant and a human.

Human: What is the capital of France?
Assistant: The capital of France is Paris.

Human: Why is the sky blue?
Assistant: Sunlight scatters off air molecules, and blue light scatters the most.

Human: <your actual question>
Assistant:
```

The base model continues the pattern and "answers". This works because of **in-context learning**: the model picks up patterns from its context window and continues them. This trick is called **few-shot prompting** (giving a few examples in the prompt).

---

## 2.7 One more key idea: compute is per token

Every token passes through **every layer** of the network. So:

- **Total computation grows with the number of tokens.**
- A single token gets a **fixed, limited amount of computation** (one pass through a fixed number of layers).

Both facts come back later: the first explains why tokenization efficiency matters for cost; the second explains why models need to "think" step by step ([03-post-training-sft](../03-post-training-sft/README.md#36-models-need-tokens-to-think)).

---

## Common mistakes (from our quizzes)

> ⚠️ **"The model always picks the highest-probability token."** Only with greedy decoding. Chat models usually **sample**; that's the source of randomness.
>
> ⚠️ **"Base models aren't assistants because they have fewer parameters."** No — they haven't been trained on conversations.
>
> ⚠️ **"Filtering includes tokenization."** No — separate stages.
>
> ⚠️ **"To make a base model act like an assistant, use 'better prompts'."** Directionally right, but name the technique: **few-shot prompting**, which works because of **in-context learning**.

---

## Interview questions for this topic

1. Where does pretraining data come from, and how is it filtered?
2. What exactly does the network output at each position, and what is the loss?
3. Why is pretraining called *self-supervised*?
4. The same prompt gives different answers each time. Why?
5. What does temperature do? What happens as it approaches 0?
6. What is a base model, and why isn't it an assistant?
7. Without fine-tuning, how could you make a base model answer questions?

Model answers → [interview-question-bank.md](../interview-question-bank.md#2-pretraining)
