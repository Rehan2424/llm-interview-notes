# Interview Question Bank — with Model Answers

> **How to use this file:** cover the answer, say yours **out loud** as if to an interviewer, then compare. Questions marked 🔁 are ones I got wrong or incomplete in practice — revise those first.
>
> **The habit that wins points:** before answering, **count the parts of the question out loud.**
> *"So you're asking (1) why letter counting fails, (2) why arithmetic fails, and (3) what I'd change. Let me take them in order."*
> In practice, I lost most points by **skipping part of a question**, not by misunderstanding.

**Sections:** [1. LLM lifecycle](#1-the-llm-lifecycle) · [2. Pretraining](#2-pretraining) · [3. Supervised Fine-Tuning](#3-post-training-supervised-fine-tuning) · [4. Reinforcement learning](#4-reinforcement-learning) · [5. Tokenization](#5-tokenization) · [6. Practical scenarios](#6-practical-scenarios)

---

## 1. The LLM lifecycle

**Q1.1 — Walk me through how ChatGPT is built, end to end.**
> Internet text is filtered for quality and tokenized with Byte-Pair Encoding. **Pretraining** teaches a transformer to predict the next token on trillions of tokens, producing a **base model** that has broad knowledge but just completes documents. **Supervised Fine-Tuning** trains the same model, with the same objective, on expert-written conversations, so it behaves like an assistant. **Preference tuning** (RLHF or DPO) polishes qualities like tone and helpfulness using human preference judgments. Finally, **RL with verifiable rewards** on math and code teaches it to reason, which produces thinking models. In one line: pretraining gives knowledge, SFT gives behavior, preference tuning gives polish, RL gives reasoning — and at inference, the context window gives reliable facts.

**Q1.2 — Why isn't a pretrained base model a useful assistant?**
> Because it has only ever seen internet documents, never conversations. It's an internet-document simulator: given a question it continues the text the way a web page would, which might mean more questions or a forum thread. It's not about size — a 405-billion-parameter base model is still not an assistant. SFT on conversation data is what teaches it the question → helpful-answer format.

**Q1.3 — Which stage is the most expensive, and why?**
> Pretraining, by far — months on thousands of GPUs, because it processes trillions of tokens. SFT is cheap in comparison (hours to days) since conversation datasets are tiny next to the internet.

**Q1.4 — What's the difference between a regular chat model and a thinking model?**
> A regular chat model is shaped mainly by SFT and preference tuning — it imitates expert answers. A thinking model has additionally been trained with RL on verifiable problems, so it has *discovered* its own reasoning strategies: it writes long chains of thought, re-checks and backtracks before answering. Thinking models are slower and costlier but much more accurate on hard math, code and logic; for simple tasks a regular model is enough.

---

## 2. Pretraining

**Q2.1 — Where does pretraining data come from, and how is it filtered?** 🔁
> It comes from web crawls such as Common Crawl. The raw crawl goes through a pipeline: URL filtering to drop spam, malware and adult sites; text extraction to strip HTML and keep readable text; language filtering; deduplication; and removal of personally identifiable information. FineWeb, for example, ends up around 44 TB, roughly 15 trillion tokens. We filter because data quality decides model quality. Tokenization is a separate step that comes *after* filtering.

**Q2.2 — What does the network output at each position, and what is the loss?** 🔁
> At every position, a probability distribution over the entire vocabulary — about 100k numbers from a softmax over the logits — predicting the **next** token. The label is simply the actual next token in the text, so labels come free from the data. The loss is cross-entropy: minus the log of the probability assigned to the correct next token, averaged over all positions. Gradients nudge the parameters to raise that probability.

**Q2.3 — Why is pretraining called self-supervised?**
> Because the labels come from the data itself — the "correct answer" at each position is just the next token in the text. No human labeling is needed, which is what makes training on trillions of tokens possible.

**Q2.4 — The same prompt gives different answers each time. Why?** 🔁
> The model itself is deterministic: the same input always gives the same probability distribution. The randomness is in the **sampling** step — instead of always taking the most likely token (greedy decoding), we sample according to the probabilities. One different token early on sends the rest of the text in a different direction.

**Q2.5 — What does temperature do? What happens as it approaches 0?**
> Temperature divides the logits before the softmax: p = softmax(z / T). Below 1 it sharpens the distribution; as T approaches 0, the top token gets nearly all the probability, so it behaves like greedy decoding. Above 1 it flattens the distribution, making unlikely tokens more common — more creative but more error-prone. Use low temperature for extraction, classification and factual Q&A; higher for brainstorming. Top-k and top-p additionally cut off the long tail of unlikely tokens before sampling.

**Q2.6 — What is a base model, and why isn't it an assistant?** 🔁
> See Q1.2. Also mention memorization: given the start of a Wikipedia article, a base model can recite it nearly word for word, because high-quality sources are seen multiple times in training.

**Q2.7 — Without fine-tuning, how could you make a base model answer questions?** 🔁
> **Few-shot prompting:** write the prompt as a document that looks like a conversation between a helpful AI and a human, with a few example question–answer turns, then add your question. The base model continues the pattern and answers. It works because of **in-context learning** — the model picks up patterns from its context window.

---

## 3. Post-training: Supervised Fine-Tuning

**Q3.1 — What is SFT, and what changes compared to pretraining?**
> Supervised Fine-Tuning continues training the pretrained model on a dataset of conversations — originally written by human labelers following "helpful, truthful, harmless" guidelines, now often LLM-generated and human-edited. The objective is **exactly the same** next-token prediction; only the data changes. The result is a model that imitates an expert labeler. It's far cheaper than pretraining.

**Q3.2 — How does a multi-turn conversation get fed to the model?**
> It's flattened into one token sequence with a chat template: special tokens like `<|im_start|>`, `<|im_sep|>` and `<|im_end|>` mark who's speaking and where each message ends. At inference, the system appends `<|im_start|>assistant<|im_sep|>` after your message, and the model continues from there — which is the answer.

**Q3.3 — Why do LLMs hallucinate confidently? Give two mitigations and explain why each works.**
> In SFT data, factual questions are almost always followed by confident answers, so the model learns that *style* and keeps it even when it lacks the knowledge — it's trained to produce a likely continuation, not to verify truth. **Mitigation 1:** probe what the model knows (ask questions several times, check answers) and add training examples where the right answer is "I don't know" — this wires the model's internal uncertainty to a refusal (Meta did this for Llama 3). **Mitigation 2:** tools or retrieval — put real facts into the context window, where the model can read them directly instead of relying on fuzzy memory.

**Q3.4 — Explain "vague recollection vs working memory" with an example.** 🔁
> Knowledge in the parameters is a lossy compression of the training data spread across billions of weights, so rarely seen facts end up fuzzy — a vague recollection. Tokens in the context window can be attended to directly at every layer — working memory. So instead of asking "summarize chapter 3 of this book", paste the chapter in. This is the principle behind Retrieval-Augmented Generation.

**Q3.5 — Why does chain-of-thought improve accuracy?**
> Each token gets a fixed amount of computation — one pass through a fixed number of layers. If the model must state the answer immediately, all the reasoning has to fit into that one token. Writing the steps out spreads the computation across many tokens, each doing an easy step, and puts intermediate results into the context for later steps to use. So training data and prompts should put reasoning before the answer, and heavy arithmetic should go to code.

**Q3.6 — How does a model "know" who it is?**
> It doesn't have a persistent self. Its answer comes from identity conversations included in SFT data or from a hidden system prompt. Without those, an open model might claim to be ChatGPT, simply because that's the most common answer on the internet.

---

## 4. Reinforcement learning

**Q4.1 — Explain SFT vs RLHF vs RL with verifiable rewards — training signal, use, and limitation of each.** 🔁🔁 *(missed twice — the loophole problem belongs to RLHF)*
> **SFT:** the signal is expert-written conversations, imitated with next-token prediction. It's used to turn a base model into an assistant. Limitation: it can only be as good as the demonstrations, and it teaches the confident style behind hallucinations.
> **RLHF (Reinforcement Learning from Human Feedback):** humans rank outputs, a **reward model** learns to predict those rankings, and the LLM is **optimized against** the reward model. Used for open-ended qualities with no checkable answer — tone, helpfulness, humor. Limitation: the reward model is just a neural network, so the LLM learns to **game it** (reward hacking) — it must be stopped after limited steps.
> **RL with verifiable rewards:** the model samples many attempts, an **automatic checker** scores the final answers, and successful reasoning is reinforced. Used for math and code — it's how thinking models are trained. Limitation: it only works where answers can be **checked automatically**, and it's compute-heavy. But because the checker is hard to fool, it can run **much longer**.
> Typical pipeline: pretraining → SFT → preference tuning (RLHF or DPO) → RL with verifiable rewards.

**Q4.2 — How does RL on verifiable problems work for LLMs?**
> Take problems with checkable answers. For each, sample many attempts, check which reach the correct answer, and train on the successful ones so those reasoning paths become more likely. Repeat over thousands of problems. The model discovers reasoning strategies that work *for it* — humans don't have to design them, which matters because the model's "cognition" differs from ours.

**Q4.3 — What did the DeepSeek-R1 paper show?**
> That RL with verifiable rewards makes reasoning emerge: as training progressed, accuracy on hard math rose, responses got much longer as the model learned on its own to think longer, and behaviors like re-checking, trying other approaches and backtracking ("wait, let me re-evaluate") appeared without being programmed. It was also important because it described LLM RL publicly when labs had kept their methods internal.

**Q4.4 — Why can't RLHF run indefinitely? What is reward hacking?**
> The reward model is an imperfect neural-network simulation of human judgment, and RL is extremely good at finding loopholes. Run long enough, the LLM finds outputs that score very high but are nonsense — adversarial examples like "the the the the" rated as the best joke. Optimizing the grader instead of the real goal is called reward hacking. So RLHF is run for a limited number of steps and then stopped.

**Q4.5 — How does DPO differ from RLHF with PPO?**
> RLHF with PPO (Proximal Policy Optimization) trains a separate reward model and then runs an RL loop against it. DPO (Direct Preference Optimization) skips both: it trains **directly on preference pairs** — prompt, preferred answer, rejected answer — raising the probability of the preferred answer relative to the rejected one. It's simpler and more stable, which is why it's now very common.

**Q4.6 — Why might RL let a model go beyond human-level reasoning?**
> AlphaGo is the analogy: trained only by imitating human experts it plateaued below the best humans, but trained with RL — rewarded for winning — it surpassed them, finding moves like the famous Move 37 that experts thought a human would almost never play. Imitation is capped by the people you copy; RL can discover strategies no human would think of.

**Q4.7 — What is GRPO?**
> Group Relative Policy Optimization, the RL algorithm DeepSeek used: it samples a group of answers to the same problem and scores each relative to the group average, instead of training a separate value model — making RL cheaper.

---

## 5. Tokenization

**Q5.1 — How does BPE tokenization work — training and encoding?** 🔁
> **Training:** convert the training text to UTF-8 bytes, so the starting vocabulary is 256. Repeatedly count all adjacent pairs and merge the most frequent pair into a new token with the next free ID, until the target vocabulary size is reached. The ordered list of merges *is* the tokenizer. **Encoding:** convert new text to bytes, split it into chunks with the regex, then within each chunk repeatedly apply the **earliest-learned** merge among the pairs present, until none applies. That guarantees the same text always tokenizes exactly as it did during training. **Decoding:** look up each ID's bytes, join, and decode as UTF-8.

**Q5.2 — Why does vocabulary size trade off against sequence length?**
> Every BPE merge adds one token to the vocabulary and shortens text. A bigger vocabulary gives shorter sequences — more text fits in the context window, attention gets cheaper since it's quadratic in length, and generation needs fewer steps. But the embedding table and the output layer both grow with vocabulary size, the softmax over the vocabulary costs more per token, rare tokens end up undertrained, and very large tokens pack too much meaning into one forward pass. A small vocabulary has the opposite problem. So models settle around 50k–200k: GPT-2 ~50k, GPT-4 ~100k, Llama 3 ~128k, GPT-4o ~200k.

**Q5.3 — Why do LLMs use byte-level BPE instead of characters or raw bytes?** 🔁
> **Raw bytes** give a tiny vocabulary of 256 but far too long sequences, wasting the context window and making attention expensive. **Characters** (Unicode code points) would need a vocabulary of ~150,000 that keeps changing, and rare characters would barely be trained. **Byte-level BPE** gets the best of both: starting from bytes means any text in any language is representable with no unknown tokens, and merging frequent pairs compresses sequences while keeping the vocabulary manageable.

**Q5.4 — Why are LLMs bad at arithmetic?** 🔁
> Numbers are tokenized by frequency, not place value — in one tokenizer `127` is one token while `677` is `6` + `77`. So digits don't line up consistently as ones, tens and hundreds, which makes column-style arithmetic hard. Modern tokenizers reduce this by capping number chunks at 3 digits (GPT-4) or splitting every digit (Llama 2). The robust fix in a product is a calculator or code tool.

**Q5.5 — Why can't LLMs spell, count letters, or reverse strings?** 🔁 *(I skipped this half twice)*
> A token can contain many letters — `.DefaultCellStyle` is a single token in GPT-4's tokenizer — so the model never sees the individual letters. Workarounds: have the model write the word out letter by letter, separated by spaces, first (then each letter is its own token), or route string tasks to code.

**Q5.6 — Why do non-English users pay more for the same API usage?**
> Pricing is per token, and non-English text takes more tokens: the tokenizer's training data was mostly English, so it learned few merges for other languages, and UTF-8 already uses 2–3 bytes per character for many scripts like Hindi or Korean versus 1 for English. More tokens also means the context window fills faster and generation is slower.

**Q5.7 — What's the difference between a token ID and an embedding?**
> A token ID is just a label — an integer assigned in order by the tokenizer, carrying no meaning, like a roll number. An embedding is a learned vector, trained by gradient descent with the LLM, where similar meanings end up close together — like a detailed profile. Meaning only appears at the embedding step.

**Q5.8 — Who learns what: tokenizer vs embeddings?** 🔁
> UTF-8 encoding learns nothing — it's a fixed rule. The tokenizer is trained once, before the LLM, **by counting** pair frequencies in its own training text; it learns the merge rules, assigns IDs in order, and is then frozen. The embedding table and transformer layers are learned **by gradient descent** as part of LLM training — the embeddings learn what each token means.

**Q5.9 — Why does attention cost grow quadratically?**
> Every token's query is compared with every token's key, producing an n × n score matrix — n² dot products of dimension d, in every head of every layer, plus n² memory. Doubling the length quadruples the cost. Feed-forward layers, by contrast, scale linearly. That's why halving token count through better tokenization cuts attention cost about 4×.

**Q5.10 — Why do you sometimes see `�` in LLM output?**
> Byte-level tokenizers can split one character — a Hindi letter or an emoji, which take 2–4 bytes in UTF-8 — across several tokens. The model picks tokens by probability without knowing UTF-8 rules, so a character can be left incomplete: a wrong token in the middle, generation stopping mid-character, or a streaming UI displaying the first token before the rest arrives. When decoding hits invalid bytes, the decoder substitutes the replacement character `�` instead of crashing. In the streaming case it's temporary and fixes itself when the next token arrives.

**Q5.11 — Why do GPT tokenizers split text with a regex before BPE?**
> It's called pre-tokenization. Without it, BPE would create tokens like `dog.`, `dog!`, `dog?`, wasting vocabulary and mixing meaning with punctuation. The regex splits text into chunks by category — words, numbers, punctuation, whitespace — and merges never cross chunks, so tokens stay meaningful and the same word gets the same token everywhere. GPT-4's pattern improved on GPT-2's: case-insensitive contractions, numbers capped at 3 digits, and merged whitespace for code.

**Q5.12 — What are special tokens? Give an example.**
> Tokens that don't come from BPE merges — the designers add them by hand to mark structure, and the tokenizer inserts them directly. `<|endoftext|>` separates documents in training data so the model learns that what follows is unrelated (ID 50256 in GPT-2). Chat models use tokens like `<|im_start|>` and `<|im_end|>` to mark turns, and code models use fill-in-the-middle tokens.

**Q5.13 — A user's message contains the literal text `<|endoftext|>`. What should your system do?**
> Treat it as ordinary text, never as the real control token — otherwise users could inject fake document boundaries or chat turns, like SQL injection. tiktoken raises an error by default; for user input you pass `disallowed_special=()` so it's encoded as harmless normal tokens. Only your own code should insert real special tokens.

**Q5.14 — How would you check whether a document fits in a model's context window?**
> Count its tokens with that model's own tokenizer — e.g. tiktoken for OpenAI models — and compare against the context limit, leaving room for the system prompt and the response. Never estimate from word count; token counts vary a lot by language and content (code, JSON, non-English).

**Q5.15 — What happens if you increase vocabulary size?** → see Q5.2.

**Q5.16 — How would you add a new special token to a pretrained model?**
> Add it to the tokenizer with the next free ID; add a new row to the embedding table, initialized to small random values or the mean of existing embeddings; add a new row to the output layer so the model can predict it; then train — often freezing the rest of the model and training only the new rows. This "model surgery" is common when turning a base model into a chat model.

**Q5.17 — What are undertrained (glitch) tokens? Explain SolidGoldMagikarp.** 🔁
> The tokenizer and the LLM are trained on different data. "SolidGoldMagikarp", a Reddit username, was frequent in the tokenizer's data, so it became its own token — but it never appeared in the LLM's training data, so its embedding row was never updated and stayed at random initial values. Feeding it at inference gives the model a meaningless vector, and behavior goes haywire. Fix: train the tokenizer on data like the LLM's, and check for undertrained tokens before release.

**Q5.18 — Why would you send YAML instead of JSON to an LLM?**
> Token efficiency. JSON spends many tokens on quotes, braces and commas; YAML expresses the same data with indentation and far less punctuation, so it's cheaper and fits more in the context window.

**Q5.19 — Why does a prompt ending in a trailing space hurt output?**
> In training data, spaces are almost always attached to the start of the next word, like `" Oh"`. A prompt ending in a space makes that space a separate token, a situation the model rarely saw — out of distribution — so quality drops. Generally, prompts that end partway through a typical token confuse the model.

**Q5.20 — How does SentencePiece differ from tiktoken-style tokenizers?**
> Both use BPE, but SentencePiece merges **characters** (code points), not bytes, so it never splits a character — while byte-level BPE must first reassemble bytes into characters. Rare characters outside SentencePiece's vocabulary use **byte fallback** (encoded as their UTF-8 bytes) instead of becoming `<unk>`. It also represents spaces as `▁`, making it language-independent. It was used by Llama 1 and 2 and T5; Llama 3 moved to a tiktoken-style tokenizer.

**Q5.21 — BPE vs WordPiece vs Unigram?**
> BPE merges the most frequent pair (GPT, Llama). WordPiece merges the pair with the highest score — frequency of the pair divided by the product of the parts' frequencies — favoring pairs that co-occur more than chance (BERT). Unigram starts from a large vocabulary and prunes the tokens that matter least (T5 via SentencePiece).

---

## 6. Practical scenarios

**Q6.1 — "Our app can't count letters in product names and makes mistakes adding prices. Why, and what would you change?"** 🔁
> *(Three parts: letters, arithmetic, the fix.)* Letter counting fails because a product name may be one or two tokens, so the model never sees its letters. Arithmetic fails because numbers are chunked inconsistently, so place values don't line up. In the product, I'd route counting and string tasks to code and all price arithmetic to a calculator tool, rather than trusting the model's output — and add tests on real product names and prices.

**Q6.2 — "We're launching a Hindi support bot on a paid LLM API. What should we worry about tokenization-wise, and what's the plan?"** 🔁
> *(Two parts: worries, plan.)* **Worries:** Hindi takes several times more tokens than English — each character is 3 bytes in UTF-8 and the tokenizer learned fewer Hindi merges. That means higher cost, a context window that fills faster, slower replies, and possibly lower quality. **Plan:** since it's an API we can't change the tokenizer, so: (1) measure real token counts on real conversations with the provider's tokenizer; (2) compare models — newer tokenizers with larger vocabularies are much more efficient for Indian languages; (3) keep context lean — English system instructions, summarized history, only relevant documents; (4) handle Hinglish, which many users type and which tokenizes differently; (5) build a Hindi test set and validate quality before launch.

**Q6.3 — "Our chatbot keeps confidently inventing product details. What would you do?"** 🔁
> First, **retrieval** (RAG): fetch the actual product documentation and put it in the context window, instructing the model to answer only from it and say it doesn't know otherwise. Why it works: our product details are rare or absent from training data, so the parameters hold at best a vague recollection, and SFT taught a confident style anyway; documents in context are working memory. Second, train or prompt it to say "I'm not sure" when retrieved documents don't contain the answer. Third, build a test set of product questions to measure the hallucination rate before and after. I would *not* fine-tune to teach facts — they change, and retrieval stays current.
