# Glossary — Every Abbreviation and Term, in Plain Words

Alphabetical. Where a term has a full explanation in the notes, the link takes you there.

| Term | Full form | Plain meaning |
|---|---|---|
| **AlphaGo** | — | DeepMind's Go-playing AI. Imitation of humans plateaued; RL surpassed the best humans ("Move 37"). [→](04-reinforcement-learning/README.md#45-the-alphago-connection--going-beyond-human) |
| **API** | Application Programming Interface | A way for programs to talk to a service. "Using an LLM through an API" = sending text to a provider's model and paying per token. |
| **ASCII** | American Standard Code for Information Interchange | 1960s standard numbering English characters 0–127. UTF-8 keeps these as single bytes. |
| **Base model** | — | The model after pretraining: a document completer, not an assistant. |
| **BERT** | Bidirectional Encoder Representations from Transformers | Google's encoder-only transformer (2018); uses the WordPiece tokenizer. |
| **BPE** | Byte-Pair Encoding | Tokenizer algorithm: repeatedly merge the most frequent adjacent pair into a new token. [→](05-tokenization/03-bpe-algorithm.md) |
| **Byte** | — | 8 bits; holds a number from 0 to 255. |
| **Byte fallback** | — | SentencePiece option: characters missing from the vocabulary are encoded as their UTF-8 bytes instead of `<unk>`. |
| **Chain-of-thought** | — | The model writing out reasoning steps before the answer; spreads computation over many tokens. |
| **Chat template** | — | The format that flattens a conversation into tokens using special tokens like `<\|im_start\|>`. |
| **cl100k_base** | — | The name of GPT-4's tokenizer (~100k tokens). |
| **Code point** | — | The number Unicode assigns to a character (`h` = 104, `न` = 2,344). |
| **Common Crawl** | — | Organization that has crawled and archived the web since 2007; the raw source of most pretraining data. |
| **Context window** | — | The maximum number of tokens the model can see at once; the model's "working memory". |
| **Cross-entropy** | — | The pretraining loss: −log(probability given to the correct next token). |
| **Decoder-only** | — | A transformer with only causal (masked) self-attention blocks — the GPT architecture. |
| **DPO** | Direct Preference Optimization | Trains directly on "A is better than B" pairs, with no separate reward model or RL loop. |
| **Embedding** | — | A learned vector for each token ID; where meaning lives. |
| **FIM** | Fill-In-the-Middle | Special tokens (`<\|fim_prefix\|>` etc.) for training code models to fill gaps inside a file. |
| **FineWeb** | — | Hugging Face's filtered web dataset: ~44 TB, ~15 trillion tokens. |
| **Few-shot prompting** | — | Giving a few examples in the prompt so the model continues the pattern. |
| **Glitch token** | — | An undertrained token whose embedding was never updated (e.g. SolidGoldMagikarp). |
| **GPT** | Generative Pre-trained Transformer | OpenAI's family of decoder-only language models. |
| **GPU** | Graphics Processing Unit | The hardware LLMs are trained and run on. |
| **Greedy decoding** | — | Always picking the single most likely next token. Deterministic. |
| **GRPO** | Group Relative Policy Optimization | DeepSeek's RL algorithm: scores each sampled answer relative to its group's average. |
| **Hallucination** | — | Fluent, confident, false output. Caused by the confident style learned in SFT. |
| **In-context learning** | — | The model picking up patterns from its prompt without any weight updates. |
| **Inference** | — | Using a trained model to generate output (as opposed to training it). |
| **InstructGPT** | — | OpenAI's 2022 paper describing SFT + RLHF with human labelers. |
| **JSON** | JavaScript Object Notation | Data format with many quotes/braces — token-expensive for LLMs. |
| **KV cache** | Key-Value cache | During generation, keys and values of earlier tokens don't change, so they're stored and reused instead of recomputed. A speed trick for inference. *(Studied later in the plan.)* |
| **Logits** | — | The raw scores the model outputs before the softmax turns them into probabilities. |
| **LLM** | Large Language Model | A large transformer trained to predict the next token. |
| **LoRA** | Low-Rank Adaptation | A cheap fine-tuning method that trains small add-on matrices instead of all weights. *(Studied later in the plan.)* |
| **Model surgery** | — | Resizing the embedding table and output layer of a trained model to add new tokens. |
| **o200k_base** | — | GPT-4o's tokenizer (~200k tokens). |
| **PII** | Personally Identifiable Information | Names, addresses, ID numbers — removed from pretraining data. |
| **PPO** | Proximal Policy Optimization | The classic RL algorithm used in RLHF; updates the model in small, clipped steps. |
| **Pre-tokenization** | — | Splitting text into chunks (with a regex) before BPE, so merges never cross chunks. |
| **Pretraining** | — | Training on trillions of internet tokens to predict the next token. Produces the base model. |
| **RAG** | Retrieval-Augmented Generation | Search documents for relevant passages and paste them into the prompt before the model answers. *(Studied in December.)* |
| **Regex** | Regular expression | A pattern language for finding pieces of text. |
| **Replacement character** | — | `�` — what a decoder shows for invalid/incomplete UTF-8 bytes. |
| **Reward hacking** | — | The model optimizing the grader instead of the real goal (e.g. nonsense that a reward model scores highly). |
| **Reward model** | — | A neural network trained to predict human rankings; stands in for humans in RLHF. |
| **RL** | Reinforcement Learning | Learning by trial and error from rewards. |
| **RLHF** | Reinforcement Learning from Human Feedback | RL against a reward model trained on human rankings. For tasks without checkable answers. Must stop early. |
| **RL with verifiable rewards** | — | RL where an automatic checker scores the answer (math, code). Can run long. Produces thinking models. |
| **Sampling** | — | Picking the next token randomly according to the model's probabilities. The source of randomness. |
| **Self-supervised learning** | — | Training where labels come from the data itself (the next token). |
| **SentencePiece** | — | Google's tokenizer library; merges characters, uses `▁` for spaces and byte fallback. |
| **SFT** | Supervised Fine-Tuning | Training the base model on expert-written conversations (same next-token objective). |
| **Softmax** | — | Turns logits into probabilities that sum to 1. |
| **Special token** | — | A hand-added token (not from BPE) marking structure, e.g. `<\|endoftext\|>`. |
| **SQL injection** | Structured Query Language injection | An attack where user input is treated as a command; analogous to injecting special tokens. |
| **System prompt** | — | Hidden instructions placed at the start of every conversation. |
| **Temperature** | — | Divides logits before softmax: low = near-greedy, high = more random. |
| **Thinking (reasoning) model** | — | A model trained with RL to reason at length before answering (DeepSeek-R1, OpenAI o-series). |
| **tiktoken** | — | OpenAI's library for encoding/decoding with their tokenizers (cannot train). |
| **Token** | — | The unit an LLM reads and writes: often a word or part of a word. |
| **Top-k / Top-p** | — | Sampling only from the k most likely tokens / the smallest set whose probabilities sum to p. |
| **Unicode** | — | The worldwide standard giving every character a number. |
| **Unigram** | — | Tokenizer algorithm that starts large and prunes tokens (T5). |
| **`<unk>`** | unknown token | What a tokenizer outputs for a character it can't represent — information lost. |
| **UTF-8** | Unicode Transformation Format, 8-bit | Encoding that stores each character in 1–4 bytes, with labels marking where characters start. [→](05-tokenization/02-utf8-from-scratch.md) |
| **WordPiece** | — | BERT's tokenizer algorithm; merges by a frequency-ratio score. |
| **YAML** | YAML Ain't Markup Language | Data format using indentation; fewer tokens than JSON. |
