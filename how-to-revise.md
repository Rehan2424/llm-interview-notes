# How to Revise (without rewriting everything by hand)

> **You don't need to handwrite these notes.** Copying is the least effective part of note-taking. What makes knowledge stick is **recalling** it — pulling it out of memory. This repo holds the explanations; your handwriting holds only the **questions**.

---

## The 3-layer system

| Layer | What | Where | Time |
|---|---|---|---|
| **1. Reference** | Full explanations, from scratch | This repo (topic folders) | Read once carefully; reread only weak spots |
| **2. Cue pages** | **Questions only**, no answers — one page per topic | **Your notebook, handwritten** | ~15 min to write per topic |
| **3. Recall** | Answer each question **out loud**, then check | Notebook + [interview-question-bank.md](interview-question-bank.md) | 1–2 hours before an interview |

---

## Layer 2: what to write on each handwritten cue page

Copy only these questions — one page per topic. Leave a small box next to each; tick it when you can answer it out loud without looking.

### Page 1 — LLM lifecycle & pretraining
- [ ] Walk me through how ChatGPT is built, end to end.
- [ ] One sentence: what does each stage give? *(knowledge / behavior / polish / reasoning / facts)*
- [ ] Where does pretraining data come from? Name the filtering steps.
- [ ] What does the network output at each position? What's the loss?
- [ ] Why is pretraining "self-supervised"?
- [ ] Same prompt, different answers — why?
- [ ] What does temperature do? T → 0?
- [ ] What's a base model? Why not an assistant? (Wikipedia memorization)
- [ ] Make a base model answer questions without fine-tuning.

### Page 2 — Supervised Fine-Tuning
- [ ] What is SFT? What changes vs pretraining?
- [ ] How is a conversation turned into tokens?
- [ ] Why do LLMs hallucinate confidently? Two fixes + why each works.
- [ ] Vague recollection vs working memory — example.
- [ ] Why does chain-of-thought work? (Emily problem)
- [ ] Why can't LLMs count letters? (Swiss cheese)

### Page 3 — Reinforcement learning
- [ ] SFT vs RLHF vs RL-verifiable: **signal / use / limitation** for each. *(Loopholes → RLHF!)*
- [ ] How does RL on verifiable problems work?
- [ ] What did DeepSeek-R1 show?
- [ ] Why must RLHF stop early? Reward hacking?
- [ ] DPO vs PPO-based RLHF.
- [ ] AlphaGo Move 37 — what's the lesson?

### Page 4 — Tokenization
- [ ] Why not raw bytes? Why not characters? Why byte-level BPE?
- [ ] BPE: training, encoding (why earliest-learned merge first?), decoding.
- [ ] Vocabulary size vs sequence length — the trade-off (3 costs of bigger vocab).
- [ ] UTF-8: English/Hindi/emoji byte counts; when does `�` appear?
- [ ] Why regex pre-tokenization? (dog example)
- [ ] Special tokens — `<|endoftext|>`; what if a user types it?
- [ ] Why bad at arithmetic? Why bad at spelling? Fix for each.
- [ ] Why does non-English cost more?
- [ ] SolidGoldMagikarp.
- [ ] Trailing whitespace; YAML vs JSON.
- [ ] SentencePiece vs tiktoken; byte fallback.

### Page 5 — Scenarios
- [ ] App can't count letters + adds prices wrong → why + fix.
- [ ] Hindi support bot on a paid API → worries + plan.
- [ ] Chatbot invents product details → fix + why + how to measure.

---

## Layer 3: the routine before an interview

1. Go down your cue pages. For each question, **answer out loud** in 3–5 sentences, as if to an interviewer.
2. **Count the parts of the question first.** ("You're asking A, B and C.")
3. Check against [interview-question-bank.md](interview-question-bank.md). Anything you stumbled on → reread that section of the notes.
4. Do the 🔁 questions (ones I got wrong in practice) **twice**.
5. Optional: paste a cue page into Claude and say *"quiz me on this — grade me like an interviewer."*

## Spaced review (so you don't forget during the November break)

| When | What | Time |
|---|---|---|
| Day after learning a topic | Answer that topic's cue page | 10 min |
| Every Sunday | All cue pages, ticks only for confident answers | 30–45 min |
| Before an interview | Full routine above | 1–2 hours |
| After a long break (e.g. December restart) | Reread the repo [README](README.md) + all cue pages | 1 hour |
