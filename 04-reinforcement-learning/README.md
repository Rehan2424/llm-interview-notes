# 04 · Post-Training (Part 2): Reinforcement Learning, Thinking Models, and RLHF

> **Source:** Karpathy, *"Deep Dive into LLMs like ChatGPT"*, the final ~hour, plus a few extras marked "beyond the video".
> **Prerequisite:** [03-post-training-sft](../03-post-training-sft/README.md).
> **This file explains:** how models learn to *reason*, what RLHF really is, and why one kind of RL can run forever while the other can't.

---

## 4.1 Imitation vs practice

Recall the textbook analogy from [01-llm-lifecycle](../01-llm-lifecycle/README.md):

| Textbook | Stage | |
|---|---|---|
| Explanations | Pretraining | knowledge |
| Worked examples | Supervised Fine-Tuning (SFT) | **imitation** |
| Practice problems (answer at the back, no solution) | Reinforcement Learning (RL) | **practice** |

SFT shows the model *how an expert would solve it*. RL gives the model only the *final answer* and lets it find its own way.

---

## 4.2 Why imitation isn't enough

Take the Emily apples problem again. There are many correct ways to solve it: algebra, step-by-step arithmetic, shortcuts. **Which solution is best for the model to imitate?**

Human labelers can't really know, because **the model's "thinking" is different from ours**:

- A step that's trivial for a human might be a **big leap** for the model (too much computation for one token).
- A step a human finds obvious and skips might be exactly what the model needs to **write out**.
- A labeler might use knowledge the model doesn't have — teaching it to **bluff**.

So instead of humans deciding how the model should reason, **let the model discover what works for itself.**

---

## 4.3 How RL works for LLMs (verifiable rewards)

1. Take a problem with a **checkable answer** (e.g. a math problem whose answer is 3).
2. Have the model generate **many attempts** — say 15 or more — by **sampling** (this is where sampling's randomness becomes *useful*: it gives variety).
3. **Check each attempt:** is the final answer correct?
4. **Train on the successful attempts** — nudge the model to make those token sequences more likely.
5. Repeat across thousands of diverse problems, for many rounds.

Over time, the model **reinforces whatever reasoning patterns lead to correct answers**. Nobody hand-designs them — they **emerge** from trial and error.

**Notes:**
- This needs **verifiable** problems: math, code (run the tests), puzzles — anything an automatic checker can score. Sometimes an LLM acts as the judge, comparing the model's answer to a reference answer.
- Pretraining and SFT are mature and standardized. RL for LLMs is **newer and harder to get right**, with lots of fiddly details. Labs kept their methods internal — which is why the public **DeepSeek-R1** paper (January 2025) was such a big deal.

---

## 4.4 DeepSeek-R1 and "thinking models"

What the DeepSeek-R1 paper showed as RL training went on:

- **Accuracy on hard math problems kept rising.**
- **Responses got much longer** — the model *learned on its own* to think longer.
- **New behaviors emerged** that nobody programmed: re-checking its work, trying a different approach, backtracking, even writing things like *"Wait, let me re-evaluate this step."*

These long **chains of thought** are the model's *discovered* problem-solving strategy. A model trained this way is a **thinking model** (also called a **reasoning model**). Examples: DeepSeek-R1, OpenAI's o-series. Some providers show only a *summary* of the thinking, partly so competitors can't copy (distill) the reasoning traces.

**When to use which:**
- Hard math, code, multi-step logic → **thinking model** (slower, more expensive, more accurate).
- Simple facts, rewriting, summarizing, chat → a **regular model** is enough.

---

## 4.5 The AlphaGo connection — going beyond human

DeepMind's **AlphaGo** shows why RL matters:

- Trained only by **imitating expert human games** (supervised learning), it improves, then **plateaus below the top humans**. You can't beat the people you're copying.
- Trained with **RL** — playing games and rewarded for winning — it **surpassed the best humans**.
- Famous example: **"Move 37"** against Lee Sedol (2016). Experts estimated a human would play it with about a 1-in-10,000 chance. It looked like a mistake — and turned out to be brilliant.

**Implication for LLMs:** with RL, models might discover reasoning strategies **no human would think of**. It's early, and it needs a huge, diverse set of practice problems — but this is why RL is seen as the frontier.

---

## 4.6 RLHF — RL where answers can't be checked

**The problem:** how do you score *"Write a funny joke about pelicans"* or *"Summarize this article well"*? There's no correct answer to check automatically. You *could* have humans rate every attempt — but RL needs **millions** of scores. Impossibly expensive.

**The solution: RLHF (Reinforcement Learning from Human Feedback)**

```
Step 1: The model writes several answers (e.g. 5 jokes) for a prompt
Step 2: Humans RANK them, best → worst
Step 3: Train a separate neural network — the REWARD MODEL — to predict those rankings
Step 4: Run RL on the LLM, using the reward model's scores (millions of cheap scores)
```

The **reward model is a simulation of human judgment**. Instead of asking humans every time, you ask the simulator.

> ⚠️ **Common mistake (from our quiz):** "RLHF reinforces the reward model" or "humans give feedback after every round of RL." No — humans rank outputs (in batches, up front), the reward model learns from those rankings, and then the **LLM** is optimized **against** the reward model. Humans aren't in the RL loop; the reward model stands in for them.

### Upside

- RL becomes possible for **open-ended tasks**: tone, helpfulness, humor, summaries.
- **Ranking is easier than writing.** You may not be able to write a great joke, but you can tell which of five jokes is funniest. This **discriminator–generator gap** means humans give higher-quality signal by ranking than by writing answers themselves.

### Downside — it can be gamed

- The reward model is **just a neural network** — an imperfect simulation of human taste.
- RL is **extremely good at finding loopholes**. Run it long enough and the LLM discovers **nonsense outputs the reward model scores very highly** — e.g. *"the the the the the"* rated as the best joke ever. These are **adversarial examples**; the general phenomenon is called **reward hacking**.
- So RLHF can only run for a **limited number of steps**, then you stop.

**Karpathy's verdict:** RLHF is **not "real" RL in the magical AlphaGo sense** — it's a useful **polishing** step. True RL that can run indefinitely and discover new things needs **verifiable** rewards.

---

## 4.7 The comparison table (memorize this)

| | **Supervised Fine-Tuning (SFT)** | **RLHF** | **RL with verifiable rewards** |
|---|---|---|---|
| **Training signal** | Expert-written example conversations (imitate them, next-token prediction) | A **reward model** trained on human **rankings** | An **automatic checker** of the final answer |
| **Used for** | Turning a base model into an assistant: format, style, behavior | Open-ended qualities with no single right answer: tone, helpfulness, humor | Math, code, logic → **thinking models** |
| **Main limitation** | Only as good as the demonstrations; teaches the confident style behind hallucinations | **Reward hacking** — must stop early | Only works where answers can be **checked automatically**; expensive (many samples per problem) |
| **How long can it run?** | Until it fits the data | **Short** | **Long** — the checker is much harder to fool |

**Memory hook:**
> *A reward model is a neural network, so it can be fooled → RLHF stops early.*
> *A checker verifies the actual answer → RL with verifiable rewards keeps running.*

> ⚠️ **Common mistake (made twice in our quizzes):** attaching "finds loopholes, must stop early" to RL with verifiable rewards. It belongs to **RLHF**.

---

## 4.8 Beyond the video (useful for interviews)

| Term | Full form | What it is |
|---|---|---|
| **PPO** | Proximal Policy Optimization | The classic RL algorithm used in RLHF. Updates the model in small, "clipped" steps so it doesn't change too much at once. |
| **DPO** | Direct Preference Optimization | Skips the reward model *and* the RL loop: trains **directly** on preference pairs ("A is better than B"), raising the probability of preferred answers relative to rejected ones. Simpler and more stable → now very common. |
| **GRPO** | Group Relative Policy Optimization | The RL algorithm DeepSeek used. Samples a **group** of answers to the same problem and scores each one **relative to the group average**, instead of training a separate "value" model. Cheaper. |

A typical modern pipeline: **pretraining → SFT → preference tuning (RLHF or DPO) → RL with verifiable rewards.**

---

## 4.9 Closing sections of the video

**What's coming:**
- **Multimodality** — audio and images are tokenized too, then go through the same next-token machinery.
- **Agents** — models doing long, multi-step tasks, with humans supervising rather than doing.
- **Computer use** — models operating a browser or desktop by seeing the screen and clicking.
- **Open problems** — models can't keep learning after deployment (no continual learning); context windows are limited.

**How to keep up:** LM Arena (a human-preference leaderboard — useful, but can be gamed), AI newsletters, researchers on X.

**Where to use models:** proprietary models on their own sites (ChatGPT, Claude, Gemini); open-weight models via inference providers (e.g. Together AI); small models locally (e.g. LM Studio).

**His final summary of what happens when you use ChatGPT:**
your text becomes tokens → a regular model gives you a **simulation of a skilled labeler** (from SFT) → a thinking model adds **reasoning discovered through RL** → either way it's **Swiss cheese**: use it, verify it.

---

## Interview questions for this topic

1. Explain the difference between SFT, RLHF, and RL with verifiable rewards — training signal, use, and limitation of each.
2. How does RL on verifiable problems work for LLMs?
3. What did the DeepSeek-R1 paper show?
4. Why can't RLHF be run indefinitely?
5. What is reward hacking?
6. How does DPO differ from RLHF with PPO?
7. Why might RL let a model go beyond human-level reasoning? (AlphaGo)

Model answers → [interview-question-bank.md](../interview-question-bank.md#4-reinforcement-learning)
