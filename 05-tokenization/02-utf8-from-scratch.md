# Tokenization 2 · UTF-8 From Scratch (Once and For All)

> **Why this file exists:** UTF-8 confused us at first, so this builds it up one step at a time. Each step uses only the steps before it.
> **Previous:** [01-why-tokenization-and-text-as-numbers.md](01-why-tokenization-and-text-as-numbers.md) · **Next:** [03-bpe-algorithm.md](03-bpe-algorithm.md)

---

## If you only remember 3 things (enough for any ML interview)

1. Every character becomes **1 to 4 bytes**: English **1**, Hindi and Korean **3**, emoji **4**.
2. **The first byte says how many bytes the character has; the rest are marked as "continuation" bytes.**
3. **Missing or out-of-place pieces → invalid → `�`.**

### The picture: a train 🚂

Think of each character as a small train:

- **English letter:** an engine alone. 🚂 (1 byte)
- **Hindi letter:** engine + 2 carriages. 🚂🚃🚃 (3 bytes)
- **Emoji:** engine + 3 carriages. 🚂🚃🚃🚃 (4 bytes)

Two rules: **the engine announces how many carriages follow**, and **every carriage is labeled "I belong to an engine in front of me."**
`�` appears when a train is broken: an engine with missing carriages (🚂🚃 ❌), or a carriage with no engine (🚃 ❌).

Everything below is the detailed version of this picture.

---

## Step 1 — Computers only store numbers

Everything is stored as **bytes**. A byte is 8 tiny on/off switches called **bits**, each written 0 or 1:

```
01101000   ← one byte (8 bits)
```

8 switches give 2⁸ = **256 patterns**, so one byte holds any number from **0 to 255**.

## Step 2 — Give every character a number (Unicode)

**Unicode** is a list giving every character in the world a number, called its **code point**:

```
h   →  104
A   →  65
न   →  2,344
안  →  50,504
👋  →  128,075
```

Text = a list of numbers. But there's a problem…

## Step 3 — Big numbers don't fit in one byte

One byte holds at most 255. `h` (104) fits. `न` (2,344) and `👋` (128,075) **don't**. Big numbers need **more than one byte**.

So we need a rule for splitting a number into several bytes and putting it back together. That rule is called an **encoding**.

## Step 4 — Two bad ideas

**Bad idea 1: always use 4 bytes per character.** Enough room for anything, but wasteful: `h` would be `0, 0, 0, 104` — three wasted bytes per English letter, 4× the storage. (This is roughly what UTF-32 does.)

**Bad idea 2: use as few bytes as needed, glued together with nothing in between.** Saves space, but it's **ambiguous**. If you see two bytes `9, 40`: are they two small characters, or two halves of **one** big character? No way to tell. You need to know **where each character starts and ends**.

## Step 5 — UTF-8's solution: label every byte

**UTF-8 (Unicode Transformation Format, 8-bit)** uses variable length (1–4 bytes, saving space) **and** reserves the first few bits of every byte as a **label**:

| Label at start of byte | Meaning |
|---|---|
| `0` | "I'm a complete character by myself" (1 byte) |
| `110` | "I **start** a character that has **2** bytes" |
| `1110` | "I **start** a character that has **3** bytes" |
| `11110` | "I **start** a character that has **4** bytes" |
| `10` | "I'm a **continuation** — I belong to the character started before me" |

The remaining bits (written `x`) carry the character's actual number:

```
1 byte:   0xxxxxxx
2 bytes:  110xxxxx 10xxxxxx
3 bytes:  1110xxxx 10xxxxxx 10xxxxxx
4 bytes:  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

### How the number of bytes relates to the character range

Count the free `x` slots in each pattern:

| Bytes | Free `x` bits | How many numbers fit | Range covered | Examples |
|---|---|---|---|---|
| 1 | 7 | 2⁷ = 128 | 0 – 127 | English, digits, punctuation |
| 2 | 5 + 6 = 11 | 2¹¹ = 2,048 | 128 – 2,047 | Greek, Arabic |
| 3 | 4 + 6 + 6 = 16 | 2¹⁶ = 65,536 | 2,048 – 65,535 | **Hindi, Korean, Chinese** |
| 4 | 3 + 6 + 6 + 6 = 21 | 2²¹ ≈ 2 million | 65,536 – 1,114,111* | **Emoji**, rare scripts |

\*Unicode deliberately stops at 1,114,111, even though 21 bits could go higher.

**Rule:** write the code point in binary, count the bits it needs, and use the **smallest pattern with enough free slots**. (Each range starts above the previous one's limit, because smaller numbers already fit in fewer bytes.)

## Step 6 — Worked example: storing `न` (2,344)

1. **Binary:** 2,344 = `100100101000` → 12 bits.
2. **Pick the pattern:** 1 byte has 7 slots (too small); 2 bytes have 11 slots, max 2,047 (too small); **3 bytes have 16 slots** ✓.
3. **Pad to 16 bits and split 4 + 6 + 6 into the slots:**

```
padded bits:      0000     100100     101000
pattern:      1110xxxx   10xxxxxx   10xxxxxx
filled:       11100000   10100100   10101000
decimal:         224        164        168
```

So `न` is stored as **224, 164, 168**.

**Reading it back (decoding):**
- First byte 224 = `11100000` starts with `1110` → "a 3-byte character".
- Next two bytes both start with `10` → continuations ✓.
- Strip the labels, join the rest: `0000` + `100100` + `101000` → 2,344 → **न** ✓

**And `h` (104):** `1101000` is 7 bits → fits the 1-byte pattern: `0` + `1101000` = `01101000` = 104. One byte, same number as its code point.

**Same idea for `안` (50,504)** — 16 bits → 3 bytes:
```
bits:       1100     010101     001000
filled:  11101100   10010101   10001000
decimal:    236        149        136
```

## Step 7 — Why English gets 1 byte

**History, not merit.** The 1960s American standard **ASCII (American Standard Code for Information Interchange)** numbered the English characters 0–127. Early computing was American, so English got the first, smallest numbers. UTF-8 (designed 1992) was built to be **backward compatible with ASCII**: those numbers stay **exactly the same, as single bytes**, so all existing English text and software kept working.

## Step 8 — Why waste bits on labels?

A 3-byte character uses only 16 of its 24 bits for the actual number. The trade-off buys **self-describing text**:
- A reader always knows where each character starts and ends.
- If you jump into the middle of a file, you can find the next character by skipping bytes that start with `10`.

## Step 9 — How the decoder detects broken text

Because every byte is labeled, broken characters are always detectable:

```
Valid:     224, 164, 168   → start("3 bytes") + cont + cont          ✓  न
Broken 1:  168 alone       → a continuation with no start            ✗  �
Broken 2:  224, 164, 97    → start promised 3 bytes, but 97 starts
                             with 0, so it's not a continuation      ✗  � then "a"
```

On a broken sequence, the decoder shows **`�`** (the Unicode **replacement character**) instead of crashing.

---

## Step 10 — Connecting it to LLMs

### When exactly does `�` appear?

**Setup.** LLM tokenizers start from UTF-8 bytes, then merge frequent byte pairs. If the tokenizer saw little Hindi, `न`'s 3 bytes might **not** be merged into one token. Say it learned:

```
Token A = [224, 164]     (these two bytes got merged)
Token B = [168]          (this byte stayed alone)
```

So **one letter = two tokens**. To write न, the model must output A then B.

**`�` appears at exactly one moment: when token IDs are decoded back into text and the bytes don't form complete, valid characters.** It never appears when encoding your input. Three situations cause it:

**1. The model picks a wrong token mid-character.** The model is just predicting "which token ID comes next" by probability — **it doesn't know the UTF-8 rules**; it has only *learned* that B usually follows A. Sampling is random, so occasionally it picks, say, the token for `a` (byte 97) after A:
```
bytes: 224, 164, 97   → letter started but never finished → �a
```

**2. Generation stops mid-character.** If the response hits its maximum length right after token A:
```
bytes: ..., 224, 164   [generation stops]   → ends with �
```

**3. Streaming shows a token before the rest of its character arrives.**
```
Token A arrives  → bytes 224, 164         → shows �         (temporary)
Token B arrives  → bytes 224, 164, 168    → � becomes न
```
Well-built apps avoid this by holding back incomplete bytes until the character finishes.

**When it never appears:** when a character is a **single token** (nothing to split — why you basically never see `�` in English), or when all of a character's tokens arrive correctly.

> ⚠️ **Common mistake:** "the model outputs a `�` token." It doesn't — `�` isn't in the vocabulary. The **model** produces an incomplete letter; the **decoder** (a separate step after the model) substitutes `�`.

**One-line rule:** *`�` = decoding produced an incomplete character — because a character's tokens got interrupted by a wrong token, an early stop, or displaying before the rest arrived.*

---

## Code to try (copy into Google Colab)

```python
for ch in ["h", "न", "안", "👋"]:
    b = ch.encode("utf-8")
    print(ch, ord(ch), list(b), [format(x, "08b") for x in b])

# h 104 [104] ['01101000']
# न 2344 [224, 164, 168] ['11100000', '10100100', '10101000']
# 안 50504 [236, 149, 136] ['11101100', '10010101', '10001000']
# 👋 128075 [240, 159, 145, 139] ['11110000', '10011111', '10010001', '10001011']

# Decoding a broken sequence:
print(bytes([224, 164, 97]).decode("utf-8", errors="replace"))   # '�a'
print(bytes([168]).decode("utf-8", errors="replace"))            # '�'
```

---

## Interview question for this file

**"Why do you sometimes see `�` in LLM output?"**

Model answer → [interview-question-bank.md](../interview-question-bank.md#5-tokenization)
