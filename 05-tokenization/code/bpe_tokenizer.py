"""
A complete Byte-Pair Encoding (BPE) tokenizer, written to be read.

Follows the notes in 03-bpe-algorithm.md and 04-regex-tiktoken-special-tokens.md.

HOW TO RUN (no installation needed for Parts 1-4):
    - Google Colab: paste this whole file into a cell and run it.
    - Or locally:   python bpe_tokenizer.py

Part 5 (regex pre-tokenization) needs the "regex" package:
    pip install regex
If it isn't installed, Part 5 is skipped automatically.

Coding level: read it top to bottom first (Level 1). Then change ONE thing
and re-run (Level 2), e.g. change VOCAB_SIZE, or the training text.
"""


# ---------------------------------------------------------------------------
# The two helper functions (notes: sections 3.3 and 3.4)
# ---------------------------------------------------------------------------

def get_stats(ids):
    """Count how often each adjacent pair appears.

    get_stats([5, 6, 6, 7, 5, 6]) -> {(5, 6): 2, (6, 6): 1, (6, 7): 1, (7, 5): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):          # every adjacent pair
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with the new token `idx`.

    merge([5, 6, 6, 7, 5, 6], (5, 6), 99) -> [99, 6, 7, 99]
    """
    newids = []
    i = 0
    while i < len(ids):
        # does the pair start at position i?  (the first check stops us
        # reading past the end of the list)
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2                          # skip both elements of the pair
        else:
            newids.append(ids[i])
            i += 1
    return newids


# ---------------------------------------------------------------------------
# The tokenizer (notes: sections 3.5 - 3.7)
# ---------------------------------------------------------------------------

class BasicTokenizer:
    """Byte-level BPE, exactly as in the notes: no regex, no special tokens."""

    def __init__(self):
        self.merges = {}                                   # (p0, p1) -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))                   # start from raw bytes

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:                                  # nothing left to merge
                break
            pair = max(stats, key=stats.get)               # most frequent pair
            idx = 256 + i                                  # next free id
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx                        # remember the merge
            # the new token's bytes = the two parts' bytes joined together
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1:2d}: {pair} -> {idx}  "
                      f"{self.vocab[idx]!r}  (seen {stats[pair]} times)")

    def decode(self, ids):
        """token ids -> text. Invalid UTF-8 becomes the replacement character."""
        raw = b"".join(self.vocab[idx] for idx in ids)
        return raw.decode("utf-8", errors="replace")

    def encode(self, text):
        """text -> token ids, applying merges in the order they were LEARNED."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # among the pairs present, the one learned earliest (lowest id);
            # pairs never learned get infinity
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                                      # no learned merge applies
            ids = merge(ids, pair, self.merges[pair])
        return ids


# ---------------------------------------------------------------------------
# Part 5: regex pre-tokenization (notes: section 4.2 - 4.3)
# ---------------------------------------------------------------------------

GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class RegexTokenizer(BasicTokenizer):
    """Same BPE, but text is first split into chunks; merges never cross chunks."""

    def __init__(self, pattern=GPT2_SPLIT_PATTERN):
        super().__init__()
        import regex                                       # pip install regex
        self.compiled = regex.compile(pattern)

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        # each chunk becomes its own list of bytes
        chunks = [list(c.encode("utf-8")) for c in self.compiled.findall(text)]

        for i in range(num_merges):
            stats = {}
            for chunk in chunks:                           # count pairs INSIDE chunks only
                for pair, n in get_stats(chunk).items():
                    stats[pair] = stats.get(pair, 0) + n
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = [merge(chunk, pair, idx) for chunk in chunks]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1:2d}: {pair} -> {idx}  {self.vocab[idx]!r}")

    def encode(self, text):
        ids = []
        for chunk in self.compiled.findall(text):         # encode each chunk separately...
            ids.extend(super().encode(chunk))              # ...then join in order
        return ids


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

TRAINING_TEXT = (
    "Tokenization is the step that turns text into the numbers a language model "
    "can process. The tokenizer starts from the bytes of the text and then merges "
    "the most frequent pairs of tokens into new tokens, again and again. Frequent "
    "words end up as single tokens, while rare words are split into smaller pieces. "
    "The tokenizer is trained before the model, on its own data, and then it is frozen. "
    "The model never sees the text itself; it only sees the token ids."
)

if __name__ == "__main__":

    print("=" * 70)
    print("PART 1: the Wikipedia example 'aaabdaaabac' (notes 3.1)")
    print("=" * 70)
    tok = BasicTokenizer()
    tok.train("aaabdaaabac", vocab_size=259, verbose=True)   # 3 merges
    ids = tok.encode("aaabdaaabac")
    print("encoded:", ids, f"-> length {len(ids)} (was 11)")
    print("decoded:", tok.decode(ids))
    # What you'll see:
    #   merge 1 is (97, 97) = 'aa'  -> exactly the 'Z' from the notes.
    #   merge 2 is (256, 97) = 'aaa', NOT 'ab' as in the hand example. Why?
    #   After merge 1 the sequence is  Z a b d Z a b a c.  Both (Z,a) and (a,b)
    #   appear twice: a TIE. The hand example broke the tie by picking 'ab';
    #   Python's max() picks whichever pair it counted first, which is (Z,a).
    #   Different tie-break, same algorithm. Both end at length 5.

    print()
    print("=" * 70)
    print("PART 2: train on a paragraph with 20 merges (notes 3.5)")
    print("=" * 70)
    tok = BasicTokenizer()
    tok.train(TRAINING_TEXT, vocab_size=276, verbose=True)
    original = len(TRAINING_TEXT.encode("utf-8"))
    compressed = len(tok.encode(TRAINING_TEXT))
    print(f"bytes: {original}  ->  tokens: {compressed}  "
          f"(compression {original / compressed:.2f}x)")

    print()
    print("=" * 70)
    print("PART 3: round trips (notes 3.7)")
    print("=" * 70)
    for s in ["the tokenizer", "hello 안녕 👋", "Tokens!!"]:
        assert tok.decode(tok.encode(s)) == s      # ALWAYS true
        print(f"decode(encode({s!r})) == original  ✓")
    # The other direction is NOT always true. ' t' (256) and 'h' (104) are
    # separate tokens, but this tokenizer also learned ' th' (259) as ONE token.
    ids = tok.encode(" t") + tok.encode("h")       # two tokens: [256, 104]
    text = tok.decode(ids)                         # ' th'
    print("encode(decode(ids)) can differ:", ids, "->", repr(text),
          "->", tok.encode(text), " (re-encoded as a single token)")

    print()
    print("=" * 70)
    print("PART 4: UTF-8 and the replacement character (notes file 02)")
    print("=" * 70)
    for ch in ["h", "न", "안", "👋"]:
        b = ch.encode("utf-8")
        print(f"{ch}: code point {ord(ch):>6}, {len(b)} byte(s): {list(b)}")
    print("bytes [224, 164, 168] ->", bytes([224, 164, 168]).decode("utf-8"))
    print("bytes [224, 164, 97]  ->", bytes([224, 164, 97]).decode("utf-8", errors="replace"),
          " (letter started, never finished)")
    print("byte  [168] alone     ->", bytes([168]).decode("utf-8", errors="replace"),
          " (continuation byte with no start)")

    print()
    print("=" * 70)
    print("PART 5: regex pre-tokenization (notes 4.2 - 4.4)")
    print("=" * 70)
    try:
        import regex
        pat = regex.compile(GPT2_SPLIT_PATTERN)
        for s in ["Hello've world123 how's are you!!!?", " how's", " HOW'S",
                  "Hello, world! It's 2024."]:
            print(f"{s!r:40} -> {pat.findall(s)}")
        rtok = RegexTokenizer()
        rtok.train(TRAINING_TEXT, vocab_size=276)
        ids = rtok.encode("The tokenizer merges pairs.")
        print("regex tokenizer pieces:", [rtok.decode([i]) for i in ids])
        print("(merges stayed inside chunks: notice '.' is its own piece, never glued to 's')")
    except ImportError:
        print("Skipped: run  pip install regex  to try this part.")
