from __future__ import annotations

import regex as re
from collections import Counter
from pathlib import Path
from typing import Iterator

GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)

def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs

def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    first, second = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    special_tokens = special_tokens or []

    if not special_tokens:
        for match in GPT2_PAT.finditer(text):
            yield match.group()
        return

    sorted_specials = sorted(special_tokens, key=len, reverse=True)

    import re as std_re
    special_pattern = "|".join(std_re.escape(s) for s in sorted_specials)
    split_pattern = f"({special_pattern})"

    parts = std_re.split(split_pattern, text)
    for part in parts:
        if part in special_tokens:
            continue
        elif part:
            for match in GPT2_PAT.finditer(part):
                yield match.group()

def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    forbidden_substrings = set()
    for special in special_tokens:
        special_bytes = special.encode("utf-8")
        for i in range(2, len(special_bytes) + 1):
            forbidden_substrings.add(special_bytes[:i])

    vocab: dict[int, bytes] = {}
    token_id = 0
    for special in special_tokens:
        vocab[token_id] = special.encode("utf-8")
        token_id += 1
    for byte_val in range(256):
        vocab[token_id] = bytes([byte_val])
        token_id += 1

    word_freqs: dict[tuple[bytes, ...], int] = {}
    for word_str in pre_tokenize(text, special_tokens):
        word_bytes = word_str.encode("utf-8")

        if any(forbidden in word_bytes for forbidden in forbidden_substrings):
            continue

        word_tuple = tuple(bytes([b]) for b in word_bytes)
        word_freqs[word_tuple] = word_freqs.get(word_tuple, 0) + 1

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for word, freq in word_freqs.items():
        for pair in get_pairs(word):
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        first, second = best_pair
        new_token = first + second

        vocab[token_id] = new_token
        token_id += 1
        merges.append(best_pair)

        affected_words = pair_to_words.pop(best_pair, set())

        for word in affected_words:
            freq = word_freqs[word]
            new_word = merge_word(word, best_pair)
            for pair in get_pairs(word):
                if pair in pair_counts:
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word)
            del word_freqs[word]
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq
            for pair in get_pairs(new_word):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(new_word)

    return vocab, merges
