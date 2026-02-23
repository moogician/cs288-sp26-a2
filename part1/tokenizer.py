from __future__ import annotations

import regex as re
from typing import Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab  # id -> bytes
        self.inverse_vocab = {v: k for k, v in vocab.items()}  # bytes -> id (also used as rank)
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

        # special token to ID mapping
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.inverse_vocab:
                self.special_token_ids[token] = self.inverse_vocab[token_bytes]

        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        tokens = [bytes([b]) for b in token_bytes]

        if len(tokens) <= 1:
            return tokens

        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            best_pair = None
            best_rank = float('inf')

            for pair in pairs:
                merged = pair[0] + pair[1]
                if merged in self.inverse_vocab:
                    rank = self.inverse_vocab[merged]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def _split_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        if not self.special_tokens_sorted:
            return [(text, False)] if text else []

        result = []
        remaining = text

        while remaining:
            earliest_pos = len(remaining)
            earliest_token = None

            for special in self.special_tokens_sorted:
                pos = remaining.find(special)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = special

            if earliest_token is None:
                if remaining:
                    result.append((remaining, False))
                break
            else:
                if earliest_pos > 0:
                    result.append((remaining[:earliest_pos], False))
                result.append((earliest_token, True))
                remaining = remaining[earliest_pos + len(earliest_token):]

        return result

    def _encode_chunk(self, text: str) -> list[int]:
        if not text:
            return []

        ids = []
        for match in self.pat.finditer(text):
            word = match.group()
            word_bytes = word.encode("utf-8")
            tokens = self._bpe(word_bytes)
            for token in tokens:
                if token in self.inverse_vocab:
                    ids.append(self.inverse_vocab[token])
                else:
                    for b in token:
                        ids.append(self.inverse_vocab[bytes([b])])

        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids = []

        parts = self._split_with_special_tokens(text)

        for part, is_special in parts:
            if is_special:
                ids.append(self.special_token_ids[part])
            else:
                ids.extend(self._encode_chunk(part))

        return ids

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""

        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            safe_end = self._find_safe_split_point(buffer)

            if safe_end > 0:
                to_process = buffer[:safe_end]
                buffer = buffer[safe_end:]

                for token_id in self.encode(to_process):
                    yield token_id

        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def _find_safe_split_point(self, text: str) -> int:
        if not text:
            return 0

        max_special_len = max((len(s) for s in self.special_tokens), default=0)

        min_keep = max_special_len - 1 if max_special_len > 0 else 0

        if len(text) <= min_keep:
            return 0

        safe_end = len(text)

        for special in self.special_tokens:
            for prefix_len in range(1, len(special)):
                prefix = special[:prefix_len]
                if text.endswith(prefix):
                    safe_end = min(safe_end, len(text) - prefix_len)

        if safe_end > 0:
            last_non_ws = safe_end - 1
            while last_non_ws >= 0 and text[last_non_ws].isspace():
                last_non_ws -= 1

            if last_non_ws >= 0 and last_non_ws < safe_end - 1:
                safe_end = last_non_ws + 1

        return safe_end


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    return Tokenizer(vocab, merges, special_tokens)
